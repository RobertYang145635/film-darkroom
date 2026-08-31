"""自动分析编排（规格 §4.5 顺序）。

阶段：解码/线性化（调用方）→ 片基取样 → Dmin+置信度 → Dmax/offset+输入剪裁
→ 中性参考点（幂指数比例、阴影色偏、高光白平衡）→ 组装参数 → 反相 → 诊断。
任何低置信度项都必须出现在 warnings 中，不得静默采用默认值（§4.5.8）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..calibration.base import base_roi_sanity_check, estimate_dmax_offset, robust_dmin
from ..calibration.border import detect_regions
from ..calibration.neutral import (estimate_exponent_ratios, estimate_wb_high,
                                   estimate_wb_low)
from .density import channel_exponents, invert, transmittance_to_z
from .params import InvertParams


@dataclass
class AutoReport:
    params: InvertParams
    confidence: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    stages: list[str] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)


def default_base_strip(t_rgb: np.ndarray, frac: float = 0.06) -> np.ndarray:
    """无用户 ROI 时的启发式：取画面外框条带作为候选未曝光片基
    （用户决策：默认要求画面含未曝光片基边缘）。Phase 2 提供可拖动修正。"""
    h, w, _ = t_rgb.shape
    b = max(4, int(min(h, w) * frac))
    strip = np.concatenate([
        t_rgb[:b, :, :].reshape(-1, 3),
        t_rgb[-b:, :, :].reshape(-1, 3),
        t_rgb[:, :b, :].reshape(-1, 3),
        t_rgb[:, -b:, :].reshape(-1, 3),
    ], axis=0)
    return strip


def _z_at(t_rgb: np.ndarray, dmin: np.ndarray, points: list[tuple[int, int]],
          radius: int = 3) -> tuple[list[np.ndarray], list[float]]:
    """在给定 (row, col) 处取 (2r+1)² 中位数的 z 值与信噪权重。

    权重 = (t_mean·|z_mean|)²：z 域噪声 σ_z ≈ σ_t/(t·ln10)，
    透射率低（亮场景/高密度）的样本对通道比例约束弱，应降权。
    """
    zs, ws = [], []
    h, w, _ = t_rgb.shape
    for (r, c) in points:
        r0, r1 = max(0, r - radius), min(h, r + radius + 1)
        c0, c1 = max(0, c - radius), min(w, c + radius + 1)
        patch = t_rgb[r0:r1, c0:c1, :].reshape(-1, 3)
        med = np.median(patch, axis=0)
        z = transmittance_to_z(med[None, None, :], dmin)[0, 0, :].astype(np.float64)
        zs.append(z)
        ws.append(float((max(med.mean(), 0.0) * max(np.abs(z).mean(), 1e-6)) ** 2))
    return zs, ws


def auto_analyze(t_rgb: np.ndarray,
                 base_roi: np.ndarray | None = None,
                 film_roi: np.ndarray | None = None,
                 neutral_bright_points: list[tuple[int, int]] | None = None,
                 neutral_dark_points: list[tuple[int, int]] | None = None,
                 prior: InvertParams | None = None) -> AutoReport:
    """执行 §4.5 自动分析。prior（胶片/设备配置）只作为初始值，
    单张分析结果不被其强行覆盖（用户决策：批处理策略）。"""
    t = np.asarray(t_rgb, dtype=np.float32)
    rep = AutoReport(params=prior or InvertParams())
    rep.stages.append("1 解码/线性化（调用方完成）")

    # 2 片基取样：用户 ROI 优先；否则自动帧/片基检测（真实翻拍外圈是黑边，
    # 直接取画面外框会命中黑边——见 docs/algorithm-notes.md），再兜底外框启发式
    # 帧检测总是执行：即使用户给了片基 ROI，Dmax 统计也要限定在帧内，
    # 否则画框黑边（占真实翻拍 ~4%）会污染暗部分位。
    det = detect_regions(t)
    rep.warnings.extend(det["warnings"])
    if base_roi is None:
        if det["base"] is not None:
            y0, x0, y1, x1 = det["base"]
            base_roi = t[y0:y1, x0:x1, :]
            rep.stages.append(
                f"2 片基 ROI：自动检测（{det['edge']} 边，对比 {det['confidence']}）")
        else:
            base_roi = default_base_strip(t)
            rep.stages.append("2 片基 ROI：自动检测失败，退回外框启发式（低置信度）")
            rep.warnings.append("片基自动检测失败，结果置信度低，请手动指定片基框")
    else:
        rep.stages.append("2 片基 ROI：用户提供")

    # 3 Dmin + 置信度
    base = robust_dmin(base_roi)
    rep.warnings.extend(base["warnings"])
    rep.warnings.extend(base_roi_sanity_check(t, base_roi))
    rep.confidence["dmin"] = base["confidence"]
    if base["confidence"] == 0.0 and prior is not None and prior.dmin != InvertParams().dmin:
        # 片基检测失败但配置文件提供了先验：用先验并如实标记（§4.6.1 兜底路径）
        rep.warnings.append("片基检测失败，使用胶片配置文件的片基先验（低置信度）")
        rep.confidence["dmin"] = 0.0
    else:
        # v3 F-ALG-3：Dmin 只能来自片基本底。禁止用场景内容（画面内
        # p99.5/p99.9）"补亮" 物理基准——那会把场景高光、齿孔漏光、
        # 灯箱和构图色注入色罩估计。暗角/导光不均的正当解法是
        # blank-light flat-field（v3 P0-C），不是改写 Dmin。
        rep.params.dmin = tuple(float(v) for v in base["dmin"])

    # 4 Dmax / offset / 输入剪裁风险（帧检测成功时只用帧内区域，
    # 排除画框黑边——黑边透射率≈0 会虚假抬高 Dmax）
    if film_roi is None and det.get("frame") is not None:
        fy0, fx0, fy1, fx1 = det["frame"]
        film_roi = t[fy0:fy1, fx0:fx1, :]
    film = film_roi if film_roi is not None else t
    dm = estimate_dmax_offset(film, np.asarray(rep.params.dmin, dtype=np.float32),
                              black=rep.params.black)
    rep.warnings.extend(dm["warnings"])
    rep.confidence["dmax"] = dm["confidence"]
    rep.params.dmax = dm["dmax"]
    rep.params.offset = dm["offset"]
    # Dmax 是胶片属性而非单帧属性：单帧内容范围只会低估胶片能力
    # （亮场景帧估 0.80 vs 标定卷先验 1.36 -> 高光冲淡，实测教训）。
    # 先验与单张估计取上界——不强行覆盖单张分析，只补足其盲区。
    if prior is not None and prior.dmax != InvertParams().dmax \
            and rep.params.dmax < prior.dmax:
        rep.warnings.append(
            f"Dmax 采用胶片先验 {prior.dmax:.2f}（本帧内容范围估计 "
            f"{rep.params.dmax:.2f} 低于标定值，单帧不足以代表胶片动态范围）")
        rep.params.dmax = float(prior.dmax)

    # 5 中性参考点（若有）
    zb, wb_ = _z_at(t, np.asarray(rep.params.dmin, dtype=np.float32),
                    neutral_bright_points or [])
    zd, wd_ = _z_at(t, np.asarray(rep.params.dmin, dtype=np.float32),
                    neutral_dark_points or [])
    allz = zb + zd
    allw = wb_ + wd_
    if allz:
        ex = estimate_exponent_ratios(allz, rep.params.reference_exponent, weights=allw)
        rep.params.red_ratio = ex["red_ratio"]
        rep.params.blue_ratio = ex["blue_ratio"]
        rep.confidence["exponent"] = ex["confidence"]
        rep.warnings.extend(ex["warnings"])
        k = channel_exponents(rep.params)
        if zb:
            wh = estimate_wb_high(zb, rep.params.dmax, rep.params.offset,
                                  rep.params.wb_low, k, weights=wb_)
            rep.params.wb_high = tuple(wh["wb_high"])
            rep.confidence["wb_high"] = wh["confidence"]
            rep.warnings.extend(wh["warnings"])
        if zd:
            # v3 P0-E 可识别性：wb_low 只通过 offset_c=wb_high*offset*wb_low 起作用，
            # |offset| 过小时该自由度不可辨识（除以近零 -> 病态解，实测 wb_low
            # 曾发散到 3.57/0.05）。锁定并降置信度。
            if abs(rep.params.offset) >= 0.08:
                wl = estimate_wb_low(zd, rep.params.dmax, rep.params.offset,
                                     rep.params.wb_high, k, weights=wd_)
                rep.params.wb_low = tuple(wl["wb_low"])
                rep.confidence["wb_low"] = wl["confidence"]
                rep.warnings.extend(wl["warnings"])
            else:
                rep.confidence["wb_low"] = 0.0
                rep.warnings.append(
                    f"offset={rep.params.offset:.3f} 过小，阴影色偏(wb_low)不可辨识，"
                    "已锁定 1.0（v3 P0-E 可识别性约束）")
        rep.stages.append("5 中性参考点：已用于幂指数/白平衡估计")
    else:
        # v3 F-ALG-4/P0-E：统计 gray-world 是自证循环（用当前参数选"看起来
        # 中性"的像素再改同一组参数），不得自动改写正式通道参数。
        # 只能作为"建议"呈现给用户，由用户确认后以 force_params 方式采纳。
        from ..calibration.neutral_stats import find_neutral_z_samples
        ns = find_neutral_z_samples(t, rep.params)
        if ns["samples"]:
            est = estimate_exponent_ratios(ns["samples"], weights=ns["weights"]) \
                if len(ns["samples"]) >= 2 else None
            rep.confidence["neutral_statistical"] = 0.0   # 建议，非标定
            rep.warnings.append(
                "无 verified 中性参考：统计中性建议已生成但【未应用】"
                f"（建议 ratios R/B="
                f"{est['red_ratio']:.3f}/{est['blue_ratio']:.3f} 仅供参考，"
                "请人工取点或提供色卡）" if est else
                "无中性参考点：仅使用片基基准（§4.6.1）；请取亮/暗中性点")
        else:
            rep.confidence["neutral"] = 0.0
            rep.warnings.append("无中性参考点：仅使用片基基准（标记“无中性参考”，§4.6.1）；"
                                "如结果异常请点击一个亮部与一个暗部中性区域（§4.6.2）")
        rep.stages.append("5 中性参考点：缺失（统计建议不自动应用，v3 P0-E）")

    # 6-7 反相 + 诊断
    res = invert(t, rep.params)
    rep.diagnostics = res.diagnostics
    rep.warnings.extend(res.warnings)
    rep.stages.append("6-7 密度域反相与诊断完成")
    rep.stages.append("8 预览生成（低置信度项已全部列出）")
    return rep
