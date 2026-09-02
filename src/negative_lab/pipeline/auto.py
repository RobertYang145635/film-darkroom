"""技术底座自动分析（v5 §6.1-§6.3）。

阶段（固定顺序）：

```
线性输入 -> 片基场（多 ROI 稳健估计，由调用方解析）-> 正密度 d
  -> 全局受约束 3x3 密度变换 A_global -> 每通道受约束单调曲线 f_c
  -> 技术端点映射 -> TechnicalPositive
```

本模块只负责**技术端点**（Dmax / offset / 黑位）与可选的中性参考采纳。
v5 §8.1：没有中性点、没有色卡、未知胶片型号都**不**是失败条件——
自动路径在没有任何中性参考时也必须给出可用结果，只记录来源与置信度。
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

from ..calibration.base import (base_roi_sanity_check, estimate_dmax_offset,
                                robust_dmin)
from ..calibration.border import detect_regions
from .density import channel_exponents, invert, transmittance_to_z
from .params import InvertParams

EPS = 1e-6


@dataclass
class AutoReport:
    params: InvertParams
    confidence: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    stages: list[str] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)


def default_base_strip(t_rgb: np.ndarray, frac: float = 0.06) -> np.ndarray:
    """无取样区时的启发式外框条带（低置信度，仅用于给出保守起点）。"""
    h, w, _ = t_rgb.shape
    b = max(4, int(min(h, w) * frac))
    return np.concatenate([
        t_rgb[:b, :, :].reshape(-1, 3),
        t_rgb[-b:, :, :].reshape(-1, 3),
        t_rgb[:, :b, :].reshape(-1, 3),
        t_rgb[:, -b:, :].reshape(-1, 3),
    ], axis=0)


def _z_at(t_rgb: np.ndarray, dmin: np.ndarray, points: list[tuple[int, int]],
          radius: int = 3) -> tuple[list[np.ndarray], list[float]]:
    """在给定 (row, col) 处取 (2r+1)² 中位数的 z 值与信噪权重。"""
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
                 prior: InvertParams | None = None,
                 resolved_dmin: tuple | list | None = None,
                 base_source: str | None = None) -> AutoReport:
    """技术端点自动分析。

    `resolved_dmin`：由 `pipeline.roll_analysis.resolve_base` 给出的片基
    （本帧 / 卷级 / 通用先验）。提供时**不再**本地重估 Dmin，来源如实记录。
    """
    t = np.asarray(t_rgb, dtype=np.float32)
    # `prior` is shared by batch/roll callers as an immutable starting point.
    # Never mutate it while estimating a frame: Dmax/offset and later model
    # assets are frame results, not edits to the caller's prior.  In-place
    # mutation here previously made the first processed frame change every
    # subsequent frame in a two-pass roll and also disabled the Dmax prior
    # upper-bound check below.
    rep = AutoReport(params=copy.deepcopy(prior) if prior is not None
                     else InvertParams())
    rep.stages.append("1 解码/线性化（调用方完成）")

    # ---- 2/3 片基
    if resolved_dmin is not None:
        rep.params.dmin = tuple(float(v) for v in resolved_dmin)
        rep.confidence["dmin"] = 1.0 if (base_source or "frame") == "frame" else 0.6
        rep.stages.append(f"2-3 片基：由卷级/帧级解析器提供（base_source={base_source}）")
    else:
        det = detect_regions(t)
        rep.warnings.extend(det["warnings"])
        if base_roi is None:
            if det.get("base") is not None:
                y0, x0, y1, x1 = det["base"]
                base_roi = t[y0:y1, x0:x1, :]
                rep.stages.append(
                    f"2 片基 ROI：自动检测（{det['edge']} 边，对比 {det['confidence']}）")
            else:
                base_roi = default_base_strip(t)
                rep.stages.append("2 片基 ROI：自动检测失败，退回外框启发式（低置信度）")
                rep.warnings.append(
                    "片基自动检测失败：已使用画面外框条带作为保守起点，"
                    "请在裁切页确认片基取样区")
        else:
            rep.stages.append("2 片基 ROI：用户提供")
        base = robust_dmin(base_roi)
        rep.warnings.extend(base["warnings"])
        rep.warnings.extend(base_roi_sanity_check(t, base_roi))
        rep.confidence["dmin"] = base["confidence"]
        rep.params.dmin = tuple(float(v) for v in base["dmin"])

    # ---- 4 Dmax / offset（限定在有效画面内，排除画框黑边）
    if film_roi is None:
        det2 = detect_regions(t)
        if det2.get("frame") is not None:
            fy0, fx0, fy1, fx1 = det2["frame"]
            film_roi = t[fy0:fy1, fx0:fx1, :]
    film = film_roi if film_roi is not None else t
    dm = estimate_dmax_offset(film, np.asarray(rep.params.dmin, dtype=np.float32),
                              black=rep.params.black)
    rep.warnings.extend(dm["warnings"])
    rep.confidence["dmax"] = dm["confidence"]
    rep.params.dmax = dm["dmax"]
    rep.params.offset = dm["offset"]
    # Dmax 是胶片属性而非单帧属性：单帧内容范围只会低估胶片能力，
    # 先验与单张估计取上界（不强行覆盖单张分析，只补足其盲区）。
    if prior is not None and prior.dmax != InvertParams().dmax \
            and rep.params.dmax < prior.dmax:
        rep.warnings.append(
            f"Dmax 采用胶片先验 {prior.dmax:.2f}（本帧内容范围估计 "
            f"{rep.params.dmax:.2f} 低于标定值，单帧不足以代表胶片动态范围）")
        rep.params.dmax = float(prior.dmax)

    # ---- 5 可选中性参考（用户主动取点才用；缺失时保持卷级起点，不阻断）
    zb, wb_ = _z_at(t, np.asarray(rep.params.dmin, dtype=np.float32),
                    neutral_bright_points or [])
    zd, wd_ = _z_at(t, np.asarray(rep.params.dmin, dtype=np.float32),
                    neutral_dark_points or [])
    if zb or zd:
        from ..calibration.neutral import (estimate_exponent_ratios,
                                           estimate_wb_high, estimate_wb_low)
        allz, allw = zb + zd, wb_ + wd_
        ex = estimate_exponent_ratios(allz, rep.params.reference_exponent,
                                      weights=allw)
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
        if zd and abs(rep.params.offset) >= 0.08:
            wl = estimate_wb_low(zd, rep.params.dmax, rep.params.offset,
                                 rep.params.wb_high, k, weights=wd_)
            rep.params.wb_low = tuple(wl["wb_low"])
            rep.confidence["wb_low"] = wl["confidence"]
            rep.warnings.extend(wl["warnings"])
        rep.stages.append("5 中性参考：用户取点已用于幂指数/白平衡估计")
    else:
        rep.confidence["neutral"] = None      # 不适用，不是失败
        rep.stages.append("5 中性参考：未提供（按卷级颜色起点继续，无需色卡）")

    # ---- 6-7 反相 + 诊断
    res = invert(t, rep.params)
    rep.diagnostics = res.diagnostics
    rep.warnings.extend(res.warnings)
    rep.stages.append("6-7 密度域技术反相与诊断完成")
    return rep
