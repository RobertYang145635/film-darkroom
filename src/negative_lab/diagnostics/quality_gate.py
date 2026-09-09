"""统一质量门禁（v5 §8 / §11.2）。

状态优先级固定：`fail > needs_review > warn > ok`。

v5 §8.1 明确删除的产品逻辑（不得再出现）：

- "彩色负片没有 verified calibration 就 needs_calibration"；
- "film_type=unknown 不得正式导出"；
- "请提供色卡"；
- "必须加载卷级校准文件"；
- "没有中性点不能完成转换"。

因此本门禁**不接受** calibration 参数，也不因缺色卡/缺标定/未知胶片型号
而降级。真正阻断导出的只有 §8.2 的硬失败（`blocks_export`）。

剪裁口径（§8.2，固定默认，逐通道、按最终 image_area_roi、**clamp 前**）：
技术正片低端或高端剪裁 >0.1% warn，>1.0% fail；输入端已发生的剪裁属于
采集事实，只 warn 并建议重扫；RenderIntent 造成的剪裁单独提示，
**不得反过来改写技术门禁**。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .colorimetry import delta_e_2000, linear_srgb_to_lab

CHANNELS = ("R", "G", "B")
CLIP_TOL = 1e-6
CLIP_WARN_FRAC = 0.001          # v5 §8.2：>0.1% warn
CLIP_FAIL_FRAC = 0.01           # v5 §8.2：>1.0% fail
QUALITY_RULES_VERSION = "v5-p0-frozen-2026-09-01"

#: 允许的用户可见门禁状态（v5 §8.1：没有 needs_calibration）
VALID_STATUS = ("ok", "warn", "needs_review", "fail")


@dataclass
class QualityGate:
    status: str = "ok"          # ok|warn|needs_review|fail
    VALID_STATUS = VALID_STATUS     # 类级常量（dataclass 非字段：无默认值注解）
    input_clip: dict = field(default_factory=dict)
    output_clip: dict = field(default_factory=dict)
    technical_preclip: dict = field(default_factory=dict)
    rendered_clip: dict = field(default_factory=dict)
    nonfinite_input: int = 0
    nonfinite_output: int = 0
    relative_delta_e2000: dict | None = None
    chroma_stats: dict = field(default_factory=dict)
    base_diagnostics: dict = field(default_factory=dict)
    mask_source: str = "none"
    review_reasons: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    thresholds: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    rules_version: str = QUALITY_RULES_VERSION

    # ------------------------------------------------------------ 评估
    @classmethod
    def evaluate(cls, *, t_in: np.ndarray | None = None,
                 positive_linear: np.ndarray | None = None,
                 technical_preclip: dict | None = None,
                 rendered_positive: np.ndarray | None = None,
                 rendered_clip: dict | None = None,
                 input_nonfinite: int = 0,
                 extra_warnings: list[str] | None = None,
                 valid_mask: np.ndarray | None = None,
                 mask_source: str = "image_area",
                 reference_positive: np.ndarray | None = None,
                 base_source: str = "frame",
                 base_confidence: float = 1.0,
                 candidate_confidence: float = 1.0,
                 crop_confidence: float = 1.0,
                 fingerprint_confidence: float = 1.0,
                 scene_flags: dict | None = None,
                 saturation_p90_ratio: float = 0.75,
                 clip_warn_frac: float = CLIP_WARN_FRAC,
                 clip_fail_frac: float = CLIP_FAIL_FRAC) -> "QualityGate":
        """评估技术正片质量。任何缺色卡/缺标定信息都不参与判定。"""
        g = cls()
        g.thresholds = {
            "clip_warn_frac": clip_warn_frac, "clip_fail_frac": clip_fail_frac,
            "saturation_p90_ratio": saturation_p90_ratio,
            "measurement": "逐通道、按最终 image_area_roi、clamp 前",
            "rules_version": QUALITY_RULES_VERSION,
        }
        if valid_mask is not None:
            _validate_mask(valid_mask, positive_linear, t_in)
            g.mask_source = mask_source

        # ---- 输入端剪裁：采集事实，只警告（v5 §8.2 末段）
        if t_in is not None:
            g.input_clip = _clip(t_in, valid_mask)
            for side in ("low", "high"):
                for ch, frac in _iter(g.input_clip, side):
                    if frac > clip_fail_frac:
                        g.warnings.append(
                            f"输入剪裁超限：通道 {ch} {side} {frac:.2%}"
                            "（扫描/翻拍端信息已丢失，算法无法恢复；如需根治请重扫）")
                    elif frac > clip_warn_frac:
                        g.warnings.append(f"输入剪裁：通道 {ch} {side} {frac:.2%}")

        if positive_linear is not None:
            g.output_clip = _clip(positive_linear, valid_mask)
            g.nonfinite_output = int((~np.isfinite(positive_linear)).sum())
            if g.nonfinite_output:
                g.failures.append(
                    f"技术正片含 {g.nonfinite_output} 个 NaN/Inf（禁止进入导出）")
            g.chroma_stats = _chroma_stats(positive_linear, valid_mask)

            # ---- 技术正片 clamp 前剪裁（§8.2 硬门禁口径）
            g.technical_preclip = dict(technical_preclip or {})
            for side, label, advice in (
                    ("low", "暗部", "请检查片基取样区或降低 Dmax"),
                    ("high", "高光", "请检查片基取样区或调整曝光起点")):
                for ch, frac in _iter(g.technical_preclip, side):
                    if frac > clip_fail_frac:
                        g.failures.append(
                            f"技术正片{label}剪裁超限(FAIL)：通道 {ch} {frac:.2%} > "
                            f"{clip_fail_frac:.1%}（{advice}）")
                    elif frac > clip_warn_frac:
                        g.warnings.append(
                            f"技术正片{label}剪裁：通道 {ch} {frac:.2%} > "
                            f"{clip_warn_frac:.1%}（{advice}）")
            above = float(g.technical_preclip.get("above_base", 0.0))
            if above > 0.0:
                g.notes.append(f"透射率超过片基的像素比例 {above:.2%}"
                               "（超基/漏光/齿孔，已保留诊断，不静默吞掉）")

            # ---- 相对参考输出差异（v5 §11.3：只能命名为"相对参考"）
            if reference_positive is not None:
                if reference_positive.shape != positive_linear.shape:
                    raise ValueError("reference_positive 形状必须与 "
                                     "positive_linear 一致")
                lab_r = linear_srgb_to_lab(reference_positive.reshape(-1, 3))
                lab_p = linear_srgb_to_lab(positive_linear.reshape(-1, 3))
                if valid_mask is not None:
                    lab_r = lab_r[valid_mask.reshape(-1)]
                    lab_p = lab_p[valid_mask.reshape(-1)]
                de = delta_e_2000(lab_r, lab_p)
                g.relative_delta_e2000 = {
                    "mean": round(float(de.mean()), 3),
                    "median": round(float(np.median(de)), 3),
                    "p95": round(float(np.quantile(de, 0.95)), 3),
                    "n": int(de.size),
                    "label": "相对参考输出差异（非绝对色准，v5 §11.3）"}
                ref_s = _chroma_stats(reference_positive, valid_mask)
                g.chroma_stats["reference_sat_p90"] = ref_s.get("sat_p90")
                mine = g.chroma_stats.get("sat_p90", 0.0)
                theirs = ref_s.get("sat_p90", 0.0)
                if theirs and mine and (mine / theirs) < saturation_p90_ratio:
                    g.failures.append(
                        f"色度塌缩：sat_p90 比率 {mine / theirs:.2f} < "
                        f"{saturation_p90_ratio}（vs 同源参考）")
            else:
                g.notes.append("无同源参考输出：未计算相对参考差异"
                               "（色彩验收不得凭此宣称通过）")

        # ---- RenderIntent 剪裁：单独提示，不改写技术门禁（§8.2）
        if rendered_clip is None and rendered_positive is not None:
            from ..metrics import channel_clip_ratios   # 叶子（上轮声称已改，实未落地）
            rendered_clip = channel_clip_ratios(rendered_positive, valid_mask)
        if rendered_clip:
            g.rendered_clip = {k: ([float(v) for v in val] if isinstance(val, list)
                                    else val)
                               for k, val in rendered_clip.items()
                               if isinstance(val, list)}
            hot = max(g.rendered_clip.get("high", [0]) or [0])
            dark = max(g.rendered_clip.get("low", [0]) or [0])
            if hot > clip_warn_frac or dark > clip_warn_frac:
                g.warnings.append(
                    f"微调后输出剪裁：高光 {hot:.2%} / 阴影 {dark:.2%}"
                    "（由用户微调产生，不影响技术门禁；请查看高光/阴影警告）")
        if rendered_positive is not None:
            bad = int((~np.isfinite(rendered_positive)).sum())
            if bad:
                g.failures.append(f"渲染正片含 {bad} 个 NaN/Inf（禁止进入导出）")

        g.nonfinite_input = int(input_nonfinite)
        if g.nonfinite_input:
            g.warnings.append(
                f"输入含 {g.nonfinite_input} 个 NaN/Inf，已替换并记录")
        g.warnings.extend(extra_warnings or [])

        # ---- 低置信度 -> needs_review（允许导出，v5 §8.3）
        g.review_reasons = _review_reasons(
            base_source=base_source, base_confidence=base_confidence,
            candidate_confidence=candidate_confidence,
            crop_confidence=crop_confidence,
            fingerprint_confidence=fingerprint_confidence,
            scene_flags=scene_flags or {})

        # ---- 状态优先级：fail > needs_review > warn > ok
        if g.failures:
            g.status = "fail"
        elif g.review_reasons:
            g.status = "needs_review"
        elif g.warnings:
            g.status = "warn"
        return g

    # ------------------------------------------------------------ 语义
    @property
    def blocks_export(self) -> bool:
        """v5 §8.2：只有硬失败阻断当前帧导出。"""
        return self.status == "fail"

    @property
    def clip_exceeded(self) -> bool:
        src = self.technical_preclip or self.output_clip
        return (any(f > CLIP_FAIL_FRAC for _, f in _iter(src, "low"))
                or any(f > CLIP_FAIL_FRAC for _, f in _iter(src, "high")))

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "blocks_export": self.blocks_export,
            "input_clip": self.input_clip,
            "output_clip": self.output_clip,
            "technical_preclip": self.technical_preclip,
            "rendered_clip": self.rendered_clip,
            "nonfinite_input": self.nonfinite_input,
            "nonfinite_output": self.nonfinite_output,
            "relative_delta_e2000": self.relative_delta_e2000,
            "chroma_stats": self.chroma_stats,
            "base_diagnostics": self.base_diagnostics,
            "mask_source": self.mask_source,
            "review_reasons": self.review_reasons,
            "notes": self.notes,
            "thresholds": self.thresholds,
            "warnings": self.warnings,
            "failures": self.failures,
            "rules_version": self.rules_version,
        }


def _review_reasons(*, base_source: str, base_confidence: float,
                    candidate_confidence: float, crop_confidence: float,
                    fingerprint_confidence: float,
                    scene_flags: dict) -> list[str]:
    """v5 §8.3：每条提示必须说明软件实际采用了什么回退。"""
    out: list[str] = []
    if base_source == "generic_prior":
        out.append("片基：本帧与本卷都不可用，已采用通用彩色负片先验（非实测 Dmin）")
    elif base_source in ("roll", "roll_partial"):
        out.append("片基：本帧无可靠取样，已采用同卷其他帧的卷级片基模型")
    elif base_confidence < 0.5:
        out.append(f"片基：取样置信度低（{base_confidence:.2f}），已采用稳健回退")
    if candidate_confidence < 0.5:
        out.append(f"颜色起点：自动估计置信度低（{candidate_confidence:.2f}），"
                   "已回退到保留氛围候选，不做自动中和")
    if crop_confidence < 0.5:
        out.append(f"裁切：自动检测置信度低（{crop_confidence:.2f}），"
                   "请在裁切页确认三类区域")
    if fingerprint_confidence < 0.5:
        out.append("采集信息：metadata 不足，采集指纹不完整（不影响转换）")
    for flag, text in (("night", "夜景/低照度"), ("monochrome_light", "单色光源"),
                       ("sunset", "夕阳/暖光"), ("high_saturation", "高饱和主导"),
                       ("large_white", "大面积白色"), ("large_green", "大面积绿色")):
        if scene_flags.get(flag):
            out.append(f"场景：疑似{text}，自动中和已降权（保留氛围优先）")
    return out


def _validate_mask(mask: np.ndarray, *arrays) -> None:
    """形状不符/空/全 False 必须报错（不产生 NaN 比例）。"""
    if mask.size == 0:
        raise ValueError("valid_mask 为空")
    if not mask.any():
        raise ValueError("valid_mask 全 False：没有有效像素参与统计")
    for a in arrays:
        if a is not None and mask.shape != a.shape[:2]:
            raise ValueError(f"valid_mask 形状 {mask.shape} 与图像 "
                             f"{a.shape[:2]} 不符")


def _clip(img: np.ndarray, mask: np.ndarray | None = None) -> dict:
    if mask is not None:
        px = img[mask]
        lo = (px <= CLIP_TOL).mean(axis=0)
        hi = (px >= 1.0 - CLIP_TOL).mean(axis=0)
    else:
        lo = (img <= CLIP_TOL).mean(axis=tuple(range(img.ndim - 1)))
        hi = (img >= 1.0 - CLIP_TOL).mean(axis=tuple(range(img.ndim - 1)))
    return {"low": [float(v) for v in lo], "high": [float(v) for v in hi]}


def _chroma_stats(pos: np.ndarray, mask: np.ndarray | None) -> dict:
    """v5 §11.2：彩度分位统计（sRGB 显示域 + 线性相对彩度）。"""
    px = pos[mask] if mask is not None else pos.reshape(-1, 3)
    px = px[np.isfinite(px).all(axis=1)]
    if len(px) == 0:
        return {}
    from ..color.management import linear_to_srgb
    s = np.clip(linear_to_srgb(np.clip(px, 0.0, 1.0)), 0.0, 1.0)
    sat = (s.max(axis=1) - s.min(axis=1)) / np.maximum(s.mean(axis=1), 1e-6)
    mx, mn = px.max(axis=1), px.min(axis=1)
    chroma = (mx - mn) / np.maximum(px.mean(axis=1), 1e-6)
    return {"sat_p10": round(float(np.quantile(sat, 0.10)), 4),
            "sat_p50": round(float(np.quantile(sat, 0.50)), 4),
            "sat_p90": round(float(np.quantile(sat, 0.90)), 4),
            "chroma_p50": round(float(np.quantile(chroma, 0.50)), 4),
            "chroma_p90": round(float(np.quantile(chroma, 0.90)), 4)}


def _iter(clip: dict, side: str):
    for ch, frac in zip(CHANNELS, (clip or {}).get(side, []) or []):
        yield ch, float(frac)
