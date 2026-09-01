"""统一质量门禁（v4 §7 P0-E 全语义）。

状态优先级固定：fail > needs_regions > needs_calibration > warn > ok。
- 彩色负片缺 verified 颜色标定：无条件 needs_calibration（与 confidence 无关）。
- 剪裁分级（有效 mask 内逐通道）：>0.1% warn；>1.0% fail（阈值入 to_dict）。
- valid_mask 形状/空校验：非法输入显式报错，不产生 NaN 比例。
- ΔE2000 只用 diagnostics/colorimetry.py 唯一实现；无参考 = None + notes。
- 色度统计（§7.5）：sRGB saturation p10/p50/p90、线性相对彩度 p50/p90；
  有同帧参考时记 p90 比率门禁（默认 0.75，可用 saturation_p90_ratio 调整）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .colorimetry import delta_e_2000, linear_srgb_to_lab

CHANNELS = ("R", "G", "B")
CLIP_TOL = 1e-6
CLIP_WARN_FRAC = 0.001          # v4 §7.2：>0.1% warn
CLIP_FAIL_FRAC = 0.01           # v4 §7.2：>1.0% fail
SAT_P90_RATIO_DEFAULT = 0.75    # v4 §7.5：同帧参考的 holdout 下限


@dataclass
class QualityGate:
    status: str = "ok"    # ok|warn|needs_regions|needs_calibration|fail
    input_clip: dict = field(default_factory=dict)
    output_clip: dict = field(default_factory=dict)
    nonfinite_input: int = 0
    nonfinite_output: int = 0
    delta_e2000: dict | None = None
    chroma_stats: dict = field(default_factory=dict)
    mask_source: str = "none"
    notes: list[str] = field(default_factory=list)
    thresholds: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    @classmethod
    def evaluate(cls, *, t_in: np.ndarray | None = None,
                 positive_linear: np.ndarray | None = None,
                 input_nonfinite: int = 0,
                 extra_warnings: list[str] | None = None,
                 calibration: dict | None = None,
                 valid_mask: np.ndarray | None = None,
                 mask_source: str = "user",
                 neutral_ref_positive: np.ndarray | None = None,
                 film_type: str = "unknown",
                 regions_ok: bool = True,
                 saturation_p90_ratio: float = SAT_P90_RATIO_DEFAULT,
                 clip_warn_frac: float = CLIP_WARN_FRAC,
                 clip_fail_frac: float = CLIP_FAIL_FRAC) -> "QualityGate":
        g = cls()
        g.thresholds = {"clip_warn_frac": clip_warn_frac,
                        "clip_fail_frac": clip_fail_frac,
                        "saturation_p90_ratio": saturation_p90_ratio}
        if valid_mask is not None:
            _validate_mask(valid_mask, positive_linear, t_in)
            g.mask_source = mask_source

        if t_in is not None:
            g.input_clip = _clip(t_in, valid_mask)
        if positive_linear is not None:
            g.output_clip = _clip(positive_linear, valid_mask)
            g.nonfinite_output = int((~np.isfinite(positive_linear)).sum())
            if g.nonfinite_output:
                g.failures.append(
                    f"输出含 {g.nonfinite_output} 个 NaN/Inf（禁止进入导出）")
            g.chroma_stats = _chroma_stats(positive_linear, valid_mask)
            # 剪裁分级（§7.2）
            for side, label, advice in (("low", "暗部", "建议降低 Dmax 或调整 offset"),
                                        ("high", "高光", "建议启用高光压缩")):
                for ch, frac in _iter(g.output_clip, side):
                    if frac > clip_fail_frac:
                        g.failures.append(
                            f"输出{label}剪裁超限(FAIL)：通道 {ch} {frac:.2%} > "
                            f"{clip_fail_frac:.1%}（{advice}）")
                    elif frac > clip_warn_frac:
                        g.warnings.append(
                            f"输出{label}剪裁：通道 {ch} {frac:.2%} > "
                            f"{clip_warn_frac:.1%}（{advice}）")
            for side in ("low", "high"):
                for ch, frac in _iter(g.input_clip or {}, side):
                    if frac > clip_fail_frac:
                        g.failures.append(
                            f"输入剪裁超限(FAIL)：通道 {ch} {side} {frac:.2%}")
                    elif frac > clip_warn_frac:
                        g.warnings.append(
                            f"输入剪裁：通道 {ch} {side} {frac:.2%}")
            # ΔE2000（有已知目标才测；§7.4.5）
            if neutral_ref_positive is not None:
                if neutral_ref_positive.shape != positive_linear.shape:
                    raise ValueError("neutral_ref_positive 形状必须与 "
                                     "positive_linear 一致")
                lab_r = linear_srgb_to_lab(neutral_ref_positive.reshape(-1, 3))
                lab_p = linear_srgb_to_lab(positive_linear.reshape(-1, 3))
                if valid_mask is not None:
                    lab_r, lab_p = lab_r[valid_mask.reshape(-1)], \
                        lab_p[valid_mask.reshape(-1)]
                de = delta_e_2000(lab_r, lab_p)
                g.delta_e2000 = {"mean": round(float(de.mean()), 3),
                                 "max": round(float(de.max()), 3),
                                 "p95": round(float(np.quantile(de, 0.95)), 3),
                                 "n": int(de.size)}
                if de.mean() > 5.0:
                    g.failures.append(
                        f"中性区 ΔE2000 均值 {de.mean():.1f} > 5（色偏不可接受）")
                elif de.mean() > 3.0:
                    g.warnings.append(
                        f"中性区 ΔE2000 均值 {de.mean():.1f} > 3（需重新取点/标定）")
            else:
                g.notes.append("无中性参考真值：ΔE2000 未测量（色彩验收不得凭此"
                               "宣称通过，v4 §7.4.5）")
            # 同帧参考的彩度比率门禁（§7.5）
            if neutral_ref_positive is not None:
                ref_s = _chroma_stats(neutral_ref_positive, valid_mask)
                g.chroma_stats["reference_sat_p90"] = ref_s.get("sat_p90")
                mine = g.chroma_stats.get("sat_p90", 0.0)
                theirs = ref_s.get("sat_p90", 0.0)
                if theirs and mine and (mine / theirs) < saturation_p90_ratio:
                    g.failures.append(
                        f"色度塌缩：sat_p90 比率 {mine/theirs:.2f} < "
                        f"{saturation_p90_ratio}（vs 同帧参考）")
        g.nonfinite_input = int(input_nonfinite)
        if g.nonfinite_input:
            g.warnings.append(f"输入含 {g.nonfinite_input} 个 NaN/Inf，已替换并记录")
        g.warnings.extend(extra_warnings or [])

        # ---- 状态优先级（§7.1）：fail > needs_regions > needs_calibration > warn > ok
        color_calibration = bool(calibration and calibration.get("level")) \
            if calibration is not None else True
        if film_type in ("bw", "b&w", "黑白", "monochrome"):
            color_calibration = True      # 黑白跳过颜色标定，仍跑剪裁/有限值门禁
        if g.failures:
            g.status = "fail"
        elif not regions_ok:
            g.status = "needs_regions"
        elif not color_calibration:
            # v4 §1.2.5：无条件 needs_calibration，confidence 不豁免
            g.status = "needs_calibration"
            g.warnings.insert(0, "NEEDS-CALIBRATION: 彩色负片缺少 verified 颜色标定"
                                 "（experimental，不得作为正式成品）")
        elif g.warnings:
            g.status = "warn"
        return g

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "input_clip": self.input_clip,
            "output_clip": self.output_clip,
            "nonfinite_input": self.nonfinite_input,
            "nonfinite_output": self.nonfinite_output,
            "delta_e2000": self.delta_e2000,
            "chroma_stats": self.chroma_stats,
            "mask_source": self.mask_source,
            "notes": self.notes,
            "thresholds": self.thresholds,
            "warnings": self.warnings,
            "failures": self.failures,
        }

    @property
    def clip_exceeded(self) -> bool:
        return any(f > CLIP_FAIL_FRAC
                   for _, f in _iter(self.output_clip, "low")) or \
            any(f > CLIP_FAIL_FRAC for _, f in _iter(self.output_clip, "high"))


def _validate_mask(mask: np.ndarray, *arrays) -> None:
    """v4 §3.3：形状不符/空/全 False 必须报错（不产生 NaN 比例）。"""
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
    """§7.5：sRGB saturation 分位 + 线性相对彩度分位。"""
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
    for ch, frac in zip(CHANNELS, clip.get(side, [])):
        yield ch, float(frac)
