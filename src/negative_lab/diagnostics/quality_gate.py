"""统一质量门禁（整改 P0-C；F-4）。

GUI、CLI、批处理、sidecar、报告共用同一结果结构；状态来自枚举，
禁止靠中文字符串包含关系猜状态（整改 P1 批处理同样遵守）。

规则（原规格 §8.4 + 整改 §2）：
- 输入/输出逐通道低端与高端剪裁比例；非蒙版输出 >1% 即 WARN，
  警告文本必须包含通道、方向与比例。
- 任何 NaN/Inf 进入导出 -> FAIL。
- 正常曝光留出样片 FAIL/WARN 会阻断发布（tests/regression）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

CHANNELS = ("R", "G", "B")
CLIP_TOL = 1e-6
CLIP_WARN_FRAC = 0.01          # 1% 目标（原规格 §8.4）


@dataclass
class QualityGate:
    status: str = "ok"                       # ok|warn|fail|needs_calibration
    input_clip: dict = field(default_factory=dict)    # {low:[r,g,b], high:[r,g,b]}
    output_clip: dict = field(default_factory=dict)
    nonfinite_input: int = 0
    nonfinite_output: int = 0
    delta_e2000: dict | None = None          # {mean,max,p95,n}（v3 P0-G）
    mask_source: str = "none"                # none|user|dust|roi
    notes: list[str] = field(default_factory=list)   # 信息性（不降级状态）
    warnings: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    # ---- 构建

    @classmethod
    def evaluate(cls, *, t_in: np.ndarray | None = None,
                 positive_linear: np.ndarray | None = None,
                 input_nonfinite: int = 0,
                 extra_warnings: list[str] | None = None,
                 clip_warn_frac: float = CLIP_WARN_FRAC,
                 calibration: dict | None = None,
                 valid_mask: np.ndarray | None = None,
                 mask_source: str = "user",
                 neutral_ref_positive: np.ndarray | None = None,
                 chroma_collapse_frac: float = 0.02) -> "QualityGate":
        """calibration（v3 P0-G）：{"level": verified-calibration 名 或 None,
        "confidence": float}。彩色负片缺卷级标定/中性参考时
        status=needs_calibration（不是 warn，v3 F-ALG-7）。
        valid_mask：True=参与统计的画面区域（蒙版区不伪装成正常）。
        neutral_ref_positive：与 positive_linear 同位置的中性参考真值块
        （色卡/verified ROI），用于 ΔE2000；缺省时如实记录 None。"""
        g = cls()
        if t_in is not None:
            g.input_clip = _clip(t_in, valid_mask)
        if positive_linear is not None:
            g.output_clip = _clip(positive_linear, valid_mask)
            g.nonfinite_output = int((~np.isfinite(positive_linear)).sum())
            if g.nonfinite_output:
                g.failures.append(
                    f"输出含 {g.nonfinite_output} 个 NaN/Inf（禁止进入导出）")
            if valid_mask is not None:
                g.mask_source = mask_source
            # 色度塌缩检查（v3 §0 失败模式）：仅在色彩被评估时启用
            # （有中性参考真值，或调用方给出标定层级）——纯灰度合成
            # 测试图不得被误判。
            color_assessed = neutral_ref_positive is not None or (
                calibration is not None and calibration.get("level"))
            if color_assessed and positive_linear.shape[-1] == 3:
                px = positive_linear[valid_mask] if valid_mask is not None \
                    else positive_linear.reshape(-1, 3)
                px = px[np.isfinite(px).all(axis=1)]
                if len(px) > 256:
                    mx, mn = px.max(axis=1), px.min(axis=1)
                    lum = np.maximum(px.mean(axis=1), 1e-6)
                    chroma = np.median((mx - mn) / lum)
                    if chroma < chroma_collapse_frac:
                        g.failures.append(
                            f"色度塌缩：中位彩度 {chroma:.3f} < {chroma_collapse_frac}"
                            "（输出接近灰度，去色罩失败）")
            # ΔE2000（有中性参考真值时）
            if neutral_ref_positive is not None:
                de = _delta_e2000_stats(neutral_ref_positive, positive_linear,
                                        valid_mask)
                g.delta_e2000 = de
                if de["mean"] > 5.0:
                    g.warnings.append(
                        f"中性区 ΔE2000 均值 {de['mean']:.1f} > 5（色偏可见，"
                        "需重新取点或标定）")
            elif calibration is None or not calibration.get("level"):
                g.notes.append(
                    "无中性参考真值：ΔE2000 未测量（色彩验收不得凭无参考结果宣称通过，"
                    "v3 P0-G）")
        g.nonfinite_input = int(input_nonfinite)
        if g.nonfinite_input:
            g.warnings.append(f"输入含 {g.nonfinite_input} 个 NaN/Inf，已替换为 0 并记录")
        for ch, frac in _iter_over(g.output_clip, "low"):
            if frac > clip_warn_frac:
                g.warnings.append(
                    f"输出暗部剪裁超限：通道 {ch} {frac:.2%} > {clip_warn_frac:.0%}"
                    f"（阴影细节丢失，建议降低 Dmax 或调整 offset）")
        for ch, frac in _iter_over(g.output_clip, "high"):
            if frac > clip_warn_frac:
                g.warnings.append(
                    f"输出高光剪裁超限：通道 {ch} {frac:.2%} > {clip_warn_frac:.0%}"
                    f"（高光细节丢失，建议启用高光压缩或提高 Dmax）")
        for ch, frac in _iter_over(g.input_clip, "low"):
            if frac > clip_warn_frac:
                g.warnings.append(
                    f"输入暗部剪裁：通道 {ch} {frac:.2%}（透射率趋零，密度信息可能丢失）")
        for ch, frac in _iter_over(g.input_clip, "high"):
            if frac > clip_warn_frac:
                g.warnings.append(
                    f"输入亮部剪裁：通道 {ch} {frac:.2%}（扫描饱和）")
        g.warnings.extend(extra_warnings or [])
        if g.failures:
            g.status = "fail"
        elif calibration is not None and not calibration.get("level") \
                and float(calibration.get("confidence", 0)) < 0.5:
            # v3：自动彩色负片缺有效标定 -> needs_calibration（阻断"成功"语义）
            g.status = "needs_calibration"
            g.warnings.insert(0,
                "NEEDS-CALIBRATION: 彩色负片自动结果缺少 verified 卷级标定/中性参考"
                "（experimental，勿作为正式成品；请提供色卡帧或人工取点）")
        elif g.warnings:
            g.status = "warn"
        return g

    # ---- 消费

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "input_clip": self.input_clip,
            "output_clip": self.output_clip,
            "nonfinite_input": self.nonfinite_input,
            "nonfinite_output": self.nonfinite_output,
            "delta_e2000": self.delta_e2000,
            "mask_source": self.mask_source,
            "notes": self.notes,
            "warnings": self.warnings,
            "failures": self.failures,
            "clip_warn_frac": CLIP_WARN_FRAC,
        }

    @property
    def clip_exceeded(self) -> bool:
        return any(f > CLIP_WARN_FRAC for _, f in _iter_over(self.output_clip, "low")) \
            or any(f > CLIP_WARN_FRAC for _, f in _iter_over(self.output_clip, "high"))


def _clip(img: np.ndarray, mask: np.ndarray | None = None) -> dict:
    if mask is not None:
        px = img[mask]
        lo = (px <= CLIP_TOL).mean(axis=0)
        hi = (px >= 1.0 - CLIP_TOL).mean(axis=0)
    else:
        lo = (img <= CLIP_TOL).mean(axis=tuple(range(img.ndim - 1)))
        hi = (img >= 1.0 - CLIP_TOL).mean(axis=tuple(range(img.ndim - 1)))
    return {"low": [float(v) for v in lo], "high": [float(v) for v in hi]}


def _linear_to_lab(lin: np.ndarray) -> np.ndarray:
    """linear sRGB(0-1) -> CIE Lab D65（sRGB 标准传递函数与 primaries）。"""
    from ..color.management import linear_to_srgb
    s = linear_to_srgb(np.clip(lin, 0.0, 1.0))
    M = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]], np.float64)
    xyz = s @ M.T
    wp = np.array([0.95047, 1.0, 1.08883])
    v = xyz / wp
    eps, kap = 216 / 24389, 24 / 116
    f = np.where(v > eps, np.cbrt(v), kap + v * (7.787))
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


def _delta_e2000_stats(ref_lin: np.ndarray, pos_lin: np.ndarray,
                       mask: np.ndarray | None) -> dict:
    """ΔE2000（Sharma 2005 公开公式）参考中性块 vs 输出同位置。"""
    lab_r = _linear_to_lab(np.asarray(ref_lin, np.float64))
    lab_p = _linear_to_lab(np.asarray(pos_lin, np.float64))
    if mask is not None:
        lab_r, lab_p = lab_r[mask], lab_p[mask]
    else:
        lab_r, lab_p = lab_r.reshape(-1, 3), lab_p.reshape(-1, 3)
    de = _de2000(lab_r[:, 0], lab_r[:, 1], lab_r[:, 2],
                 lab_p[:, 0], lab_p[:, 1], lab_p[:, 2])
    return {"mean": round(float(de.mean()), 3),
            "max": round(float(de.max()), 3),
            "p95": round(float(np.quantile(de, 0.95)), 3),
            "n": int(de.size)}


def _de2000(L1, a1, b1, L2, a2, b2):
    kL = kC = kH = 1.0
    C1 = np.hypot(a1, b1); C2 = np.hypot(a2, b2)
    Cb = (C1 + C2) / 2
    G = 0.5 * (1 - np.sqrt(Cb**7 / (Cb**7 + 25.0**7))) if np.any(Cb > 0) else 0.5
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360
    dLp = L2 - L1
    dCp = C2p - C1p
    with np.errstate(invalid="ignore"):
        dhp = np.where(C1p * C2p == 0, 0.0,
                       np.where(np.abs(h2p - h1p) <= 180, h2p - h1p,
                                np.where(h2p - h1p > 180, h2p - h1p - 360,
                                         h2p - h1p + 360)))
    dHp = 2 * np.sqrt(np.maximum(C1p * C2p, 0)) * np.sin(np.radians(dhp / 2))
    Lbp = (L1 + L2) / 2
    Cbp = (C1p + C2p) / 2
    with np.errstate(invalid="ignore"):
        hbp = np.where(C1p * C2p == 0, h1p + h2p,
                       np.where(np.abs(h1p - h2p) <= 180, (h1p + h2p) / 2,
                                np.where(h1p + h2p < 360, (h1p + h2p + 360) / 2,
                                         (h1p + h2p - 360) / 2)))
    T = (1 - 0.17 * np.cos(np.radians(hbp - 30)) + 0.24 * np.cos(np.radians(2 * hbp))
         + 0.32 * np.cos(np.radians(3 * hbp + 6))
         - 0.20 * np.cos(np.radians(4 * hbp - 63)))
    SL = 1 + 0.015 * (Lbp - 50) ** 2 / np.sqrt(20 + (Lbp - 50) ** 2)
    SC = 1 + 0.045 * Cbp
    SH = 1 + 0.015 * Cbp * T
    RT = -2 * np.sqrt(Cbp**7 / (Cbp**7 + 25.0**7)) \
        * np.sin(np.radians(60 * np.exp(-((hbp - 275) / 25) ** 2)))
    return np.sqrt((dLp / (kL * SL)) ** 2 + (dCp / (kC * SC)) ** 2
                   + (dHp / (kH * SH)) ** 2
                   + RT * (dCp / (kC * SC)) * (dHp / (kH * SH)))


def _iter_over(clip: dict, side: str):
    for ch, frac in zip(CHANNELS, clip.get(side, [])):
        yield ch, float(frac)
