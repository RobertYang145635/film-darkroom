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
    status: str = "ok"                       # ok|warn|fail
    input_clip: dict = field(default_factory=dict)    # {low:[r,g,b], high:[r,g,b]}
    output_clip: dict = field(default_factory=dict)
    nonfinite_input: int = 0
    nonfinite_output: int = 0
    warnings: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    # ---- 构建

    @classmethod
    def evaluate(cls, *, t_in: np.ndarray | None = None,
                 positive_linear: np.ndarray | None = None,
                 input_nonfinite: int = 0,
                 extra_warnings: list[str] | None = None,
                 clip_warn_frac: float = CLIP_WARN_FRAC,
                 calibration: dict | None = None) -> "QualityGate":
        """calibration（v3 P0-G）：{"level": verified-calibration 名 或 None,
        "confidence": float}。彩色负片缺卷级标定/中性参考时
        status=needs_calibration（不是 warn，v3 F-ALG-7）。"""
        g = cls()
        if t_in is not None:
            g.input_clip = _clip(t_in)
        if positive_linear is not None:
            g.output_clip = _clip(positive_linear)
            g.nonfinite_output = int((~np.isfinite(positive_linear)).sum())
            if g.nonfinite_output:
                g.failures.append(
                    f"输出含 {g.nonfinite_output} 个 NaN/Inf（禁止进入导出）")
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
            "warnings": self.warnings,
            "failures": self.failures,
            "clip_warn_frac": CLIP_WARN_FRAC,
        }

    @property
    def clip_exceeded(self) -> bool:
        return any(f > CLIP_WARN_FRAC for _, f in _iter_over(self.output_clip, "low")) \
            or any(f > CLIP_WARN_FRAC for _, f in _iter_over(self.output_clip, "high"))


def _clip(img: np.ndarray) -> dict:
    lo = (img <= CLIP_TOL).mean(axis=tuple(range(img.ndim - 1)))
    hi = (img >= 1.0 - CLIP_TOL).mean(axis=tuple(range(img.ndim - 1)))
    return {"low": [float(v) for v in lo], "high": [float(v) for v in hi]}


def _iter_over(clip: dict, side: str):
    for ch, frac in zip(CHANNELS, clip.get(side, [])):
        yield ch, float(frac)
