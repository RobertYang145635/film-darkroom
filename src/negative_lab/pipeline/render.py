"""技术正片 vs 风格渲染分离（v4 §4.4/§6 P0-D）。

TechnicalPositive（backend 输出）：只做密度->纸张反射的线性还原。
RenderIntent：风格层（grade/contrast/exposure/highlight/lift/saturation），
默认恒等；风格参数不参与标定拟合或技术质量判定（v4 §2 边界）。

v4 §1.2.10 实锤迁移：exposure/gamma/highlight_* 从 InvertParams 技术链路
迁移至此（from_legacy_parameters 提供旧 sidecar 一次性兼容映射）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RenderIntent:
    grade: float = 1.0          # 纸级（旧 gamma 语义迁移；1.0=技术基线）
    contrast: float = 0.0       # 绕 pivot 的 S 曲线强度 [0,1]
    pivot: float = 0.35
    exposure: float = 1.0       # 线性乘子（旧 InvertParams.exposure 迁移）
    highlight_knee: float = 0.9
    highlight_strength: float = 0.0   # 旧 InvertParams.highlight_* 迁移
    lift: float = 0.0           # 黑位提升 [0, 0.2]
    saturation: float = 1.0     # 彩度缩放（仅风格）

    def is_neutral(self) -> bool:
        return (abs(self.grade - 1.0) < 1e-9 and self.contrast == 0.0
                and abs(self.exposure - 1.0) < 1e-9
                and self.highlight_strength == 0.0
                and self.lift == 0.0 and abs(self.saturation - 1.0) < 1e-9)

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in
                ("grade", "contrast", "pivot", "exposure", "highlight_knee",
                 "highlight_strength", "lift", "saturation")}

    @classmethod
    def from_legacy_parameters(cls, d: dict) -> "RenderIntent":
        """旧 sidecar parameters（含 gamma/exposure/highlight_*）的一次性映射。"""
        return cls(grade=float(d.get("gamma", 1.0)),
                   exposure=float(d.get("exposure", 1.0)),
                   highlight_knee=float(d.get("highlight_knee", 0.9)),
                   highlight_strength=float(d.get("highlight_strength", 0.0)))


def _highlight_compress(x: np.ndarray, knee: float, strength: float) -> np.ndarray:
    """单调高光软压缩：x>knee 部分向 1 压缩，保持端点与单调。"""
    if strength <= 0:
        return x
    hi = np.clip((x - knee) / max(1e-6, 1.0 - knee), 0.0, 1.0)
    compressed = 1.0 - (1.0 - hi) ** (1.0 + 2.0 * strength)
    return x + strength * (compressed - hi) * (1.0 - knee)


def apply_render(technical: np.ndarray, intent: RenderIntent) -> np.ndarray:
    """(H,W,3) linear 0-1 -> 渲染后 linear 0-1。除 lift/exposure 外保 0/1 端点。"""
    x = np.clip(np.asarray(technical, np.float32), 0.0, 1.0)
    if intent.is_neutral():
        return x
    if intent.exposure != 1.0:
        x = np.clip(x * np.float32(intent.exposure), 0.0, 1.0)
    if intent.grade != 1.0:
        x = np.power(x, intent.grade)
    if intent.highlight_strength > 0:
        x = _highlight_compress(x, intent.highlight_knee, intent.highlight_strength)
    if intent.contrast > 0:
        k = 1.0 + 2.0 * intent.contrast
        x = np.clip((x - intent.pivot) * k + intent.pivot, 0.0, 1.0)
    if intent.lift > 0:
        x = x * (1.0 - intent.lift) + intent.lift
    if intent.saturation != 1.0:
        lum = x @ np.array([0.2126, 0.7152, 0.0722], np.float32)[..., None]
        x = np.clip(lum + (x - lum) * np.float32(intent.saturation), 0.0, 1.0)
    return np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
