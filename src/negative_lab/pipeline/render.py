"""技术正片 vs 风格渲染分离（整改 v3 P0-F）。

TechnicalPositive（invert 输出）：只做密度->纸张反射的线性还原，
gamma=1、无对比度曲线——色彩正确性的可验证基线。

RenderIntent（本模块）：纸级/对比度/饱和度的风格层，默认恒等。
darktable negadoctor 的 paper grade（默认 4）与 NLP tone profile 都
位于转换与色彩分析之外——本实现同样禁止风格参数参与标定/质量判定。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RenderIntent:
    grade: float = 1.0          # 纸级曲线（>1 加深中间调）；1.0=技术基线
    contrast: float = 0.0       # 绕 pivot 的 S 曲线强度 [0,1]
    pivot: float = 0.35
    lift: float = 0.0           # 黑位提升 [0,0.2]
    saturation: float = 1.0     # 彩度缩放（仅风格；1.0=不变）

    def is_neutral(self) -> bool:
        return (abs(self.grade - 1.0) < 1e-9 and self.contrast == 0.0
                and self.lift == 0.0 and abs(self.saturation - 1.0) < 1e-9)

    def to_dict(self) -> dict:
        return {"grade": self.grade, "contrast": self.contrast,
                "pivot": self.pivot, "lift": self.lift,
                "saturation": self.saturation}


def apply_render(technical: np.ndarray, intent: RenderIntent) -> np.ndarray:
    """(H,W,3) linear 0-1 -> 渲染后 linear 0-1。单调、保 0/1 端点（除 lift）。"""
    x = np.clip(np.asarray(technical, np.float32), 0.0, 1.0)
    if intent.is_neutral():
        return x
    if intent.grade != 1.0:
        x = np.power(x, intent.grade)
    if intent.contrast > 0:
        k = 1.0 + 2.0 * intent.contrast
        x = np.clip((x - intent.pivot) * k + intent.pivot, 0.0, 1.0)
    if intent.lift > 0:
        x = x * (1.0 - intent.lift) + intent.lift
    if intent.saturation != 1.0:
        lum = x @ np.array([0.2126, 0.7152, 0.0722], np.float32)[..., None]
        x = np.clip(lum + (x - lum) * np.float32(intent.saturation), 0.0, 1.0)
    return x.astype(np.float32)
