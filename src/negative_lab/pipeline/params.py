"""反相参数（规格 §4.3 / §6 旁车 schema_version=1 的 parameters 字段）。

字段命名与规格 §6 的 JSON 完全一致，保证可迁移。
`black` 保留用户界面“纸黑/密度修正”语义（非负标量）；
内部带符号量 `black_internal = -exposure * (1 + black)` 只在 density.py 中出现，
两者不得混用（规格 §4.3）。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class InvertParams:
    # 片基与密度域
    dmin: tuple[float, float, float] = (0.5, 0.5, 0.5)  # 未曝光片基每通道线性透射率
    dmax: float = 2.0            # 胶片有效密度范围，必须为正
    offset: float = 0.0          # 扫描曝光偏置（黑位控制）
    # 通道白平衡
    wb_low: tuple[float, float, float] = (1.0, 1.0, 1.0)   # 阴影色偏校正
    wb_high: tuple[float, float, float] = (1.0, 1.0, 1.0)  # 高光/照明白平衡校正
    # 每通道幂指数（规格 §4.4）：k_G=reference_exponent, k_R=k_G*red_ratio, k_B=k_G*blue_ratio
    reference_exponent: float = 1.0
    red_ratio: float = 1.0
    blue_ratio: float = 1.0
    # 相纸式正片映射
    black: float = 0.0     # 纸黑 >= 0
    gamma: float = 1.0     # 必须 > 0
    exposure: float = 1.0  # 必须 > 0
    # 高光软压缩（§4.3 末：单调高光映射）。strength=0 时恒等。
    highlight_knee: float = 0.9
    highlight_strength: float = 0.0

    def to_parameters_dict(self) -> dict:
        """输出规格 §6 parameters 字段（含本实现新增的高光控制项，附加字段向后兼容）。"""
        return {
            "dmin": [float(v) for v in self.dmin],
            "dmax": float(self.dmax),
            "offset": float(self.offset),
            "wb_low": [float(v) for v in self.wb_low],
            "wb_high": [float(v) for v in self.wb_high],
            "reference_exponent": float(self.reference_exponent),
            "red_ratio": float(self.red_ratio),
            "blue_ratio": float(self.blue_ratio),
            "black": float(self.black),
            "gamma": float(self.gamma),
            "exposure": float(self.exposure),
            "highlight_knee": float(self.highlight_knee),
            "highlight_strength": float(self.highlight_strength),
        }

    @classmethod
    def from_parameters_dict(cls, d: dict) -> "InvertParams":
        p = cls()
        for k, v in d.items():
            if not hasattr(p, k):
                continue  # 忽略未知字段（前向兼容）
            if isinstance(getattr(p, k), tuple):
                v = tuple(float(x) for x in v)
            setattr(p, k, v)
        return p
