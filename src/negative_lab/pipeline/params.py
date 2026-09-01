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
    # 相纸式正片映射（技术端点；v4 §4.4：black 属技术范围映射）
    black: float = 0.0     # 纸黑 >= 0
    # ---- 以下为已废止的技术字段（v4 §1.2.10/§4.4 裁决：风格迁移 RenderIntent）----
    # gamma/exposure/highlight_* 保留仅为旧 sidecar 加载兼容，技术链路不再读取；
    # 一次性兼容映射见 RenderIntent.from_legacy_parameters()。
    gamma: float = 1.0             # 已废弃：-> RenderIntent.grade
    exposure: float = 1.0          # 已废弃：-> RenderIntent.exposure
    highlight_knee: float = 0.9    # 已废弃：-> RenderIntent.highlight_*
    highlight_strength: float = 0.0
    # 密度域解混矩阵（行主 3x3；单位阵 = 路线 A 基线）。
    density_matrix: tuple = (1, 0, 0, 0, 1, 0, 0, 0, 1)
    # 路线 C 通道曲线（(x,y) 节点三元组；空 = 线性 = 路线 B）
    channel_curves: tuple = ()
    # v4 §4.3：显式 backend（默认=旧兼容基线；不得自称彩色负片最优默认，
    # 生产默认由 bake-off 留出集证据确定）
    backend_name: str = "power-reciprocal"

    FORMULA_VERSION = "2026-09-v4-technical-contract"

    def to_parameters_dict(self) -> dict:
        """技术参数（v4 sidecar：只含技术字段；风格在 render_intent 字段）。"""
        return {
            "formula_version": self.FORMULA_VERSION,
            "backend_name": self.backend_name,
            "dmin": [float(v) for v in self.dmin],
            "dmax": float(self.dmax),
            "offset": float(self.offset),
            "wb_low": [float(v) for v in self.wb_low],
            "wb_high": [float(v) for v in self.wb_high],
            "reference_exponent": float(self.reference_exponent),
            "red_ratio": float(self.red_ratio),
            "blue_ratio": float(self.blue_ratio),
            "black": float(self.black),
            "density_matrix": [float(v) for v in self.density_matrix],
            "channel_curves": [
                ([list(map(float, k)) for k in curve] if curve else None)
                for curve in self.channel_curves
            ] if self.channel_curves else [],
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
