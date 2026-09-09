"""卷级校准对象（整改 v3 P0-E）。

层级（高优先到低）：
1. user-points        用户在单张上取的 verified 中性点（当场生效，不跨卷）
2. chart-calibration  色卡/已知目标监督标定（跨卷复用，含解混矩阵）
3. base-only          仅片基基准（needs_calibration，experimental）

统计 gray-world 永远不进入本对象——只能作为 suggestion 呈现，
用户显式采纳后以 user-points/chart 形式落库（v3 F-ALG-4）。
"""

from __future__ import annotations

from .. import storage

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

CALIB_DIR = storage.repo_path("fixtures", "calibrations")
SCHEMA = 1


@dataclass
class RollCalibration:
    roll: str
    film_type: str = "unknown"
    source: str = "base-only"          # user-points|chart-calibration|base-only
    verified: bool = False
    dmin: tuple | None = None
    dmax: float | None = None
    density_matrix: tuple | None = None      # 9 元素，行主；None=单位阵
    red_ratio: float | None = None
    blue_ratio: float | None = None
    reference_exponent: float | None = None
    wb_high: tuple | None = None
    offset: float | None = None
    confidence: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)   # 样本文件/方法/版本
    schema_version: int = SCHEMA

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RollCalibration":
        if int(d.get("schema_version", 0)) != SCHEMA:
            raise ValueError(f"校准文件 schema_version={d.get('schema_version')} "
                             f"与当前 {SCHEMA} 不兼容")
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: (tuple(v) if isinstance(v, list) and k in
                          ("dmin", "wb_high", "density_matrix") else v)
                      for k, v in d.items() if k in known})

    @property
    def usable_for_color(self) -> bool:
        """彩色通道参数是否达到可用标准（v3：base-only 不算可用）。"""
        return (self.verified and self.source in ("user-points", "chart-calibration")
                and float(self.confidence.get("neutral", 0)) >= 0.5)

    def save(self, directory: Path | None = None) -> Path:
        d = directory or CALIB_DIR
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"{self.roll}.calib.json"
        from ..io.atomic import atomic_write_bytes    # 原子写（v6.2 审查批次 3）
        atomic_write_bytes(p, json.dumps(self.to_dict(), ensure_ascii=False,
                                         indent=2).encode("utf-8"),
                           overwrite=True)
        return p

    @classmethod
    def load(cls, path: Path) -> "RollCalibration":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def build_from_neutral_samples(roll: str, film_type: str,
                               measured_z: np.ndarray,
                               target_density: np.ndarray | None = None,
                               base_dmin: tuple | None = None,
                               provenance: dict | None = None) -> RollCalibration:
    """从 verified 中性/色卡样本构建校准。

    - target_density 提供（色卡监督）-> 解混矩阵 + 指数（chart-calibration）
    - 仅中性样本 -> 指数比值/白平衡（user-points），解混保持单位阵
      （单射线不可辨识，v3 F-ALG-5，见 backends.fit_decrosstalk）
    """
    from ..pipeline.backends import fit_decrosstalk_supervised
    Z = np.asarray(measured_z, np.float64)
    T = None if target_density is None else np.asarray(target_density, np.float64)
    cal = RollCalibration(roll=roll, film_type=film_type,
                          dmin=base_dmin,
                          provenance=provenance or {})
    if T is not None:
        fit = fit_decrosstalk_supervised(Z, T)
        if fit["confidence"] > 0:
            cal.density_matrix = tuple(fit["matrix"])
            cal.source = "chart-calibration"
            cal.verified = True
            cal.confidence["neutral"] = 1.0
            cal.confidence["decrosstalk"] = fit["confidence"]
            cal.provenance["method"] = "supervised-ridge"
            cal.provenance["n_samples"] = fit["n_samples"]
            return cal
        cal.provenance["decrosstalk_warning"] = fit["warnings"]
    # 中性 -> 指数比值（对角）：要求跨密度
    from ..calibration.neutral import estimate_exponent_ratios
    if len(Z) >= 2:
        zl = Z.mean(axis=1)
        if float(np.ptp(zl)) >= 0.05:
            # 用 z 分位对构造两密度组（暗组/亮组中位）
            lo = Z[zl < np.quantile(zl, 0.3)].mean(axis=0)
            hi = Z[zl > np.quantile(zl, 0.7)].mean(axis=0)
            est = estimate_exponent_ratios([tuple(lo), tuple(hi)])
            if est["confidence"] > 0:
                cal.red_ratio = est["red_ratio"]
                cal.blue_ratio = est["blue_ratio"]
                cal.source = "user-points"
                cal.verified = True
                cal.confidence["neutral"] = est["confidence"]
                cal.provenance["method"] = "neutral-pair-exponents"
                return cal
    cal.source = "base-only"
    cal.verified = False
    cal.confidence["neutral"] = 0.0
    cal.provenance.setdefault("reason", "样本不足或不可辨识")
    return cal
