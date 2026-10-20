r"""片种 → 技术级通道斜率（**走资产通道**）。

为什么是资产通道而不是配置文件（`FilmProfile.exponent_prior`）：
`docs/blockers.md` B-16 追加⑩ 三臂实测——用户先验 `blue_ratio=1.30` 配上默认资产
解析出来仍是 0.850，因为 `process_image` 在 `auto_analyze` 之后**无条件**
`asset.apply_to(params)`，而 `deterministic_asset()` 的 payload 里带着
`red_ratio=LOOK_RED_EXPONENT / blue_ratio=LOOK_BLUE_EXPONENT`。
⇒ 往 profile 里填斜率**不生效也没有警告**；只有资产这一路是活的。

为什么需要这一层（追加⑨ 的结论）：技术级蓝斜率按片种给才有解——
全量 81 帧上四卷的 `Δcast_bg` 过零点分别落在 **0.47 / 1.02 / 1.17 / >1.5**
（5219 / 5207 / 1298 / 1299），跨度 3 倍以上，而且 5219 与其余三卷在蓝轴**反号**。
一个全局常数服务不了它们，这是"必须按片种给参数"的第一次正面证据。

纪律（三条，全部有锁）：
 ① **未声明 = 逐比特不变**：`resolve(None/"")` 返回 `None`，调用方保持原有
    `model_asset=None` 的路径（走 `deterministic_asset`）。
 ② **每个表项必须带出处与帧数**：`provenance` 为空或 `frames<=0` 直接抛错——
    这条仓库里已经吃过亏（常数没有出处，下一轮就没人敢动）。
 ③ 资产里 `red_ratio`/`blue_ratio` **两个都给**：`InvertParams` 的默认值是 1.0，
    而产品默认值 1.30/0.85 是由默认资产注入的；只给一个会把另一个悄悄退回 1.0。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .model_assets import ModelAsset, canonical_hash


@dataclass(frozen=True)
class FilmType:
    key: str                      # 机器可读，写进 `FrameRecord.film_type`
    label: str                    # 给人看的（GUI 下拉、CLI --film-type 的说明）
    red_ratio: float
    blue_ratio: float
    provenance: str               # 在谁的样片上、怎么量的；空字符串直接抛错
    frames: int                   # 定值用的帧数（置信度证据）

    def __post_init__(self):
        if not self.key or not self.label:
            raise ValueError(f"片种 {self.key!r} 缺 key/label")
        if not self.provenance.strip():
            raise ValueError(f"片种 {self.key} 没有出处——不许进表"
                             f"（常数不带出处，下一轮就没人敢动）")
        if int(self.frames) <= 0:
            raise ValueError(f"片种 {self.key} 的定值帧数必须 >0")
        for name in ("red_ratio", "blue_ratio"):
            v = float(getattr(self, name))
            if not (0.2 <= v <= 4.0):
                raise ValueError(f"片种 {self.key}.{name}={v} 出界（0.2~4.0）")


#: 表。值由 `tmp_slope_cast_sweep.py` 在全量帧上扫出来（每个都要是**网格内点最小**，
#: 端点值不进表），出处写在各条的 `provenance` 里。
FILM_TYPES: dict[str, FilmType] = {}


def resolve(key: str | None) -> FilmType | None:
    """`None`/空/`unknown` ⇒ `None`（不改变任何行为）。未知键也返回 `None`，
    但调用方应当用 `known_keys()` 自己校验，别把拼错的片种当成"未声明"。"""
    if not key or key == "unknown":
        return None
    return FILM_TYPES.get(key)


def asset_for(ft: FilmType) -> ModelAsset:
    """片种 → 资产。矩阵仍是单位阵、曲线为空 ⇒ 除了两个斜率什么都不动。"""
    payload = {"density_matrix": np.eye(3).tolist(), "channel_curves": [None, None, None],
               "red_ratio": float(ft.red_ratio), "blue_ratio": float(ft.blue_ratio),
               "reference_exponent": None, "dmax": None,
               "film_type": ft.key, "provenance": ft.provenance, "frames": int(ft.frames)}
    return ModelAsset(asset_version=f"film-type-{ft.key}-v1", payload=payload,
                      sha256=canonical_hash(payload), enabled=True,
                      meta={"kind": "film_type", "note": ft.provenance})


def resolve_keys(keys: Iterable[str | None]) -> FilmType | None:
    """一卷里允许混着填：取**第一个已声明**的键；全都未声明 ⇒ `None`。
    混合声明不静默平均——那会造出一个没有任何出处的第三套值。"""
    for k in keys:
        ft = resolve(k)
        if ft is not None:
            return ft
    return None


def known_keys() -> list[str]:
    return sorted(FILM_TYPES)
