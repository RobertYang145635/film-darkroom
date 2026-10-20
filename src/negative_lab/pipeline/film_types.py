r"""片种 → **资产**（`library/film_types.py` 那张表的可执行部分）。

表本身（`FilmType` / `FILM_TYPES` / `resolve` / `known_keys`）在
`library/film_types.py`——那份是纯数据，`library/models.py` 校验卷声明时要用它，
而分层门禁禁止 `library → pipeline`（`tests/unit/test_layering.py`）。
这一层只多做一件事：把表项变成 `ModelAsset`。

为什么必须走资产而不是 `FilmProfile.exponent_prior`：
`docs/blockers.md` B-16 追加⑩ 三臂实测——用户先验 `blue_ratio=1.30` 配默认资产
解析出来仍是 0.850，因为 `process_image` 在 `auto_analyze` 之后**无条件**
`asset.apply_to(params)`，而 `deterministic_asset()` 的 payload 带着
`red_ratio=LOOK_RED_EXPONENT / blue_ratio=LOOK_BLUE_EXPONENT`。
⇒ 往 profile 里填斜率**不生效也没有警告**；只有资产这一路是活的。

⚠ 资产里 `red_ratio`/`blue_ratio` **两个都必须给**：`InvertParams` 的默认值是 1.0，
而产品默认 1.30/0.85 是由默认资产注入的；只声明一个会把另一个悄悄退回 1.0。
"""
from __future__ import annotations

import numpy as np

from ..library.film_types import (FILM_TYPES, FilmType, known_keys, resolve,  # noqa: F401
                                  resolve_keys)
from .model_assets import ModelAsset, canonical_hash

__all__ = ["FILM_TYPES", "FilmType", "asset_for", "known_keys", "resolve", "resolve_keys"]


def asset_for(ft: FilmType) -> ModelAsset:
    """片种 → 资产。矩阵仍是单位阵、曲线为空 ⇒ 除了两个斜率什么都不动。"""
    payload = {"density_matrix": np.eye(3).tolist(), "channel_curves": [None, None, None],
               "red_ratio": float(ft.red_ratio), "blue_ratio": float(ft.blue_ratio),
               "reference_exponent": None, "dmax": None,
               "film_type": ft.key, "provenance": ft.provenance, "frames": int(ft.frames)}
    return ModelAsset(asset_version=f"film-type-{ft.key}-v1", payload=payload,
                      sha256=canonical_hash(payload), enabled=True,
                      meta={"kind": "film_type", "note": ft.provenance})
