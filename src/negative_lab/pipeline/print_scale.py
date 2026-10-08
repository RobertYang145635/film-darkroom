"""自动打印尺度在**管线侧**的入口（纯数学已下沉到叶子 `negative_lab.print_math`）。

为什么要拆：`tests/unit/test_layering.py` 禁止下层 import 上层，
而施加曲线的 `apply_render` 位于 `library/render.py`——
所以曲线本体在根目录叶子模块（与 `constants.py`、`metrics.py` 同族），
这里只保留需要 `InvertResult` 的那个包装，并把名字重导出，
让既有调用点与测试的 import 路径不变。
"""

from __future__ import annotations

import numpy as np

from ..print_math import (LOOK_CHROMA_GAIN, MIN_SPAN, P_HIGH, P_LOW, P_MID,
                          PRINT_AIM, PRINT_WHITE, PRINT_REGION_CENTRE_FRAC,
                          anchor_refusal_reason, centre_region, density_anchor,
                          picture_density, print_curve, print_remap_array, remap_luma)

__all__ = ["LOOK_CHROMA_GAIN", "MIN_SPAN", "P_HIGH", "P_LOW", "P_MID", "PRINT_AIM",
           "PRINT_WHITE", "PRINT_REGION_CENTRE_FRAC", "anchor_refusal_reason",
           "centre_region", "density_anchor", "picture_density", "print_curve",
           "print_remap_array", "remap_luma", "apply_print_remap"]


def apply_print_remap(res, anchor: tuple[float, float, float] | None,
                      dmax_old: float, mask: np.ndarray | None = None):
    """`InvertResult` 版包装：换掉 `positive_linear` 与 `preview_srgb`，其余照抄。

    实现只有一份（`print_math.print_remap_array`）。生产路径**不用**这个入口——
    它改的是技术正片，而技术正片是门禁口径（v5 §8.2）与
    `test_rebate_independence` 的比较对象；生产走 `RenderIntent.print_anchor`，
    由 `library/render.py` 的 `apply_render` 施加。
    """
    import dataclasses

    from .density import encode_srgb

    out, info = print_remap_array(res.positive_linear, anchor, dmax_old, mask)
    if info is None:
        return res, None
    return dataclasses.replace(res, positive_linear=out,
                               preview_srgb=encode_srgb(out)), info
