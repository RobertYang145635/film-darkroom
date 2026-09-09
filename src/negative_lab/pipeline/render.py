"""v5 §2.2 / §7.2：唯一微调层 `RenderIntent`（实现 moved 到 library.render）。

历史：本模块原本在 pipeline/render.py。io/sidecar.py（下层）做 legacy
sidecar 迁移需要 `RenderIntent`，构成 io -> pipeline 的反向依赖（分层门禁
tests/unit/test_layering.py 的 ALLOW 待办）。RenderIntent 与其色温/影调
数学只依赖 numpy，是**库层能力**而非管线编排，故整体下沉到 library 层。

兼容：pipeline.render 以显式自别名重导出全部既有公开名（noqa: F401），
既有调用方与测试零改动。新代码请 import negative_lab.library.render。
"""

from __future__ import annotations

from ..library.render import (  # noqa: F401 —— 显式重导出（兼容旧路径）
    CANDIDATE_TEMP_CAP, CANDIDATE_TINT_CAP, P0_FIELDS, RENDER_ORDER,
    RANGES, RenderIntent, adaptation_matrix, apply_render, cct_from_temperature,
    clamp_candidate_intent, daylight_xy, planckian_xy, render_clip_ratios,
    tone_curve, white_xy, xy_to_xyz,
)
from ..library.render import _BRADFORD as _BRADFORD  # noqa: F401
from ..library.render import _apply_chroma as _apply_chroma  # noqa: F401
from ..library.render import _apply_tone as _apply_tone  # noqa: F401
from ..library.render import _bump as _bump  # noqa: F401
from ..library.render import _curve_for as _curve_for  # noqa: F401
from ..library.render import _pivot_u as _pivot_u  # noqa: F401
