"""锁：**卷级片基继承**的可达性与边界（B-11 追加⑪）。

被修的缺陷（实测）：`resolve_base` 里 `elif roll_ok:` 那条分支的文案是
"本帧没有可靠片基，已使用同卷 N 帧的卷级片基模型"——它**就是**为
"自己什么都没测到"的帧准备的；但准入判据写的是 `frame_id in shared_with_frames`，
而那个列表只装"自己测到并参与汇总"的帧，于是需要继承的帧永远进不去，
整卷退到 `generic_prior`。实测 `1299_G200` 9/11 帧、`1298_G200::G200-7` 都是这样，
青偏的直接来源（红通道均值偏 −0.35~−0.61 对用户参考）。

同时钉住**不能走过头**：卷模型只由 1 帧建成时（`roll_partial`）不许覆盖没实测的帧。
实测 `1299_G200` 的卷 dmin 由单帧 G200-11 决定（R=0.1967），
而另一帧实测 G200-10 是 0.2672（差 0.13 decade，且被判漂移排除），
用它覆盖 9 帧后三帧参考的距离 0.128 → **0.164**，比通用先验更糟。
"""

from __future__ import annotations

import numpy as np

from negative_lab.library.models import RollAnalysis
from negative_lab.pipeline.roll_analysis import resolve_base

T = np.full((8, 8, 3), 0.12, np.float32)
ROLL_DMIN = [0.28, 0.11, 0.03]


def _roll(*, base_source="roll", shared=(), excluded=(), n=3):
    return RollAnalysis(
        base_statistics={"dmin": list(ROLL_DMIN), "n_frames": n},
        confidence=0.8 if base_source == "roll" else 0.5,
        base_source=base_source,
        shared_with_frames=list(shared),
        excluded_from_share=list(excluded),
        source_frames=list(shared) or ["A", "B"],
        generic_scale=0.42)


def _own(conf=0.9):
    return {"dmin": [0.25, 0.09, 0.02], "confidence": conf,
            "contamination": 0.0}


def test_unmeasured_frame_inherits_a_real_roll_model():
    """**这条就是被修死的分支**：本帧没测到片基，但同卷有真卷模型 ⇒ 继承。

    旧实现要求 `frame_id in shared_with_frames`，而"没测到"的帧按定义不在里面，
    所以本例在修复前必然得到 `generic_prior`。
    """
    out = resolve_base({"dmin": None, "confidence": 0.0}, _roll(),
                       t_oriented=T, frame_id="roll::me-less")
    assert out["base_source"] == "roll", (
        f"没实测的帧仍没继承卷级片基：{out['base_source']} —— "
        "v5 §6.2.3 的这条路径又死了")
    assert np.allclose(out["dmin"], ROLL_DMIN, atol=1e-9)
    assert any("卷级" in w for w in out["warnings"]), "继承必须留下可追溯的话"


def test_single_frame_roll_model_must_not_overwrite_unmeasured_frames():
    """`roll_partial`（参与帧不足 `min_roll_frames`）**不许**覆盖没实测的帧。

    依据是实测而不是原则：1299 的卷模型由单帧决定，与另一帧实测差 0.13 decade，
    覆盖之后三帧参考的距离 0.128→0.164，比通用先验更差。
    """
    out = resolve_base({"dmin": None, "confidence": 0.0},
                       _roll(base_source="roll_partial", shared=["only-one"]),
                       t_oriented=T, frame_id="roll::me-less")
    assert out["base_source"] == "generic_prior", (
        f"单帧卷模型被当成卷级真值用了：{out['base_source']}")


def test_drift_excluded_frame_does_not_inherit():
    """自己测到过、但与同卷漂移不符的帧**不共享**（v5 §4.4 不得静默共享错误参数）。"""
    out = resolve_base({"dmin": None, "confidence": 0.0},
                       _roll(excluded=["roll::drifted"]),
                       t_oriented=T, frame_id="roll::drifted")
    assert out["base_source"] == "generic_prior", out["base_source"]


def test_own_good_measurement_is_unchanged_by_the_fix():
    """本帧自己有可靠片基 ⇒ 仍走自己的（加权或不加权），继承改动不许越界。"""
    out = resolve_base(_own(), _roll(base_source="roll_partial",
                                     shared=["A", "B"]),
                       t_oriented=T, frame_id="roll::self")
    assert out["base_source"] in ("frame", "frame+roll"), out["base_source"]
    assert out["base_source"] != "generic_prior"


def test_no_roll_at_all_still_falls_back():
    """没有任何卷模型时照旧退回通用先验——这条锁保证上面几条不是把回退删了。"""
    out = resolve_base({"dmin": None, "confidence": 0.0}, None,
                       t_oriented=T, frame_id="roll::x")
    assert out["base_source"] == "generic_prior"
