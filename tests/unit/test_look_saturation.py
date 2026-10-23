r"""㊹ 的锁：外观级饱和度按片种（值来自用户自己的 XMP，不是在我方误差上选的）。

单独成文件有一层实际原因（㊹⑥ 记过）：把这几条锁塞进 `test_film_types.py` 之后，
`pytest tests/unit` 单进程会在 `ui/theme.apply` 的 `setPalette` 段错误（B-11 追加⑦ 记过的
同一个失效模式，触发条件是测试条数/顺序）；拆出来让全量能单进程跑完，
**这不是把 GUI 测试 `--ignore` 掉**（那是不许的，见交付规矩），只是安排文件边界。
"""
from __future__ import annotations

import numpy as np
import pytest

from negative_lab.library.models import FrameRecord, GeometryRegions
from negative_lab.print_math import LOOK_BLUE_EXPONENT, LOOK_RED_EXPONENT
from negative_lab.pipeline.film_types import FILM_TYPES, FilmType
from negative_lab.pipeline.service import process_roll


def _neg(h: int = 160, w: int = 200) -> np.ndarray:
    rng = np.random.default_rng(11)
    base = rng.uniform(0.02, 0.30, (h, w, 3)).astype(np.float32)
    base[..., 0] *= 1.9
    base[..., 2] *= 0.45
    base[h // 3: 2 * h // 3] *= 2.4
    return np.clip(base, 1e-4, 0.98).astype(np.float32)


def _rec(film_type: str) -> FrameRecord:
    return FrameRecord(source_path="probe/x.tif", source_sha256="0" * 64,
                       geometry=GeometryRegions(), orientation_decided=True,
                       film_type=film_type)


def _roll(film_type: str):
    img = _neg()
    _, results = process_roll([(_rec(film_type), img)])
    return results[0]


def _ft_sat(key: str, sat: float) -> FilmType:
    return FilmType(key=key, label=f"探针饱和度 {sat}", roll="probe_roll",
                    red_ratio=LOOK_RED_EXPONENT, blue_ratio=LOOK_BLUE_EXPONENT,
                    look_saturation=sat,
                    provenance="本测试：外观级饱和度按片种", frames=1)


def test_look_saturation_out_of_range_is_rejected():
    """0 = 不施加；非零只许在 ±50 内声明（控件量程 ±100，表里不许写没量过的档）。"""
    for bad in (-80.0, 70.0, 120.0):
        with pytest.raises(ValueError, match="look_saturation"):
            _ft_sat(f"probe-sat-bad-{bad}", bad)
    assert _ft_sat("probe-sat-zero", 0.0).look_saturation == 0.0


def test_look_saturation_split_between_the_two_ecn2_rolls():
    """**值的出处是他自己的 XMP**：逐帧配对 70 对里 `crs:Saturation`
    只有 5219 的 35 对不同（我方 0 → 他 −10），5207 的 35 对是 0/0 全同
    ⇒ 这一维**只许给 5219**，5207 与 G200 各卷必须是 0（不施加）。"""
    assert FILM_TYPES["ecn2-5219"].look_saturation == pytest.approx(-10.0)
    assert FILM_TYPES["ecn2-5207"].look_saturation == 0.0
    others = {k: ft.look_saturation for k, ft in FILM_TYPES.items()
              if k != "ecn2-5219"}
    assert not any(others.values()), \
        f"除 5219 之外的片种都不许带 look_saturation（没有锚的一格）：{others}"


def test_look_saturation_reaches_the_render_and_is_recorded(monkeypatch):
    """声明带这一维的片种 ⇒ 意图看得见、像素真的变、sidecar 记的是生效值。"""
    monkeypatch.setitem(FILM_TYPES, "probe-sat-on", _ft_sat("probe-sat-on", -10.0))
    pr = _roll("probe-sat-on")
    assert float(pr.render_intent.saturation) == pytest.approx(-10.0)
    off = _roll("")
    assert float(off.render_intent.saturation) == 0.0
    assert not np.array_equal(pr.rendered_positive_linear,
                              off.rendered_positive_linear), "这一维是空转"
    rec = pr.meta["print_scale"]
    assert rec["look_saturation"] == pytest.approx(-10.0)
    assert rec["look_saturation_asset"] == pytest.approx(-10.0)


def test_stock_look_saturation_equals_writing_the_intent_directly(monkeypatch):
    """规矩 26 家族：**片种表这条路**的像素，必须与**把同一个值直接写进那条锚意图**
    逐比特相同。不同就说明表值没有走被量过的那条路（中途被改写、被重复施加、
    或根本没有进产品路径）。"""
    from negative_lab.library.render import RenderIntent, apply_render
    monkeypatch.setitem(FILM_TYPES, "probe-sat-eq", _ft_sat("probe-sat-eq", -10.0))
    via_stock = _roll("probe-sat-eq")
    plain = _roll("")
    it = RenderIntent.from_dict(plain.render_intent.to_dict())
    assert float(it.saturation) == 0.0, "未声明的卷不该带饱和度偏移"
    it.saturation = -10.0
    assert np.array_equal(via_stock.rendered_positive_linear,
                          apply_render(plain.result.positive_linear, it)), \
        "表值路线与直传 intent 的像素不一致 ⇒ 两条路不是同一个语义"
    assert float(via_stock.render_intent.saturation) == pytest.approx(-10.0)


def test_look_saturation_zero_is_bit_identical_to_undeclared(monkeypatch):
    """退化控制：`look_saturation=0` 必须与"什么都不声明"逐比特相同
    ⇒ 5207 与 G200 这些没有锚的卷一个字都不受影响。"""
    monkeypatch.setitem(FILM_TYPES, "probe-sat-off", _ft_sat("probe-sat-off", 0.0))
    a = _roll("probe-sat-off")
    b = _roll("")
    assert np.array_equal(a.rendered_positive_linear, b.rendered_positive_linear)
    assert float(a.render_intent.saturation) == 0.0


def test_candidate_saturation_wins_over_the_stock_value():
    """`saturation` 是用户能动的控件：候选/用户写了非零值时片种值不许覆盖它。
    直接测生产里那条合并规则本身（`effective_look_saturation`），不是测它的复制品。"""
    from negative_lab.pipeline.service import effective_look_saturation
    assert effective_look_saturation(0.0, -10.0) == pytest.approx(-10.0)
    assert effective_look_saturation(None, -10.0) == pytest.approx(-10.0)
    assert effective_look_saturation(20.0, -10.0) == pytest.approx(20.0), \
        "用户设的饱和度被片种值顶掉了"
    assert effective_look_saturation(20.0, 0.0) == pytest.approx(20.0)
    assert effective_look_saturation(0.0, 0.0) == 0.0


def test_intent_with_only_look_saturation_does_not_claim_identity():
    """与 `white_balance`/`chroma_highlight_gain` 同型：能单独施加的处理不许自称恒等。"""
    from negative_lab.library.render import RenderIntent
    assert RenderIntent(saturation=-10.0).is_identity() is False
    assert RenderIntent().is_identity() is True
