r"""片种参数表 + 声明通道的锁（㊶ 追加⑨/⑩ 的实现层）。

四条纪律各自有锁，其中**退化控制**是最重要的一条：
声明一个斜率恰好等于现产默认（red 1.30 / blue 0.85）的片种，
渲染结果必须与"什么都不声明"**逐比特相同** —— 这条过了，
才可以说"未声明的用户不受影响"，而不是"看起来没影响"。
"""
from __future__ import annotations

import numpy as np
import pytest

from negative_lab.library.models import FrameRecord, GeometryRegions
from negative_lab.pipeline.model_assets import ModelAsset
from negative_lab.pipeline.params import InvertParams
from negative_lab.print_math import LOOK_BLUE_EXPONENT, LOOK_RED_EXPONENT
from negative_lab.pipeline.service import process_roll
from negative_lab.library.models import RollGroup as RollGroupBack
from negative_lab.pipeline.film_types import (FILM_TYPES, FilmType, asset_for,
                                              known_keys, resolve, resolve_keys)


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


def test_undeclared_is_not_a_film_type():
    assert resolve(None) is None and resolve("") is None and resolve("unknown") is None
    assert resolve_keys(["", None, "unknown"]) is None
    assert resolve("no-such-stock") is None            # 未知键也不静默造新值


def test_table_entries_must_carry_provenance_and_frames():
    with pytest.raises(ValueError, match="出处"):
        FilmType(key="k", label="L", roll="probe_roll", red_ratio=1.3, blue_ratio=0.9,
                 provenance="   ", frames=35)
    with pytest.raises(ValueError, match="帧数"):
        FilmType(key="k", label="L", roll="probe_roll", red_ratio=1.3, blue_ratio=0.9,
                 provenance="实测", frames=0)
    for k, ft in FILM_TYPES.items():
        assert ft.provenance.strip() and ft.frames > 0 and k == ft.key


def test_asset_sets_both_slopes_and_touches_nothing_else():
    ft = FilmType(key="probe", label="探针片种", roll="probe_roll",
                  red_ratio=1.42, blue_ratio=0.63,
                  provenance="本测试的合成输入", frames=1)
    a = asset_for(ft)
    assert isinstance(a, ModelAsset) and a.enabled
    p = InvertParams()
    before = (p.reference_exponent, p.dmax, tuple(p.density_matrix))
    info = a.apply_to(p)
    assert info["applied"] is True
    assert (p.red_ratio, p.blue_ratio) == pytest.approx((1.42, 0.63))
    assert (p.reference_exponent, p.dmax, tuple(p.density_matrix)) == before, \
        "片种资产只许动两个斜率，别的字段一动就必须补新的出处与锁"


def test_declaring_production_slopes_is_bit_identical_to_declaring_nothing(monkeypatch):
    """退化控制：表里放一个"等于现产默认"的片种 ⇒ 像素必须逐比特相同。"""
    base = _roll("")
    ft = FilmType(key="probe-same", label="与现产同值", roll="probe_roll",
                  red_ratio=LOOK_RED_EXPONENT, blue_ratio=LOOK_BLUE_EXPONENT,
                  provenance="本测试：证明声明通道本身不引入差异", frames=1)
    monkeypatch.setitem(FILM_TYPES, "probe-same", ft)
    got = _roll("probe-same")
    assert np.array_equal(got.result.positive_linear, base.result.positive_linear), \
        "声明通道改变了像素 ⇒ 它不只是替换斜率（检查 reference_exponent/dmax 是否被带进 payload）"
    assert got.report.params.blue_ratio == pytest.approx(LOOK_BLUE_EXPONENT)


def test_declaring_a_different_slope_actually_reaches_the_backend(monkeypatch):
    """非空控制：换一个蓝斜率，解析后的参数与像素都必须变（否则声明又是死的）。"""
    base = _roll("")
    ft = FilmType(key="probe-blue", label="探针蓝斜率", roll="probe_roll",
                  red_ratio=LOOK_RED_EXPONENT, blue_ratio=0.55,
                  provenance="本测试：追加⑩ 三臂实测的最小复现", frames=1)
    monkeypatch.setitem(FILM_TYPES, "probe-blue", ft)
    got = _roll("probe-blue")
    assert got.report.params.blue_ratio == pytest.approx(0.55)
    assert not np.array_equal(got.result.positive_linear, base.result.positive_linear)
    assert str(got.model_asset.get("asset_version", "")).startswith("film-type-probe-blue")


def test_known_keys_are_sorted_and_match_the_table():
    assert known_keys() == sorted(FILM_TYPES)


# ---------------------------------------------------------------- 卷级声明（㊶ 追加⑪）

def _roll_group(film_type: str):
    from negative_lab.library.models import RollGroup
    return RollGroup(id="r1", display_name="探针卷", frame_ids=["a", "b"],
                     film_type=film_type)


def test_roll_group_roundtrip_keeps_the_declaration():
    g = _roll_group("probe")
    assert RollGroupBack.from_dict(g.to_dict()).film_type == "probe"
    # 旧会话文件里没有这个键 ⇒ 未声明，不是"未知片种"
    d = g.to_dict()
    d.pop("film_type")
    assert RollGroupBack.from_dict(d).film_type == ""
    assert RollGroupBack.from_dict(g.to_dict() | {"film_type": 42}).film_type == ""


def test_frame_level_declaration_wins_over_roll_level():
    from negative_lab.library.models import FrameRecord, GeometryRegions
    g = _roll_group("probe")
    frames = [FrameRecord(source_path="a", geometry=GeometryRegions()),
              FrameRecord(source_path="b", geometry=GeometryRegions(), film_type="other")]
    g.apply_film_type(frames)
    assert [f.film_type for f in frames] == ["probe", "other"]


def test_unknown_roll_declaration_is_rejected_not_silently_dropped():
    assert any("未知片种" in e for e in _roll_group("no-such-stock").validate())
    assert _roll_group("").validate() == []          # 未声明是合法状态


def test_roll_level_declaration_reaches_the_backend(monkeypatch):
    """整条链：卷声明 → 落到帧 → process_roll 解析成资产 → 后端参数变。"""
    from negative_lab.library.models import FrameRecord, GeometryRegions
    ft = FilmType(key="probe-roll", label="卷级探针", roll="probe_roll",
                  red_ratio=LOOK_RED_EXPONENT, blue_ratio=0.62,
                  provenance="本测试：卷级声明端到端", frames=1)
    monkeypatch.setitem(FILM_TYPES, "probe-roll", ft)
    g = _roll_group("probe-roll")
    img = _neg()
    recs = [FrameRecord(source_path="a", source_sha256="0" * 64,
                        geometry=GeometryRegions(), orientation_decided=True)]
    g.apply_film_type(recs)
    _, results = process_roll([(recs[0], img)])
    assert results[0].report.params.blue_ratio == pytest.approx(0.62)
    assert str(results[0].model_asset.get("asset_version")) == "film-type-probe-roll-v1"


def test_declare_film_type_is_the_single_entry(tmp_path, monkeypatch):
    """`ImportSession.declare_film_type` 是 CLI 与将来 GUI 共用的唯一入口。"""
    from negative_lab.library.models import FrameRecord, ImportSession, RollGroup
    ft = FilmType(key="probe-sess", label="会话探针", roll="probe_roll",
                  red_ratio=LOOK_RED_EXPONENT, blue_ratio=0.80,
                  provenance="本测试：会话往返", frames=1)
    monkeypatch.setitem(FILM_TYPES, "probe-sess", ft)
    rec = FrameRecord(source_path="a.NEF", source_sha256="0" * 64,
                      geometry=GeometryRegions(), roll_id="r1")
    sess = ImportSession(files=[rec],
                         roll_groups=[RollGroup(id="r1", display_name="卷一",
                                                frame_ids=["a.NEF"])])
    with pytest.raises(ValueError, match="未知片种"):
        sess.declare_film_type("r1", "no-such")
    with pytest.raises(KeyError):
        sess.declare_film_type("没有这卷", "probe-sess")
    sess.declare_film_type("卷一", "probe-sess")          # 按显示名也要能找到
    p = sess.save(tmp_path / "s.json")
    back = ImportSession.load(p)
    assert back.roll("r1").film_type == "probe-sess"
    sess.declare_film_type("r1", "")                       # 空键 = 取消声明
    assert sess.roll("r1").film_type == ""
