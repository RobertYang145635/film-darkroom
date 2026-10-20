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


def test_roll_caption_shows_declared_stock(monkeypatch):
    """左栏卷名必须带上已声明的片种（声明了却看不见 = 用户下一轮只能猜）。

    只调静态方法，不建 widget、不建 QApplication —— 这条锁要能在无头环境跑，
    不能把 B-8 那类 GUI 崩溃引进单测全量门。
    """
    from types import SimpleNamespace
    from negative_lab.ui.library_panel import LibraryPanel
    monkeypatch.setitem(FILM_TYPES, "probe-cap", FilmType(
        key="probe-cap", label="探针卷", roll="probe_roll",
        red_ratio=LOOK_RED_EXPONENT, blue_ratio=LOOK_BLUE_EXPONENT,
        provenance="本测试：卷名可见性", frames=1))
    cap = LibraryPanel._roll_caption
    assert cap(SimpleNamespace(display_name="5207", film_type="probe-cap")) == "5207｜探针卷"
    # 未声明 / 键被从表里删掉：都只显示卷名，不显示空壳分隔符
    assert cap(SimpleNamespace(display_name="5207", film_type="")) == "5207"
    assert cap(SimpleNamespace(display_name="5207", film_type="gone")) == "5207"


# ------------------------------------------------------- 印相中灰目标按片种（㊶ 追加⑯）
def _ft_aim(key: str, aim: float) -> FilmType:
    return FilmType(key=key, label=f"探针 aim {aim}", roll="probe_roll",
                    red_ratio=LOOK_RED_EXPONENT, blue_ratio=LOOK_BLUE_EXPONENT,
                    print_aim=aim, provenance="本测试：印相中灰目标按片种", frames=1)


def test_print_aim_out_of_range_is_rejected():
    """0 表示"不施加"；非零必须落在纸级反射率的合理区间。"""
    for bad in (0.01, 0.60, -0.2):
        with pytest.raises(ValueError, match="print_aim"):
            _ft_aim(f"probe-aim-bad-{bad}", bad)
    assert _ft_aim("probe-aim-zero", 0.0).print_aim == 0.0


def test_print_aim_reaches_the_render_and_changes_pixels(monkeypatch):
    """声明带 aim 的片种 ⇒ 意图里看得见、像素真的变了（机制不是装饰）。"""
    monkeypatch.setitem(FILM_TYPES, "probe-aim-on", _ft_aim("probe-aim-on", 0.10))
    pr = _roll("probe-aim-on")
    assert float(pr.render_intent.print_aim) == pytest.approx(0.10)
    assert pr.render_intent.to_dict().get("print_aim") == pytest.approx(0.10)
    off = _roll("")
    assert float(off.render_intent.print_aim) == 0.0
    assert not np.array_equal(pr.rendered_positive_linear,
                              off.rendered_positive_linear)


def test_print_aim_zero_is_bit_identical_to_undeclared(monkeypatch):
    """退化控制：`print_aim=0` 的片种必须与"什么都不声明"**逐比特相同**。

    这条过了才可以说"老用户与未声明的卷不受这次改动影响"——
    斜率那一维已有同型锁（`test_declaring_production_slopes_is_a_no_op`），
    这一维是新加的通道，必须单独钉。
    """
    monkeypatch.setitem(FILM_TYPES, "probe-aim-off", _ft_aim("probe-aim-off", 0.0))
    a = _roll("probe-aim-off")
    b = _roll("")
    assert np.array_equal(a.rendered_positive_linear, b.rendered_positive_linear)
    assert "print_aim" not in a.render_intent.to_dict()


def test_old_intent_schema_without_print_aim_still_loads():
    """旧 sidecar/草稿里没有 `print_aim` ⇒ 读成 0（不施加），不是报错也不是猜值。"""
    from negative_lab.library.render import RenderIntent
    it = RenderIntent.from_dict({"print_anchor": [0.05, 0.40, 1.20], "print_dmax": 1.3})
    assert float(it.print_aim) == 0.0
    assert float(RenderIntent.from_dict(it.to_dict()).print_aim) == 0.0


def test_print_remap_array_default_targets_are_bit_identical():
    """`aim`/`white` 不传 ⇒ 与模块默认逐比特同（新形参不许改变现产行为）。"""
    from negative_lab import print_math
    from negative_lab.print_math import print_remap_array
    anchor = (0.05, 0.40, 1.20)
    x = np.random.default_rng(7).uniform(0.02, 0.9, (24, 40, 3)).astype(np.float32)
    d0, _ = print_remap_array(x, anchor, 1.3)
    d1, _ = print_remap_array(x, anchor, 1.3, aim=print_math.PRINT_AIM,
                              white=print_math.PRINT_WHITE)
    assert np.array_equal(d0, d1)
    d2, _ = print_remap_array(x, anchor, 1.3, aim=0.10)
    assert not np.array_equal(d0, d2)


def test_print_scale_helpers_must_use_the_same_aim_as_the_print():
    """㊶ 追加⑯ 上线时抓到的真缺陷：辅助判断各自偷偷用**默认 aim** 验曲线。

    片种把 aim 降到 0.10 之后，"三点曲线能不能解"这道闸在 `service` 里按 0.10 过了，
    紧接着 `window_y_med` 又按默认 0.18 去解同一条曲线 ⇒ 直接抛
    `ValueError: 打印曲线斜率未递增（趾部 0.709 ≥ 高光 0.574）`，整卷导出崩。
    这条锁钉两件事：① aim 必须**真的被转发**（不是签名里有、体里不用）；
    ② 解不出时 `window_y_med` 按它自己的契约返回 `None`，不许把异常抛给调用方。
    """
    from negative_lab.print_math import (print_curve, window_y_med,
                                         white_window_floor, anchor_with_level)
    good = (0.05, 0.50, 1.30)
    #: ① 转发有效：同一根锚，换 aim 必须换出不同的纸级亮度
    y_def = window_y_med(good, 0.50)
    y_low = window_y_med(good, 0.50, aim=0.10)
    assert y_def is not None and y_low is not None and y_def != y_low
    xs, ys = print_curve(good, aim=0.10)
    import math as _m
    assert _m.isclose(y_low, float(1.0 - 10.0 ** (-float(np.interp(0.50, xs, ys)))),
                      rel_tol=1e-9)
    #: 其余两个辅助也不能忽略 aim（它们内部要验曲线，且必须能正常返回）
    assert anchor_with_level(good, 0.50, 0.2, aim=0.10)[1] == \
        anchor_with_level(good, 0.50, 0.2)[1]          # δ 与 aim 无关，但不许抛
    assert white_window_floor(good, density_median=0.50, aim=0.10)[0] is not None
    #: ② 反例：这根锚在默认 aim 下解不出（趾部斜率 > 高光），必须返回 None 而不是抛
    bad = (0.05, 0.06, 1.30)
    with pytest.raises(ValueError):
        print_curve(bad)
    assert window_y_med(bad, 0.50) is None
    #: 负控制：把 aim 传进去也不许把异常漏出去
    assert window_y_med(bad, 0.50, aim=0.10) is None
