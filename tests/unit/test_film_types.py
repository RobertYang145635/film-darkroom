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


def test_declared_asset_provenance_is_visible_in_the_delivered_record(monkeypatch):
    """交付记录必须答得出"这帧用了哪个片种、打印目标是多少"。

    只有 `asset_version` 是不够的：`print_aim` 不改任何反相参数，
    它在 sidecar 里会完全隐形（㊶ 追加⑯ 建探针时实测 `film_type` 读出来是 `None`，
    只能靠解析版本字符串反推——那正是"证据链在交付记录里看不见"这一类缺陷）。
    """
    monkeypatch.setitem(FILM_TYPES, "probe-vis", FilmType(
        key="probe-vis", label="溯源探针", roll="probe_roll",
        red_ratio=LOOK_RED_EXPONENT, blue_ratio=LOOK_BLUE_EXPONENT,
        print_aim=0.10, provenance="本测试：交付记录可见性", frames=7))
    info = _roll("probe-vis").model_asset
    assert info.get("film_type") == "probe-vis"
    assert float(info.get("print_aim")) == pytest.approx(0.10)
    assert int(info.get("frames")) == 7
    assert "溯源探针" not in str(info.get("provenance"))   # 出处是测量描述，不是 label
    assert str(info.get("provenance")).startswith("本测试")
    #: 负控制：未声明的默认资产没有这些字段，不该被凭空造出来
    off = _roll("").model_asset
    assert "film_type" not in off and "print_aim" not in off


# ------------------------------------------- 纸白端彩度收回档按片种（㊶ 追加/㉟）
def _ft_chg(key: str, chg: float) -> FilmType:
    return FilmType(key=key, label=f"探针高光档 {chg}", roll="probe_roll",
                    red_ratio=LOOK_RED_EXPONENT, blue_ratio=LOOK_BLUE_EXPONENT,
                    chroma_highlight_gain=chg,
                    provenance="本测试：纸白端彩度收回档按片种", frames=1)


def test_chroma_highlight_gain_out_of_range_is_rejected():
    """0 = 不施加；非零必须落在量过的区间（表里不许出现没量过的档）。"""
    for bad in (0.01, 1.80, -0.5):
        with pytest.raises(ValueError, match="chroma_highlight_gain"):
            _ft_chg(f"probe-chg-bad-{bad}", bad)
    assert _ft_chg("probe-chg-zero", 0.0).chroma_highlight_gain == 0.0


def test_chroma_highlight_gain_reaches_the_render_and_is_recorded(monkeypatch):
    """声明带这一档的片种 ⇒ 意图看得见、像素真的变、**sidecar 记的是生效值**。"""
    monkeypatch.setitem(FILM_TYPES, "probe-chg-on", _ft_chg("probe-chg-on", 0.55))
    pr = _roll("probe-chg-on")
    assert float(pr.render_intent.print_chroma_highlight_gain) == pytest.approx(0.55)
    assert (pr.render_intent.to_dict().get("print_chroma_highlight_gain")
            == pytest.approx(0.55))
    off = _roll("")
    assert float(off.render_intent.print_chroma_highlight_gain) == 0.0
    assert not np.array_equal(pr.rendered_positive_linear,
                              off.rendered_positive_linear)
    rec = pr.meta["print_scale"]
    assert rec["chroma_highlight_gain"] == pytest.approx(0.55)


def test_chroma_highlight_gain_zero_is_bit_identical_to_undeclared(monkeypatch):
    """退化控制：`chroma_highlight_gain=0` 必须与"什么都不声明"逐比特相同
    ⇒ 未声明的卷（含 G200 那两卷）一个字都不受影响。"""
    monkeypatch.setitem(FILM_TYPES, "probe-chg-off", _ft_chg("probe-chg-off", 0.0))
    a = _roll("probe-chg-off")
    b = _roll("")
    assert np.array_equal(a.rendered_positive_linear, b.rendered_positive_linear)
    assert "print_chroma_highlight_gain" not in a.render_intent.to_dict()


def test_print_remap_records_effective_targets_not_module_defaults():
    """㉟ 补的真缺陷：`info` 从前写死模块常数，于是按片种传了 `aim` 的帧
    在交付记录里被记成 0.18（现产），**记录说一套、像素做一套**。"""
    from negative_lab import print_math
    from negative_lab.print_math import print_remap_array
    x = np.random.default_rng(5).uniform(0.02, 0.9, (16, 24, 3)).astype(np.float32)
    anchor = (0.05, 0.40, 1.20)
    _out, info = print_remap_array(x, anchor, 1.3, aim=0.10, white=0.75,
                                   highlight_gain=0.55)
    assert info["aim"] == pytest.approx(0.10), "生效的 aim 没记进 sidecar"
    assert info["white"] == pytest.approx(0.75), "生效的 white 没记进 sidecar"
    assert info["chroma_gain_highlight"] == pytest.approx(0.55)
    _o2, i2 = print_remap_array(x, anchor, 1.3)
    assert i2["aim"] == pytest.approx(print_math.PRINT_AIM)
    assert i2["chroma_gain_highlight"] == pytest.approx(
        print_math.PRINT_CHROMA_HIGHLIGHT_GAIN)


def test_chroma_gain_at_default_is_bit_identical_and_only_moves_highlights():
    """不传 = 现产逐比特；传值只动纸级末端以上，中调一个比特都不许变。"""
    from negative_lab import print_math
    from negative_lab.print_math import chroma_gain_at
    y = np.linspace(0.0, print_math.PRINT_WHITE, 400).astype(np.float32)
    a = chroma_gain_at(y, 2.50)
    b = chroma_gain_at(y, 2.50, highlight_gain=print_math.PRINT_CHROMA_HIGHLIGHT_GAIN)
    assert np.array_equal(a, b)
    c = chroma_gain_at(y, 2.50, highlight_gain=0.55)
    mid = y < print_math.PRINT_WHITE * print_math.PRINT_CHROMA_HIGHLIGHT_START
    assert np.array_equal(a[mid], c[mid]), "斜坡起点以下就被动了"
    assert float(np.abs(a[~mid] - c[~mid]).max()) > 0.05, "纸白端没收到 ⇒ 这一维是空转"


def test_intent_with_only_chroma_highlight_gain_does_not_claim_identity():
    """与 `white_balance` 同型：可脱离锚点单独施加的处理不许自称恒等。"""
    from negative_lab.library.render import RenderIntent
    assert RenderIntent(print_chroma_highlight_gain=0.55).is_identity() is False
    assert RenderIntent().is_identity() is True


# ------------------------------------------- 外观级饱和度按片种（B-16 追加㊹）
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


# ------------------------------------------- 纸白输出目标按片种（㊫）
def _ft_white(key: str, pw: float, aim: float = 0.0) -> FilmType:
    return FilmType(key=key, label=f"探针纸白 {pw}", roll="probe_roll",
                    red_ratio=LOOK_RED_EXPONENT, blue_ratio=LOOK_BLUE_EXPONENT,
                    print_white=pw, print_aim=aim,
                    provenance="本测试：纸白输出目标按片种", frames=1)


def test_print_white_out_of_range_is_rejected():
    """0 = 不施加；非零必须落在量过的区间（0.60~0.98），表里不许出现没量过的档。"""
    for bad in (0.30, 1.20, -0.5):
        with pytest.raises(ValueError, match="print_white"):
            _ft_white(f"probe-pw-bad-{bad}", bad)
    assert _ft_white("probe-pw-zero", 0.0).print_white == 0.0


def test_print_white_reaches_the_render_and_is_recorded(monkeypatch):
    """生产路径：声明纸白 ⇒ 意图看得见、像素真的变、sidecar 记的是生效值。"""
    monkeypatch.setitem(FILM_TYPES, "probe-pw-on", _ft_white("probe-pw-on", 0.845))
    pr = _roll("probe-pw-on")
    assert float(pr.render_intent.print_white) == pytest.approx(0.845)
    assert pr.render_intent.to_dict().get("print_white") == pytest.approx(0.845)
    off = _roll("")
    assert float(off.render_intent.print_white) == 0.0
    assert not np.array_equal(pr.rendered_positive_linear,
                              off.rendered_positive_linear), "纸白没改变像素 ⇒ 这一维是空转"
    assert pr.meta["print_scale"]["print_white"] == pytest.approx(0.845)


def test_print_white_zero_is_bit_identical_to_undeclared(monkeypatch):
    """退化控制：`print_white=0` 必须与"什么都不声明"逐比特相同
    ⇒ 没标纸白的卷（含 G200 两卷）一个字都不受影响。"""
    monkeypatch.setitem(FILM_TYPES, "probe-pw-off", _ft_white("probe-pw-off", 0.0))
    a = _roll("probe-pw-off")
    b = _roll("")
    assert np.array_equal(a.rendered_positive_linear, b.rendered_positive_linear)
    assert "print_white" not in a.render_intent.to_dict()


def test_print_white_is_the_same_ruler_in_all_four_reads(monkeypatch):
    """㊫ 的真风险：纸白在 `_remap_band` 里是**四件事**的尺（曲线高光端点、整像素同除的
    天花板、彩度斜坡的归一点、顶破纸白的剩余空间）。探针当年改的是**模块全局**，
    一次动四件；上线走**逐帧参数**，只转发给曲线的话只动一件 ⇒
    交付出去的像素不等于量过的那组数（纪律 12：组合口径就是模型）。

    两条对照都要过：
    ① 显式 `white=` == 改全局（同一条语义的两条入口，逐比特）；
    ② 显式 `white=PRINT_WHITE` == 不传（"用默认"必须是死值，不是另一个臂）。
    """
    from negative_lab import print_math
    from negative_lab.print_math import print_remap_array
    #: 激励必须**同时**打到四个读取点：够亮（`lum` 让曲线把它抬过纸白 ⇒ `gain>1` ⇒ `cap` 生效）、
    #: 通道差够大（顶破纸白 ⇒ `slack`/`k2` 生效）、亮度落在彩度斜坡以上（`ramp` 生效）。
    #: 第一版用 `uniform(0.02, 0.97)` + 一条把大部分像素**压暗**的锚 ⇒ `gain ≤ 1`，
    #: `np.minimum(gain, cap, where=gain>1)` 根本不写，把 `cap` 那一处改回模块全局
    #: 测试照样绿（当场用"逐处回退"的负控制抓到，四只回退三处才红）。
    rng = np.random.default_rng(7)
    x = rng.uniform(0.62, 0.99, (32, 48, 3)).astype(np.float32)
    anchor = (0.05, 0.30, 0.62)      # `d_hi` 小 ⇒ 曲线把这批亮像素**提亮**到纸白附近
    dmax_old = 1.3
    dflt = print_math.PRINT_WHITE
    out_default, _i0 = print_remap_array(x, anchor, dmax_old, aim=0.07)
    out_explicit_default, _i1 = print_remap_array(x, anchor, dmax_old, aim=0.07, white=dflt)
    assert np.array_equal(out_default, out_explicit_default), "② 传默认值就不等于默认值"

    monkeypatch.setattr(print_math, "PRINT_WHITE", 0.845)
    out_patch, _i2 = print_remap_array(x, anchor, dmax_old, aim=0.07)      # 探针口径
    monkeypatch.undo()
    out_arg, i3 = print_remap_array(x, anchor, dmax_old, aim=0.07, white=0.845)
    assert np.array_equal(out_patch, out_arg), \
        "① 改全局与逐帧传值不等价 ⇒ 纸白还有没被转发的读取点，上线的不是量到的那个东西"
    assert not np.array_equal(out_arg, out_default), "非空控制：0.845 必须真的改变像素"
    assert i3["white"] == pytest.approx(0.845)
    #: 激励本身要非空转：这批像素必须真的**碰过天花板**（`chroma_clip_frac>0` ⇒ `slack` 被走过），
    #: 且必须真的有 `gain>1` 的像素（否则 `cap` 那一处又是空的）。
    assert i3["chroma_clip_frac"] > 0.0, "激励没顶破纸白 ⇒ `slack` 那一处没被这个测试覆盖"
    _o, i_dflt = print_remap_array(x, anchor, dmax_old, aim=0.07)
    assert i_dflt["chroma_clip_frac"] != i3["chroma_clip_frac"], \
        "收回系数与纸白无关 ⇒ 这一维又是空转"
    #: 斜坡的尺也要跟着挪：同一档 `highlight_gain` 下，把纸白抬高应让**高光末端**变化，
    #: 而纸级中段（远低于两把尺的斜坡起点）一个比特都不许动。
    y = np.linspace(0.0, 0.845, 300).astype(np.float32)
    g_lo = print_math.chroma_gain_at(y, 2.50, highlight_gain=0.45, white=dflt)
    g_hi = print_math.chroma_gain_at(y, 2.50, highlight_gain=0.45, white=0.845)
    mid = y < dflt * print_math.PRINT_CHROMA_HIGHLIGHT_START
    assert np.array_equal(g_lo[mid], g_hi[mid]), "斜坡起点以下就被纸白动了"
    assert float(np.abs(g_lo[~mid] - g_hi[~mid]).max()) > 1e-6, "纸白没参与斜坡 ⇒ 转发了个空值"


def test_intent_with_only_print_white_does_not_claim_identity():
    from negative_lab.library.render import RenderIntent
    assert RenderIntent(print_white=0.845).is_identity() is False
    assert RenderIntent(print_white=0.845).copy().print_white == pytest.approx(0.845)


def test_ecn2_stocks_split_on_the_paper_white_cell():
    """㊯：这一格**按片种分道**——5207 保留，5219 退回。

    #36 把两臂放到逐帧对过网格的裁块上重跑（对齐变换只由基线臂定一次、两臂共用），
    5207 的 HOLD 降幅在对齐口径下原样站住（−14.9%，暗/亮六端 6/6 变好），
    5219 从 −17.2% 塌到 **−3.0%** ⇒ ㊫ 登记的本卷数字约 82% 是 ㊭ 那个 2~4% 网格错位的产物。
    按 #36 预写的 A3 判据依据公开撤回，产品去留交看图（`ship_vs_base_5219.png` 8 帧读不出
    「明显更像他」）⇒ 本卷退回 `print_white=0`（不施加）与 `print_aim=0.10`（㊶ 原值）。
    这条锁的存在本身就是结论：**同格要求的是效果同格，不是参数同格**（规矩 25）。
    """
    keep = FILM_TYPES["ecn2-5207"]
    assert float(keep.print_white) == pytest.approx(0.845)
    assert float(keep.print_aim) == pytest.approx(0.07)
    back = FILM_TYPES["ecn2-5219"]
    assert float(back.print_white) == 0.0, "本卷纸白必须退回不施加（㊯）"
    assert float(back.print_aim) == pytest.approx(0.10), "本卷 aim 必须退回 ㊶ 的原值"
    #: 退回之后本卷的其余三项按片种参数不许被牵连（㊶/㉚/㊱ 的账各自独立）
    assert (float(back.red_ratio), float(back.blue_ratio)) == pytest.approx((1.70, 0.65))
    assert float(back.white_balance) == pytest.approx(1.80)
    assert float(back.chroma_highlight_gain) == pytest.approx(0.45)
