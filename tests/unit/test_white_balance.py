r"""高光带白点蓝黄校正（`white_balance`）的锁（㊶ 追加㉙/㉚ 的实现层）。

这个机制的两条主张各自要有锁，而且**都要有反方向的控制**：
 ① 把"该是中性的那批高光"的 `b*` 拉向 0 —— 反向（把方向取负）必须**更差**；
 ② 不许动整幅的红绿账（`a*`）—— 这是它区别于 ㉘/㉙ 前两版的地方，
    前两版就是死在这条上（5219 整幅 `|a|` 付 +0.15）。
另外三条是通道层的：`strength=0` 逐比特等于不施加（未声明的卷不受影响）、
旧 schema 没有这个键要读成 0、应用步**逐像素**所以分块必须与整帧一致
（`print_math` 那条同型锁管不到这一步）。
"""
from __future__ import annotations

import numpy as np
import pytest

from negative_lab.diagnostics.colorimetry import srgb_to_lab
from negative_lab.library.render import RenderIntent, apply_render
from negative_lab.library.models import FrameRecord, GeometryRegions
from negative_lab.pipeline.density import encode_srgb
from negative_lab.pipeline.film_types import FILM_TYPES, FilmType
from negative_lab.pipeline.service import process_roll
from negative_lab.print_math import (LOOK_BLUE_EXPONENT, LOOK_RED_EXPONENT,
                                     PRINT_WHITE)
from negative_lab.white_balance import (apply_white_balance, highlight_luma_start,
                                        highlight_weight, neutral_highlight_mask,
                                        white_gain_direction)


# ---------------------------------------------------------------- 合成输入
def _positive_with_blue_whites(h: int = 240, w: int = 320) -> np.ndarray:
    """一张"白大褂发蓝"的正片：横向渐变到纸白，亮部抬蓝压红。"""
    ramp = np.linspace(0.02, 0.80, w, dtype=np.float32)[None, :]              # (1,w)
    x = np.repeat(np.repeat(ramp[..., None], 3, axis=2), h, axis=0)           # (h,w,3)
    hi = (ramp > 0.58)[..., None] * np.array([-0.10, 0.0, 0.13], np.float32)
    x = x * (1.0 + hi)
    # 顶部一条暗暖带：让"最亮 20%"这一筛真的有东西被筛掉（全集会掩盖选取逻辑）
    x[: h // 4] = np.array([0.30, 0.26, 0.20], np.float32)
    return np.clip(x, 0.0, 1.0).astype(np.float32)


def _white_ab(lin: np.ndarray) -> tuple[float, float]:
    disp = encode_srgb(np.clip(lin, 0, 1))[..., :3]
    lab = srgb_to_lab(disp).reshape(*disp.shape[:2], 3)
    sel = neutral_highlight_mask(lab)
    if not sel.any():
        return 0.0, 0.0
    return float(lab[..., 1][sel].mean()), float(lab[..., 2][sel].mean())


def _global_a(lin: np.ndarray) -> float:
    disp = encode_srgb(np.clip(lin, 0, 1))[..., :3]
    return float(srgb_to_lab(disp).reshape(*disp.shape[:2], 3)[..., 1].mean())


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


def _ft_wb(key: str, wb: float) -> FilmType:
    return FilmType(key=key, label=f"探针白平衡 {wb}", roll="probe_roll",
                    red_ratio=LOOK_RED_EXPONENT, blue_ratio=LOOK_BLUE_EXPONENT,
                    white_balance=wb, provenance="本测试：高光带白点蓝黄校正", frames=1)


def _roll(film_type: str):
    _, results = process_roll([(_rec(film_type), _neg())])
    return results[0]


# ---------------------------------------------------------------- 机制本身
def test_solve_pulls_white_toward_neutral_and_holds_global_a_star():
    """① 白的 `b*` 被拉向 0；② 整幅平均 `a*` 不许跟着跑（这是版 D 的立身之本）。"""
    x = _positive_with_blue_whites()
    a0, b0 = _white_ab(x)
    assert b0 < -0.5, f"合成输入本身不够蓝（b*={b0}），这条锁会空转"
    g = white_gain_direction(x, PRINT_WHITE)
    out, info = apply_white_balance(x, 1.0, PRINT_WHITE, direction=g)
    a1, b1 = _white_ab(out)
    assert info["applied"] is True
    assert abs(b1) < abs(b0) * 0.5, f"白点没被拉近中性：{b0:.3f} → {b1:.3f}"
    assert abs(b1) < 1.0, f"白点仍偏蓝：b*={b1:.3f}"
    assert abs(_global_a(out) - _global_a(x)) < 0.05, "整幅红绿账被动了"


def test_reversed_direction_is_worse_than_doing_nothing():
    """方向控制：把解取反必须**更蓝**。没有这条，上一条可能只是"随便动了一下"。"""
    x = _positive_with_blue_whites()
    _a0, b0 = _white_ab(x)
    g = white_gain_direction(x, PRINT_WHITE)
    worse, _ = apply_white_balance(x, 1.0, PRINT_WHITE, direction=(-g[0], -g[1]))
    assert abs(_white_ab(worse)[1] - 0.0) > abs(_white_ab(x)[1] - 0.0) * 1.2, (
        f"反向没有变差（{b0:.3f} → {_white_ab(worse)[1]:.3f}）⇒ 解的方向是噪声")


def test_already_neutral_frame_self_disables():
    """没有蓝罩的帧不许被硬修：解 ≈ 0 ⇒ `applied=False`，像素逐比特不动。"""
    ramp = np.linspace(0.05, 0.75, 200, dtype=np.float32)[None, :]
    x = np.clip(np.repeat(ramp[..., None], 3, axis=2), 0, 1).astype(np.float32)
    g = white_gain_direction(x, PRINT_WHITE)
    assert max(abs(g[0]), abs(g[1])) < 1e-3, f"中性输入却解出了方向 {g}"
    out, info = apply_white_balance(x, 1.3, PRINT_WHITE)
    assert info["applied"] is False and info["reason"] == "white_already_neutral"
    assert np.array_equal(out, x)


def test_strength_is_zero_at_zero_and_monotone_in_highlights():
    """`k=0` 逐比特不变；`k` 越大白的 `b*` 越往正方向走（单峰扫描的前提）。"""
    x = _positive_with_blue_whites()
    g = white_gain_direction(x, PRINT_WHITE)
    off, _info = apply_white_balance(x, 0.0, PRINT_WHITE, direction=g)
    assert np.array_equal(off, x)
    b0 = _white_ab(x)[1]
    bs = []
    for k in (0.5, 1.0, 1.5):
        y, _ = apply_white_balance(x, k, PRINT_WHITE, direction=g)
        bs.append(_white_ab(y)[1])
    # 剂量-反应：半程只走一半、全程落到中性、超程不许飞出去（`k` 是"打到几成"）
    assert b0 < bs[0] < -0.5 * abs(b0), f"半程位移不对：{b0} → {bs[0]}"
    assert abs(bs[1]) < 0.5, f"整程没落到中性：{bs[1]}"
    assert abs(bs[2]) < 1.0, f"超程过冲太大：{bs[2]}"


def test_weight_is_a_per_pixel_function_and_chunks_match_whole_frame_bit_exact():
    """应用是**逐像素**的（只用该像素自己的亮度与两个常数）⇒ 分块必须逐比特一致。

    与 `test_print_scale.py::test_chunked_matches_whole_frame_bit_exact` 同型：
    那一条管印相重映射，这一步在它之后、独立的处理，必须单独钉。
    """
    x = _positive_with_blue_whites(h=211, w=337)      # 奇数高，切块必然不等分
    w = highlight_weight(x, PRINT_WHITE)
    y = np.exp(np.clip(w[..., None] * np.array([0.07, 0.0, -0.12], np.float32),
                       -0.5, 0.5)).astype(np.float32)
    whole = np.clip(x * y, 0.0, 1.0)
    parts = []
    for r0 in range(0, x.shape[0], 37):
        ww = highlight_weight(x[r0:r0 + 37], PRINT_WHITE)
        yy = np.exp(np.clip(ww[..., None] * np.array([0.07, 0.0, -0.12], np.float32),
                            -0.5, 0.5)).astype(np.float32)
        parts.append(np.clip(x[r0:r0 + 37] * yy, 0.0, 1.0))
    assert np.array_equal(whole, np.concatenate(parts, axis=0))


def test_apply_white_balance_is_row_blocked_and_matches_whole_frame_bit_exact():
    """应用是**按行分块**执行的（45MP 帧一次性做会重演 B-11 追加⑦ 的段错误），
    而分块不许改变一个比特——奇数高度配 `cap=1` 把每行都切成独立块。"""
    x = _positive_with_blue_whites(h=211, w=337)
    g = white_gain_direction(x, PRINT_WHITE)
    whole, rec = apply_white_balance(x, 1.3, PRINT_WHITE, direction=g,
                                     max_chunk_pixels=10 ** 9)
    assert rec["weight_pixels"] == x.shape[0] * x.shape[1]
    for cap in (1, 7, 33 * 337, 211 * 337):
        part, _ = apply_white_balance(x, 1.3, PRINT_WHITE, direction=g,
                                      max_chunk_pixels=cap)
        assert np.array_equal(whole, part), f"cap={cap} 的分块结果与整帧不同"


def test_band_edges_come_from_existing_constants():
    """权重带沿不许是拍出来的数：起点 = `L*=80` 的严格反解，终点 = 纸白。"""
    lo = highlight_luma_start(PRINT_WHITE)
    assert 0.7 < lo < 0.85, f"带起点异常：{lo}"
    hi = float(encode_srgb(np.array([PRINT_WHITE], np.float64))[0])
    #: ㊫ 之后纸白可以按片种给（`RenderIntent.print_white`），这两个带沿就是它唯一的落点，
    #: 所以带次序必须钉住：起点在纸白的编码值之下，且整个带还在显示域量程内。
    assert lo < hi < 1.0, f"权重带沿次序不对：{lo} / {hi}"
    x = _positive_with_blue_whites()
    w = highlight_weight(x, PRINT_WHITE)
    assert w.min() == 0.0 and w.max() > 0.9, "权重没有覆盖到纸白端"
    assert float(np.percentile(w, 50)) < 0.2, "一半像素都在高光带里 ⇒ 带太宽，等于全帧"


# ---------------------------------------------------------------- 通道与 schema
def test_white_balance_reaches_the_render_and_changes_pixels(monkeypatch):
    """声明带强度的片种 ⇒ 意图看得见、像素真的变、交付元数据留痕。"""
    monkeypatch.setitem(FILM_TYPES, "probe-wb-on", _ft_wb("probe-wb-on", 1.3))
    pr = _roll("probe-wb-on")
    assert float(pr.render_intent.white_balance) == pytest.approx(1.3)
    assert pr.render_intent.to_dict().get("white_balance") == pytest.approx(1.3)
    off = _roll("")
    assert float(off.render_intent.white_balance) == 0.0
    assert not np.array_equal(pr.rendered_positive_linear,
                              off.rendered_positive_linear)
    assert pr.meta["print_scale"]["white_balance_strength"] == pytest.approx(1.3)


def test_white_balance_zero_is_bit_identical_to_undeclared(monkeypatch):
    """退化控制：`white_balance=0` 的片种必须与"什么都不声明"逐比特相同。

    未声明的卷（含 G200 那两卷）靠这条才敢说"一个字都没变"。
    """
    monkeypatch.setitem(FILM_TYPES, "probe-wb-off", _ft_wb("probe-wb-off", 0.0))
    a = _roll("probe-wb-off")
    b = _roll("")
    assert np.array_equal(a.rendered_positive_linear, b.rendered_positive_linear)
    assert "white_balance" not in a.render_intent.to_dict()
    assert a.meta["print_scale"].get("white_balance_strength") in (None, 0.0)


def test_old_intent_schema_without_white_balance_still_loads():
    """旧 sidecar/草稿里没有这个键 ⇒ 读成 0（不施加），不报错也不猜值。"""
    it = RenderIntent.from_dict({"print_anchor": [0.05, 0.40, 1.20], "print_dmax": 1.3})
    assert float(it.white_balance) == 0.0
    assert float(RenderIntent.from_dict(it.to_dict()).white_balance) == 0.0


def test_intent_with_only_white_balance_does_not_claim_identity():
    """只改白平衡的意图不许自称恒等——否则 `apply_render` 的恒等短路会把它悄悄丢掉。"""
    assert RenderIntent(white_balance=1.3).is_identity() is False
    assert RenderIntent(print_aim=0.10).is_identity() is False
    assert RenderIntent().is_identity() is True


def test_out_of_range_strength_is_rejected():
    """0 表示"不施加"；非零必须落在扫描过的区间内（表里不许出现没量过的强度）。"""
    for bad in (0.01, 6.0, -1.3):
        with pytest.raises(ValueError, match="white_balance"):
            _ft_wb(f"probe-wb-bad-{bad}", bad)
    assert _ft_wb("probe-wb-zero", 0.0).white_balance == 0.0


def test_apply_render_with_zero_strength_is_bit_identical():
    """现产行为：`white_balance` 不设 ⇒ 渲染逐比特不变（新旧代码同一输出）。"""
    x = _positive_with_blue_whites(h=120, w=160)
    it = RenderIntent(print_anchor=(0.05, 0.40, 1.20), print_dmax=1.3)
    a = apply_render(x, it)
    b = apply_render(x, RenderIntent.from_dict({**it.to_dict(), "white_balance": 0.0}))
    assert np.array_equal(a, b)
