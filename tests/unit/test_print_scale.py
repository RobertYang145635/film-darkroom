"""锁：自动打印尺度（B-11 的"一键就要好看"那一级）。

**为什么是曲线而不是指数**（2026-10-08 实测）：纸级形式 `paper = 1 − 10^(−v)`
要求"黑→中灰"与"中灰→白"的 v 比是 **1 : 8.1**，而实测画面密度里这两段几乎等宽
（0.479 vs 0.512 decade）⇒ `(dmax, offset)` 只能整体缩放，表达不了这个比例。
三个横坐标来自**这一帧自己的分位**（p5/p50/p95），且取在 `picture_density` 域
（技术正片亮度按 `−dmax·log10(1−y)` 反解，即重映射自己会读到的那个域）——
取在逐通道密度 luma 上时黑点只咬住名义像素的 1/50，见
`test_black_anchor_really_bites_in_its_own_domain`。
两个纵坐标由纸级常数反解：`AIM=0.18`（horizon `Paper.printAim` 与 filmeon 同值）、
`WHITE=0.80375`（实测 Endura 相纸最大反射率）——**没有一个数拟合自用户图片**。

**最重要的一条锁是"色度不变"**：第一版把同一条曲线逐通道施加到密度上，
对比确实上来了，但天空变成深青蓝（饱和 p95 0.528 vs 目标 0.405）。
颜色改动必须由**已验证的颜色模型**做，不能由影调曲线夹带
（`service.py:383-388` 同一条纪律）。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from negative_lab.pipeline.density import InvertResult, encode_srgb, invert
from negative_lab.pipeline.params import InvertParams
from negative_lab.pipeline.print_scale import (MIN_SPAN, P_HIGH, P_LOW, P_MID,
                                               PRINT_AIM, PRINT_LEVEL_AREA_STRENGTH,
                                               PRINT_LEVEL_HIGHLIGHT_AREA_REFERENCE,
                                               PRINT_LEVEL_HIGHLIGHT_WINDOW,
                                               PRINT_LEVEL_MAX_SHIFT,
                                               PRINT_LEVEL_REFERENCE_DENSITY,
                                               PRINT_LEVEL_STRENGTH,
                                               PRINT_MIN_DENSITY_WINDOW, PRINT_WHITE,
                                               PRINT_WINDOW_YMED_REFERENCE,
                                               anchor_refusal_reason, anchor_with_level,
                                               apply_print_remap,
                                               centre_region, density_anchor,
                                               highlight_area_fraction, level_shift,
                                               picture_density, print_curve,
                                               print_remap_array, remap_luma,
                                               white_window_floor, window_y_med)

_LUM = np.array([0.2126, 0.7152, 0.0722])
_BASE = (0.25, 0.25, 0.25)


def _ramp(n: int = 256, base: float = 0.25) -> np.ndarray:
    """线性透射率 0.02→base 的一幅"灰阶负片"（每行一个值）。"""
    t = np.linspace(0.02, base, n, dtype=np.float64)
    return np.broadcast_to(t[:, None, None], (n, 4, 3)).copy()


def _tech(t: np.ndarray, base=_BASE, dmax: float = 2.0) -> np.ndarray:
    """透射率 → 技术正片（与生产同一 `invert`）。"""
    return invert(np.clip(t, 0, None),
                  InvertParams(dmin=tuple(base), dmax=dmax)).positive_linear


def _anchor(positive: np.ndarray, dmax: float = 2.0):
    """锚点取在**重映射自己会读到的域**——不是逐通道密度的 luma。"""
    return density_anchor(picture_density(positive, dmax))


# ------------------------------------------------- 闭式落点（本模块的承诺）
def test_anchors_land_on_the_paper_scale():
    """画面密度的 `P_LOW/P_MID/P_HIGH` 三个分位必须分别印到 0 / AIM / WHITE。

    分位写的是**常数本身**而不是 5/50/95 字面量：`P_LOW` 与 `P_MID`
    都对着用户手工参考改过（5→2、50→35，见 `print_math` 的注释与
    docs/blockers.md B-13 追加①②），写死数字的锁会在改常数时"自己变红"，
    而那恰好是它最该继续成立的时候。

    ⚠ 比较在**线性反射率域**做。我第一版拿显示域去比线性常数，
    看到"实测 0.4625 应为 0.18"以为落点错了——其实是比错了域
    （0.4625 显示编码恰好等于 0.18 线性）。
    """
    old = _tech(_ramp())
    a = _anchor(old)
    assert a is not None
    new, info = apply_print_remap(
        InvertResult(positive_linear=old, preview_srgb=encode_srgb(old)), a, 2.0)
    assert info is not None
    lum = np.clip(new.positive_linear, 0, 1)[..., :3] @ _LUM
    got = np.percentile(lum, [P_LOW, P_MID, P_HIGH])
    want = np.array([0.0, PRINT_AIM, PRINT_WHITE])
    assert np.allclose(got, want, atol=0.03), f"锚点没落准：{np.round(got,4)} vs {want}"


def test_black_anchor_really_bites_in_its_own_domain():
    """**这一条是实测逼出来的**：锚点必须和施加域同一个域。

    旧实现从 `−log10(t/base)` 的**密度 luma** 取 p5，而 `remap_luma` 读的是
    技术正片 luma 并用 `−dmax·log10(1−y)` 反解。橙色片基让三通道密度天然不等，
    加上 `density.py:132` 的 0 截断，两个域并不重合：
    标定卷 G200-1 名义 p5 以下按真实密度有 5.00% 像素，按反解域只有 **0.10%**，
    G200-8 是 5.00% 对 **0.00%**——黑点等于没咬，趾部整段没压，
    这就是曲线接上之后暗部偏差仍是 `p05 +0.110` 的原因。
    """
    rng = np.random.default_rng(7)
    base = np.array([0.28, 0.11, 0.031])           # 橙片基：三通道密度天然不等
    d3 = -0.05 + 1.05 * rng.random((160, 160, 3))  # 含"比片基还亮"的负密度尾
    t = base[None, None, :] * np.power(10.0, -d3)
    old = _tech(t, base)
    a = _anchor(old)
    assert a is not None
    s = picture_density(old, 2.0)
    frac = float((s <= a[0]).mean())
    assert abs(frac - P_LOW / 100.0) < 0.01, (
        f"锚点在自己的域里只咬住 {frac:.3%} 的像素，名义 p5={P_LOW}")
    new, info = apply_print_remap(
        InvertResult(positive_linear=old, preview_srgb=encode_srgb(old)), a, 2.0)
    assert info is not None
    dark = float((np.clip(new.positive_linear, 0, 1)[..., :3] @ _LUM <= 1e-4).mean())
    assert dark >= 0.8 * frac, f"印到黑的只有 {dark:.3%}，锚点没落到黑"


def test_wrong_domain_is_only_demonstrable_on_real_frames():
    """**负控制的位置说明**：合成帧里两个域几乎重合（实测咬合率 5.86% 对 5%），
    所以"域错配到底有多大"只能在真实底片上量——放在
    `tests/regression/test_print_anchor_domain.py`，那条才是这条锁的反向守门人。
    这里保留一条**合成帧必须能算出锚点**的下界，否则上面那条正锁没有检验对象。
    """
    rng = np.random.default_rng(7)
    base = np.array([0.28, 0.11, 0.031])
    d3 = -0.05 + 1.05 * rng.random((160, 160, 3))
    t = base[None, None, :] * np.power(10.0, -d3)
    old = _tech(t, base)
    assert _anchor(old) is not None
    dl = -np.log10(np.clip(t, 1e-6, None) / base[None, None, :]) @ _LUM
    lo_old = float(np.percentile(dl.ravel(), P_LOW))
    frac_old = float((picture_density(old, 2.0) <= lo_old).mean())
    assert abs(frac_old - P_LOW / 100.0) < 0.01, (
        f"合成帧的两域偏差 {frac_old:.3%} 已经变大了，本条说明需要重写")
    assert float((picture_density(old, 2.0) < 0.0).mean()) == 0.0, (
        "反解域不该出现负密度：0 截断正是两域错配的来源")


def test_chroma_ratios_are_preserved_bit_for_bit():
    """**核心锁**：打印重映射只改亮度，三通道按同一增益缩放。

    这条就是"天空变青蓝"那次的守门人——逐通道施加曲线时它必然红。
    """
    rng = np.random.default_rng(3)
    base = np.array([0.28, 0.11, 0.031])
    d = 0.15 + 1.0 * rng.random((120, 120))
    t = base[None, None, :] * np.power(10.0, -d)[..., None]
    old = _tech(t, base)
    a = _anchor(old)
    assert a is not None
    new, info = apply_print_remap(
        InvertResult(positive_linear=old, preview_srgb=encode_srgb(old)), a, 2.0)
    assert info is not None
    x, y = old[..., :3], new.positive_linear[..., :3]
    # 白端钳位**本来就会**改变通道比例（纸不可能反射超过最大反射率），
    # 所以只在"没被削顶"的像素上要求逐比特保比例，另外单独钉削顶比例的上界。
    interior = (y > 1e-6).all(2) & (y < 1.0 - 1e-6).all(2) & (x[..., 2] > 1e-4)
    assert interior.mean() > 0.5, (
        f"未削顶像素只剩 {interior.mean():.1%}，这张图不足以检验比例保持")
    r_in = x[..., 0] / np.maximum(x[..., 2], 1e-6)
    r_out = y[..., 0] / np.maximum(y[..., 2], 1e-6)
    dev = float(np.max(np.abs(r_out[interior] - r_in[interior])
                       / np.maximum(r_in[interior], 1e-6)))
    assert dev < 1e-5, f"通道比例被改动（最大相对偏差 {dev:.2e}）——影调曲线夹带了颜色"
    clipped = float((~((y > 1e-6).all(2) & (y < 1.0 - 1e-6).all(2))).mean())
    assert clipped < 0.30, f"削顶 {clipped:.1%} 太多：打印尺度在制造剪裁"


def test_saturation_does_not_run_away_from_the_reference():
    """饱和度的**分布**不该被影调曲线推高（越饱和的地方曲线斜率越陡）。"""
    rng = np.random.default_rng(11)
    base = np.array([0.28, 0.11, 0.031])
    d = 0.15 + 1.0 * rng.random((120, 120))
    t = base[None, None, :] * np.power(10.0, -d)[..., None]
    old = _tech(t, base)
    a = _anchor(old)
    assert a is not None, "合成帧应当仍有可用锚点，否则这条锁失去检验对象"
    new, info = apply_print_remap(
        InvertResult(positive_linear=old, preview_srgb=encode_srgb(old)), a, 2.0)
    assert info is not None
    s_old = (encode_srgb(old)[..., :3].max(2) - encode_srgb(old)[..., :3].min(2))
    s_new = (new.preview_srgb[..., :3].max(2) - new.preview_srgb[..., :3].min(2))
    p95_old, p95_new = np.percentile(s_old, 95), np.percentile(s_new, 95)
    assert p95_new <= p95_old * 1.05, (
        f"影调曲线把饱和 p95 从 {p95_old:.3f} 抬到 {p95_new:.3f}——那是颜色不是影调")


def test_negative_control_old_path_is_flat_and_bright():
    """**负控制**：不施加时必须更平、暗部更浮，否则"有效"无从判断。

    原来这条还要求"中间调被压暗"（`mn < mo`）。`P_MID` 从 50 改成 35 之后
    中间调**本来就该更亮**（把第 35 分位放到相纸中灰，第 50 分位自然在它上面），
    所以那条断言量的已经不是"打印有没有生效"，而是"中灰放在第几分位"——
    换成与中灰位置无关的对症量：**趾部必须被压住**（暗部分位变暗，
    这正是"奶白雾"的量）加上对比必须提升。
    """
    old = _tech(_ramp())
    a = _anchor(old)
    new, info = apply_print_remap(
        InvertResult(positive_linear=old, preview_srgb=encode_srgb(old)), a, 2.0)
    lo = float((encode_srgb(old)[..., :3] @ _LUM).std())
    ln = float((new.preview_srgb[..., :3] @ _LUM).std())
    assert ln > lo, f"对比没提升：{lo:.3f} → {ln:.3f}"
    old_lum = encode_srgb(old)[..., :3] @ _LUM
    new_lum = new.preview_srgb[..., :3] @ _LUM
    toe_old = float(np.percentile(old_lum, P_LOW))
    toe_new = float(np.percentile(new_lum, P_LOW))
    assert toe_new < toe_old, f"趾部没压住：p{P_LOW} {toe_old:.3f} → {toe_new:.3f}"


def test_curve_is_monotone_increasing():
    """映射本身必须单调不减——否则高光里出现反转（局部伪色）。"""
    a = (0.0, 0.5, 1.0)
    y = remap_luma(np.linspace(0.0, 0.999, 400), a, 2.0)
    assert np.all(np.diff(y) >= -1e-12), "打印映射非单调"


def test_degenerate_anchor_is_rejected_without_raising():
    """趾部斜率超过高光段 ⇒ 直接调用报 ValueError，生产入口静默退回。"""
    with pytest.raises(ValueError):
        print_curve((0.0, 0.02, 1.0))
    img = _ramp()
    old = invert(img, InvertParams(dmin=(0.25,) * 3, dmax=2.0))
    res, info = apply_print_remap(old, (0.0, 0.02, 1.0), 2.0)
    assert res is old and info is None, "退化曲线没有安全退回"


def test_no_anchor_means_untouched():
    img = _ramp()
    old = invert(img, InvertParams(dmin=(0.25,) * 3, dmax=2.0))
    res, info = apply_print_remap(old, None, 2.0)
    assert res is old and info is None


@pytest.mark.parametrize("bad", [np.nan, 0.0, -1.0])
def test_invalid_dmax_refuses_to_guess(bad):
    """曝光尺度非法 ⇒ 反解不出密度，不算分位数，不套曲线。

    原来这条锁的是"片基非法（回退 generic_prior 的帧）"；锚点改从**要被重映射
    的那个数组**取之后，片基不再是取样输入，所以那份保护转移到了这里——
    剩下的真实输入只有 `dmax` 与数组本身。
    """
    assert picture_density(_tech(_ramp()), bad) is None


def test_non_finite_positive_or_wrong_shape_refuses():
    """整帧非有限（解码/平场出过错）或维度不对 ⇒ 取样 None，而不是抛错。"""
    x = np.full((16, 16, 3), np.nan, np.float32)
    assert picture_density(x, 2.0) is None
    assert picture_density(np.zeros((16, 16), np.float32), 2.0) is None


def test_thin_negative_is_rejected():
    """密度跨度 < MIN_SPAN 时三点曲线近乎阶跃，必须拒绝。"""
    tight = np.full((64, 64, 3), 0.25) * (1.0 + 1e-5 * np.arange(64)[:, None, None])
    assert density_anchor(picture_density(_tech(tight), 2.0)) is None
    assert MIN_SPAN > 0


def test_dark_pixels_are_not_divided_by_zero():
    """亮度≈0 的像素增益取 1（不放大噪声、不产生 NaN/Inf）。"""
    x = np.zeros((8, 8, 3), np.float32)
    x[0, 0] = 0.5
    res = InvertResult(positive_linear=x, preview_srgb=encode_srgb(x))
    out, _info = apply_print_remap(res, (0.0, 0.5, 1.0), 2.0)
    assert np.isfinite(out.positive_linear).all()
    assert np.allclose(out.positive_linear[1:, 1:], 0.0), "全黑像素被抬亮了"


def test_preview_field_matches_the_remapped_positive():
    """`preview_srgb` 必须与新正片一致——否则预览和导出会分叉。"""
    old = invert(_ramp(), InvertParams(dmin=(0.25,) * 3, dmax=2.0))
    out, info = apply_print_remap(old, _anchor(old.positive_linear), 2.0)
    assert info is not None
    assert np.allclose(out.preview_srgb, encode_srgb(out.positive_linear), atol=1e-6)
    assert out.diagnostics == old.diagnostics
    assert abs(math.log10(PRINT_AIM)) > 0


def test_remap_introduces_no_new_highlight_clipping():
    """**这条锁是被实测逼出来的**：接上打印尺度后，非保留片边那次
    B 通道高光越界 **2.29%**，已越过门禁 1% 的失败线（`CLIP_FAIL_FRAC`）
    ——等于让整帧被挡导出。根因是"按亮度算增益再乘三通道"会把比亮度更亮的
    通道推过天花板；而把增益封到 1.0 也没用，像素会正好堆在 1.0 上，
    仍被 `_clip`（按 ≥1−tol 计数）判成削顶。上限必须是**相纸最大反射率**。
    """
    rng = np.random.default_rng(5)
    base = np.array([0.28, 0.11, 0.031])
    d = 0.05 + 0.9 * rng.random((120, 120))
    t = base[None, None, :] * np.power(10.0, -d)[..., None]
    t[:40, :, 2] *= 2.6                      # 人造"比亮度更亮的 B 通道"高光
    old = invert(np.clip(t, 0, None), InvertParams(dmin=tuple(base), dmax=2.0))
    a = _anchor(old.positive_linear)
    assert a is not None
    new, info = apply_print_remap(old, a, 2.0)
    assert info is not None
    y = new.positive_linear[..., :3]
    assert float(y.max()) <= PRINT_WHITE + 1e-6, (
        f"重映射后最大反射率 {y.max():.4f} 超过相纸天花板 {PRINT_WHITE}")
    at_ceiling = float((y >= 1.0 - 1e-6).any(axis=2).mean())
    assert at_ceiling < 0.005, (
        f"{at_ceiling:.2%} 的像素堆在 1.0 上，会被门禁判成高光削顶")


def test_chunked_matches_whole_frame_bit_exact():
    """**分块不得改变结果**（逐比特）。

    `density.py:231` 已为反相立了同一条锁（"mask 随行切片，保证与整帧一次
    算完全相同"），新函数不能免检：分块实现最容易出的错就是边界行用了不同的
    临时量或不同的归约顺序，而这种错在统计上看不出来。
    """
    rng = np.random.default_rng(2)
    x = np.clip(rng.random((5000, 7, 3)) * 0.9, 0, 1).astype(np.float32)
    mask = np.zeros((5000, 7), bool)
    mask[:, 1:6] = True
    a = (0.02, 0.45, 1.10)
    whole, i1 = print_remap_array(x, a, 2.0, mask, max_chunk_pixels=10 ** 12)
    chunked, i2 = print_remap_array(x, a, 2.0, mask, max_chunk_pixels=3 * 7 * 3)
    assert i1 is not None and i2 is not None
    assert i1 == i2
    assert np.array_equal(whole, chunked), "分块与整帧结果不一致（必须逐比特相同）"
    assert whole.dtype == np.float32, "整帧运算不该把数组抬到 float64"


def test_chroma_gain_lifts_saturation_without_breaking_the_ceiling():
    """色度增益：把"偏离亮度的部分"放大，亮度不变、天花板不越。

    这条锁管两件事：① 它确实起作用（饱和上升）；② 它不许把最亮通道顶过
    相纸最大反射率——上一版逐通道钳位就是在这里造出 2.29% 高光越界的。
    """
    import negative_lab.print_math as pm

    rng = np.random.default_rng(9)
    x = np.clip(rng.random((200, 120, 3)) * 0.85, 0, 1).astype(np.float32)
    a = (0.02, 0.45, 1.10)
    on, i1 = print_remap_array(x, a, 2.0)
    saved = pm.LOOK_CHROMA_GAIN
    try:
        pm.LOOK_CHROMA_GAIN = 1.0
        off, i2 = print_remap_array(x, a, 2.0)
    finally:
        pm.LOOK_CHROMA_GAIN = saved
    assert i1 is not None and i2 is not None
    s_on = float(np.percentile(on[..., :3].max(2) - on[..., :3].min(2), 95))
    s_off = float(np.percentile(off[..., :3].max(2) - off[..., :3].min(2), 95))
    assert s_on > s_off * 1.2, f"色度增益没起作用：{s_off:.3f} → {s_on:.3f}"
    assert float(on[..., :3].max()) <= PRINT_WHITE + 1e-6, "色度放大顶破了相纸天花板"
    assert float((on >= 1.0 - 1e-6).any(axis=2).mean()) < 0.005, "堆在 1.0 的像素过多"


def test_centre_region_ignores_rebate_padding():
    """取样区由**画面区包围盒**决定 ⇒ "保留片边"改变数组尺寸也不能移动它。

    这是 v5 §11.1（保留片边不得改变画面统计）在美学层的对应物：
    如果中心框按**数组**中心算，加一圈片边就会把黑点挪走，
    同一帧的印相决定就随开关漂移。
    """
    small = np.ones((100, 140), bool)
    r_small = centre_region(small)
    big = np.zeros((190, 270), bool)
    big[30:130, 60:200] = small                      # 等效于保留片边/不同裁切
    r_big = centre_region(big)
    assert np.array_equal(r_big[30:130, 60:200], r_small), (
        "加了一圈片边之后取样区挪位了")
    assert r_big.sum() == r_small.sum()


def test_centre_region_is_subset_and_keeps_area():
    m = np.zeros((200, 300), bool)
    m[20:120, 40:190] = True                          # 包围盒 100x150
    r = centre_region(m)
    assert (r & ~m).sum() == 0, "取样区跑到画面区外"
    want = 0.70 * 100 * 0.70 * 150
    assert abs(int(r.sum()) - want) <= 2 * (100 + 150), (
        f"中心区面积 {int(r.sum())} 与包围盒的 70%²={want:.0f} 差太远")


def test_centre_region_without_mask_and_tiny_bbox():
    """无掩膜时用给定形状取中心；包围盒小到不能再切时**原样返回**而不是给空集。"""
    r = centre_region(None, shape=(100, 200))
    assert r is not None and r.shape == (100, 200)
    assert abs(int(r.sum()) - 70 * 140) <= 2 * (100 + 200)
    tiny = np.zeros((40, 40), bool)
    tiny[10:11, 5:40] = True                          # 包围盒高 1px：不能再切
    assert np.array_equal(centre_region(tiny), tiny), "退化包围盒不该把取样区清空"
    thin = np.zeros((40, 40), bool)
    thin[10:12, 5:7] = True                           # 2x2：同样原样返回
    assert np.array_equal(centre_region(thin), thin)


def test_refusal_reason_states_the_measured_cause():
    """被拒必须**说得出是哪一个闸**，且健康样本要闭嘴（负控制）。"""
    assert "像素不足" in anchor_refusal_reason(np.zeros(10, np.float64))
    assert anchor_refusal_reason(None)
    flat = np.full(4096, 0.5, np.float64)
    assert "跨度" in anchor_refusal_reason(flat)
    # 负控制：跨度、次序、斜率都合法 ⇒ 空串（不许把这条路当装饰）。
    # 样本必须是**三分位都落在不同簇**的健康分布：旧版用 50/50 双峰就够了，
    # 但 `P_MID` 从 50 改成 35 之后，双峰样本的 p35 会掉进暗簇，
    # 它**真的**退化了（见下面"次序退化"的第一例），不能再当健康样本用。
    ok = np.concatenate([np.full(205, 0.05), np.full(1638, 0.60),
                         np.full(2253, 1.05)])
    assert anchor_refusal_reason(ok) == "", anchor_refusal_reason(ok)
    # 次序退化：暗簇占 40% ⇒ p2 与 p35 都落在暗簇里（跨度合法、次序不合法）
    assert "次序不成立" in anchor_refusal_reason(
        np.concatenate([np.full(1600, 0.05), np.full(2496, 1.05)]))
    assert "次序不成立" in anchor_refusal_reason(
        np.concatenate([np.full(2048, 0.5), np.full(2048, 0.5),
                        np.full(2048, 1.0)]))


def test_degenerate_curve_is_refused_before_any_work():
    """曲线不合法时**先拒再算**：不能处理到一半才抛错，留下半张图。"""
    x = np.full((64, 8, 3), 0.4, np.float32)
    out, info = print_remap_array(x, (0.0, 0.02, 1.0), 2.0)   # 趾部斜率 > 高光
    assert info is None
    assert np.array_equal(out, x), "退化时应当原样返回，而不是改一半"


# ---------------------------------------------------------------------------
# 电平的绝对项（B-14⑭）：锚点的中灰/白按取样总体的密度中位平移，黑锚不动。
# ---------------------------------------------------------------------------

def test_level_constants_are_the_measured_values():
    """钉住上线值本身：动这三个常数必须同时动 `PRINT_LEVEL_STRENGTH` 的实测依据。"""
    assert PRINT_LEVEL_STRENGTH == pytest.approx(0.25)
    assert PRINT_LEVEL_REFERENCE_DENSITY == pytest.approx(0.5539)
    assert PRINT_LEVEL_MAX_SHIFT == pytest.approx(0.12)


def test_level_shift_is_linear_then_clamped():
    assert level_shift(PRINT_LEVEL_REFERENCE_DENSITY) == 0.0
    assert level_shift(0.6539) == pytest.approx(0.025)          # 浓 ⇒ 正 ⇒ 印得更亮
    assert level_shift(0.3539) == pytest.approx(-0.05)          # 薄 ⇒ 负 ⇒ 印得更暗
    # 限幅：病态帧（片基错让密度中位飞掉）不该被电平项二次放大
    assert level_shift(1.55) == pytest.approx(PRINT_LEVEL_MAX_SHIFT)
    assert level_shift(0.01) == pytest.approx(-PRINT_LEVEL_MAX_SHIFT)


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf")])
def test_level_shift_refuses_to_guess_on_missing_input(bad):
    assert level_shift(bad) == 0.0


def test_level_moves_mid_and_white_but_never_the_black_anchor():
    anchor = (0.05, 0.50, 0.95)
    out, delta, note = anchor_with_level(anchor, 0.7539)
    assert delta == pytest.approx(0.05) and note == ""
    assert out[0] == anchor[0], "黑锚跟着抬 = 把已经偏高的阴影(Δp05 +0.0354)再抬一次"
    assert out[1] == pytest.approx(anchor[1] - delta)
    assert out[2] == pytest.approx(anchor[2] - delta)
    # δ=0 时必须**逐比特**是原锚（不加这一项 == 现产）
    same, d0, _ = anchor_with_level(anchor, PRINT_LEVEL_REFERENCE_DENSITY)
    assert d0 == 0.0 and same == anchor


def test_positive_level_prints_brighter():
    """方向锁：δ>0（本帧比标称浓）必须让同一个密度点**印得更亮**。
    方向错的话整条机制只会把逐帧亮度误差放大一倍，而统计量照样"变好"。"""
    y = np.full((8, 8), 0.5, np.float64)          # 反解出 d = 0.301
    base = remap_luma(y, (0.05, 0.50, 0.95), 1.0)
    lift = remap_luma(y, (0.05, 0.45, 0.90), 1.0)  # δ=+0.05
    assert float(lift.mean()) > float(base.mean())


def test_two_point_anchor_gets_no_level_and_says_so():
    """两点退化锚（`1298_G200::G200-10` 就走这条路）**不加未测过的电平项**，
    但必须留下说明——静默跳过等于没有这回事。"""
    out, delta, note = anchor_with_level((0.10, 0.90), 0.85)
    assert out == (0.10, 0.90) and delta == 0.0 and note
    assert "两点退化" in note


def test_level_is_dropped_when_the_shifted_curve_is_illegal():
    """平移把趾部压没时**保持原锚**，绝不让这帧从"印"退成"不印"。

    条件是 `mid − lo > 0.0563·(hi − mid)`（`0.0862/(mid−lo) < 1.531/(hi−mid)`），
    而抬电平只会压缩 `mid − lo`（`hi − mid` 不动），所以**高调帧**最容易越界：
    `tmp_level_absolute.py` 在 `strength=0.75` 那一格就真的出现过 1 帧非法。
    """
    anchor = (0.30, 0.40, 0.70)                    # 本身合法：0.10 > 0.0563·0.30
    print_curve(anchor)                            # 自检：前提成立
    # δ = 0.25·(0.9139−0.5539) = 0.09 ⇒ mid 落到 0.31、lo 仍 0.30 ⇒ 趾部斜率超过高光段
    out, delta, note = anchor_with_level(anchor, 0.9139)
    with pytest.raises(ValueError):
        print_curve((anchor[0], anchor[1] - 0.09, anchor[2] - 0.09))
    assert out == anchor and delta == 0.0 and note
    assert "不合法" in note and "+0.0900" in note


def test_level_negative_control_zero_strength_is_exactly_the_old_product(monkeypatch):
    """**负控制**：`strength=0` 必须让整条路径退回"没有这一项"——
    否则"上线后变好"分不清是机制还是别的改动。"""
    monkeypatch.setattr("negative_lab.print_math.PRINT_LEVEL_STRENGTH", 0.0)
    anchor = (0.05, 0.50, 0.95)
    out, delta, note = anchor_with_level(anchor, 0.95)
    assert out == anchor and delta == 0.0 and note == ""
    assert level_shift(0.95) == 0.0
    # 反方向（把浓的印暗）也必须真的把方向反过来：符号是从密度中位来的
    monkeypatch.setattr("negative_lab.print_math.PRINT_LEVEL_STRENGTH", -0.25)
    assert level_shift(0.95) < 0.0


# ---------------------------------------------------------------------------
# 色度增益的**纸级末端收回**（B-14⑲）：中间调浓、高光淡。
# ---------------------------------------------------------------------------

def test_chroma_highlight_constants_are_the_measured_values():
    from negative_lab.print_math import (PRINT_CHROMA_HIGHLIGHT_GAIN,
                                         PRINT_CHROMA_HIGHLIGHT_START)
    assert PRINT_CHROMA_HIGHLIGHT_GAIN == pytest.approx(0.80)
    assert PRINT_CHROMA_HIGHLIGHT_START == pytest.approx(0.85)


def test_chroma_ramp_is_flat_until_the_paper_scale_end():
    """`t0n` 归一在**纸级刻度**上：起点以下必须逐比特等于原增益。
    第一版把它当显示亮度扫到 0.88，结果一个像素都没进斜坡（三档 k_hi 数字完全相同），
    而 0.45 已经咬到纸级 56% = 中间调——这条锁钉住归一后的语义。"""
    from negative_lab.print_math import PRINT_WHITE, chroma_gain_at
    y = np.array([[0.0, 0.2 * PRINT_WHITE, 0.5 * PRINT_WHITE,
                   0.85 * PRINT_WHITE, 0.925 * PRINT_WHITE, PRINT_WHITE]])
    k = chroma_gain_at(y, 2.50)
    assert np.allclose(k[:, :4], 2.50), k          # 起点之前恒等（含端点 0.85）
    assert float(k[0, 4]) == pytest.approx(2.50 - (2.50 - 0.80) * 0.5)
    assert float(k[0, 5]) == pytest.approx(0.80)   # 纸级顶端收到高光档
    assert np.all(np.diff(k) <= 1e-9), "增益必须随亮度单调不增"


def test_chroma_ramp_direction_is_exactly_the_ramp():
    """公式不做钳位是**故意的**：`k_hi` 高于基增益时末端反而放大（负控制要用它），
    低于时收回。这里把两个方向都钉住，防止有人"好心"加回 `min/max` 钳位
    ——加了钳位之后反向控制会与恒等档逐比特相同（本轮实测踩过）。"""
    from negative_lab.print_math import PRINT_WHITE, chroma_gain_at
    y = np.linspace(0.0, PRINT_WHITE, 33).reshape(1, 33)
    down = chroma_gain_at(y, 2.50)                 # 生产：2.50 → 0.80
    up = chroma_gain_at(y, 0.5)                    # 反向：0.50 → 0.80（末端更放大）
    assert float(down[0, -1]) == pytest.approx(0.80) and down.max() <= 2.50 + 1e-9
    assert float(up[0, -1]) == pytest.approx(0.80) and up.max() >= 0.5
    assert np.all(np.diff(up) >= -1e-9), "反向档必须单调上升，否则斜坡写反了"
    inert = chroma_gain_at(y, 0.80)                # 高光档 == 基增益 ⇒ 整帧恒等
    assert np.allclose(inert, 0.80)


def test_chroma_ramp_preserves_luma_by_construction():
    """`out = y + k_eff(x − y)` 的权重和恒为 1 ⇒ **色度这一步不碰亮度**
    （亮度只由前面的纸级曲线决定）。随机像素上直接验恒等式。"""
    rng = np.random.default_rng(20261012)
    rgb = rng.random((96, 64, 3)).astype(np.float32) * 0.9
    y = rgb @ _LUM.astype(np.float32)
    from negative_lab.print_math import chroma_gain_at
    out = y[..., None] + chroma_gain_at(y, 2.50)[..., None] * (rgb - y[..., None])
    assert np.allclose(out @ _LUM, y, rtol=0, atol=2e-6)


def test_chroma_ramp_negative_control_off_is_the_old_product(monkeypatch):
    """**负控制**：这一项关掉之后必须逐比特等于"改之前的产品"，打开时必须不同。

    "关掉"有两种等价写法，这里**互相校验**：
    ① 把高光档钉成与总增益相同（斜坡恒等）；
    ② 把 `chroma_gain_at` 直接换成"返回常数"的旧实现。
    两者必须逐比特一致，且与生产默认值（0.80）**不一致**——
    否则要么这一项根本没进产品路径，要么这条负控制是空的。
    """
    from negative_lab.print_math import print_remap_array
    rgb = (np.linspace(0.05, 0.95, 200, dtype=np.float32)
           .reshape(20, 10, 1) * np.array([[[0.9, 1.0, 0.8]]], dtype=np.float32))
    anchor = (0.05, 0.50, 0.95)

    monkeypatch.setattr("negative_lab.print_math.PRINT_CHROMA_HIGHLIGHT_GAIN", 2.50)
    inert_const, _ = print_remap_array(rgb, anchor, 1.0, None, chroma_gain=2.50)

    monkeypatch.setattr("negative_lab.print_math.PRINT_CHROMA_HIGHLIGHT_GAIN", 0.80)

    def _old(y, base):     # 改之前的实现：整帧同一个标量
        return np.full(np.asarray(y).shape, base, dtype=np.float32)

    monkeypatch.setattr("negative_lab.print_math.chroma_gain_at", _old)
    inert_patch, _ = print_remap_array(rgb, anchor, 1.0, None, chroma_gain=2.50)
    assert np.array_equal(inert_const, inert_patch), "两种'关掉'写法不一致，其中一条是空的"

    monkeypatch.undo()
    shipped, info = print_remap_array(rgb, anchor, 1.0, None, chroma_gain=2.50)
    assert not np.array_equal(shipped, inert_const), "生产默认值下这一项没进产品路径"
    assert info["chroma_gain_highlight"] == pytest.approx(0.80)
    assert info["chroma_ramp_start"] == pytest.approx(0.85)


# ---------------------------------------------------------------------------
# 窄跨度负片不拉满纸级（B-14⑳）：只抬白锚，黑点与中灰一字不动。
# ---------------------------------------------------------------------------

def test_window_constant_is_the_measured_value():
    """0.60 = 工业先验值（正常曝光 135 彩色负片有用密度范围 0.6~0.9 decade），
    ㉕ 用符号闸把 ㉔ 被迫降到的 0.55 换回来。
    ⚠ 加闸之后**留出卷对 `W_min` 不再有否决权**（那一刀不再落到它头上），
    所以这个值不是本语料选出来的，是钉住的——依据与绊线见
    `PRINT_MIN_DENSITY_WINDOW` / `PRINT_WINDOW_YMED_REFERENCE` 两块与
    `tests/regression/test_window_sign_gate.py`。"""
    assert PRINT_MIN_DENSITY_WINDOW == pytest.approx(0.60)
    assert PRINT_WINDOW_YMED_REFERENCE == pytest.approx(0.43)


def test_window_y_med_is_the_curve_read_at_the_median():
    """**手算可核对**的身份：`window_y_med` 就是出厂曲线在密度中位处的读数。
    取 `d = mid` 必须逐比特等于 `PRINT_AIM`、`d = lo` 等于 0、`d = hi` 等于 `PRINT_WHITE`
    （曲线本身是按这三个锚点解出来的）；没有携带者时必须返回 `None` 而不是猜一个。"""
    a = (0.363, 0.520, 0.725)
    assert window_y_med(a, a[1]) == pytest.approx(PRINT_AIM, abs=1e-12)
    assert window_y_med(a, a[0]) == pytest.approx(0.0, abs=1e-12)
    assert window_y_med(a, a[2]) == pytest.approx(PRINT_WHITE, abs=1e-9)
    assert window_y_med(a, None) is None
    assert window_y_med(a, float("nan")) is None
    assert window_y_med((0.10, 0.90), 0.5) is None, "两点退化锚没有中灰控制点，闸无从判读"
    # 单调：中位越往窗口高端走，纸级亮度只能升不能降
    ys = [window_y_med(a, d) for d in np.linspace(a[0], a[2], 9)]
    assert all(x <= y + 1e-12 for x, y in zip(ys, ys[1:])), "y_med 对密度中位不单调"


def test_floor_and_level_do_not_commute():
    """下限与电平项动的是**同一个 `hi`**，所以次序不是风格：
    `先下限再电平` 与 `先电平再下限` 在 δ>0 时相差整整一个 δ。
    生产走前者（`service.py` step 9b，㉔）；这条锁防的是"把两步合并成一个
    `max(hi − δ, lo + W)`"——那一步合并就是旧次序，窄跨度帧的电平校正会被整笔收回。"""
    lo, hi, delta = 0.343, 0.7086, 0.0216    # 1298::G200-5 的出厂锚（mid=0.541）与 δ
    w = PRINT_MIN_DENSITY_WINDOW
    assert hi < lo + w, "这帧必须是被抬的那一类，否则两种次序相等"
    fl = max(hi, lo + w) - delta          # 下限 → 电平（生产）
    lf = max(hi - delta, lo + w)          # 电平 → 下限（旧出厂）
    assert fl == pytest.approx(lo + w - delta)
    assert lf == pytest.approx(lo + w)
    assert abs(fl - lf) == pytest.approx(delta), "两种次序没区别 ⇒ 这条锁失去区分力"


def test_window_floor_only_raises_the_white_anchor():
    """几何锁：只抬 `hi`。用 `tau=inf` 把 ㉕ 的符号闸**摘掉**，
    这样这一条测的是"下限改哪几个坐标"，而不是"闸放不放行"。"""
    lo, mid, hi = 0.343, 0.541, 0.726          # 1298::G200-5 的真实锚（跨度 0.383）
    out, raised, note = white_window_floor(anchor=(lo, mid, hi), tau=math.inf)
    assert raised and note == ""
    assert out[0] == lo and out[1] == mid, "黑点/中灰动了 = 会把电平项与阴影雾一起推翻"
    assert out[2] == pytest.approx(lo + PRINT_MIN_DENSITY_WINDOW)
    # 跨度本来就够的帧必须**逐比特**不变
    wide = (0.064, 0.247, 1.051)               # 1298::G200-7，跨度 0.987
    out2, raised2, note2 = white_window_floor(anchor=wide, tau=math.inf)
    assert out2 == wide and not raised2 and note2 == ""


def test_window_floor_negative_control_zero_is_the_old_product():
    """`w_min=0` 必须整条路径恒等，否则"上线后变好"没有对照。"""
    a = (0.343, 0.541, 0.726)
    out, raised, _ = white_window_floor(a, w_min=0.0, tau=math.inf)
    assert out == a and not raised


def test_window_floor_refuses_to_break_the_curve():
    """抬白锚会让高光段斜率掉下来；掉到趾部以下时**保持原锚并说明**，
    与电平项同一条纪律：宁可少一个窗口下限，不许从"印"退成"不印"。"""
    a = (0.50, 0.52, 0.53)                     # 抬到 1.10 后 s2=2.64 < s1=4.31
    print_curve(a)                             # 自检：原锚本身合法
    out, raised, note = white_window_floor(a, tau=math.inf)
    assert out == a and not raised and note
    assert "不合法" in note


def test_two_point_anchor_gets_no_window_floor():
    out, raised, note = white_window_floor((0.10, 0.90))
    assert out == (0.10, 0.90) and not raised and note == ""


# ---------------------------------------------------------------------------
# ㉕ 符号闸：这一刀只会压暗 ⇒ 只允许落在本批语料上"会被印亮"的那一侧
# ---------------------------------------------------------------------------

def test_gate_spares_the_frame_whose_median_prints_high():
    """**方向锁**：同一张窄跨度锚，密度中位越靠窗口高端 ⇒ `y_med` 越高 ⇒ 必须**放过**
    并写明原因。这一刀只会压暗，而 `y_med` 高那一侧在本批语料里全是被测出
    "已经印暗了"的帧（`1299` 三帧 `y_med` 0.475 / 0.476 / 0.665，Δmean 全负）。
    真实帧的逐帧数字在 `tests/regression/test_window_sign_gate.py` 里，
    本条只钉"方向"，不依赖任何我没核过的 `mid` 值。"""
    a = (0.265, 0.520, 0.818)                  # 跨度 0.553 < 0.60：本该挨这一刀
    assert a[2] < a[0] + PRINT_MIN_DENSITY_WINDOW, "这帧必须窄到本该挨刀，否则闸没在做事"
    d_high = a[0] + 0.90 * (a[2] - a[0])
    assert window_y_med(a, d_high) >= PRINT_WINDOW_YMED_REFERENCE
    out, raised, note = white_window_floor(a, density_median=d_high)
    assert not raised and out == a
    assert "符号闸" in note, f"跳过原因必须写明是闸，而不是悄悄不动：{note!r}"


def test_gate_allows_the_thin_frame_that_prints_bright():
    """同一张锚，中位靠窗口低端（`y_med` 低 = 本批语料里"会被印亮"那一侧）⇒ 必须放行。"""
    a = (0.363, 0.520, 0.725)                  # 1298::G200-5 那一类薄负片
    d_low = a[0] + 0.30 * (a[2] - a[0])
    assert window_y_med(a, d_low) < PRINT_WINDOW_YMED_REFERENCE
    out, raised, note = white_window_floor(a, density_median=d_low)
    assert raised and note == ""
    assert out[2] == pytest.approx(a[0] + PRINT_MIN_DENSITY_WINDOW)
    #: 闸的两侧必须**真的能互换**：同一张锚只挪中位，一侧抬一侧不抬
    spared = white_window_floor(a, density_median=a[0] + 0.99 * (a[2] - a[0]))[1]
    assert raised and not spared, "两侧同结果 ⇒ 闸没有在读这个携带者"


def test_gate_missing_carrier_does_not_fire():
    """没有携带者就**不施加单向钳**（说不清方向宁可少一刀），
    与"曲线不合法就保持原锚"同一条纪律；且必须写明原因。"""
    a = (0.363, 0.520, 0.725)
    out, raised, note = white_window_floor(a, density_median=None)
    assert out == a and not raised and "携带者" in note


def test_gate_infinite_tau_reproduces_the_pre_gate_product():
    """**恒等负控制**：`tau=inf`（闸门恒开）必须逐比特等于"没有闸"的 ㉔ 产品形状；
    而带闸的结果只允许是"抬"或"不抬"**两种之一**，不许出现第三种数。
    这条锁保证"加闸"这件事本身可以被干净地关掉对照——
    反过来若有人把闸写成默认关闭，这一条会红。"""
    cases = [(0.363, 0.520, 0.725, 0.576), (0.265, 0.520, 0.818, 0.714),
             (0.064, 0.247, 1.051, 0.339), (0.205, 0.470, 0.750, 0.472)]
    for lo, mid, hi, d in cases:
        gated = white_window_floor((lo, mid, hi), density_median=d)[0]
        open_gate = white_window_floor((lo, mid, hi), tau=math.inf)[0]
        off = white_window_floor((lo, mid, hi), w_min=0.0)[0]
        assert open_gate == (lo, mid, max(hi, lo + PRINT_MIN_DENSITY_WINDOW))
        assert off == (lo, mid, hi), "w_min=0 必须恒等"
        assert gated in (open_gate, off), (
            f"闸只能二选一（抬 / 不抬），不许产生第三个锚：{(lo, mid, hi, d)}")


def test_gate_threshold_sits_in_an_empty_band():
    """**取值边界锁**：本批语料里 `y_med` 的最高"该抬"值是 **0.404**
    （`1298::G200-9`），最低"该放"值是 **0.458**（`1298::G200-1`），
    闸值 0.43 落在这段**没有帧**的间隔里。
    ⚠ 这条同时说明闸值是**一侧的证据**：`y_med ≥ 0.458` 的四帧里三帧来自 1299，
    所以"闸有效"与"闸 = 让 1299 永不开火"在两卷语料上分不开（常量块里写死了）。
    第三卷一到必须重扫"闸值 τ × `W_min`"这一对。"""
    assert 0.404 < PRINT_WINDOW_YMED_REFERENCE < 0.458
    assert PRINT_WINDOW_YMED_REFERENCE - 0.404 > 0.02, "闸值贴到该抬那一侧了"
    assert 0.458 - PRINT_WINDOW_YMED_REFERENCE > 0.02, "闸值贴到该放那一侧了"


# ---------------------------------------------------------------------------
# 电平的第二个携带者：高调面积（B-14㉓，与 P_LOW 联立才通过判据）
# ---------------------------------------------------------------------------

def test_highlight_area_fraction_is_a_band_share():
    """占比 = 落在自身 p99 之下 `window` decade 以内的像素比。
    构造两簇：90% 在 0.1、10% 在 1.0 ⇒ p99=1.0，band=[0.8,1.0] 只有那 10%。"""
    s = np.concatenate([np.full(9000, 0.1), np.full(1000, 1.0)])
    assert highlight_area_fraction(s) == pytest.approx(0.1, abs=2e-3)
    # 全平的一幅：p99 与自身同值 ⇒ 全部落在带内（占比 1.0，不是 0）
    assert highlight_area_fraction(np.full(2048, 0.5)) == pytest.approx(1.0)
    # 样本不足不许猜
    assert highlight_area_fraction(np.full(200, 0.5)) is None
    assert highlight_area_fraction(None) is None
    assert highlight_area_fraction(np.array([0.1, np.nan, 0.9] * 300)) is not None


def test_level_shift_adds_two_carriers_and_clamps_the_sum():
    base = PRINT_LEVEL_REFERENCE_DENSITY
    area_ref = PRINT_LEVEL_HIGHLIGHT_AREA_REFERENCE
    assert level_shift(base, area_ref) == 0.0
    # 只给中位 ⇒ 与单携带者时代逐比特相同（向后兼容，`None` 就是不加第二项）
    assert level_shift(base + 0.2) == pytest.approx(PRINT_LEVEL_STRENGTH * 0.2)
    assert level_shift(base + 0.2, None) == level_shift(base + 0.2)
    # 两项同向 ⇒ 相加
    got = level_shift(base + 0.2, area_ref + 0.2)
    assert got == pytest.approx(PRINT_LEVEL_STRENGTH * 0.2
                                + PRINT_LEVEL_AREA_STRENGTH * 0.2)
    # 反向抵消到 0 就是"不加这一项"
    k = PRINT_LEVEL_AREA_STRENGTH * 0.4 / PRINT_LEVEL_STRENGTH
    assert level_shift(base + 0.4, area_ref - 0.4) == pytest.approx(
        PRINT_LEVEL_STRENGTH * 0.4 - PRINT_LEVEL_AREA_STRENGTH * 0.4)
    assert k > 0
    # 限幅作用在**和**上
    assert level_shift(base + 5.0, area_ref + 5.0) == pytest.approx(PRINT_LEVEL_MAX_SHIFT)
    assert level_shift(base - 5.0, area_ref - 5.0) == pytest.approx(-PRINT_LEVEL_MAX_SHIFT)


def test_level_constants_are_the_measured_pair():
    assert PRINT_LEVEL_AREA_STRENGTH == pytest.approx(0.10)
    assert PRINT_LEVEL_HIGHLIGHT_AREA_REFERENCE == pytest.approx(0.3122)
    assert PRINT_LEVEL_HIGHLIGHT_WINDOW == pytest.approx(0.20)
    assert P_LOW == pytest.approx(3.0), "黑锚分位改值必须同时改 ㉓ 的依据"


def test_area_carrier_negative_control_zero_is_the_single_carrier(monkeypatch):
    """**负控制**：κ₂=0 必须逐比特退回 ⑮ 的单携带者；
    κ₂ 反号必须把 δ 反方向推（否则"面积大 ⇒ 印得亮"这句话没有区分力）。"""
    monkeypatch.setattr("negative_lab.print_math.PRINT_LEVEL_AREA_STRENGTH", 0.0)
    assert level_shift(0.70, 0.70) == level_shift(0.70, None)
    monkeypatch.setattr("negative_lab.print_math.PRINT_LEVEL_AREA_STRENGTH", -0.10)
    assert level_shift(0.5539, 0.70) < 0.0


#: 一片"又亮又饱和"的合成天（8 列 × 6 行，线性正片）。
#: 挑这些值是当场试出来的：`chroma_gain=1.0` 时裁剪**一帧都不触发**，
#: 换成生产增益 2.50 时 **75% 的像素顶破纸白** ⇒ 裁剪这一级在这里真的有事做。
_BRIGHT_SATURATED = np.ascontiguousarray(np.tile(np.array(
    [[0.62, 0.86, 0.75], [0.55, 0.90, 0.42], [0.88, 0.52, 0.63], [0.44, 0.86, 0.80],
     [0.70, 0.72, 0.35], [0.35, 0.62, 0.90], [0.80, 0.80, 0.80], [0.20, 0.30, 0.75]],
    np.float32)[None, :, :], (6, 1, 1)))


def test_chroma_boost_spends_saturation_not_luma_when_it_hits_the_ceiling():
    """㉝：彩度放大顶破纸白时**只许收色度、不许收电平**。

    旧写法是"整像素同除一个系数"——保通道比例，但把亮度一起拖下去。
    分层量出来的代价（`tmp_layer_blame.py`，11 帧、验收同一把尺，
    Δ(B→A) = 彩度这一级吃掉多少 luma）：1299 三帧 p50 **−0.065** / p75 **−0.059** /
    p95 **−0.037**，而 1298 八帧只有 −0.007/−0.015/−0.008
    ⇒ 它专打"又亮又饱和"的像素，后果就是用户说的"云不发光、像蒙了层灰纱"
    （`outputs/ref_look_probe/look11_part2.jpg` 1299 三帧）。
    新写法解"最大的 k′≤1 使 max(y+k′·db) ≤ 纸白"，仍是 `y + k′·db` 的形式
    ⇒ 亮度逐比特不动（实测最大变化 2.2e-08），牺牲的是那一像素的饱和度。
    """
    anchor = (0.30, 0.60, 0.88)
    ref, info_ref = print_remap_array(_BRIGHT_SATURATED, anchor, 2.0, chroma_gain=1.0)
    got, info = print_remap_array(_BRIGHT_SATURATED, anchor, 2.0)
    assert info["chroma_clip_frac"] > 0.5, (
        f"非空对照：这个输入必须真的触发裁剪，实得 {info['chroma_clip_frac']}")
    assert info_ref["chroma_clip_frac"] == 0.0, (
        "**负控制**：不放大彩度就不该有像素被天花板收——否则这条锁练的不是裁剪")
    w = np.array([0.2126, 0.7152, 0.0722], np.float32)
    l_ref, l_got = ref @ w, got @ w
    assert float(np.abs(l_got - l_ref).max()) < 1e-6, (
        f"彩度放大把高光拖暗了（旧缺陷）：逐像素 luma 最大变化 "
        f"{float(np.abs(l_got - l_ref).max()):.3e}")
    assert float(got.max()) <= PRINT_WHITE + 1e-6, "纸白天花板必须仍然守住"
    sat_ref = float((ref.max(2) - ref.min(2)).mean())
    sat_got = float((got.max(2) - got.min(2)).mean())
    assert sat_got > sat_ref, (
        f"收色度之后增益仍然要真的买到饱和：{sat_ref:.4f} → {sat_got:.4f}")
