"""锁：自动打印尺度（B-11 的"一键就要好看"那一级）。

**为什么是曲线而不是指数**（2026-10-08 实测）：纸级形式 `paper = 1 − 10^(−v)`
要求"黑→中灰"与"中灰→白"的 v 比是 **1 : 8.1**，而实测画面密度里这两段几乎等宽
（0.479 vs 0.512 decade）⇒ `(dmax, offset)` 只能整体缩放，表达不了这个比例。
三个横坐标来自**这一帧自己的密度分位**（p5/p50/p95，luma 域），
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
from negative_lab.pipeline.print_scale import (MIN_SPAN, PRINT_AIM, PRINT_WHITE,
                                               apply_print_remap, density_anchor,
                                               positive_density, print_curve,
                                               remap_luma)

_LUM = np.array([0.2126, 0.7152, 0.0722])


def _ramp(n: int = 256, base: float = 0.25) -> np.ndarray:
    """线性透射率 0.02→base 的一幅"灰阶负片"（每行一个值）。"""
    t = np.linspace(0.02, base, n, dtype=np.float64)
    return np.broadcast_to(t[:, None, None], (n, 4, 3)).copy()


def _anchor(img: np.ndarray, base=(0.25, 0.25, 0.25)):
    return density_anchor(positive_density(img, np.asarray(base, float)))


# ------------------------------------------------- 闭式落点（本模块的承诺）
def test_anchors_land_on_the_paper_scale():
    """画面密度的 p5/p50/p95 必须分别印到 0 / AIM / WHITE。

    ⚠ 比较在**线性反射率域**做。我第一版拿显示域去比线性常数，
    看到"实测 0.4625 应为 0.18"以为落点错了——其实是比错了域
    （0.4625 显示编码恰好等于 0.18 线性）。
    """
    img = _ramp()
    a = _anchor(img)
    assert a is not None
    old = invert(img, InvertParams(dmin=(0.25,) * 3, dmax=2.0)).positive_linear
    new, info = apply_print_remap(
        InvertResult(positive_linear=old, preview_srgb=encode_srgb(old)), a, 2.0)
    assert info is not None
    lum = np.clip(new.positive_linear, 0, 1)[..., :3] @ _LUM
    got = np.percentile(lum, [5, 50, 95])
    want = np.array([0.0, PRINT_AIM, PRINT_WHITE])
    assert np.allclose(got, want, atol=0.03), f"锚点没落准：{np.round(got,4)} vs {want}"


def test_chroma_ratios_are_preserved_bit_for_bit():
    """**核心锁**：打印重映射只改亮度，三通道按同一增益缩放。

    这条就是"天空变青蓝"那次的守门人——逐通道施加曲线时它必然红。
    """
    rng = np.random.default_rng(3)
    base = np.array([0.28, 0.11, 0.031])
    d = 0.15 + 1.0 * rng.random((120, 120))
    t = base[None, None, :] * np.power(10.0, -d)[..., None]
    a = density_anchor(positive_density(t, base))
    assert a is not None
    old = invert(t, InvertParams(dmin=tuple(base), dmax=2.0)).positive_linear
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
    a = density_anchor(positive_density(t, base))
    old = invert(t, InvertParams(dmin=tuple(base), dmax=2.0)).positive_linear
    new, info = apply_print_remap(
        InvertResult(positive_linear=old, preview_srgb=encode_srgb(old)), a, 2.0)
    assert info is not None
    s_old = (encode_srgb(old)[..., :3].max(2) - encode_srgb(old)[..., :3].min(2))
    s_new = (new.preview_srgb[..., :3].max(2) - new.preview_srgb[..., :3].min(2))
    p95_old, p95_new = np.percentile(s_old, 95), np.percentile(s_new, 95)
    assert p95_new <= p95_old * 1.05, (
        f"影调曲线把饱和 p95 从 {p95_old:.3f} 抬到 {p95_new:.3f}——那是颜色不是影调")


def test_negative_control_old_path_is_flat_and_bright():
    """**负控制**：不施加时必须更平、中间调更亮，否则"有效"无从判断。"""
    img = _ramp()
    a = _anchor(img)
    old = invert(img, InvertParams(dmin=(0.25,) * 3, dmax=2.0)).positive_linear
    new, info = apply_print_remap(
        InvertResult(positive_linear=old, preview_srgb=encode_srgb(old)), a, 2.0)
    lo = float((encode_srgb(old)[..., :3] @ _LUM).std())
    ln = float((new.preview_srgb[..., :3] @ _LUM).std())
    mo = float(np.percentile(encode_srgb(old)[..., :3] @ _LUM, 50))
    mn = float(np.percentile(new.preview_srgb[..., :3] @ _LUM, 50))
    assert ln > lo, f"对比没提升：{lo:.3f} → {ln:.3f}"
    assert mn < mo, f"中间调没压回：{mo:.3f} → {mn:.3f}"


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


@pytest.mark.parametrize("bad", [
    np.array([np.nan, 0.1, 0.03]), np.array([0.0, 0.1, 0.03]),
    np.array([-0.2, 0.1, 0.03])])
def test_non_finite_or_non_positive_base_refuses_to_guess(bad):
    """片基非法（回退 generic_prior 的帧）⇒ 不算分位数，不套曲线。"""
    assert positive_density(_ramp(), bad) is None


def test_thin_negative_is_rejected():
    """密度跨度 < MIN_SPAN 时三点曲线近乎阶跃，必须拒绝。"""
    tight = np.full((64, 64, 3), 0.25) * (1.0 + 1e-5 * np.arange(64)[:, None, None])
    assert density_anchor(positive_density(tight, np.full(3, 0.25))) is None
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
    img = _ramp()
    old = invert(img, InvertParams(dmin=(0.25,) * 3, dmax=2.0))
    out, info = apply_print_remap(old, _anchor(img), 2.0)
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
    a = density_anchor(positive_density(np.clip(t, 0, None), base))
    assert a is not None
    old = invert(np.clip(t, 0, None), InvertParams(dmin=tuple(base), dmax=2.0))
    new, info = apply_print_remap(old, a, 2.0)
    assert info is not None
    y = new.positive_linear[..., :3]
    assert float(y.max()) <= PRINT_WHITE + 1e-6, (
        f"重映射后最大反射率 {y.max():.4f} 超过相纸天花板 {PRINT_WHITE}")
    at_ceiling = float((y >= 1.0 - 1e-6).any(axis=2).mean())
    assert at_ceiling < 0.005, (
        f"{at_ceiling:.2%} 的像素堆在 1.0 上，会被门禁判成高光削顶")
