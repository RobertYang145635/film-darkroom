"""锁：自动打印尺度（B-11 的"一键就要好看"那一级）。

**为什么这些锁长这样**（全部 2026-10-08 实测）：
- 纸级形式 `paper = 1 − 10^(−v)` 要求"黑→中灰"与"中灰→高光"的 v 比是 **1 : 8.1**，
  而实测画面密度里这两段几乎等宽 ⇒ `(dmax, offset)` 只能整体缩放，**表达不了**；
  所以必须是一条密度域单调折线。这条曲线由三个实测分位点 + 两个**外部**纸级常数
  （`AIM=0.18`、`WHITE=0.80375`）唯一确定，**没有一个数是从用户图片拟合的**。
- 曲线是**影调**，不是**颜色**：`density_matrix` 与色温/色调一律不许被它改动
  （`service.py:383-388` 明令禁止静默启用未验证的颜色矩阵）。
- 生产路径**不许因为曲线退化而抛错**：宁可退回"没有自动打印尺度"。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from negative_lab.pipeline.density import encode_srgb, invert
from negative_lab.pipeline.params import InvertParams
from negative_lab.pipeline.print_scale import (MIN_SPAN, PRINT_AIM, PRINT_WHITE,
                                               apply_print_scale, density_anchor,
                                               positive_density, print_knots)

_LUM = np.array([0.2126, 0.7152, 0.0722])


def _ramp(n: int = 256, base: float = 0.25) -> np.ndarray:
    """线性透射率从 0.02 到 base（即密度 0 → −log10(base) 的一幅"灰阶负片"）。"""
    t = np.linspace(0.02, base, n, dtype=np.float64)
    return np.broadcast_to(t[:, None, None], (n, 4, 3)).copy()


def _apply(img: np.ndarray, base=(0.25, 0.25, 0.25)):
    p0 = InvertParams(dmin=base, dmax=2.0)
    samples = positive_density(img, np.asarray(base, float))
    anchor = density_anchor(samples)
    p1, info = apply_print_scale(p0, anchor)
    out = invert(img, p1).positive_linear
    return out, p0, p1, info, anchor


# ------------------------------------------------------- 闭式落点（核心承诺）
def test_anchors_land_on_the_paper_scale():
    """套曲线后，画面密度的 p5/p50/p95 必须分别印到 0 / AIM / WHITE。

    ⚠ 比较必须在**线性反射率域**做：锚点是纸级反射率，
    拿显示域数值去比会差一整条 sRGB 曲线（我第一版就犯了这个错，
    当时"实测 p50=0.4625 应为 0.18"看着像落点错了，其实是比错了域——
    0.4625 编码值恰好等于 0.18 线性）。
    """
    out, _p0, _p1, info, anchor = _apply(_ramp())
    assert info is not None, "灰阶样本本应能定出锚点"
    lum = np.clip(out, 0, 1)[..., :3] @ _LUM
    got = np.percentile(lum, [5, 50, 95])
    want = np.array([0.0, PRINT_AIM, PRINT_WHITE])
    assert np.allclose(got, want, atol=0.03), (
        f"锚点没落准：实测 {np.round(got,4)} 应为 {want}")


def test_without_the_curve_the_same_image_is_flat_and_bright():
    """**负控制**：同一幅图不套曲线时必须明显更平、中间调更亮——
    否则本模块的"有效"无从判断（曲线可能是空操作）。"""
    img = _ramp()
    out, p0, p1, _info, _a = _apply(img)
    flat = invert(img, p0).positive_linear
    std_new = float((encode_srgb(out)[..., :3] @ _LUM).std())
    std_old = float((encode_srgb(flat)[..., :3] @ _LUM).std())
    mid_old = float(np.percentile(encode_srgb(flat)[..., :3] @ _LUM, 50))
    mid_new = float(np.percentile(encode_srgb(out)[..., :3] @ _LUM, 50))
    assert std_new > std_old, f"套曲线后对比没提升：{std_old:.3f} → {std_new:.3f}"
    assert mid_new < mid_old, (
        f"中间调没被压回 aim 一侧：{mid_old:.3f} → {mid_new:.3f}")
    # 注：这里**不写**"未套曲线时中间调必须 > 0.6"之类的阈值——
    # 合成灰阶的密度跨度比真实翻拍帧宽，那个数是拍的而不是测的。
    # "发灰"的量化证据在语料上（阴影 p05 0.708 → 0.257），见 B-11 与验收文档。
    assert encode_srgb(out)[..., :3].shape[-1] == 3


def test_curve_is_monotone_and_shared_by_all_channels():
    """三通道必须是**同一条**曲线：逐通道曲线等于偷偷引入颜色比例变化。"""
    _out, _p0, p1, info, _a = _apply(_ramp())
    a, b, c = p1.channel_curves
    assert a == b == c, "三通道曲线不一致——那是在改颜色不是在改影调"
    ys = [y for _x, y in a]
    assert all(y2 > y1 for y1, y2 in zip(ys, ys[1:])), f"曲线非单调：{ys}"
    assert info["aim"] == PRINT_AIM and info["white"] == PRINT_WHITE


def test_colour_matrix_and_white_balance_are_untouched():
    """打印尺度**不得**碰 3×3 矩阵、色温色调或候选——那是颜色层的事。"""
    img = _ramp()
    p0 = InvertParams(dmin=(0.25, 0.25, 0.25), dmax=2.0,
                      density_matrix=(1, 0, 0, 0, 1, 0, 0, 0, 1))
    p1, info = apply_print_scale(
        p0, density_anchor(positive_density(img, np.asarray(p0.dmin, float))))
    assert tuple(p1.density_matrix) == tuple(p0.density_matrix)
    assert p1.dmin == p0.dmin
    assert info is not None
    assert sorted(info) == ["aim", "d_hi", "d_lo", "d_mid", "knots", "white"]


# --------------------------------------------------------------- 退化与拒绝
def test_non_finite_base_refuses_to_guess():
    """片基非有限（回退 generic_prior 的帧）⇒ 不套曲线，而不是拿假 base 算分位。"""
    img = _ramp()
    assert positive_density(img, np.array([np.nan, 0.1, 0.03])) is None
    assert density_anchor(positive_density(img, np.array([0.0, 0.1, 0.03]))) is None
    p0 = InvertParams()
    p1, info = apply_print_scale(p0, None)
    assert p1 is p0 and info is None, "无锚点却改了参数"


def test_thin_negative_below_min_span_is_rejected():
    """密度跨度不足 MIN_SPAN ⇒ 三点解出的曲线近乎阶跃，必须拒绝。"""
    tight = np.full((64, 64, 3), 0.25, np.float64) * (1.0 + 1e-4 * np.arange(64)[:, None])
    samples = positive_density(tight, np.array([0.25, 0.25, 0.25]))
    assert samples is not None
    assert density_anchor(samples) is None, "跨度极窄还定出了锚点"
    assert MIN_SPAN > 0


def test_degenerate_curve_does_not_raise_in_production():
    """中灰贴到 p5 会让趾部斜率超过高光段（曲线变 S 反）——
    直接调用必须报出来，而生产入口必须**静默退回原参数**，
    不能抛错打断整卷导出。"""
    with pytest.raises(ValueError):
        print_knots((0.0, 0.02, 1.0))          # 中灰几乎等于黑点
    p0 = InvertParams()
    p1, info = apply_print_scale(p0, (0.0, 0.02, 1.0))
    assert p1 is p0 and info is None, "退化曲线没有安全退回"


def test_exponent_prior_disables_the_print_scale():
    """带指数先验时曲线自变量是 `d·s` 且逐通道不同，
    共用一条绝对密度曲线会**悄悄改掉通道比例**（= 色偏）⇒ 必须不套。"""
    p0 = InvertParams(reference_exponent=1.2, red_ratio=1.17, blue_ratio=1.06)
    img = _ramp()
    p1, info = apply_print_scale(p0, density_anchor(positive_density(img, np.full(3, 0.25))))
    assert p1 is p0 and info is None, (
        "有指数先验还套了共用曲线——这会引入未验证的色偏")


def test_anchor_uses_luma_not_per_channel_percentiles():
    """锚点必须来自 **luma** 密度：逐通道分位会改变通道间比例（= 色偏）。
    horizon 实测放宽端点让中间调 R-G 漂了 14.4 codes，就是这个坑。"""
    img = _ramp()
    b = np.array([0.30, 0.25, 0.20])           # 三通道片基不同
    s = positive_density(img, b)
    a = density_anchor(s)
    assert a is not None
    per_channel = [np.percentile(-np.log10(np.clip(img[..., c], 1e-6, None) / b[c]),
                                 [5, 50, 95]) for c in range(3)]
    assert not any(np.allclose(a, pc) for pc in per_channel), (
        "锚点竟然等于某个单通道的分位数——说明没按 luma 加权")


def test_mid_grey_of_a_real_ish_scene_moves_toward_the_aim():
    """端到端：一个中密度场景，套曲线后显示域中位亮度必须**降**向 aim 邻域。"""
    base = np.array([0.28, 0.11, 0.031])       # 典型 C-41 橙色片基（实测值）
    rng = np.random.default_rng(7)
    t = base[None, None, :] * np.clip(10.0 ** -(0.2 + 0.9 * rng.random((96, 96))),
                                      1e-4, 1.0)[..., None]
    p0 = InvertParams(dmin=tuple(base), dmax=2.0)
    old = float(np.percentile(encode_srgb(invert(t, p0).positive_linear)
                              [..., :3] @ _LUM, 50))
    p1, info = apply_print_scale(p0, density_anchor(positive_density(t, base)))
    assert info is not None
    new = float(np.percentile(encode_srgb(invert(t, p1).positive_linear)
                              [..., :3] @ _LUM, 50))
    assert new < old, f"套曲线后中间调反而更亮：{old:.3f} → {new:.3f}"
    assert abs(math.log10(max(new, 1e-4)) - math.log10(PRINT_AIM)) < 0.6, (
        f"中间调离 aim 太远：display {new:.3f}")
