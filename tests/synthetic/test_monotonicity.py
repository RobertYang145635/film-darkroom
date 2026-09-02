"""§8.1.3 + v4 §4.4：技术参数单调性（offset/dmax/black）+ RenderIntent 单调性。

exposure/gamma 已从 InvertParams 迁移到 RenderIntent（v4 §1.2.10 实锤：
旧技术链路中它们是假旋钮）。本测试同时锁定：技术链路改风格字段无效。
"""

import numpy as np
import pytest

from negative_lab.pipeline.density import invert
from negative_lab.pipeline.params import InvertParams
from negative_lab.pipeline.render import RenderIntent, apply_render
from negative_lab.testing.synthetic import make_synthetic_negative


@pytest.fixture(scope="module")
def base_case():
    s = make_synthetic_negative(noise=0.001, seed=31)
    return s["negative"], s["truth"]


def _mean_tech(neg, **over):
    p = InvertParams(dmin=(0.5, 0.5, 0.5), dmax=2.0)
    for k, v in over.items():
        setattr(p, k, v)
    return float(invert(neg, p).positive_linear.mean())


def _mean_render(neg, tech_p, **over):
    intent = RenderIntent(**over)
    return float(apply_render(invert(neg, tech_p).positive_linear,
                              intent).mean())


def test_exposure_render_monotonic(base_case):
    """exposure 现在是 RenderIntent 风格参数：单调且只影响 rendered。"""
    neg, truth = base_case
    p = InvertParams(dmin=truth["dmin"], dmax=truth["dmax"])
    tech = invert(neg, p).positive_linear
    vals = [_mean_render(neg, p, exposure=e)
            for e in (0.5, 0.8, 1.0, 1.3, 2.0)]
    assert all(b > a - 1e-6 for a, b in zip(vals, vals[1:])), vals
    # 技术正片不因 intent 改变
    assert np.array_equal(tech, invert(neg, p).positive_linear)


def test_contrast_increases_spread_and_stays_monotonic(base_case):
    """v5 §7.2：对比度绕受保护支点扩展影调；旧 grade 字段已删除。"""
    neg, truth = base_case
    p = InvertParams(dmin=truth["dmin"], dmax=truth["dmax"])
    tech = invert(neg, p).positive_linear
    assert "grade" not in RenderIntent.FIELD_NAMES
    lum_w = np.array([0.2126, 0.7152, 0.0722], np.float32)
    in_lum = (tech @ lum_w).ravel()
    spreads = []
    for c in (-60.0, -20.0, 0.0, 40.0, 80.0):
        y = apply_render(tech, RenderIntent(contrast=c))
        out_lum = (y @ lum_w).ravel()
        spreads.append(float(out_lum.std()))
        # 顺序保持（单调映射的像素级表现）
        order_in = np.argsort(in_lum, kind="stable")
        order_out = np.argsort(out_lum, kind="stable")
        assert np.array_equal(order_in[:200], order_out[:200]), \
            f"contrast={c} 破坏了亮度次序"
    assert all(b > a - 1e-6 for a, b in zip(spreads, spreads[1:])), spreads


def test_offset_monotonic_decreasing(base_case):
    """offset 增大 -> 黑位加深 -> 平均亮度下降（技术参数）。"""
    neg, truth = base_case
    vals = [_mean_tech(neg, dmin=truth["dmin"], offset=o)
            for o in (-0.3, 0.0, 0.2, 0.5, 0.8)]
    assert all(b < a - 1e-6 for a, b in zip(vals, vals[1:])), vals


def test_dmax_monotonic_decreasing(base_case):
    """Dmax 增大 -> 同一密度被压缩到更靠近黑位 -> 平均亮度下降。"""
    neg, truth = base_case
    vals = [_mean_tech(neg, dmin=truth["dmin"], dmax=d)
            for d in (1.2, 1.6, 2.0, 2.6, 3.2)]
    assert all(b < a - 1e-6 for a, b in zip(vals, vals[1:])), vals


def test_black_monotonic_increasing(base_case):
    """纸黑 black 增大 -> 整体抬亮（技术端点）。"""
    neg, truth = base_case
    vals = [_mean_tech(neg, dmin=truth["dmin"], black=b)
            for b in (0.0, 0.1, 0.25, 0.5)]
    assert all(b > a + 1e-6 for a, b in zip(vals, vals[1:])), vals


def test_legacy_style_fields_no_effect_on_technical(base_case):
    """v4 §1.2.10 回归锁：InvertParams 的 gamma/exposure/highlight_* 是
    废弃字段，技术正片不得随其变化。"""
    neg, truth = base_case
    a = invert(neg, InvertParams(dmin=truth["dmin"], dmax=truth["dmax"],
                                 gamma=1.8, exposure=2.0)).positive_linear
    b = invert(neg, InvertParams(dmin=truth["dmin"], dmax=truth["dmax"])).positive_linear
    assert np.array_equal(a, b), "废弃风格字段仍在影响技术反相（假旋钮）"


def test_determinism(base_case):
    neg, truth = base_case
    p = InvertParams(dmin=truth["dmin"], dmax=truth["dmax"])
    a = invert(neg, p).positive_linear
    b = invert(neg, p).positive_linear
    assert np.array_equal(a, b)
