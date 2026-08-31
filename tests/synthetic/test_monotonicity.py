"""§8.1.3：改变 Dmax、offset、gamma、exposure 产生单调且可解释的变化。"""

import numpy as np
import pytest

from negative_lab.pipeline.density import invert
from negative_lab.pipeline.params import InvertParams
from negative_lab.testing.synthetic import make_synthetic_negative


@pytest.fixture(scope="module")
def base_case():
    s = make_synthetic_negative(noise=0.001, seed=31)
    return s["negative"], s["truth"]


def _mean(neg, **over):
    p = InvertParams(dmin=(0.5, 0.5, 0.5), dmax=2.0)
    for k, v in over.items():
        setattr(p, k, v)
    return float(invert(neg, p).positive_linear.mean())


def test_exposure_monotonic(base_case):
    neg, truth = base_case
    vals = [_mean(neg, dmin=truth["dmin"], exposure=e)
            for e in (0.5, 0.8, 1.0, 1.3, 2.0)]
    assert all(b > a + 1e-6 for a, b in zip(vals, vals[1:])), vals


def test_gamma_monotonic_decreasing(base_case):
    """paper 值域在 [0,1)：gamma 越大输出越暗（相纸对比语义）。"""
    neg, truth = base_case
    vals = [_mean(neg, dmin=truth["dmin"], gamma=g)
            for g in (0.6, 0.8, 1.0, 1.4, 1.8)]
    assert all(b < a - 1e-6 for a, b in zip(vals, vals[1:])), vals


def test_offset_monotonic_decreasing(base_case):
    """offset 增大 -> 黑位加深 -> 平均亮度下降（扫描曝光偏置语义）。"""
    neg, truth = base_case
    vals = [_mean(neg, dmin=truth["dmin"], offset=o)
            for o in (-0.3, 0.0, 0.2, 0.5, 0.8)]
    assert all(b < a - 1e-6 for a, b in zip(vals, vals[1:])), vals


def test_dmax_monotonic_decreasing(base_case):
    """Dmax 增大 -> 同一密度被压缩到更靠近黑位 -> 平均亮度下降。"""
    neg, truth = base_case
    vals = [_mean(neg, dmin=truth["dmin"], dmax=d)
            for d in (1.2, 1.6, 2.0, 2.6, 3.2)]
    assert all(b < a - 1e-6 for a, b in zip(vals, vals[1:])), vals


def test_black_monotonic_increasing(base_case):
    """纸黑 black 增大 -> 整体抬亮（密度修正语义）。"""
    neg, truth = base_case
    vals = [_mean(neg, dmin=truth["dmin"], black=b)
            for b in (0.0, 0.1, 0.25, 0.5)]
    assert all(b > a + 1e-6 for a, b in zip(vals, vals[1:])), vals


def test_determinism(base_case):
    """稳定性指标（§8.4）：同输入同参数，逐比特一致。"""
    neg, truth = base_case
    p = InvertParams(dmin=truth["dmin"], dmax=truth["dmax"], gamma=1.1)
    a = invert(neg, p).positive_linear
    b = invert(neg, p).positive_linear
    assert np.array_equal(a, b)
