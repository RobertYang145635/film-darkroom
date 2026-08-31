"""合成负片方向测试：锁定 §4.2 符号约定与反相方向。"""

import numpy as np

from negative_lab.pipeline.density import invert
from negative_lab.pipeline.params import InvertParams
from negative_lab.testing.synthetic import make_synthetic_negative


def test_ramp_is_monotonic_after_inversion():
    """灰度斜坡场景经反相后输出亮度必须随场景亮度单调递增。"""
    s = make_synthetic_negative(noise=0.001, seed=21)
    neg, truth = s["negative"], s["truth"]
    p = InvertParams(dmin=truth["dmin"], dmax=truth["dmax"])
    out = invert(neg, p).positive_linear.mean(axis=2)  # 亮度

    a, b = truth["ramp_cols"]
    cols = np.arange(a + 4, b - 4, 8)
    prof = out[:, cols].mean(axis=0)
    diffs = np.diff(prof)
    assert (diffs > -1e-4).all(), f"斜坡非单调: {diffs.min()}"
    assert prof[-1] > prof[0] * 5, "亮场景输出应显著亮于暗场景"


def test_film_base_maps_to_black():
    """未曝光片基（z=0）在默认参数下映射到纸黑附近。

    使用无尘点样本：片基上的尘斑属于 Phase 4 除尘职责，不由反相模型负责。
    """
    s = make_synthetic_negative(noise=0.001, dust=0, seed=22)
    neg, truth = s["negative"], s["truth"]
    p = InvertParams(dmin=truth["dmin"], dmax=truth["dmax"], offset=0.0, black=0.0)
    out = invert(neg, p).positive_linear
    x0, x1 = truth["base_cols"]
    base_mean = out[:, x0:x1, :].mean()
    assert base_mean < 0.02, f"片基应映射到黑位，实际 {base_mean}"


def test_bright_scene_near_white():
    """最亮场景（L=1，密度 -dmax）应映射到高光区。"""
    s = make_synthetic_negative(noise=0.0, seed=23)
    neg, truth = s["negative"], s["truth"]
    p = InvertParams(dmin=truth["dmin"], dmax=truth["dmax"])
    out = invert(neg, p).positive_linear
    a, b = truth["ramp_cols"]
    assert out[:, b - 2, :].mean() > 0.7
