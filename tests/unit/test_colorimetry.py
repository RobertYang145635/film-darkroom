"""v4 §7.4/§3.3：colorimetry 唯一实现测试（L* 锁定 + Sharma 标准向量）。"""

from __future__ import annotations

import numpy as np

from negative_lab.diagnostics.colorimetry import (delta_e_2000,
                                                  linear_srgb_to_lab,
                                                  srgb_to_lab)

# Sharma et al. 2005, Table 1 公开测试向量（节选，含难例 1/17/19/24）
SHARMA = [
    ((50.0000, 2.6772, -79.7751), (50.0000, 0.0000, -82.7485), 2.0425),
    ((50.0000, 3.1571, -77.2803), (50.0000, 0.0000, -82.7485), 2.8615),
    ((50.0000, 2.8361, -74.0200), (50.0000, 0.0000, -82.7485), 3.4412),
    ((50.0000, -1.3802, -84.2814), (50.0000, 0.0000, -82.7485), 1.0000),
    ((50.0000, 2.5000, 0.0000), (50.0000, 3.1736, 0.5854), 1.0000),
    ((50.0000, -1.0000, 2.0000), (50.0000, 0.0000, 0.0000), 2.3669),
    ((50.0000, 2.4900, -0.0010), (50.0000, -2.4900, 0.0009), 7.1792),
    ((50.0000, 2.4900, -0.0010), (50.0000, -2.4900, 0.0010), 7.1792),
    ((50.0000, 2.4900, -0.0010), (50.0000, -2.4900, 0.0011), 7.2195),
    ((50.0000, 2.4900, -0.0010), (50.0000, -2.4900, 0.0012), 7.2195),
]


def test_linear_gray_half_L76():
    """v4 §3.3：linear [0.5,0.5,0.5] -> L* ≈ 76.069（错误实现给 88.70）。"""
    lab = linear_srgb_to_lab(np.array([[0.5, 0.5, 0.5]]))
    assert abs(lab[0, 0] - 76.069) <= 0.02


def test_linear_black_and_white():
    lab0 = linear_srgb_to_lab(np.array([[0.0, 0.0, 0.0]]))
    lab1 = linear_srgb_to_lab(np.array([[1.0, 1.0, 1.0]]))
    assert abs(lab0[0, 0]) < 1e-7
    assert abs(lab1[0, 0] - 100.0) < 1e-4   # 白点浮点尾差 ~4e-5


def test_sharma_vectors():
    for (l1, a1, b1), (l2, a2, b2), expected in SHARMA:
        got = delta_e_2000(np.array([[l1, a1, b1]]),
                           np.array([[l2, a2, b2]]))[0]
        assert abs(got - expected) < 0.011, ((l1, a1, b1), got, expected)


def test_srgb_to_lab_consistent_with_linear_path():
    """srgb_to_lab 必须等于"先 TRC 解码再 linear_srgb_to_lab"。"""
    rng = np.random.default_rng(7)
    x = rng.uniform(0, 1, (32, 3))
    lin = np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    assert np.allclose(srgb_to_lab(x), linear_srgb_to_lab(lin), atol=1e-9)


def test_delta_e_zero_for_identical():
    lab = linear_srgb_to_lab(np.array([[0.3, 0.5, 0.7]]))
    assert abs(delta_e_2000(lab, lab)[0]) < 1e-9
