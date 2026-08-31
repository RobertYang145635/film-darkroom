"""ΔE2000 与 sRGB->Lab 单元测试（规格 §8.4 色彩指标）。

已知值来源：Sharma et al. 2005 表中可独立复核的两个代表对
（#1 蓝区含 RT 旋转项、#8 跨 180° 色相环项），其余轴对期望值由
公式手工独立推导（低彩度中性区 ΔE00 经 a' 拉伸可大于 ΔE76，属已知特性）。
"""

import numpy as np
import pytest

from negative_lab.diagnostics.colorimetry import delta_e_2000, srgb_to_lab

# (lab1, lab2, expected dE2000)
SHARMA = [
    ((50.0, 2.6772, -79.7751), (50.0, 0.0, -82.7485), 2.0425),   # Sharma #1
    ((50.0, 2.5, 0.0), (50.0, 0.0, -2.5), 4.3065),               # Sharma #8
    ((50.0, 2.5, 0.0), (50.0, 0.0, 0.0), 3.4582),                # 手推
    ((50.0, 0.0, 2.5), (50.0, 0.0, 0.0), 2.3669),                # 手推
    ((50.0, 0.0, -2.5), (50.0, 0.0, 0.0), 2.3669),               # 手推
    ((50.0, -1.0, 2.0), (50.0, 0.0, 0.0), 2.3669),               # 手推
]


@pytest.mark.parametrize("a,b,exp", SHARMA)
def test_delta_e_2000_known_values(a, b, exp):
    got = float(delta_e_2000(np.array(a), np.array(b)).item())
    assert abs(got - exp) < 1e-3, f"{got} != {exp}"


def test_identity_and_batch():
    labs = np.tile([50.0, 2.5, -75.0], (8, 1))
    de = delta_e_2000(labs, labs)
    assert np.allclose(de, 0.0, atol=1e-9)


def test_srgb_to_lab_endpoints():
    lab = srgb_to_lab(np.array([[0, 0, 0], [1, 1, 1], [1, 0, 0]], float))
    assert lab[0, 0] == pytest.approx(0.0, abs=1e-6)          # 黑 L=0
    assert lab[1, 0] == pytest.approx(100.0, abs=1e-4)        # 白 L=100
    assert lab[1, 1] == pytest.approx(0.0, abs=1e-4)          # 白 a≈0
    assert lab[2, 1] > 70                                      # 红 a*≈80.8
    assert lab[2, 2] > 55                                      # 红 b*≈67.9


def test_neutral_metric_module_imports():
    from negative_lab.diagnostics.colorimetry import neutral_delta_e
    a = np.full((4, 4, 3), 0.5, float)
    r = neutral_delta_e(a, a)
    assert r["max"] == pytest.approx(0.0, abs=1e-9)
