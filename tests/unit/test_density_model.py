"""密度域模型单元测试（规格 §8.1 条 1、2）。"""

import numpy as np
import pytest

from negative_lab.pipeline.density import (EPS, channel_exponents, invert,
                                           sanitize_dmax, transmittance_to_z)
from negative_lab.pipeline.params import InvertParams
from negative_lab.testing.synthetic import ORANGE_BASE, make_synthetic_negative


def _cast(img: np.ndarray, rows: slice, cols: slice) -> float:
    """中性区域的通道偏差：(max-min)/mean。"""
    patch = img[rows, cols, :].reshape(-1, 3)
    m = patch.mean(axis=0)
    return float((m.max() - m.min()) / max(m.mean(), 1e-6))


class TestSignConvention:
    """§4.2 符号约定：片基 z=0，画面内 z<=0，场景越亮 z 越负。"""

    def test_base_is_zero(self):
        dmin = np.array(ORANGE_BASE, dtype=np.float32)
        z = transmittance_to_z(dmin[None, None, :], dmin)
        assert np.allclose(z, 0.0, atol=1e-6)

    def test_darker_transmittance_is_negative_z(self):
        dmin = np.array(ORANGE_BASE, dtype=np.float32)
        t = dmin * 0.1  # 密度更高（场景更亮）
        z = transmittance_to_z(t[None, None, :], dmin)
        assert (z < -0.9).all()

    def test_direction_positive_output_increases_with_scene(self):
        """场景越亮 -> 负片越密 -> 反相输出越亮（方向锁，禁止翻转符号）。"""
        dmin = np.array(ORANGE_BASE, dtype=np.float32)
        p = InvertParams(dmin=tuple(dmin.tolist()), dmax=2.0)
        # 暗场景：t 接近片基；亮场景：t 远低于片基
        dark = np.full((4, 4, 3), dmin * 0.9, np.float32)
        bright = np.full((4, 4, 3), dmin * 0.05, np.float32)
        out_dark = invert(dark, p).positive_linear.mean()
        out_bright = invert(bright, p).positive_linear.mean()
        assert out_bright > out_dark


class TestDminBaseline:
    """§8.1.1：Dmin 等于片基值时，中性场景输出不产生额外色偏。"""

    def test_correct_dmin_no_cast(self):
        s = make_synthetic_negative(noise=0.0015, seed=11)
        neg, truth = s["negative"], s["truth"]
        p = InvertParams(dmin=truth["dmin"], dmax=truth["dmax"])
        out = invert(neg, p).positive_linear
        r, c = truth["neutral_bright"]
        assert _cast(out, slice(r - 2, r + 3), slice(c - 2, c + 3)) < 0.01
        # 斜坡中段（中性灰）
        a, b = truth["ramp_cols"]
        mid = (a + b) // 2
        assert _cast(out, slice(0, neg.shape[0]), slice(mid, mid + 4)) < 0.01

    def test_wrong_dmin_creates_cast(self):
        s = make_synthetic_negative(noise=0.0, seed=12)
        neg, truth = s["negative"], s["truth"]
        bad = tuple(v * f for v, f in zip(truth["dmin"], (1.5, 1.0, 1.0)))
        p = InvertParams(dmin=bad, dmax=truth["dmax"])
        out = invert(neg, p).positive_linear
        r, c = truth["neutral_bright"]
        assert _cast(out, slice(r - 2, r + 3), slice(c - 2, c + 3)) > 0.02


class TestRobustness:
    """§8.1.2：单通道接近零/一、超范围、NaN/Inf 不崩溃，输出无 NaN/Inf。"""

    def test_extreme_values(self):
        p = InvertParams(dmin=(0.5, 0.5, 0.5), dmax=2.0)
        vals = [0.0, 1e-9, 1.0, 1.5, -0.2]
        img = np.array([[v for v in vals] for _ in range(3)], dtype=np.float32)
        img = np.stack([img] * 3, axis=-1)
        out = invert(img, p).positive_linear
        assert np.isfinite(out).all()

    def test_nan_inf_replaced_and_counted(self):
        p = InvertParams()
        img = np.full((8, 8, 3), 0.3, np.float32)
        img[0, 0, 0] = np.nan
        img[1, 1, 1] = np.inf
        img[2, 2, 2] = -np.inf
        res = invert(img, p)
        assert np.isfinite(res.positive_linear).all()
        assert res.diagnostics["input_nonfinite"] == 3
        assert res.diagnostics["output_nonfinite"] == 0
        assert any("NaN/Inf" in w for w in res.warnings)

    def test_zero_dmin_does_not_diverge(self):
        p = InvertParams(dmin=(0.0, 0.0, 0.0), dmax=2.0)
        img = np.full((4, 4, 3), 0.5, np.float32)
        out = invert(img, p).positive_linear
        assert np.isfinite(out).all()

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), 1e-12, "x", None])
    def test_dmax_invalid_falls_back(self, bad):
        v, w = sanitize_dmax(bad)
        assert v == 2.0 and w is not None
        p = InvertParams(dmax=bad if not isinstance(bad, str) else 2.0)
        res = invert(np.full((4, 4, 3), 0.4, np.float32), p)
        assert np.isfinite(res.positive_linear).all()


class TestExponentVector:
    """§4.4：k_G=reference, k_R=ref*red_ratio, k_B=ref*blue_ratio。"""

    def test_channel_exponents(self):
        p = InvertParams(reference_exponent=0.9, red_ratio=1.1, blue_ratio=0.8)
        k = channel_exponents(p)
        assert np.allclose(k, [0.9 * 1.1, 0.9, 0.9 * 0.8])
