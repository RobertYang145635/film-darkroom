"""§8.1.4：幂指数只改变通道曲线时，参考中性点的色偏可被校正。"""

import numpy as np

from negative_lab.calibration.neutral import estimate_exponent_ratios
from negative_lab.pipeline.density import invert, transmittance_to_z
from negative_lab.pipeline.params import InvertParams
from negative_lab.testing.synthetic import make_synthetic_negative


def _z_at(neg, dmin, point, radius=3):
    r, c = point
    patch = neg[r - radius:r + radius + 1, c - radius:c + radius + 1, :].reshape(-1, 3)
    med = np.median(patch, axis=0)
    z = transmittance_to_z(med[None, None, :], np.asarray(dmin, np.float32))[0, 0, :].astype(np.float64)
    weight = float((max(med.mean(), 0.0) * max(np.abs(z).mean(), 1e-6)) ** 2)
    return z, weight


def _cast_at(out, point, half=2):
    r, c = point
    patch = out[r - half:r + half + 1, c - half:c + half + 1, :].reshape(-1, 3)
    m = patch.mean(axis=0)
    return float((m.max() - m.min()) / max(m.mean(), 1e-6))


def test_exponent_correction_removes_cast():
    k_true = (1.15, 1.0, 0.85)  # 乳剂/老化导致的通道差异
    s = make_synthetic_negative(exponents=k_true, noise=0.001, seed=41)
    neg, truth = s["negative"], s["truth"]
    dmin = truth["dmin"]
    bright, dark = truth["neutral_bright"], truth["neutral_dark"]

    # 未校正：中性点存在可测色偏
    p0 = InvertParams(dmin=dmin, dmax=truth["dmax"])
    out0 = invert(neg, p0).positive_linear
    cast0 = _cast_at(out0, bright)
    assert cast0 > 0.03, f"预期未校正有色偏，实际 {cast0}"

    # 用亮、暗两个中性点估计比例（SNR 加权，与生产路径一致）
    zb, wb = _z_at(neg, dmin, bright)
    zd, wd = _z_at(neg, dmin, dark)
    est = estimate_exponent_ratios([zb, zd], weights=[wb, wd])
    assert est["confidence"] > 0.5, est["warnings"]
    # 理论值：k_c 需与真值互反才能拉平 z' = k_c * z_c
    assert abs(est["red_ratio"] - 1 / k_true[0]) < 0.05, est
    assert abs(est["blue_ratio"] - 1 / k_true[2]) < 0.05, est

    p1 = InvertParams(dmin=dmin, dmax=truth["dmax"],
                      red_ratio=est["red_ratio"], blue_ratio=est["blue_ratio"])
    out1 = invert(neg, p1).positive_linear
    cast1 = _cast_at(out1, bright)
    assert cast1 < 0.01, f"校正后色偏应 <1%，实际 {cast1}"
    assert _cast_at(out1, dark) < 0.01


def test_insufficient_reference_points_lowers_confidence():
    """参考点不足时不得猜测：ratio=1 + 低置信度（§4.4 末段）。"""
    est = estimate_exponent_ratios([])
    assert est["red_ratio"] == 1.0 and est["blue_ratio"] == 1.0
    assert est["confidence"] == 0.0
    assert any("无有效中性参考点" in w for w in est["warnings"])

    # 全部样本太浅（接近片基）同样拒绝估计
    shallow = [np.array([-0.01, -0.012, -0.008])]
    est2 = estimate_exponent_ratios(shallow)
    assert est2["confidence"] == 0.0
