"""v3 P0-C：flat-field 测试（blank 2D / 条带 1D / 诚实的部分覆盖声明）。"""

from __future__ import annotations

import numpy as np
import pytest

from negative_lab.calibration.base import robust_dmin
from negative_lab.calibration.flatfield import apply_flat, fit_base_surface
from negative_lab.calibration.border import detect_regions
from negative_lab.testing.spectral import make_spectral_negative


@pytest.fixture()
def vig_pair():
    """非对称漏光场：片基条带在暗侧、画面亮侧更亮——画面内透明区会超过
    边缘片基测量值（G200-1 真实几何：边缘片基比画面内 clear film 暗）。"""
    s = make_spectral_negative(seed=21, crosstalk=False, vignetting=False)
    t = s["negative"]
    h, w, _ = t.shape
    xx = np.mgrid[0:h, 0:w][1].astype(np.float32)
    gain = 1.27 - 0.55 * (xx / w)              # 左 1.27 -> 右（片基侧）0.72
    blank = (np.ones((h, w, 3), np.float32) *
             np.array([0.62, 0.45, 0.30], np.float32)) * gain[..., None]
    neg = t * gain[..., None]
    # 画面内近透明带（片头 leader，d≈0）：亮侧 clear film 超过暗侧边缘
    # 片基测量值 -> 未校正时必然黑位剪裁（G200-1 雾化带的等价物）
    neg[:, 200:260, :] = (np.array([0.62, 0.45, 0.30], np.float32)
                          * gain[:, 200:260, None])
    return neg, blank, s


def test_blank2d_flattens_base_uniformly(vig_pair):
    neg, blank, s = vig_pair
    flat = apply_flat(neg, fit_base_surface(neg, blank=blank, order=2))
    bx0, bx1 = s["truth"]["base_cols"]
    strip = flat[:, bx0:bx1]
    top = robust_dmin(strip[: strip.shape[0] // 3])["dmin"]
    bot = robust_dmin(strip[-strip.shape[0] // 3:])["dmin"]
    rel = np.abs(top - bot) / np.maximum(np.minimum(top, bot), 1e-6)
    assert rel.max() < 0.03, (top, bot)          # 校正后上下片基一致
    base = np.array(s["truth"]["base"])
    assert np.allclose(robust_dmin(strip)["dmin"], base, rtol=0.05)


def test_strip1d_partial_coverage_honest(vig_pair):
    neg, _, s = vig_pair
    det = detect_regions(neg)
    fit = fit_base_surface(neg, base_roi=det["base"], order=2)
    assert fit["coverage"] == "strip-1d"
    assert any("不可观测" in w or "部分校正" in w for w in fit["warnings"]), \
        "1D 拟合必须声明部分覆盖（v3：不得假装全图已校正）"


def test_flat_reduces_output_clipping(vig_pair):
    """漏光场下物理指标"透射率超过片基(z>0)的像素占比"（必然落入黑位以下）：
    未 flat 校正 4%+；校正后应降到噪声水平（G200-1 B 通道问题的合成等价物）。"""
    from negative_lab.pipeline.auto import auto_analyze
    from negative_lab.pipeline.density import transmittance_to_z

    neg, blank, s = vig_pair
    bx0, bx1 = s["truth"]["base_cols"]
    rep0 = auto_analyze(neg, base_roi=neg[:, bx0:bx1])
    z0 = transmittance_to_z(neg, np.array(rep0.params.dmin)).mean(axis=2)
    # 阈值 z>0.04：超出片基 9% 才真正落入黑点以下（clear film z≈0 属正常）
    frac0 = float((z0[:, :bx0] > 0.04).mean())
    flat = apply_flat(neg, fit_base_surface(neg, blank=blank))
    rep1 = auto_analyze(flat, base_roi=flat[:, bx0:bx1])
    z1 = transmittance_to_z(flat, np.array(rep1.params.dmin)).mean(axis=2)
    frac1 = float((z1[:, :bx0] > 0.04).mean())
    assert frac0 > 0.02, f"场景未产生预期的黑位问题（{frac0}）"
    assert frac1 < frac0 * 0.3, (frac0, frac1)


def test_no_source_silently_overwrites_dmin(vig_pair):
    """flat 之后 Dmin 仍来自片基本身（数值上等于 blank 中位），
    不得被画面内容改写。"""
    neg, blank, s = vig_pair
    fit = fit_base_surface(neg, blank=blank)
    flat = apply_flat(neg, fit)
    bx0, bx1 = s["truth"]["base_cols"]
    got = robust_dmin(flat[:, bx0:bx1])["dmin"]
    assert np.allclose(got, np.array(s["truth"]["base"]), rtol=0.05)
