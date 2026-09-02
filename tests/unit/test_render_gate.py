"""v5 §7.2 / §11.3：RenderIntent 渲染层 + 相对参考差异测试。"""

from __future__ import annotations

import numpy as np

from negative_lab.diagnostics.quality_gate import QualityGate
from negative_lab.pipeline.render import (P0_FIELDS, RenderIntent,
                                          adaptation_matrix, apply_render,
                                          tone_curve)

_LUM = np.array([0.2126, 0.7152, 0.0722], np.float32)


def _ramp(n=64):
    return (np.linspace(0, 1, n, dtype=np.float32)[None, :, None]
            * np.ones((1, 1, 3), np.float32))


class TestRender:
    def test_identity_is_bit_exact(self):
        x = np.random.default_rng(1).uniform(0, 1, (8, 8, 3)).astype(np.float32)
        assert np.array_equal(apply_render(x, RenderIntent()), np.clip(x, 0, 1))

    def test_contrast_darkens_shadows_brightens_highlights(self):
        x = _ramp()
        y = apply_render(x, RenderIntent(contrast=60.0))
        assert y[0, 10, 0] < x[0, 10, 0]          # 支点（0.18）以下更暗
        assert y[0, 50, 0] > x[0, 50, 0]          # 支点以上更亮
        assert y[0, 0, 0] == 0.0                  # 黑端保持
        assert abs(float(y[0, -1, 0]) - 1.0) < 1e-4   # 白端保持

    def test_every_control_is_monotonic_both_signs(self):
        """v5 §11.2：三候选与所有滑杆均保持亮度单调。"""
        x = _ramp()
        for name in P0_FIELDS:
            for sign in (1.0, -1.0):
                it = RenderIntent()
                setattr(it, name, sign * (1.0 if name == "exposure" else 60.0))
                y = apply_render(x, it)[0, :, 0]
                assert (np.diff(y) >= -1e-6).all(), f"{name}@{sign} 非单调"

    def test_tone_curve_endpoints_and_pivot(self):
        c = tone_curve(contrast=80, highlights=-40, shadows=30,
                       whites=-20, blacks=15)
        assert c[0] < c[-1]
        assert (np.diff(c) >= -1e-9).all(), "曲线必须非递减"
        assert abs(float(c[0]) - (-4.0)) < 1e-9   # 黑端映射回黑端
        assert abs(float(c[-1]) - 2.0) < 1e-9     # 白端映射回白端

    def test_saturation_preserves_luminance(self):
        """彩度缩放只改变色度；亮度只在触及色域保护（剪裁）时才变化。"""
        x = np.random.default_rng(2).uniform(0.1, 0.9, (16, 16, 3)).astype(np.float32)
        y = apply_render(x, RenderIntent(saturation=40.0))
        lx, ly = x @ _LUM, y @ _LUM
        in_gamut = (y > 1e-6).all(axis=2) & (y < 1.0 - 1e-6).all(axis=2)
        assert in_gamut.mean() > 0.5, "样本应大多落在色域内"
        assert np.abs(lx - ly)[in_gamut].max() < 1e-4

    def test_temperature_direction_is_warm_positive(self):
        g = np.full((1, 4, 3), 0.5, np.float32)
        warm = apply_render(g, RenderIntent(temperature=50.0))[0, 0]
        cool = apply_render(g, RenderIntent(temperature=-50.0))[0, 0]
        assert warm[0] > warm[2] and cool[2] > cool[0]

    def test_tint_direction_is_green_positive(self):
        g = np.full((1, 4, 3), 0.5, np.float32)
        green = apply_render(g, RenderIntent(tint=50.0))[0, 0]
        magenta = apply_render(g, RenderIntent(tint=-50.0))[0, 0]
        assert green[1] > green[0] and green[1] > green[2]
        assert magenta[1] < magenta[0]

    def test_adaptation_preserves_luminance_of_gray(self):
        m = adaptation_matrix(60.0, 40.0)
        xyz_gray = np.array([0.5, 0.5, 0.5]) @ m.T
        assert abs(float(xyz_gray.mean()) - 0.5) < 0.08

    def test_legacy_parameters_map_to_intent_read_only(self):
        it = RenderIntent.from_legacy_parameters(
            {"gamma": 1.4, "exposure": 2.0, "highlight_strength": 0.5})
        assert it.origin == "legacy"
        assert it.contrast > 0 and it.exposure > 0 and it.highlights < 0
        assert "grade" not in RenderIntent.FIELD_NAMES


class TestRelativeReference:
    def test_no_reference_is_honest_none(self):
        pos = np.random.default_rng(3).uniform(0, 1, (32, 32, 3)).astype(np.float32)
        g = QualityGate.evaluate(t_in=None, positive_linear=pos)
        assert g.relative_delta_e2000 is None
        assert any("未计算相对参考差异" in n for n in g.notes), g.notes

    def test_perfect_reference_zero_delta(self):
        ref = np.full((8, 8, 3), 0.5, np.float32)
        g = QualityGate.evaluate(t_in=None, positive_linear=ref.copy(),
                                 reference_positive=ref)
        assert g.relative_delta_e2000 is not None
        assert g.relative_delta_e2000["mean"] < 0.1
        assert "相对参考输出差异" in g.relative_delta_e2000["label"]

    def test_cyan_cast_detected(self):
        ref = np.full((8, 8, 3), 0.5, np.float32)
        cast = ref.copy()
        cast[..., 2] *= 1.15
        cast[..., 0] *= 0.9
        g = QualityGate.evaluate(t_in=None, positive_linear=cast,
                                 reference_positive=ref)
        assert g.relative_delta_e2000["mean"] > 2.0

    def test_mask_excludes_region_from_clip(self):
        t = np.full((16, 16, 3), 0.2, np.float32)
        pos = np.ones((16, 16, 3), np.float32)
        pos[:8, :, :] = -1.0
        mask = np.zeros((16, 16), bool)
        mask[:8, :] = True
        g = QualityGate.evaluate(t_in=t, positive_linear=pos, valid_mask=~mask)
        assert max(g.output_clip["low"]) < 0.01
        g2 = QualityGate.evaluate(t_in=t, positive_linear=pos)
        assert max(g2.output_clip["low"]) > 0.4
