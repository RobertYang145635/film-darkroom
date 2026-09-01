"""v3 P0-F/P0-G：渲染分离 + 质量门禁 ΔE2000/mask 测试。"""

from __future__ import annotations

import numpy as np

from negative_lab.diagnostics.quality_gate import QualityGate
from negative_lab.pipeline.render import RenderIntent, apply_render


class TestRender:
    def test_neutral_intent_is_identity(self):
        x = np.random.default_rng(1).uniform(0, 1, (8, 8, 3)).astype(np.float32)
        assert np.array_equal(apply_render(x, RenderIntent()), x)

    def test_grade_darkens_midtones_keeps_endpoints(self):
        x = np.linspace(0, 1, 11, dtype=np.float32)[None, :, None] * np.ones((1, 1, 3), np.float32)
        y = apply_render(x, RenderIntent(grade=1.6))
        mid = (x[0, 5, 0], y[0, 5, 0])
        assert mid[1] < mid[0]
        assert y[0, 0, 0] == 0 and abs(float(y[0, -1, 0]) - 1) < 1e-6

    def test_monotonic(self):
        x = np.linspace(0, 1, 64, dtype=np.float32)[None, :, None] * np.ones((1, 1, 3), np.float32)
        y = apply_render(x, RenderIntent(grade=1.4, contrast=0.5, lift=0.05))
        assert (np.diff(y[0, :, 0]) >= -1e-7).all()

    def test_saturation_preserves_luminance(self):
        x = np.random.default_rng(2).uniform(0.1, 0.9, (16, 16, 3)).astype(np.float32)
        y = apply_render(x, RenderIntent(saturation=0.5))
        lx = x @ np.array([0.2126, 0.7152, 0.0722], np.float32)
        ly = y @ np.array([0.2126, 0.7152, 0.0722], np.float32)
        assert np.abs(lx - ly).max() < 1e-4


class TestGateDeltaE:
    def test_no_reference_is_honest_none(self):
        pos = np.random.default_rng(3).uniform(0, 1, (32, 32, 3)).astype(np.float32)
        g = QualityGate.evaluate(t_in=None, positive_linear=pos)
        assert g.delta_e2000 is None
        assert any("ΔE2000 未测量" in n for n in g.notes), g.notes

    def test_perfect_reference_zero_delta(self):
        from negative_lab.color.management import linear_to_srgb
        ref_lin = np.full((8, 8, 3), 0.5, np.float32)
        pos = ref_lin.copy()
        g = QualityGate.evaluate(t_in=None, positive_linear=pos,
                                 neutral_ref_positive=ref_lin)
        assert g.delta_e2000 is not None
        assert g.delta_e2000["mean"] < 0.1

    def test_cyan_cast_detected(self):
        ref = np.full((8, 8, 3), 0.5, np.float32)
        cast = ref.copy()
        cast[..., 2] *= 1.15
        cast[..., 0] *= 0.9
        g = QualityGate.evaluate(t_in=None, positive_linear=cast,
                                 neutral_ref_positive=ref)
        assert g.delta_e2000["mean"] > 2.0

    def test_mask_excludes_region_from_clip(self):
        t = np.full((16, 16, 3), 0.2, np.float32)
        pos = np.ones((16, 16, 3), np.float32)
        pos[:8, :, :] = -1.0                      # 上半部全剪裁
        mask = np.zeros((16, 16), bool)
        mask[:8, :] = True                        # 蒙版排除上半部
        g = QualityGate.evaluate(t_in=t, positive_linear=pos,
                                 valid_mask=~mask)
        assert max(g.output_clip["low"]) < 0.01
        g2 = QualityGate.evaluate(t_in=t, positive_linear=pos)
        assert max(g2.output_clip["low"]) > 0.4
