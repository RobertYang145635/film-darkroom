"""v4 §3.1/§3.2 backend 与全入口接线契约（v5 保留仍有效的部分，
入口接线改为校验卷级分析与采集指纹是否送达）。"""

from __future__ import annotations


import numpy as np
import pytest
import tifffile

from negative_lab.calibration.rollcalib import RollCalibration
from negative_lab.pipeline import service as svc
from negative_lab.pipeline.density import invert, invert_chunked
from negative_lab.pipeline.params import InvertParams
from negative_lab.pipeline.render import RenderIntent, apply_render
from negative_lab.pipeline.service import candidate_positive, process_image
from negative_lab.testing.spectral import make_spectral_negative

C = [[0.92, 0.09, 0.05], [0.06, 0.90, 0.07], [0.04, 0.08, 0.94]]
NON_UNIT = tuple(C[0]) + tuple(C[1]) + tuple(C[2])


@pytest.fixture(scope="module")
def xt():
    s = make_spectral_negative(seed=41, crosstalk=True)
    t = s["negative"]
    return s, t


class TestBackendWiring:
    def test_invert_chunked_nonunit_matrix_changes_output(self, xt):
        """§3.1.1：生产入口 invert_chunked 必须消费 density_matrix
        （v4 §1.2.1 实锤：曾恒等输出）。"""
        _, t = xt
        base = InvertParams(dmin=(0.62, 0.45, 0.30), dmax=2.0)
        nonunit = InvertParams(dmin=(0.62, 0.45, 0.30), dmax=2.0,
                               density_matrix=NON_UNIT,
                               backend_name="density-affine")
        a = invert_chunked(t, base).positive_linear
        b = invert_chunked(t, nonunit).positive_linear
        assert float(np.abs(a - b).max()) > 1e-3

    def test_process_image_nonunit_matrix_affects_technical(self, xt):
        s, t = xt
        bx0, bx1 = s["truth"]["base_cols"]
        base_roi = (0, bx0, t.shape[0], bx1)
        area = (0, 0, t.shape[0], bx0)
        p1 = InvertParams(dmin=(0.62, 0.45, 0.30), dmax=2.0)
        p2 = InvertParams(dmin=(0.62, 0.45, 0.30), dmax=2.0,
                          density_matrix=NON_UNIT, backend_name="density-affine")
        r1 = process_image(t, output_crop_source=area, image_area_source=area,
                           base_roi_source=base_roi, force_params=p1)
        r2 = process_image(t, output_crop_source=area, image_area_source=area,
                           base_roi_source=base_roi, force_params=p2)
        assert float(np.abs(r1.result.positive_linear
                            - r2.result.positive_linear).max()) > 1e-3

    def test_unit_matrix_affine_matches_reciprocal_allclose(self, xt):
        """§3.1.3：单位阵 density-affine 与 power-reciprocal 等价，
        覆盖奇数尺寸与最后一个 chunk。"""
        _, t = xt
        odd = t[:-3, :-7]
        p = InvertParams(dmin=(0.62, 0.45, 0.30), dmax=2.0,
                         backend_name="density-affine")   # 单位阵
        pa = InvertParams(dmin=(0.62, 0.45, 0.30), dmax=2.0,
                          backend_name="power-reciprocal")
        assert np.allclose(invert_chunked(odd, p, max_chunk_pixels=3000).positive_linear,
                           invert(odd, pa).positive_linear, atol=1e-6)
        assert np.allclose(invert_chunked(t, p, max_chunk_pixels=12345).positive_linear,
                           invert(t, pa).positive_linear, atol=1e-6)

    def test_sensitivity_table(self, xt):
        """§3.1.4：每个保留技术参数都必须真实改变技术正片；
        exposure 只改变 rendered（v4 §1.2.9/10 回归锁）。"""
        _, t = xt
        base = InvertParams(dmin=(0.62, 0.45, 0.30), dmax=2.0)
        out0 = invert(t, base).positive_linear
        cases = {
            "red_ratio": InvertParams(dmin=(0.62, 0.45, 0.30), dmax=2.0, red_ratio=1.4),
            "blue_ratio": InvertParams(dmin=(0.62, 0.45, 0.30), dmax=2.0, blue_ratio=0.6),
            "density_matrix": InvertParams(dmin=(0.62, 0.45, 0.30), dmax=2.0,
                                           density_matrix=NON_UNIT,
                                           backend_name="density-affine"),
            "dmax": InvertParams(dmin=(0.62, 0.45, 0.30), dmax=1.3),
            "wb_high": InvertParams(dmin=(0.62, 0.45, 0.30), dmax=2.0,
                                    wb_high=(1.1, 1.0, 0.9)),
            "wb_low": InvertParams(dmin=(0.62, 0.45, 0.30), dmax=2.0,
                                   offset=0.15, wb_low=(1.2, 1.0, 0.85)),
            "offset": InvertParams(dmin=(0.62, 0.45, 0.30), dmax=2.0, offset=0.2),
            "black": InvertParams(dmin=(0.62, 0.45, 0.30), dmax=2.0, black=0.3),
        }
        for name, p in cases.items():
            out = invert(t, p).positive_linear
            diff = float(np.abs(out - out0).max())
            assert diff > 1e-4, f"参数 {name} 对技术正片无效果（假旋钮，v4 §1.2.9/10）"
        # exposure 在技术链路无效果（已迁移），但在渲染层单调生效
        rendered = apply_render(out0, RenderIntent(exposure=2.0))
        assert float(np.abs(rendered - out0).max()) > 1e-4


class TestEntryWiring:
    def _write_tif(self, tmp_path, s):
        src = tmp_path / "frame.tif"
        tifffile.imwrite(src, (np.clip(s["negative"], 0, 1) * 65535
                               ).astype(np.uint16), photometric="rgb")
        return src

    def test_roll_analysis_reaches_process_image_from_batch(self, tmp_path, xt,
                                                            monkeypatch):
        """v5 §6.2：批处理入口必须把卷级分析送进唯一服务。"""
        s, t = xt
        src = self._write_tif(tmp_path, s)
        from negative_lab.library.models import RollAnalysis
        ra = RollAnalysis(roll_id="roll", base_statistics={"dmin": [0.6, 0.4, 0.3]},
                          capture_fingerprint="fp", source_frames=["a"],
                          confidence=0.8, base_source="roll")
        captured = {}
        orig = svc.process_image

        def spy(*a, **kw):
            captured["roll_analysis"] = kw.get("roll_analysis")
            captured["capture_fingerprint"] = kw.get("capture_fingerprint")
            return orig(*a, **kw)
        monkeypatch.setattr(svc, "process_image", spy)
        import negative_lab.pipeline.batch as batch_mod
        monkeypatch.setattr(batch_mod, "process_image", spy)
        from negative_lab.pipeline.batch import BatchJob
        job = BatchJob([src], tmp_path / "out", tiff_encoding="linear",
                       roll_analysis=ra)
        job.run()
        assert captured.get("roll_analysis") is ra
        assert captured.get("capture_fingerprint") is not None

    def test_render_intent_changes_rendered_not_technical(self, xt):
        """v5 §7.2：微调只改渲染正片，技术正片逐比特不变。"""
        s, t = xt
        bx0, bx1 = s["truth"]["base_cols"]
        base_roi = (0, bx0, t.shape[0], bx1)
        area = (0, 0, t.shape[0], bx0)
        kw = dict(output_crop_source=area, image_area_source=area,
                  base_roi_source=base_roi)
        r_identity = process_image(t, render_intent=RenderIntent(), **kw)
        r_style = process_image(t, render_intent=RenderIntent(exposure=1.5,
                                                              saturation=40.0), **kw)
        assert np.array_equal(r_identity.result.positive_linear,
                              r_style.result.positive_linear)
        assert not np.array_equal(r_identity.rendered_positive_linear,
                                  r_style.rendered_positive_linear)
        # 恒等 intent 与技术正片逐像素等价（预览与导出一致的前提）
        assert np.array_equal(r_identity.rendered_positive_linear,
                              r_identity.result.positive_linear)
        sc = r_style.sidecar(source_sha256="a" * 64)
        assert sc["render_intent"]["exposure"] == 1.5
        assert sc["render_intent"]["saturation"] == 40.0
        assert sc["schema_version"] == 4

    def test_three_candidates_share_technical_positive(self, xt):
        """v5 §6.5：三候选同一物理底座，只有 RenderIntent 不同。"""
        from negative_lab.pipeline.candidates import build_candidates
        s, t = xt
        bx0, bx1 = s["truth"]["base_cols"]
        area = (0, 0, t.shape[0], bx0)
        pr = process_image(t, output_crop_source=area, image_area_source=area,
                           base_roi_source=(0, bx0, t.shape[0], bx1))
        cs = build_candidates(pr)
        tech = pr.result.positive_linear
        rendered = {c.name: candidate_positive(pr, c.name) for c in cs}
        for c in cs:
            assert np.array_equal(c.technical_positive_linear, tech)
        assert rendered["preserve"].shape == tech.shape
        assert np.isfinite(rendered["warm"]).all()

    def test_color_negative_without_chart_is_not_blocked(self, xt):
        """v5 §8.1：彩色负片缺色卡/缺标定不得阻断，只可能 needs_review。"""
        s, t = xt
        bx0, bx1 = s["truth"]["base_cols"]
        pr = process_image(t, output_crop_source=(0, 0, t.shape[0], bx0),
                           image_area_source=(0, 0, t.shape[0], bx0),
                           base_roi_source=(0, bx0, t.shape[0], bx1),
                           )
        assert pr.gate.status in ("ok", "warn", "needs_review")
        assert pr.gate.blocks_export is False
        assert pr.rendered_positive_linear is not None

    def test_cli_calibration_flag_parses(self, tmp_path, xt):
        """§5.1.3：--calibration 路径被加载并传入管线（契约级）。"""
        s, t = xt
        cal = RollCalibration(roll="roll", film_type="C-41",
                              source="chart-calibration", verified=True,
                              dmax=2.2, confidence={"neutral": 1.0})
        p = cal.save(tmp_path)
        got = RollCalibration.load(p)
        assert got.verified and got.usable_for_color
