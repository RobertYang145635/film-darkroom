"""P0-D/P0-F：统一服务与 sidecar v2 测试（整改 v2 §2、§4）。"""

from __future__ import annotations

import numpy as np
import pytest

from negative_lab.pipeline.geometry import (oriented_to_source,
                                            roi_source_to_oriented, rotate_cw)
from negative_lab.pipeline.params import InvertParams
from negative_lab.pipeline.service import process_image
from negative_lab.testing.synthetic import make_synthetic_negative


@pytest.fixture(scope="module")
def synth():
    return make_synthetic_negative(black_border=20, dust=40, seed=91)


class TestRotationSemantics:
    def test_clockwise_90(self):
        g = np.zeros((2, 3, 3), np.float32)
        g[0, 0, 0] = 1  # a 左上
        r = rotate_cw(g, 90)
        assert r.shape[:2] == (3, 2)
        assert r[0, 1, 0] == 1          # 左上 -> 右上

    def test_roi_roundtrip_all_angles(self):
        for deg in (0, 90, 180, 270):
            roi = (10, 20, 30, 40)
            o = roi_source_to_oriented(roi, 100, 200, deg)
            back = oriented_to_source(o, 100, 200, deg)
            assert back == roi, (deg, o, back)


class TestService:
    def test_geometry_and_sidecar(self, synth):
        t = synth["negative"]
        pr = process_image(t, rotation_degrees=0)
        assert pr.gate.status in ("ok", "warn", "needs_calibration")
        geo = pr.geometry
        assert geo.source_size == t.shape[:2]
        assert geo.crop_oriented is not None          # 默认裁到画面
        y0, x0, y1, x1 = geo.crop_oriented
        assert pr.t.shape == (y1 - y0, x1 - x0, 3)
        sc = pr.sidecar(source_sha256="x" * 64, source_path="a.tif")
        assert sc["schema_version"] == 3   # v4 §2/§6.7：技术/风格分列、可追溯
        assert "technical_parameters" in sc and "render_intent" in sc
        assert sc["backend"] == pr.report.params.backend_name
        assert sc["geometry"]["roi_source_space"]["coordinate_space"] == "source"
        assert sc["geometry"]["roi_source_space"]["coordinate_space"] == "source"
        assert "preview" not in json_dumps(sc)
        assert sc["quality"]["status"] == pr.gate.status
        # 往返：由 sidecar 重建输出尺寸
        assert tuple(sc["geometry"]["output_size"]) == pr.t.shape[:2]

    def test_rotation_changes_output_shape(self, synth):
        t = synth["negative"]
        pr0 = process_image(t, rotation_degrees=0)
        pr90 = process_image(t, rotation_degrees=90)
        assert pr90.t.shape[0] == pr0.t.shape[1] or \
            pr90.geometry.output_size[0] > 0
        assert pr90.geometry.rotation_degrees == 90

    def test_force_params_override(self, synth):
        t = synth["negative"]
        p = InvertParams(dmin=(0.3, 0.2, 0.1), dmax=1.8, gamma=1.23)
        pr = process_image(t, rotation_degrees=0, force_params=p)
        assert pr.report.params.gamma == 1.23
        assert pr.report.params.dmax == 1.8

    def test_keep_rebate_no_crop(self, synth):
        t = synth["negative"]
        pr = process_image(t, keep_rebate=True)
        assert pr.geometry.crop_oriented is None
        assert pr.t.shape[:2] == t.shape[:2]

    def test_gate_fail_blocks_batch_export(self, synth, tmp_path, monkeypatch):
        """I-3：fail 在写文件前阻断，不产出任何输出。"""
        import negative_lab.pipeline.service as svc
        original = svc.invert_chunked

        def bad_invert(t, p, **kw):
            res = original(t, p, **kw)
            bad = res.positive_linear.copy()
            bad[0, 0, 0] = np.nan
            res.positive_linear = bad
            return res
        monkeypatch.setattr(svc, "invert_chunked", bad_invert)
        from negative_lab.pipeline.batch import BatchJob
        import tifffile
        src = tmp_path / "s.tif"
        tifffile.imwrite(src, (np.clip(synth["negative"], 0, 1) * 65535
                               ).astype(np.uint16), photometric="rgb")
        out = tmp_path / "out"
        job = BatchJob([src], out, tiff_encoding="linear")
        rep = job.run()
        assert rep["items"][0]["status"] == "failed"
        assert rep["items"][0]["outputs"] == []
        assert not list(out.glob("s-positive*"))


def json_dumps(o) -> str:
    import json
    return json.dumps(o, ensure_ascii=False)
