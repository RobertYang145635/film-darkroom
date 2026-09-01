"""v3 用户裁决：手动区域为主（保存/跳过检测/批处理策略）。"""

from __future__ import annotations

import numpy as np
import tifffile

from negative_lab.calibration.annotations import (Annotation, load_annotation,
                                                  save_annotation)
from negative_lab.pipeline.service import process_image
from negative_lab.testing.synthetic import make_synthetic_negative


def test_save_load_roundtrip(tmp_path, monkeypatch):
    import negative_lab.calibration.annotations as A
    monkeypatch.setattr(A, "ANNOTATIONS_DIR", tmp_path)
    ann = Annotation(file="roll/a.NEF", roll="roll", rotation_degrees=90,
                     frame_roi=(10, 20, 300, 400), base_roi=(0, 0, 50, 500),
                     verified=True, verified_by="user")
    save_annotation("roll/a.NEF", ann)
    got = load_annotation("roll/a.NEF")
    assert got.rotation_degrees == 90
    assert tuple(got.frame_roi) == (10, 20, 300, 400)
    assert got.verified and got.verified_by == "user"
    assert got.verified_date          # 自动补日期


def test_manual_regions_skip_detection(monkeypatch):
    """手动 frame+base 齐全 -> 绝不调用 detect_regions（用户为准）。"""
    import negative_lab.pipeline.service as svc

    def boom(*a, **k):
        raise AssertionError("手动区域时不得自动检测")
    monkeypatch.setattr(svc, "detect_regions", boom)
    s = make_synthetic_negative(seed=61)
    t = s["negative"]
    h, w = t.shape[:2]
    pr = process_image(t, frame_roi_source=(30, 30, h - 30, w - 100),
                       base_roi_source=(0, w - 90, h, w))
    assert pr.gate.status in ("ok", "warn", "needs_calibration")
    assert "手动区域已指定" in " ".join(pr.report.warnings)


def test_batch_require_regions(tmp_path):
    from negative_lab.pipeline.batch import BatchJob
    s = make_synthetic_negative(seed=62)
    src = tmp_path / "x.tif"
    tifffile.imwrite(src, (np.clip(s["negative"], 0, 1) * 65535
                           ).astype(np.uint16), photometric="rgb")
    job = BatchJob([src], tmp_path / "out", tiff_encoding="linear",
                   regions_policy="require")
    rep = job.run()
    assert rep["items"][0]["status"] == "needs_regions"
    assert rep["items"][0]["outputs"] == []
