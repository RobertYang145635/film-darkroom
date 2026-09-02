"""v5 §3.2/§5.2：三类独立区域的用户为准语义 + 批处理不再因缺区域阻断。"""

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
                     output_crop_roi=(10, 20, 300, 400),
                     image_area_roi=(20, 30, 290, 390),
                     base_sample_rois=[(0, 0, 50, 500), (250, 0, 300, 500)],
                     verified=True, verified_by="user")
    save_annotation("roll/a.NEF", ann)
    got = load_annotation("roll/a.NEF")
    assert got.rotation_degrees == 90
    assert tuple(got.output_crop_roi) == (10, 20, 300, 400)
    assert tuple(got.image_area_roi) == (20, 30, 290, 390)
    assert [tuple(r) for r in got.base_sample_rois] == \
        [(0, 0, 50, 500), (250, 0, 300, 500)]
    assert got.verified and got.verified_by == "user"
    assert got.verified_date          # 自动补日期


def test_manual_regions_skip_detection(monkeypatch):
    """三类区域齐全 -> 绝不调用 detect_regions（用户为准）。"""
    import negative_lab.pipeline.service as svc

    def boom(*a, **k):
        raise AssertionError("手动区域时不得自动检测")
    monkeypatch.setattr(svc, "detect_regions", boom)
    s = make_synthetic_negative(seed=61)
    t = s["negative"]
    h, w = t.shape[:2]
    pr = process_image(t, output_crop_source=(30, 30, h - 30, w - 100),
                       image_area_source=(30, 30, h - 30, w - 100),
                       base_sample_source=[(0, w - 90, h, w)])
    assert pr.gate.status in ("ok", "warn", "needs_review")
    assert "三类区域已按用户/标注确认" in " ".join(pr.report.warnings)


def test_batch_processes_frames_without_verified_regions(tmp_path):
    """v5 §3.3：缺少人工确认区域不再是阻断状态，走自动/回退并标记复核。"""
    from negative_lab.pipeline.batch import BatchJob
    s = make_synthetic_negative(seed=62)
    src = tmp_path / "x.tif"
    tifffile.imwrite(src, (np.clip(s["negative"], 0, 1) * 65535
                           ).astype(np.uint16), photometric="rgb")
    job = BatchJob([src], tmp_path / "out", tiff_encoding="linear")
    rep = job.run()
    item = rep["items"][0]
    assert item["status"] in ("ok", "warn", "needs_review"), item
    assert item["outputs"], "缺人工区域不得阻止导出"
