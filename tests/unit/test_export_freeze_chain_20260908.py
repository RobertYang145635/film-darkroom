"""freeze_export_inputs → worker → apply_status_writeback 全链路回归
（冷读 minor-4：GUI 组装路径此前无直接测试；2026-09-08）。

同时锁定冷读修复后的新契约：
- snapshot_frame 预捕获确认标注（frozen_annotation）：导出中途签字/改盘
  不影响本次成品的几何（快照模式 worker 零盘读）；
- 同步模式标注逐帧读盘的行为不变（L5-② 契约由
  test_export_uses_annotations_20260908.py 继续锁定）；
- export_roll_frames(presnapshot=True, session=...) 不再经 session 兜底
  读活对象（冷读 nit：结构性保证）。
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
import tifffile


@pytest.fixture(scope="module")
def synth_tiff(tmp_path_factory):
    from negative_lab.testing.synthetic import make_synthetic_negative

    d = tmp_path_factory.mktemp("freeze")
    s = make_synthetic_negative(black_border=20, dust=10, seed=53)
    src = d / "frame.tif"
    tifffile.imwrite(src, (np.clip(s["negative"], 0, 1) * 65535).astype(np.uint16),
                     photometric="rgb")
    return src


def _img_shape(src):
    return tifffile.imread(str(src)).shape[:2]


def test_freeze_worker_writeback_chain(synth_tiff, tmp_path):
    """GUI 组装路径：freeze（主线程模拟）→ export_roll_frames(session=None)
    → apply_status_writeback；中途改活对象不影响成品。"""
    from negative_lab.library.models import (FrameStatus, FrameRecord,
                                             GeometryRegions, ImportSession)
    from negative_lab.pipeline.roll_export import (apply_status_writeback,
                                                   export_roll_frames,
                                                   freeze_export_inputs)

    h, w = _img_shape(synth_tiff)
    sha = hashlib.sha256(synth_tiff.read_bytes()).hexdigest()
    rec = FrameRecord(
        source_path=synth_tiff.name, source_sha256=sha,
        input_facts={"abs_path": str(synth_tiff), "kind": "tiff"},
        geometry=GeometryRegions(
            output_crop_roi=(h // 4, w // 4, 3 * h // 4, 3 * w // 4),
            image_area_roi=(h // 4, w // 4, 3 * h // 4, 3 * w // 4),
            base_sample_rois=[(0, 0, 8, w)]))
    rec.status = FrameStatus.READY_TO_CONVERT.value
    sess = ImportSession(files=[rec])
    frozen = freeze_export_inputs(sess, None, [rec])
    assert frozen["intents"] and synth_tiff.name in frozen["intents"]

    def progress(_m):
        # 导出中途改活对象（用户编辑）
        rec.geometry = GeometryRegions(output_crop_roi=(0, 0, 8, 8),
                                       image_area_roi=(0, 0, 8, 8),
                                       base_sample_rois=[])

    rep = export_roll_frames(
        frozen["records"], tmp_path / "out", session=None, roll_id=None,
        intent_resolver=lambda r: frozen["intents"].get(
            r.source_path) or frozen["intents"].get(""),
        tiff_assumption="linear",
        presnapshot=True, frozen_roll_stats=frozen["roll_stats"],
        frozen_batch_assumption=frozen["batch_assumption"],
        progress=progress)
    assert rep["failed"] == 0, rep["items"]
    sidecar = json.loads(next((tmp_path / "out").glob("*.json"))
                         .read_text(encoding="utf-8"))
    got = sidecar["geometry"]["roi_source_space"]["output_crop_roi"]
    assert tuple(got) == (h // 4, w // 4, 3 * h // 4, 3 * w // 4), \
        f"冻结几何必须生效：{got}"
    # worker 零写
    assert rec.status == FrameStatus.READY_TO_CONVERT.value
    # 主线程回写
    apply_status_writeback(sess, rep["items"])
    entry = next(e for e in rep["items"] if e["file"] == rec.source_path)
    assert rec.status == (FrameStatus.NEEDS_REVIEW.value
                          if entry["status"] == "needs_review"
                          else FrameStatus.EXPORTED.value)


def test_frozen_annotation_wins_over_mid_export_confirmation(synth_tiff,
                                                             tmp_path):
    """几何不完整 + 冻结后有确认标注：成品用冻结时刻之前的（无标注）结果，
    中途签字不进本次导出（快照模式零盘读）。"""
    from negative_lab import storage
    from negative_lab.calibration.annotations import Annotation, \
        save_annotation
    from negative_lab.library.models import FrameRecord, GeometryRegions
    from negative_lab.pipeline.roll_export import (export_roll_frames,
                                                   snapshot_frame)

    h, w = _img_shape(synth_tiff)
    sha = hashlib.sha256(synth_tiff.read_bytes()).hexdigest()
    rec = FrameRecord(source_path=synth_tiff.name, source_sha256=sha,
                      input_facts={"abs_path": str(synth_tiff), "kind": "tiff"},
                      geometry=GeometryRegions())
    # 冻结时刻：无标注 → frozen_annotation=None
    snap = snapshot_frame(rec)
    assert snap.frozen_annotation is None
    # 导出开始后用户签字（写盘）
    with storage.isolated_storage(tmp_path):
        save_annotation(rec.source_path, Annotation(
            file=rec.source_path, roll="misc", rotation_degrees=0,
            coordinate_space="source", status="verified",
            verified=True, verified_by="user",
            image_area_roi=(10, 10, 90, 90),
            base_sample_rois=[(0, 0, 6, w)],
            output_crop_roi=(10, 10, 90, 90),
            source_sha256=sha))
        rep = export_roll_frames([snap], tmp_path / "out", presnapshot=True,
                                 tiff_assumption="linear")
        assert rep["failed"] == 0, rep["items"]
        sidecar = json.loads(next((tmp_path / "out").glob("*.json"))
                             .read_text(encoding="utf-8"))
        got = sidecar["geometry"]["roi_source_space"]["output_crop_roi"]
        assert got is None or tuple(got) != (10, 10, 90, 90), \
            "导出中途的签字不得进入本次成品（冻结语义）"


def test_frozen_annotation_captured_when_present(synth_tiff, tmp_path):
    """冻结时刻已有确认标注：进入 frozen_annotation，成品用它（L5-②）。"""
    from negative_lab import storage
    from negative_lab.calibration.annotations import Annotation, \
        save_annotation
    from negative_lab.library.models import FrameRecord, GeometryRegions
    from negative_lab.pipeline.roll_export import (export_roll_frames,
                                                   snapshot_frame)

    sha = hashlib.sha256(synth_tiff.read_bytes()).hexdigest()
    rec = FrameRecord(source_path=synth_tiff.name, source_sha256=sha,
                      input_facts={"abs_path": str(synth_tiff), "kind": "tiff"},
                      geometry=GeometryRegions())
    with storage.isolated_storage(tmp_path):
        save_annotation(rec.source_path, Annotation(
            file=rec.source_path, roll="misc", rotation_degrees=0,
            coordinate_space="source", status="verified",
            verified=True, verified_by="user",
            image_area_roi=(10, 10, 90, 90),
            base_sample_rois=[(0, 0, 6, 6)],
            output_crop_roi=(10, 10, 90, 90),
            source_sha256=sha))
        snap = snapshot_frame(rec)
        assert snap.frozen_annotation is not None, "确认标注必须被预捕获"
        rep = export_roll_frames([snap], tmp_path / "out", presnapshot=True,
                                 tiff_assumption="linear")
        assert rep["failed"] == 0, rep["items"]
        sidecar = json.loads(next((tmp_path / "out").glob("*.json"))
                             .read_text(encoding="utf-8"))
        got = sidecar["geometry"]["roi_source_space"]["output_crop_roi"]
        assert tuple(got) == (10, 10, 90, 90), \
            "冻结时刻的确认标注必须驱动导出几何"
