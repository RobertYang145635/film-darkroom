"""导出必须尊重已落盘的用户标定 —— 冷读审查 L5-②（2026-09-08）。

只有本会话打开过的帧才有 rec.geometry；用户上个会话签过字、这次没点开就整卷
导出的帧，此前会按自动检测几何出片，签字基线被静默忽略。
"""

from __future__ import annotations

from negative_lab import storage
from negative_lab.calibration.annotations import Annotation, save_annotation
from negative_lab.library.models import FrameRecord, GeometryRegions
from negative_lab.pipeline.roll_export import _export_geometry


def _rec():
    return FrameRecord(source_path="rolls/R1/0001.nef", source_sha256="a" * 64,
                       geometry=GeometryRegions(rotation_degrees=0))


def _ann(rel, verified=True, by="user", sha="a" * 64):
    return Annotation(file=rel, roll="R1", rotation_degrees=0,
                      coordinate_space="source",
                      status="verified" if verified else "draft",
                      verified=verified, verified_by=by,
                      image_area_roi=(10, 10, 90, 90),
                      base_sample_rois=[(0, 0, 6, 6)],
                      output_crop_roi=(10, 10, 90, 90),
                      source_sha256=sha)


def test_incomplete_geometry_is_restored_from_confirmed_annotation(tmp_path):
    with storage.isolated_storage(tmp_path):
        rec = _rec()
        save_annotation(rec.source_path, _ann(rec.source_path))
        geo = _export_geometry(rec)
        assert geo.is_complete(), "签字几何没被回读，导出仍走自动检测"
        assert geo.image_area_roi == (10, 10, 90, 90)


def test_unverified_annotation_does_not_drive_export(tmp_path):
    with storage.isolated_storage(tmp_path):
        rec = _rec()
        save_annotation(rec.source_path,
                        _ann(rec.source_path, verified=False, by=""))
        assert not _export_geometry(rec).is_complete()


def test_stale_source_hash_is_not_applied(tmp_path):
    with storage.isolated_storage(tmp_path):
        rec = _rec()
        save_annotation(rec.source_path, _ann(rec.source_path, sha="b" * 64))
        assert not _export_geometry(rec).is_complete(), "源图已换还套用旧标定"


def test_geometry_already_in_session_wins(tmp_path):
    with storage.isolated_storage(tmp_path):
        rec = _rec()
        rec.geometry = GeometryRegions(
            rotation_degrees=0, image_area_roi=(1, 2, 3, 4),
            base_sample_rois=[(0, 0, 2, 2)], output_crop_roi=(1, 2, 3, 4))
        save_annotation(rec.source_path, _ann(rec.source_path))
        assert _export_geometry(rec).image_area_roi == (1, 2, 3, 4)
