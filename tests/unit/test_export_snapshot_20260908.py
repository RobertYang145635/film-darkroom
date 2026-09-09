"""§4-1 回归（2026-09-08 交接文档最高危未修项）：导出输入快照冻结 + 状态回写。

历史缺陷：后台导出线程逐帧读活对象 `rec.geometry / keep_rebate /
render_intent_override / frame_analysis`，导出期间的 UI 编辑会漏进成品
（错误产物安静出厂）；worker 还直接写 `rec.status / capture_fingerprint`。

锁定契约：
- presnapshot=True：入口冻结输入，导出期间改活对象，成品仍用启动前几何；
- 快照模式下 worker 零写 rec（status/capture_fingerprint 原样保留）；
- apply_status_writeback 按 items 在调用方回写状态（ok / needs_review /
  failed 按 decode 前缀分 DECODE_ERROR，其余 PROCESSING_ERROR）；
  cancelled 与未知帧不碰；
- 同步路径（presnapshot=False）保持旧语义，由既有
  test_v6_keep_rebate / test_v6_2_fixes / test_review_fixes 覆盖。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import tifffile


@pytest.fixture(scope="module")
def synth_tiff(tmp_path_factory):
    from negative_lab.testing.synthetic import make_synthetic_negative

    d = tmp_path_factory.mktemp("snapshot")
    s = make_synthetic_negative(black_border=20, dust=10, seed=43)
    src = d / "frame.tif"
    tifffile.imwrite(src, (np.clip(s["negative"], 0, 1) * 65535).astype(np.uint16),
                     photometric="rgb")
    return src


def _complete_geometry(img_shape):
    from negative_lab.library.models import GeometryRegions

    h, w = img_shape[:2]
    # 三类区域齐全（is_complete）-> _export_geometry 直接采用 rec.geometry，
    # 输出构图完全由 output_crop_roi 决定，便于断言冻结生效。
    return GeometryRegions(
        output_crop_roi=(h // 4, w // 4, 3 * h // 4, 3 * w // 4),
        image_area_roi=(h // 4, w // 4, 3 * h // 4, 3 * w // 4),
        base_sample_rois=[(0, 0, 8, w)])


def _frame_for(src: Path, geo):
    from negative_lab.library.models import FrameRecord, FrameStatus

    sha = hashlib.sha256(src.read_bytes()).hexdigest()
    rec = FrameRecord(source_path=src.name, source_sha256=sha,
                      input_facts={"abs_path": str(src), "kind": "tiff"},
                      geometry=geo)
    rec.status = FrameStatus.READY_TO_CONVERT.value
    return rec


def test_snapshot_freezes_geometry_against_live_edits(synth_tiff, tmp_path):
    """导出期间改活对象几何/keep_rebate，成品仍用启动前输入（§4-1 验收）。"""
    from negative_lab.library.models import GeometryRegions
    from negative_lab.pipeline.roll_export import export_roll_frames

    img = tifffile.imread(str(synth_tiff))
    h, w = img.shape[:2]
    out = tmp_path / "out"
    rec = _frame_for(synth_tiff, _complete_geometry(img.shape))
    orig_roi = rec.geometry.output_crop_roi
    mutated = {"done": False}

    def progress(_msg):
        """模拟 UI 线程在导出进行中编辑活对象。"""
        if not mutated["done"]:
            mutated["done"] = True
            rec.geometry = GeometryRegions(
                output_crop_roi=(0, 0, max(8, h // 2), max(8, w // 2)),
                image_area_roi=(0, 0, max(8, h // 2), max(8, w // 2)),
                base_sample_rois=[(0, 0, 8, w)])
            rec.keep_rebate = True

    rep = export_roll_frames([rec], out, presnapshot=True, progress=progress,
                             tiff_assumption="linear")
    assert rep["failed"] == 0, rep["items"]
    entry = next(e for e in rep["items"] if e["file"] == rec.source_path)
    assert entry["status"] in ("ok", "needs_review")
    assert mutated["done"], "测试自证：活对象确实在导出中途被改"
    sidecar = json.loads(next(out.glob("*.json")).read_text(encoding="utf-8"))
    got = sidecar["geometry"]["roi_source_space"]["output_crop_roi"]
    assert tuple(got) == tuple(orig_roi), \
        f"成品必须用启动前几何：{got} != {orig_roi}"
    assert sidecar["geometry"]["keep_rebate"] is False, \
        "导出中途勾选保留片边不得影响本次成品"


def test_snapshot_mode_worker_writes_nothing_and_writeback_reports(synth_tiff,
                                                                  tmp_path):
    """快照模式 worker 零写 rec；主线程 writeback 按 items 回写状态。"""
    from negative_lab.library.models import FrameStatus, ImportSession
    from negative_lab.pipeline.roll_export import (apply_status_writeback,
                                                   export_roll_frames)

    img = tifffile.imread(str(synth_tiff))
    out = tmp_path / "out"
    rec = _frame_for(synth_tiff, _complete_geometry(img.shape))
    sess = ImportSession(files=[rec])
    rep = export_roll_frames([rec], out, presnapshot=True,
                             tiff_assumption="linear")
    assert rec.status == FrameStatus.READY_TO_CONVERT.value, \
        "快照模式下导出线程不得写 rec.status"
    entry = next(e for e in rep["items"] if e["file"] == rec.source_path)
    if entry.get("capture_fingerprint"):
        assert rec.capture_fingerprint != entry["capture_fingerprint"], \
            "快照模式下 capture_fingerprint 只进 items，不得即写"
    apply_status_writeback(sess, rep["items"])
    expected = (FrameStatus.NEEDS_REVIEW.value
                if entry["status"] == "needs_review"
                else FrameStatus.EXPORTED.value)
    assert rec.status == expected
    if entry.get("capture_fingerprint"):
        assert rec.capture_fingerprint == entry["capture_fingerprint"]


def test_writeback_maps_failure_classes_and_ignores_cancelled():
    """failed 按 decode 前缀分类；cancelled/未知帧不改状态（不伪造结论）。"""
    from negative_lab.library.models import (FrameRecord, FrameStatus,
                                             ImportSession)
    from negative_lab.pipeline.roll_export import apply_status_writeback

    r1 = FrameRecord(source_path="a.tif")
    r2 = FrameRecord(source_path="b.tif")
    r3 = FrameRecord(source_path="c.tif")
    r4 = FrameRecord(source_path="d.tif")
    for r in (r1, r2, r3, r4):
        r.status = FrameStatus.IMPORTED.value
    sess = ImportSession(files=[r1, r2, r3, r4])
    apply_status_writeback(sess, [
        {"file": "a.tif", "status": "needs_review"},
        {"file": "b.tif", "status": "failed", "error": "decode: 坏图"},
        {"file": "c.tif", "status": "failed", "error": "export: 磁盘满"},
        {"file": "d.tif", "status": "cancelled"},
        {"file": "not-in-session.tif", "status": "ok"},   # 未知帧：跳过
    ])
    assert r1.status == FrameStatus.NEEDS_REVIEW.value
    assert r2.status == FrameStatus.DECODE_ERROR.value
    assert r3.status == FrameStatus.PROCESSING_ERROR.value
    assert r4.status == FrameStatus.IMPORTED.value, "cancelled 不得改状态"


def test_writeback_classifies_real_decode_failure(synth_tiff, tmp_path):
    """真实失败路径：源文件缺失 -> ReadError -> 主线程回写 DECODE_ERROR。"""
    from negative_lab.library.models import (FrameRecord, FrameStatus,
                                             ImportSession)
    from negative_lab.pipeline.roll_export import (apply_status_writeback,
                                                   export_roll_frames)

    rec = FrameRecord(source_path="missing_frame.tif",
                      input_facts={"abs_path": str(tmp_path / "nope.tif"),
                                   "kind": "tiff"})
    rec.status = FrameStatus.IMPORTED.value
    sess = ImportSession(files=[rec])
    rep = export_roll_frames([rec], tmp_path / "out", presnapshot=True)
    assert rep["failed"] == 1 and rep["ok"] == 0
    assert rec.status == FrameStatus.IMPORTED.value, "worker 不得写状态"
    apply_status_writeback(sess, rep["items"])
    assert rec.status == FrameStatus.DECODE_ERROR.value
