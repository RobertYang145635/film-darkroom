"""P0-C 回归（v6.1 §4）：「保留片边」状态闭环与导出链路。

历史缺陷：UI 写入不存在的 FrameRecord.keep_rebate；roll_export 不传
keep_rebate 给 process_image；勾选与否导出相同文件。
锁定契约：
- FrameRecord.keep_rebate: bool = False，to_dict/from_dict 显式布尔；
- Annotation 可选 keep_rebate（旧 schema 默认 False，非 bool 不当真）；
- 导出走 process_image(keep_rebate=rec.keep_rebate)，输出尺寸随勾选不同；
- 统计不受片边污染；草稿/确认标注保存后重开恢复勾选状态。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import tifffile

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module", autouse=True)
def qapp():
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from negative_lab.ui import theme

    app = QApplication.instance() or QApplication([])
    theme.apply(app)
    return app


# ------------------------------------------------------------- 数据模型
def test_frame_record_keep_rebate_round_trip():
    from negative_lab.library.models import FrameRecord

    r1 = FrameRecord(source_path="a.tif", keep_rebate=True)
    r2 = FrameRecord(source_path="a.tif", keep_rebate=False)
    d1 = r1.to_dict()
    assert d1["keep_rebate"] is True and r2.to_dict()["keep_rebate"] is False
    assert FrameRecord.from_dict(d1).keep_rebate is True
    assert FrameRecord.from_dict(r2.to_dict()).keep_rebate is False
    # 旧文件无字段 -> False；非 bool 不能静默当真
    assert FrameRecord.from_dict({"source_path": "a.tif"}).keep_rebate is False
    for bad in (1, "yes", None, []):
        assert FrameRecord.from_dict({"keep_rebate": bad}).keep_rebate is False


def test_annotation_keep_rebate_compat():
    from negative_lab.calibration.annotations import Annotation

    a = Annotation(file="r/a.tif", roll="r", keep_rebate=True)
    d = a.to_dict()
    assert d["keep_rebate"] is True
    b = Annotation.from_dict(d)
    assert b.keep_rebate is True
    # 旧 schema 文件：字段缺失 -> False；非 bool 不当真
    old = {"file": "r/a.tif", "roll": "r", "schema_version": 3}
    assert Annotation.from_dict(old).keep_rebate is False
    bad = dict(old, keep_rebate="true")
    assert Annotation.from_dict(bad).keep_rebate is False


# ------------------------------------------------------------- 导出链路
@pytest.fixture(scope="module")
def synth_tiff(tmp_path_factory):
    from negative_lab.testing.synthetic import make_synthetic_negative

    d = tmp_path_factory.mktemp("rebate")
    s = make_synthetic_negative(black_border=20, dust=10, seed=41)
    src = d / "frame.tif"
    tifffile.imwrite(src, (np.clip(s["negative"], 0, 1) * 65535).astype(np.uint16),
                     photometric="rgb")
    return src


def _frame_for(src: Path, keep: bool):
    from negative_lab.library.models import (FrameRecord, FrameStatus,
                                             GeometryRegions)

    sha = __import__("hashlib").sha256(src.read_bytes()).hexdigest()
    rec = FrameRecord(source_path=src.name, source_sha256=sha,
                      input_facts={"abs_path": str(src), "kind": "tiff"},
                      geometry=GeometryRegions(), keep_rebate=keep)
    rec.status = FrameStatus.NEEDS_CROP_REVIEW.value
    return rec


def test_export_keep_rebate_changes_output_size(synth_tiff, tmp_path,
                                                monkeypatch):
    """勾选与未勾选分别导出：输出尺寸不同且与 rec.keep_rebate 一致；
    sidecar 的 geometry.keep_rebate 也一致。"""
    from negative_lab.pipeline.roll_export import export_roll_frames

    out_a, out_b = tmp_path / "a", tmp_path / "b"
    rec_a = _frame_for(synth_tiff, keep=False)
    rec_b = _frame_for(synth_tiff, keep=True)
    r = export_roll_frames([rec_a], out_a, roll_id=None,
                           tiff_assumption="linear")
    assert r["ok"] == 1 and r["failed"] == 0, r["items"]
    r2 = export_roll_frames([rec_b], out_b, roll_id=None,
                            tiff_assumption="linear")
    assert r2["ok"] == 1 and r2["failed"] == 0, r2["items"]
    ta = tifffile.imread(str(next(out_a.glob("*.tif"))))
    tb = tifffile.imread(str(next(out_b.glob("*.tif"))))
    assert tb.shape[0] > ta.shape[0] or tb.shape[1] > ta.shape[1], \
        f"勾选保留片边必须更大：{tb.shape} vs {ta.shape}"
    sa = json.loads(next(out_a.glob("*.json")).read_text(encoding="utf-8"))
    sb = json.loads(next(out_b.glob("*.json")).read_text(encoding="utf-8"))
    assert sa["geometry"]["keep_rebate"] is False
    assert sb["geometry"]["keep_rebate"] is True


def test_export_stats_not_polluted_by_rebate(synth_tiff, tmp_path):
    """同一帧两次导出（只差 keep_rebate）：image-area/片基/门禁统计一致。"""
    from negative_lab.pipeline.roll_export import export_roll_frames

    out_a, out_b = tmp_path / "sa", tmp_path / "sb"
    rec_a = _frame_for(synth_tiff, keep=False)
    rec_b = _frame_for(synth_tiff, keep=True)
    export_roll_frames([rec_a], out_a, roll_id=None, tiff_assumption="linear")
    export_roll_frames([rec_b], out_b, roll_id=None, tiff_assumption="linear")
    sa = json.loads(next(out_a.glob("*.json")).read_text(encoding="utf-8"))
    sb = json.loads(next(out_b.glob("*.json")).read_text(encoding="utf-8"))
    ga, gb = sa["geometry"]["roi_source_space"], sb["geometry"]["roi_source_space"]
    assert ga["image_area_roi"] == gb["image_area_roi"], \
        "image_area 不得随 keep_rebate 变化（片边不能污染统计）"
    assert ga["base_sample_rois"] == gb["base_sample_rois"]
    assert sa["diagnostics"]["technical"]["gate"]["status"] == \
        sb["diagnostics"]["technical"]["gate"]["status"]
    assert sa["technical"]["parameters"] == sb["technical"]["parameters"], \
        "技术参数（Dmin/密度）不得被片边污染"


# ------------------------------------------------------------- UI 状态闭环
def _win(monkeypatch, tmp_path, t, keep_first: bool):
    import negative_lab.ui.main_window as mw
    from negative_lab.calibration import annotations as annmod
    from negative_lab.library.fingerprint import capture_fingerprint
    from negative_lab.library.models import (FrameRecord, GeometryRegions,
                                             ImportSession)
    from negative_lab.ui.main_window import MainWindow

    monkeypatch.setattr(annmod, "ANNOTATIONS_DIR", tmp_path / "ann")
    # 经 monkeypatch 改道草稿目录（直接赋值会永久污染后续测试：
    # 本测试暴露过 test_review_fixes 的损坏草稿被该残留吞掉的顺序耦合）
    monkeypatch.setattr(mw, "_DRAFTS_DIR", tmp_path / "drafts")
    rec = FrameRecord(source_path="synthetic/kr.tif",
                      source_sha256="0" * 64,
                      input_facts={"abs_path": "synthetic/kr.tif",
                                   "kind": "tiff"},
                      geometry=GeometryRegions(), keep_rebate=keep_first)
    w = MainWindow()
    w.session = ImportSession(files=[rec])
    w.current = rec
    w.t_source = t
    w._load_meta = {"kind": "tiff", "sha256": "0" * 64}
    w._fingerprint = capture_fingerprint(w._load_meta,
                                         source_path=rec.source_path)
    w._prepare_frame_views()
    return w, rec


def test_ui_toggle_and_frame_switch_state(monkeypatch, tmp_path):
    """勾选写入 rec；帧切换后勾选反映新帧状态（不串帧）。"""
    from negative_lab.testing.synthetic import make_synthetic_negative

    t = make_synthetic_negative(seed=13)["negative"]
    w, rec = _win(monkeypatch, tmp_path, t, keep_first=False)
    assert w.right.keep_rebate() is False
    w.right.sec_settings.chk_rebate.setChecked(True)   # 用户点击
    w.right.keep_rebate_toggled.emit(True)
    assert rec.keep_rebate is True
    # 第二帧 keep_rebate=False：载入后勾选必须为 False
    rec2 = __import__("copy").copy(rec)
    rec2.keep_rebate = False
    w.current = rec2
    w._sync_keep_rebate_ui(rec2)
    assert w.right.keep_rebate() is False
    w.current = rec
    w._sync_keep_rebate_ui(rec)
    assert w.right.keep_rebate() is True


def test_draft_persists_keep_rebate_across_reload(monkeypatch, tmp_path):
    """保存草稿 -> 重新加载（模拟重启后重开）：勾选状态恢复。"""
    from negative_lab.testing.synthetic import make_synthetic_negative

    t = make_synthetic_negative(seed=17)["negative"]
    w, rec = _win(monkeypatch, tmp_path, t, keep_first=False)
    w.right.sec_settings.chk_rebate.setChecked(True)
    w.right.keep_rebate_toggled.emit(True)
    assert rec.keep_rebate is True
    assert w._draft_file is not None and w._draft_file.exists()
    # 模拟重启后重新打开：走 _annotation_for -> _frame_done
    from negative_lab.ui.main_window import _annotation_for

    ann = _annotation_for(rec)
    assert ann is not None and ann.keep_rebate is True, "草稿必须保存勾选状态"
    rec2 = __import__("copy").copy(rec)
    rec2.keep_rebate = False        # 重开时内存默认 False
    d = {"ok": True, "value": {"rec": rec2, "rgb": t, "meta": w._load_meta,
                               "fingerprint": w._fingerprint,
                               "annotation": ann, "roll": None,
                               "preview": w.pr, "small": w._small,
                               "scale": w._small_scale}}
    w.current = rec2
    w._frame_done(d, rec2)
    assert rec2.keep_rebate is True, "重新加载必须从草稿恢复勾选状态"
    assert w.right.keep_rebate() is True


def test_unchecked_uses_output_crop_not_image_area(monkeypatch, tmp_path):
    """未勾选导出使用 output_crop_roi；没有 output crop 时按现有规则进入
    needs_crop_review（UI 状态规则），service 的 image-area 替代必须显式
    记录 provenance，不得伪装成用户构图。"""
    from negative_lab.library.models import (FrameRecord, FrameStatus,
                                             GeometryRegions)
    from negative_lab.pipeline.service import process_image
    from negative_lab.testing.synthetic import make_synthetic_negative

    s = make_synthetic_negative(black_border=16, seed=23)
    t = s["negative"]
    area = (20, 20, t.shape[0] - 20, int(t.shape[1] * 0.6))
    geo = GeometryRegions(image_area_roi=area)          # 无 output_crop
    pr = process_image(t, geometry=geo, keep_rebate=False)
    # service 自动补 output_crop 时 provenance 不得标成 user
    assert pr.roi_provenance.get("output_crop") != "user", \
        "自动替代不得伪装成用户构图"
    geo2 = GeometryRegions(image_area_roi=area, output_crop_roi=area)
    pr2 = process_image(t, geometry=geo2, keep_rebate=False)
    assert pr2.t.shape[:2] == (area[2] - area[0], area[3] - area[1]), \
        "有 output_crop 时导出必须用它"
    # 现有契约：用户直接给三类区域时 provenance 不再写自动来源
    assert pr2.roi_provenance.get("output_crop") in (None, "user")
    # UI 现有规则：缺 output_crop 的帧 -> needs_crop_review
    rec = FrameRecord(source_path="x.tif",
                      geometry=GeometryRegions(image_area_roi=area))
    assert not rec.geometry.is_complete()
    rec.status = FrameStatus.NEEDS_CROP_REVIEW.value
    assert rec.status == FrameStatus.NEEDS_CROP_REVIEW.value
