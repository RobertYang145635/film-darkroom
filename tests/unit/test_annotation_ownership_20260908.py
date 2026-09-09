"""标注/草稿归属（撞名）回归锁 —— 2026-09-08 冷读审查 L2-④。

ann_path_for 用相对路径的 parts[1] 当"卷名"，因此浅层与深层路径都会落到同一个
文件名。实测本仓库自己的两个顶层样片目录：
    20260506_SAMPLE-ID/G200-1.nef     -> misc__G200-1.json
    20260709_SAMPLE-ID(1)/G200-1.nef  -> misc__G200-1.json
"文件名命中"绝不等于"这帧的标注"。此前既不校验也不拒绝：B 帧会静默套用 A 帧的
签字几何，保存时还会直接覆盖掉 A 的标注。
"""

from __future__ import annotations

import json

import pytest

A_REL = "20260506_SAMPLE-ID/G200-1.nef"
B_REL = "20260709_SAMPLE-ID(1)/G200-1.nef"


def _ann(rel, roi):
    from negative_lab.calibration.annotations import Annotation
    return Annotation(file=rel, roll="r", rotation_degrees=0,
                      coordinate_space="source", status="draft",
                      image_area_roi=roi)


class TestCollisionIsReal:
    def test_two_frames_share_one_filename(self):
        from negative_lab.calibration.annotations import ann_path_for
        assert ann_path_for(A_REL).name == ann_path_for(B_REL).name


class TestOwnershipEnforced:
    def test_load_refuses_another_frames_annotation(self, tmp_path):
        from negative_lab import storage
        from negative_lab.calibration.annotations import load_annotation
        with storage.isolated_storage(tmp_path):
            (storage.annotations_dir() / "misc__G200-1.json").write_text(
                json.dumps(_ann(A_REL, (0, 0, 8, 8)).to_dict()),
                encoding="utf-8")
            assert load_annotation(B_REL) is None, "B 帧吃下了 A 帧的标注"
            assert load_annotation(A_REL) is not None, "归属相符时仍须可用"

    def test_save_refuses_to_clobber(self, tmp_path):
        from negative_lab import storage
        from negative_lab.calibration.annotations import save_annotation
        with storage.isolated_storage(tmp_path):
            save_annotation(A_REL, _ann(A_REL, (0, 0, 8, 8)))
            with pytest.raises(ValueError, match="命名冲突"):
                save_annotation(B_REL, _ann(B_REL, (1, 1, 9, 9)))
            kept = json.loads((storage.annotations_dir() /
                               "misc__G200-1.json").read_text("utf-8"))
            assert kept["file"] == A_REL, "A 帧的标注被覆盖了"

    def test_foreign_draft_ignored_and_reported(self, tmp_path):
        from negative_lab import storage
        from negative_lab.library.models import FrameRecord
        from negative_lab.ui import main_window as mw
        with storage.isolated_storage(tmp_path):
            name = mw.ann_path_for(B_REL).name
            (storage.drafts_dir() / name).write_text(
                json.dumps(_ann(A_REL, (0, 0, 8, 8)).to_dict()),
                encoding="utf-8")
            st = mw.resolve_persistent_state(
                FrameRecord(source_path=B_REL, source_sha256="c" * 64))
            assert st.annotation is None, "别人的草稿成了生效几何"
            assert "归属不符" in st.reason, st.reason
