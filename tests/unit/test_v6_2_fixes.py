"""v6.2 修复的回归测试（审查 20260907 + v6.2 §9.3 回归矩阵）。

覆盖：
- F1 冷启动先用持久状态（不再"界面显示用户标定、算法跑自动检测"）
- F2 有效画面区编辑必须让分析失效（此前只有片基会重跑）
- F3 三个曾抛 AttributeError 的回调（重新检测 / 候选 / 同步作用域）
- F4 确认基线 vs 工作草稿的六种状态组合（含旧草稿无指纹 -> conflict）
- F5 画布坐标 -> 整幅技术正片坐标（裁切生效时偏移非零，此前直接错一个裁切原点）
- F9 离开调整阶段退出拾点；分析阶段点击不得按正片求解
- 导出计数覆盖全部帧（取消不再静默消失）
- 存储解析优先级：显式改道 > 环境变量 > 仓库默认
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module", autouse=True)
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from negative_lab.ui import theme

    app = QApplication.instance() or QApplication([])
    theme.apply(app)
    return app


def _win(cast=(0.0, 0.0, 0.06)):
    """调整阶段的窗口：技术正片带注入蓝偏（模拟自动翻车）。"""
    from negative_lab.library.models import (FrameRecord, GeometryRegions,
                                             ImportSession)
    from negative_lab.pipeline.render import RenderIntent
    from negative_lab.testing.synthetic import make_synthetic_negative
    from negative_lab.ui.main_window import MainWindow

    w = MainWindow()
    rec = FrameRecord(source_path="synthetic/v62/frame.tif",
                      source_sha256="0" * 64,
                      input_facts={"abs_path": "synthetic/v62/frame.tif",
                                   "kind": "tiff"},
                      geometry=GeometryRegions())
    w.session = ImportSession(files=[rec])
    w.current = rec
    w.t_source = make_synthetic_negative(seed=11)["negative"]
    w._load_meta = {"kind": "tiff", "sha256": "0" * 64}
    w._prepare_frame_views()
    w._set_stage(1)
    w._positive_full = np.clip(w._positive_full + np.array(cast, np.float32),
                               0, 1)
    w._intent = RenderIntent()
    return w, rec


class TestF5CoordinateMapping:
    """审查 §2.1：显示图是整幅正片的子矩形时，画布坐标必须换算。"""

    def test_full_frame_view_has_zero_origin(self):
        w, rec = _win()
        rec.geometry.output_crop_roi = None
        rec.keep_rebate = True
        w._compose_view()
        assert w._view_origin == (0, 0)
        assert w._view_space == "positive_full"

    def test_cropped_view_records_origin_and_pick_uses_it(self, monkeypatch):
        import negative_lab.pipeline.rescue as rescue
        from negative_lab.pipeline.render import RenderIntent

        w, rec = _win()
        h, wd = w._positive_full.shape[:2]
        # 输出裁切 = SOURCE 内偏右下的子矩形（未勾选保留片边）
        src_h, src_w = w.t_source.shape[:2]
        rec.keep_rebate = False
        rec.geometry.output_crop_roi = (
            int(src_h * 0.25), int(src_w * 0.30),
            int(src_h * 0.85), int(src_w * 0.92))
        w._compose_view()
        ox, oy = w._view_origin
        assert w._view_space == "positive_full_cropped"
        assert ox > 0 and oy > 0, "裁切生效时显示原点必须非零"

        seen = {}

        def fake_exact(tech, py, px, **kw):
            seen["py"], seen["px"] = py, px
            return RenderIntent(), []

        monkeypatch.setattr(rescue, "neutral_pick_intent_exact", fake_exact)
        w._on_point_picked("neutral", 10.0, 10.0)      # 画布左上角
        assert seen == {"py": 10 + oy, "px": 10 + ox}, \
            "画布 (10,10) 必须采样整幅正片的 (10+oy, 10+ox)"

    def test_out_of_bounds_after_offset_is_rejected(self, monkeypatch):
        import negative_lab.pipeline.rescue as rescue

        w, rec = _win()
        src_h, src_w = w.t_source.shape[:2]
        rec.keep_rebate = False
        rec.geometry.output_crop_roi = (0, 0, int(src_h * 0.6), int(src_w * 0.6))
        w._compose_view()
        called = []
        monkeypatch.setattr(rescue, "neutral_pick_intent_exact",
                            lambda *a, **k: called.append(1))
        w._on_point_picked("neutral", 1e6, 1e6)
        assert not called


class TestF9StageGating:
    def test_leaving_adjust_stage_clears_pick_mode(self):
        w, _ = _win()
        w._on_pick_neutral(True)
        assert w.view._pick_mode == "neutral"
        w._set_stage(0)
        assert w.view._pick_mode is None, "切回分析阶段必须退出拾点"
        assert not w.right.btn_pick_neutral.isChecked()

    def test_pick_in_analysis_stage_does_not_solve_on_positive(self, monkeypatch):
        import negative_lab.pipeline.rescue as rescue

        w, _ = _win()
        w._set_stage(0)
        called = []
        monkeypatch.setattr(rescue, "neutral_pick_intent_exact",
                            lambda *a, **k: called.append(1))
        w._on_point_picked("neutral", 20.0, 20.0)
        assert not called, "分析阶段显示的是负片，不得按正片像素求解"


class TestF3CallbacksDoNotRaise:
    def test_auto_crop_button(self):
        w, _ = _win()
        w._auto_crop()          # 此前 AttributeError: _set_banner_status

    def test_candidate_button_syncs_sliders(self):
        w, _ = _win()
        w._choose_candidate("neutral")
        assert w._candidate == "neutral"
        assert w.current.chosen_candidate == "neutral"
        # 滑杆必须跟着候选走（此前 convert_panel 不存在直接抛错）
        assert abs(w.right.panel.intent().temperature
                   - w._intent.temperature) < 0.5, \
            "选候选后右栏滑杆必须跟着走（v6.1 这里直接抛 AttributeError）"

    def test_sync_intent_reads_scope_combo(self):
        w, _ = _win()
        w.right.panel.combo_sync_scope.setCurrentText("当前帧")
        w._sync_intent(["temperature"])      # 此前 AttributeError: convert_panel


class TestF2Invalidation:
    def _spy(self, w, monkeypatch):
        calls = []
        monkeypatch.setattr(w, "_refresh_preview",
                            lambda *a, **k: calls.append(k.get("reason")))
        return calls

    def test_base_edit_reruns(self, monkeypatch):
        w, _ = _win()
        from negative_lab.ui.preview_view import RoiKind
        calls = self._spy(w, monkeypatch)
        w._on_roi_edited(RoiKind.BASE)
        assert calls

    def test_image_area_edit_reruns(self, monkeypatch):
        """F2：有效画面区决定统计范围，改完必须重算分析。"""
        w, _ = _win()
        from negative_lab.ui.preview_view import RoiKind
        calls = self._spy(w, monkeypatch)
        w._on_roi_edited(RoiKind.IMAGE_AREA)
        assert calls, "改有效画面后仍挂着按旧范围算出的分析结果"

    def test_output_crop_stays_fast_path(self, monkeypatch):
        w, _ = _win()
        from negative_lab.ui.preview_view import RoiKind
        calls = self._spy(w, monkeypatch)
        w._on_roi_edited(RoiKind.OUTPUT_CROP)
        assert not calls, "输出裁切只改构图，不该触发管线"


class TestF4StateResolution:
    def _ann(self, **kw):
        from negative_lab.calibration.annotations import Annotation
        base = dict(file="rolls/roll/a.tif", roll="roll",
                    image_area_roi=(10, 10, 90, 90),
                    base_sample_rois=[(0, 0, 5, 5)],
                    output_crop_roi=(5, 5, 95, 95),
                    source_sha256="a" * 64)
        base.update(kw)
        return Annotation(**base)

    def test_none(self):
        from negative_lab.calibration.state import SOURCE_NONE, resolve_state
        r = resolve_state(confirmed=None, draft=None)
        assert r.source == SOURCE_NONE and r.annotation is None

    def test_confirmed_only(self):
        from negative_lab.calibration.state import (SOURCE_CONFIRMED,
                                                    resolve_state)
        c = self._ann(status="verified", verified_by="user")
        r = resolve_state(confirmed=c, draft=None)
        assert r.source == SOURCE_CONFIRMED and r.annotation is c

    def test_draft_only(self):
        from negative_lab.calibration.state import (SOURCE_WORK_DRAFT,
                                                    resolve_state)
        d = self._ann()
        r = resolve_state(confirmed=None, draft=d)
        assert r.source == SOURCE_WORK_DRAFT and r.annotation is d

    def test_draft_on_current_baseline_wins(self):
        from negative_lab.calibration.state import (SOURCE_WORK_DRAFT,
                                                    content_fingerprint,
                                                    resolve_state)
        c = self._ann(status="verified", verified_by="user")
        d = self._ann(base_annotation_sha=content_fingerprint(c),
                      draft_revision=3, image_area_roi=(20, 20, 80, 80))
        r = resolve_state(confirmed=c, draft=d)
        assert r.source == SOURCE_WORK_DRAFT and r.annotation is d
        assert r.draft_fingerprint_match is True and r.draft_revision == 3

    def test_draft_without_provenance_is_conflict(self):
        """旧版草稿没有基线指纹 -> 绝不静默合并，确认基线生效。"""
        from negative_lab.calibration.state import (SOURCE_CONFLICT,
                                                    resolve_state)
        c = self._ann(status="verified", verified_by="user")
        d = self._ann(image_area_roi=(20, 20, 80, 80))
        r = resolve_state(confirmed=c, draft=d)
        assert r.source == SOURCE_CONFLICT and r.annotation is c
        assert r.needs_reconfirm and r.draft is d, "草稿必须原地保留"

    def test_draft_on_stale_baseline_is_conflict(self):
        from negative_lab.calibration.state import (SOURCE_CONFLICT,
                                                    resolve_state)
        c = self._ann(status="verified", verified_by="user")
        d = self._ann(base_annotation_sha="deadbeefdeadbeef")
        r = resolve_state(confirmed=c, draft=d)
        assert r.source == SOURCE_CONFLICT and r.annotation is c

    def test_changed_source_invalidates_both(self):
        from negative_lab.calibration.state import (SOURCE_STALE, resolve_state)
        c = self._ann(status="verified", verified_by="user")
        r = resolve_state(confirmed=c, draft=None, source_sha256="b" * 64)
        assert r.source == SOURCE_STALE and r.annotation is None

    def test_remark_does_not_change_fingerprint(self):
        from negative_lab.calibration.state import content_fingerprint
        a = self._ann(status="verified", verified_by="user")
        b = self._ann(status="verified", verified_by="user", notes="改了备注")
        assert content_fingerprint(a) == content_fingerprint(b)

    def test_geometry_change_does_change_fingerprint(self):
        from negative_lab.calibration.state import content_fingerprint
        a = self._ann(status="verified", verified_by="user")
        b = self._ann(status="verified", verified_by="user",
                      image_area_roi=(11, 10, 90, 90))
        assert content_fingerprint(a) != content_fingerprint(b)

    def test_non_user_verified_is_not_a_signature(self):
        from negative_lab.calibration.state import is_confirmed
        assert not is_confirmed(self._ann(status="verified",
                                          verified_by="agent-inspected"))
        assert is_confirmed(self._ann(status="verified", verified_by="user"))


class TestF1ColdStartUsesPersistentState:
    """F1：有工作草稿时，算法必须用草稿几何，而不是重新自动检测。"""

    def test_draft_geometry_reaches_the_job_seed(self, monkeypatch, tmp_path):
        from negative_lab.calibration.annotations import Annotation, ann_path_for
        from negative_lab.library.models import FrameRecord
        from negative_lab.ui import main_window as mw

        rec = FrameRecord(source_path="rolls/roll/a.tif",
                          source_sha256="a" * 64)
        assert not rec.geometry.is_complete()
        draft = Annotation(
            file=rec.source_path, roll="roll", status="draft",
            image_area_roi=(10, 10, 90, 90),
            base_sample_rois=[(0, 0, 5, 5)],
            output_crop_roi=(5, 5, 95, 95), source_sha256="a" * 64,
            draft_revision=2)
        d = tmp_path / "drafts"
        d.mkdir()
        monkeypatch.setattr(mw, "_DRAFTS_DIR", d)
        (d / ann_path_for(rec.source_path).name).write_text(
            json.dumps(draft.to_dict()), encoding="utf-8")
        got = mw._annotation_for(rec)
        assert got is not None and got.draft_revision == 2
        geo = mw._geometry_from_annotation(got)
        assert geo.is_complete(), "草稿几何必须完整可用作种子"

    def test_confirmed_wins_over_provenance_less_draft_end_to_end(
            self, monkeypatch, tmp_path):
        from negative_lab.calibration import annotations as annmod
        from negative_lab.calibration.annotations import Annotation
        from negative_lab.library.models import FrameRecord
        from negative_lab.ui import main_window as mw

        rec = FrameRecord(source_path="rolls/roll/b.tif",
                          source_sha256="a" * 64)
        confirmed = Annotation(
            file=rec.source_path, roll="roll", status="verified",
            verified_by="user", verified_date="2026-09-01",
            image_area_roi=(10, 10, 90, 90), base_sample_rois=[(0, 0, 5, 5)],
            output_crop_roi=(5, 5, 95, 95), source_sha256="a" * 64)
        a_dir = tmp_path / "ann"
        a_dir.mkdir()
        monkeypatch.setattr(annmod, "ANNOTATIONS_DIR", a_dir)
        (a_dir / "roll__b.json").write_text(
            json.dumps(confirmed.to_dict()), encoding="utf-8")
        d_dir = tmp_path / "drafts"
        d_dir.mkdir()
        monkeypatch.setattr(mw, "_DRAFTS_DIR", d_dir)
        older = Annotation(
            file=rec.source_path, roll="roll", status="draft",
            image_area_roi=(40, 40, 60, 60), base_sample_rois=[(0, 0, 5, 5)],
            output_crop_roi=(40, 40, 60, 60), source_sha256="a" * 64)
        (d_dir / "roll__b.json").write_text(
            json.dumps(older.to_dict()), encoding="utf-8")

        state = mw.resolve_persistent_state(rec)
        assert state.source == "conflict"
        assert state.annotation.image_area_roi == (10, 10, 90, 90)
        assert state.draft.image_area_roi == (40, 40, 60, 60)


class TestExportAccounting:
    def test_cancelled_frames_are_counted(self, tmp_path):
        from negative_lab.library.models import FrameRecord
        from negative_lab.pipeline.roll_export import export_roll_frames

        frames = [FrameRecord(source_path=f"r/a{i}.tif", source_sha256="a" * 64)
                  for i in range(3)]
        ev = threading.Event()
        ev.set()
        rep = export_roll_frames(frames, tmp_path, cancel=ev)
        assert rep["cancelled"] == 3
        assert sum(k in rep and rep[k] for k in
                   ("ok", "needs_review", "failed", "cancelled")) == 3
        assert len(rep["items"]) == 3      # 面板要能跳到具体帧

    def test_gui_smoke_assertion_target_exists(self):
        """gui_smoke 曾断言 rep["items"]，而返回体里没有这个键（审计 F7）。"""
        from negative_lab.pipeline.roll_export import export_roll_frames
        import inspect

        src = inspect.getsource(export_roll_frames)
        assert 'result["items"]' in src


class TestStorageResolution:
    def test_precedence_explicit_beats_env_beats_default(self, monkeypatch,
                                                         tmp_path):
        from negative_lab import storage

        monkeypatch.setenv(storage.ENV_DRAFTS, str(tmp_path / "env"))
        assert storage.drafts_dir() == tmp_path / "env"
        explicit = tmp_path / "explicit"
        got = storage.drafts_dir(module_global=explicit)
        assert got == explicit, "显式改道（测试 monkeypatch）必须赢过环境变量"
        monkeypatch.delenv(storage.ENV_DRAFTS)
        assert storage.drafts_dir() == storage.DEFAULTS["drafts"]

    def test_isolated_storage_redirects_and_restores(self):
        from negative_lab import storage

        before = os.environ.get(storage.ENV_DRAFTS)
        with storage.isolated_storage() as st:
            assert storage.drafts_dir() == st.drafts
            assert storage.annotations_dir() == st.annotations
        assert os.environ.get(storage.ENV_DRAFTS) == before
