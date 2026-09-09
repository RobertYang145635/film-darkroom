"""2026-09-08 审查修复的回归锁。

来源：三份审查（0907 自查 / 0908 外部静态扫描 / 本轮运行时复核）合并后的
逐条实测复现。只钉「已经在脚本里跑出过错误结果」的项，避免把猜测写成契约。
"""

from __future__ import annotations

import json
import threading

import numpy as np
import pytest


class TestPickPatchEdgeClamp:
    """rescue._patch_median 此前写 arr[y-r:y+r+1]：负索引对 numpy 是「从尾部
    计数」——大图边缘合法坐标直接抛 ValueError，小图更糟：静默取到图像
    **另一侧**的像素。正边界 numpy 自动钳、负边界不钳，这个不对称是根因。"""

    def _img(self):
        return np.random.RandomState(0).rand(40, 60, 3).astype(np.float32)

    @pytest.mark.parametrize("y,x", [(0, 30), (1, 30), (2, 30), (20, 0),
                                      (20, 1), (20, 2), (39, 30), (20, 59),
                                      (0, 0), (39, 59)])
    def test_legal_picks_near_any_edge_work(self, y, x):
        from negative_lab.pipeline.rescue import _patch_median
        v = _patch_median(self._img(), y, x, 3)
        assert v.shape == (3,) and np.all(np.isfinite(v))

    @pytest.mark.parametrize("y,x", [(-1, 0), (40, 0), (0, 60), (0, -5)])
    def test_true_out_of_range_still_rejected(self, y, x):
        from negative_lab.pipeline.rescue import _patch_median
        with pytest.raises(ValueError):
            _patch_median(self._img(), y, x, 3)

    def test_small_image_never_reads_the_far_side(self):
        from negative_lab.pipeline.rescue import _patch_median
        img = np.zeros((4, 60, 3), np.float32)
        img[0], img[1], img[2], img[3] = [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]
        # 修复前：y=1,r=3 -> arr[-2:5] 抓到第 2/3 行，返回 [0.5,0.5,1.0]
        v = _patch_median(img, 1, 30, 3)
        assert np.allclose(v, [0.5, 0.5, 0.5]), f"补丁取到了错误区域：{v}"


class TestFingerprintIsContentSensitive:
    """§5.2 的命门：base_annotation_sha 必须只对**内容**敏感。
    原实现含 status（且 from_dict 会就地改写它），并把 numpy 整数经
    default=str 写成 "100" 而 python int 写成 100 —— 同一份几何实测
    可算出三个不同指纹，一旦盖戳侧与比对侧表示不同，草稿永久判 conflict。"""

    def _ann(self, roi):
        from negative_lab.calibration.annotations import Annotation
        return Annotation(file="r/f.tif", roll="r", rotation_degrees=0,
                          coordinate_space="oriented", status="verified",
                          verified=True, verified_by="user",
                          output_crop_roi=roi, image_area_roi=roi,
                          base_sample_rois=[roi], source_sha256="a" * 64)

    def test_numeric_representation_does_not_change_fingerprint(self):
        from negative_lab.calibration.state import content_fingerprint
        fps = {content_fingerprint(self._ann(r)) for r in (
            (100.0, 200.0, 300.0, 400.0),
            (100, 200, 300, 400),
            tuple(np.int64(v) for v in (100, 200, 300, 400)),
            tuple(np.float64(v) for v in (100.0, 200.0, 300.0, 400.0)),
            [100, 200.0, np.int64(300), 400])}
        assert len(fps) == 1, f"同一内容算出 {len(fps)} 个指纹：{fps}"

    def test_saved_then_reloaded_fingerprint_is_stable(self):
        """签字时的内存对象 vs 下次启动读回的对象，必须同一个值。"""
        from negative_lab.calibration.annotations import Annotation
        from negative_lab.calibration.state import content_fingerprint
        a = self._ann((100, 200, 300, 400))
        b = Annotation.from_dict(json.loads(json.dumps(a.to_dict())))
        assert content_fingerprint(a) == content_fingerprint(b)

    def test_to_dict_is_json_safe_for_numpy_scalars(self):
        a = self._ann(tuple(np.int64(v) for v in (1, 2, 3, 4)))
        json.dumps(a.to_dict())          # 修复前：TypeError int64 not serializable

    def test_real_geometry_change_still_changes_fingerprint(self):
        from negative_lab.calibration.state import content_fingerprint
        assert content_fingerprint(
            self._ann((100, 200, 300, 400))) != content_fingerprint(
            self._ann((100, 200, 300, 401)))


class TestCorruptStateIsVisible:
    """损坏的标注/草稿此前被吞成 None，与「用户从没标定过」不可区分 ->
    界面回落自动检测 -> 用户再签字 -> 覆盖掉可救回的旧基线。"""

    def test_truncated_draft_is_reported_not_treated_as_absent(self, tmp_path):
        from negative_lab import storage
        from negative_lab.library.models import FrameRecord
        from negative_lab.ui import main_window as mw
        with storage.isolated_storage(tmp_path):
            rec = FrameRecord(source_path="r/a.tif", source_sha256="a" * 64)
            name = mw.ann_path_for(rec.source_path).name
            (storage.drafts_dir() / name).write_text(
                '{"file": "r/a.tif", "roll"', encoding="utf-8")   # 截断 JSON
            st = mw.resolve_persistent_state(rec)
            assert "读取失败" in st.reason, f"损坏被静默吞掉：{st.reason!r}"
            assert st.needs_reconfirm
            # 干净目录不得留下同样的痕迹（诊断不能跨帧/跨会话残留）
            st2 = mw.resolve_persistent_state(
                FrameRecord(source_path="r/clean.tif", source_sha256="b" * 64))
            assert "读取失败" not in st2.reason


class TestExportResultContract:
    """v6.2 §6.6 的取消计数必须能被**调用方**读到：此前 summary_complete 只
    写进落盘报告、返回体里没有，UI 想如实提示也拿不到。"""

    def test_cancelled_and_completeness_are_returned(self, tmp_path):
        from negative_lab.library.models import FrameRecord
        from negative_lab.pipeline.roll_export import export_roll_frames
        frames = [FrameRecord(source_path=f"r/a{i}.tif", source_sha256="a" * 64)
                  for i in range(3)]
        ev = threading.Event()
        ev.set()
        rep = export_roll_frames(frames, tmp_path, cancel=ev)
        assert rep["cancelled"] == 3
        assert rep["summary_complete"] is True
        assert sum(rep[k] for k in
                   ("ok", "needs_review", "failed", "cancelled")) == 3
        # 落盘报告与返回体必须一致
        doc = json.loads((tmp_path / "roll-report.json").read_text("utf-8"))
        assert doc["summary"]["cancelled"] == 3


class TestRescueReportsNoConvergence:
    """二自由度控件族本来就救不平强色偏点；此前求解顶到边界、残差 68%
    仍然**零警告**，UI 显示「已把该点拉向中性灰」+ 滑杆拉满。"""

    def test_saturated_pick_warns_instead_of_claiming_success(self):
        from negative_lab.pipeline.render import RenderIntent
        from negative_lab.pipeline.rescue import neutral_pick_intent_exact
        p = np.zeros((40, 40, 3), np.float32)
        p[:] = [0.90, 0.30, 0.20]
        _it, warns = neutral_pick_intent_exact(
            p, 20, 20, base_intent=RenderIntent(), return_warnings=True)
        assert any(("未能拉平" in w) or ("边界" in w) for w in warns), warns

    def test_genuinely_fixable_pick_stays_silent(self):
        """反向护栏：正常可救的点不得被新警告骚扰。"""
        from negative_lab.pipeline.render import RenderIntent
        from negative_lab.pipeline.rescue import neutral_pick_intent_exact
        rng = np.random.RandomState(5)
        p = np.clip(rng.rand(40, 40, 3) * 0.02 + np.array([0.30, 0.42, 0.62]),
                    0, 1).astype(np.float32)
        _it, warns = neutral_pick_intent_exact(
            p, 20, 20, base_intent=RenderIntent(), return_warnings=True)
        assert warns == [], f"可正常收敛的拾点被误报警：{warns}"


class TestConsistencyOutputIsValidJson:
    """库层此前 median_ab/outlier_score 落裸 NaN（de_matrix 却用 None），
    实测 json.dumps(allow_nan=False) 直接抛 —— 缺失值必须一套约定。"""

    def test_missing_frames_yield_none_not_nan(self):
        from negative_lab.diagnostics.consistency import roll_consistency
        rng = np.random.RandomState(1)
        frames = [rng.rand(64, 64, 3).astype(np.float32) for _ in range(2)]
        frames.append(rng.rand(6, 6, 3).astype(np.float32))   # 有效像素不足
        out = roll_consistency(frames)
        assert any(v is None for v in out["outlier_score"]), out["outlier_score"]
        json.dumps(out, allow_nan=False)     # 不得抛

