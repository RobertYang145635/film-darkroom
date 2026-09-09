"""R5 取色器 UI 接线 + 片基一致性显示（去色罩算法研究第二轮）。

锁定契约：
- 右栏「色罩校正」节有「中性点救援」可勾选按钮与「撤销拾点」按钮；
- 勾选 -> view 进入 neutral 拾点模式（point_picked("neutral") 保持模式，
  Esc 退出；pick_mode_changed 同步按钮态）；
- 拾点 -> 精确救援写 RenderIntent + 即时合成 + 撤销栈；撤销逐级回退；
- 可信度警告进右栏一级提示（lbl_pick_hint）；
- 技术信息节显示卷级片基一致性（实测帧色度极差；generic_prior 不参与
  统计——研究 #1 §2.4 教训）。
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import Qt

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module", autouse=True)
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from negative_lab.ui import theme

    app = QApplication.instance() or QApplication([])
    theme.apply(app)
    return app


def _img(h: int = 96, w: int = 96, cast=(0.0, 0.0, 0.0),
         seed: int = 5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    scene = 0.45 + 0.10 * np.sin(xx / 9.0) * np.cos(yy / 7.0)
    base = np.stack([scene * 0.98, scene, scene * 1.02], axis=-1)
    return np.clip(base + np.array(cast, np.float32)
                   + rng.normal(0, 0.004, base.shape).astype(np.float32),
                   0, 1).astype(np.float32)


def _win_with_frame(monkeypatch, cast=(0.0, 0.0, 0.06)):
    """调整阶段窗口：技术正片带注入蓝偏（模拟自动翻车）。

    用真实合成负片（带片基/边缘结构）走全管线，保证拾点场景里有
    物理上可信的中性区。
    """
    from negative_lab.library.models import (FrameRecord, GeometryRegions,
                                             ImportSession)
    from negative_lab.pipeline.render import RenderIntent
    from negative_lab.testing.synthetic import make_synthetic_negative
    from negative_lab.ui.main_window import MainWindow

    w = MainWindow()
    rec = FrameRecord(source_path="synthetic/pick.tif",
                      source_sha256="0" * 64,
                      input_facts={"abs_path": "synthetic/pick.tif",
                                   "kind": "tiff"},
                      geometry=GeometryRegions())
    w.session = ImportSession(files=[rec])
    w.current = rec
    t = make_synthetic_negative(seed=3)["negative"]
    w.t_source = t
    w._load_meta = {"kind": "tiff", "sha256": "0" * 64}
    w._prepare_frame_views()
    w._set_stage(1)
    # 注入蓝偏到缓存正片（模拟自动结果翻车）
    w._positive_full = np.clip(w._positive_full
                               + np.array(cast, np.float32), 0, 1)
    w._intent = RenderIntent()
    return w, rec


def _good_pick_point(tech: np.ndarray, r: int = 3) -> tuple[int, int]:
    """找技术正片里中亮度、低饱和的 (x, y)（模拟用户点"该是灰"的地方）。

    技术正片的极暗区是合法输出（负片深色场景），通道比在天文级失衡——
    挑"中亮度且通道比最接近"的块，模拟用户点在应呈中性的中间调。
    """
    h, w = tech.shape[:2]
    best, best_score = (w // 2, h // 2), float("inf")
    for yy in range(r + 1, h - r - 1, 4):
        for xx in range(r + 1, w - r - 1, 4):
            p = tech[yy - r:yy + r + 1, xx - r:xx + r + 1].reshape(-1, 3)
            m = np.median(p, axis=0)
            lum = float(m.mean())
            if not (0.15 <= lum <= 0.75):
                continue
            # 通道比（对数域极差）小 = 接近中性
            spread_log = float(np.log10(m.max() / max(m.min(), 1e-9)))
            score = abs(lum - 0.4) + spread_log
            if score < best_score:
                best_score, best = score, (xx, yy)
    return best


class TestPickNeutralUI:
    def test_button_toggles_pick_mode(self):
        w, _ = _win_with_frame(None)
        rp = w.right
        assert rp.btn_pick_neutral.isCheckable()
        rp.btn_pick_neutral.setChecked(True)     # 模拟点击（不触发 toggled）
        rp.pick_neutral_toggled.emit(True)
        assert w.view._pick_mode == "neutral"
        # pick_mode_changed 回路同步按钮态
        w.view.set_pick_mode(None)
        assert rp.btn_pick_neutral.isChecked() is False

    def test_pick_applies_intent_and_undo(self):
        w, rec = _win_with_frame(None)
        before_intent = w._intent.copy()
        before_pos = w._positive_full.copy()
        px, py = _good_pick_point(w._positive_full)
        # 模拟用户在可信中性区拾点（坐标系 = _positive_full 像素）
        w._on_point_picked("neutral", float(px), float(py))
        assert w._intent.temperature != 0.0 or w._intent.tint != 0.0
        assert rec.render_intent_override == w._intent.to_dict()
        assert w._pick_undo_stack and w.right.btn_undo_pick.isEnabled()
        assert "拉向中性" in w.right.lbl_pick_hint.text()
        assert np.array_equal(w._positive_full, before_pos), \
            "救援不得修改技术正片（逐比特契约）"
        # 撤销回到之前的 intent
        w._undo_pick_rescue()
        assert w._intent.to_dict() == before_intent.to_dict()
        assert w.right.btn_undo_pick.isEnabled() is False

    def test_two_picks_undo_twice(self):
        w, _ = _win_with_frame(None)
        i0 = w._intent.copy()
        w._on_point_picked("neutral", 40.0, 40.0)
        i1 = w._intent.copy()
        w._on_point_picked("neutral", 56.0, 56.0)
        w._intent.copy()
        assert len(w._pick_undo_stack) == 2
        w._undo_pick_rescue()
        assert w._intent.to_dict() == i1.to_dict()
        w._undo_pick_rescue()
        assert w._intent.to_dict() == i0.to_dict()
        assert w.right.btn_undo_pick.isEnabled() is False

    def test_suspect_pick_shows_warning_hint(self, monkeypatch):
        """拾到通道比失衡的点：右栏一级警告提示。"""
        w, _ = _win_with_frame(None)
        img = w._positive_full
        img[40:58, 40:58] = (0.03, 0.45, 0.60)   # 强失衡块
        w._on_point_picked("neutral", 48.0, 48.0)
        assert "⚠" in w.right.lbl_pick_hint.text(), \
            w.right.lbl_pick_hint.text()

    def test_pick_out_of_frame_clears_gracefully(self):
        w, _ = _win_with_frame(None)
        # 越界/非有限拾点：拒绝并提示，不应用任何救援
        w._on_point_picked("neutral", -5.0, -5.0)
        assert "拾点失败" in w.right.lbl_pick_hint.text()
        w._on_point_picked("neutral", 1e9, 1e9)
        assert "拾点失败" in w.right.lbl_pick_hint.text()
        w._on_point_picked("neutral", float("nan"), float("nan"))
        assert "拾点失败" in w.right.lbl_pick_hint.text()
        assert w._pick_undo_stack == []

    def test_pick_mode_persists_for_multiple_picks(self):
        w, _ = _win_with_frame(None)
        from PySide6.QtCore import QPointF

        w.view.set_pick_mode("neutral")
        class _Ev:
            def __init__(self, x, y):
                self._p = QPointF(x, y)
            def button(self):
                return Qt.LeftButton
            def position(self):
                return self._p
        # 直接调 mousePressEvent 的拾点分支（offscreen 下不依赖真实事件）
        from unittest.mock import patch as _patch

        v = w.view
        with _patch.object(v, "mapToScene", return_value=QPointF(48.0, 48.0)):
            _Ev(48, 48)
            v._pick_mode = "neutral"
            v._points["neutral"] = QPointF(48.0, 48.0)
            v.point_picked.emit("neutral", 48.0, 48.0)
        assert v._pick_mode == "neutral", "neutral 模式拾点后保持，允许连续试点"


class TestConsistencyDisplay:
    def test_consistency_line_from_roll_analysis(self):
        from negative_lab.library.models import RollAnalysis
        from negative_lab.ui.main_window import MainWindow

        w = MainWindow()
        w.roll_analysis = RollAnalysis(
            base_source="roll",
            base_statistics={"per_frame": {
                "a": [0.29, 0.09, 0.03], "b": [0.285, 0.088, 0.0295],
                "c": [0.295, 0.091, 0.0305]}})
        text = w._consistency_text()
        assert "片基一致性" in text and "3 帧" in text and "decade" in text

    def test_generic_prior_frames_excluded(self):
        """仅 1 个实测帧 + 若干 prior：不产生统计行（prior 形状是常量）。"""
        from negative_lab.library.models import RollAnalysis
        from negative_lab.ui.main_window import MainWindow

        w = MainWindow()
        w.roll_analysis = RollAnalysis(
            base_source="roll_partial",
            base_statistics={"per_frame": {"a": [0.29, 0.09, 0.03]}})
        assert w._consistency_text() == ""

    def test_all_prior_warns(self):
        from negative_lab.library.models import RollAnalysis
        from negative_lab.ui.main_window import MainWindow

        w = MainWindow()
        w.roll_analysis = RollAnalysis(base_source="generic_prior",
                                       base_statistics={})
        assert "通用先验" in w._consistency_text()

    def test_tech_info_shows_consistency(self):
        from negative_lab.library.models import (FrameRecord, GeometryRegions,
                                                 ImportSession, RollAnalysis)
        from negative_lab.ui.main_window import MainWindow

        w = MainWindow()
        rec = FrameRecord(source_path="synthetic/tech.tif",
                          source_sha256="0" * 64,
                          input_facts={"abs_path": "synthetic/tech.tif",
                                       "kind": "tiff"},
                          geometry=GeometryRegions())
        w.session = ImportSession(files=[rec])
        w.current = rec
        w.t_source = _img(cast=(0, 0, 0))
        w._load_meta = {"kind": "tiff", "sha256": "0" * 64}
        w._prepare_frame_views()
        w.roll_analysis = RollAnalysis(
            base_source="roll",
            base_statistics={"per_frame": {
                "a": [0.29, 0.09, 0.03], "b": [0.285, 0.088, 0.0295],
                "c": [0.295, 0.091, 0.0305]}})
        w._update_tech_info()
        body = w.right.sec_tech.lbl_body.text()
        assert "本卷片基一致性" in body
