"""C8 实测（2026-09-08 交接「已知未验证」第一条）。

待验证前提：「按住看原片」按钮被按住时，Qt 在画布点击的时刻
`btn.isDown()` 仍为真——若为假，`_on_point_picked` 里
`self._orig_held or self.btn_hold_orig.isDown()` 的第二子句是死代码；
若为真，守卫前提成立。两种结果都记录，不做中途结论。

同时锁定守卫行为本身：
- 按住按钮（_orig_held=True）时拾点被拒绝并提示，不改 intent；
- 松开后同一坐标拾点正常应用救援。
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from PySide6.QtCore import Qt

ROOT = None  # 保持与其他 UI 测试一致的结构；无需额外路径处理


@pytest.fixture(scope="module", autouse=True)
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from negative_lab.ui import theme

    app = QApplication.instance() or QApplication([])
    theme.apply(app)
    return app


def _win_with_frame():
    """与 test_v6_pick_rescue_ui 同构的最小调整阶段窗口。"""
    from negative_lab.library.models import (FrameRecord, GeometryRegions,
                                             ImportSession)
    from negative_lab.pipeline.render import RenderIntent
    from negative_lab.testing.synthetic import make_synthetic_negative
    from negative_lab.ui.main_window import MainWindow

    w = MainWindow()
    rec = FrameRecord(source_path="synthetic/c8.tif",
                      source_sha256="0" * 64,
                      input_facts={"abs_path": "synthetic/c8.tif",
                                   "kind": "tiff"},
                      geometry=GeometryRegions())
    w.session = ImportSession(files=[rec])
    w.current = rec
    t = make_synthetic_negative(seed=3)["negative"]
    w.t_source = t
    w._load_meta = {"kind": "tiff", "sha256": "0" * 64}
    w._prepare_frame_views()
    w._set_stage(1)
    w._positive_full = np.clip(w._positive_full
                               + np.array((0.0, 0.0, 0.06), np.float32), 0, 1)
    w._intent = RenderIntent()
    return w, rec


def _good_pick_point(tech: np.ndarray, r: int = 3) -> tuple[int, int]:
    h, w = tech.shape[:2]
    best, best_score = (w // 2, h // 2), float("inf")
    for yy in range(r + 1, h - r - 1, 4):
        for xx in range(r + 1, w - r - 1, 4):
            p = tech[yy - r:yy + r + 1, xx - r:xx + r + 1].reshape(-1, 3)
            m = np.median(p, axis=0)
            lum = float(m.mean())
            if not (0.15 <= lum <= 0.75):
                continue
            spread_log = float(np.log10(m.max() / max(m.min(), 1e-9)))
            score = abs(lum - 0.4) + spread_log
            if score < best_score:
                best_score, best = score, (xx, yy)
    return best


class TestHoldOrigGuardPremise:
    def test_isdown_true_while_button_held(self, qapp):
        """核心前提实测：按住（press 不 release）期间 isDown() 为真。"""
        from PySide6.QtTest import QTest

        w, _ = _win_with_frame()
        btn = w.btn_hold_orig
        assert not btn.isDown()
        QTest.mousePress(btn, Qt.LeftButton)
        try:
            held_down = btn.isDown()
            # pressed 信号必须已把状态置位（守卫第一子句的事实来源）
            assert w._orig_held is True, "按住必须同步置 _orig_held"
            assert held_down is True, (
                "前提实测：按住期间 isDown() 应为真——"
                "守卫第二子句 isDown() 有效")
        finally:
            QTest.mouseRelease(btn, Qt.LeftButton)
        assert btn.isDown() is False
        assert w._orig_held is False

    def test_pick_rejected_while_held_then_works_after_release(self):
        w, rec = _win_with_frame()
        px, py = _good_pick_point(w._positive_full)
        # 模拟按住状态（与 _hold_original(True) 后的状态一致）
        w._orig_held = True
        before = w._intent.copy()
        w._on_point_picked("neutral", float(px), float(py))
        assert "松开" in w.right.lbl_pick_hint.text(), \
            w.right.lbl_pick_hint.text()
        assert w._intent.to_dict() == before.to_dict(), "按住时拾点不得生效"
        assert rec.render_intent_override is None
        # 松开后同一坐标正常救援
        w._orig_held = False
        w._on_point_picked("neutral", float(px), float(py))
        assert w._intent.temperature != 0.0 or w._intent.tint != 0.0
        assert rec.render_intent_override == w._intent.to_dict()
