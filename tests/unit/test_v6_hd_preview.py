"""P0-B 回归（v6.1 §3）：高清预览 API、worker 解包与竞态。

历史缺陷：
1. wheelEvent()/current_zoom() 调用不存在的 QGraphicsView.m11()
   -> AttributeError；
2. _hd_tile_done() 把 {"ok": True, "value": ndarray} 当数组传 set_image()；
3. 旧任务结果可在帧切换/缩回后覆盖新状态；
4. 调整阶段曾把原始负片数组当正片上屏。
锁定契约：ENABLE_HD_PREVIEW 默认 False；缩放信号统一；worker 契约
{"ok","value"}；失败不替换当前图；过期任务丢弃。
"""

from __future__ import annotations

import os
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


@pytest.fixture()
def view():
    from negative_lab.ui.preview_view import PreviewView

    v = PreviewView()
    v.resize(900, 700)
    v.set_image(np.zeros((400, 600, 3), np.float32))
    return v


def test_current_zoom_no_attribute_error(view):
    z = view.current_zoom()      # 曾抛 AttributeError: m11
    assert isinstance(z, float) and z > 0


def test_wheel_zoom_emits_signal_and_updates(view):
    got = []
    view.zoom_changed.connect(got.append)
    from PySide6.QtCore import QPoint, QPointF, Qt
    from PySide6.QtGui import QWheelEvent

    ev = QWheelEvent(QPointF(300, 300), QPointF(300, 300),
                     QPoint(0, 0), QPoint(0, 120),
                     Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
                     Qt.ScrollUpdate, False)
    view.wheelEvent(ev)
    assert got and got[-1] > 1.0, "滚轮放大必须发出 zoom_changed 且 >1"
    assert abs(view.current_zoom() - got[-1]) < 1e-6
    # fit()/zoom_100() 同一信号源
    view.fit()
    assert got[-1] == pytest.approx(view.current_zoom())
    view.zoom_100()
    assert got[-1] == pytest.approx(1.0)


def test_hd_disabled_by_default():
    import negative_lab.ui.main_window as mw

    assert mw.ENABLE_HD_PREVIEW is False, \
        "同场景高清 tile 实现前必须保持关闭（v6.1 §3）"


def _win_with_frame():
    from negative_lab.library.models import (FrameRecord, GeometryRegions,
                                             ImportSession)
    from negative_lab.testing.synthetic import make_synthetic_negative
    from negative_lab.ui.main_window import MainWindow

    t = make_synthetic_negative(seed=3)["negative"]
    rec = FrameRecord(source_path="synthetic/hd.tif",
                      source_sha256="0" * 64,
                      input_facts={"abs_path": "synthetic/hd.tif",
                                   "kind": "tiff"},
                      geometry=GeometryRegions())
    w = MainWindow()
    w.session = ImportSession(files=[rec])
    w.current = rec
    w.t_source = t
    w._load_meta = {"kind": "tiff", "sha256": "0" * 64}
    w._prepare_frame_views()
    return w, rec


def test_hd_done_failure_keeps_small_image_and_shows_error():
    w, rec = _win_with_frame()
    w._hd_generation = 1
    before = w.view._pix_item
    w._hd_tile_done({"ok": False, "error": "RuntimeError: 解码失败"},
                    rec, 2.0, generation=1)
    assert w.view._pix_item is before, "失败结果不得替换当前图"
    assert "高清预览失败" in w.lbl_banner.text()


def test_hd_done_success_unwraps_value_and_checks_generation():
    w, rec = _win_with_frame()
    w._hd_generation = 5
    good = np.full((200, 300, 3), 0.5, np.float32)
    # 旧一代任务：必须丢弃
    w._hd_tile_done({"ok": True, "value": good}, rec, 2.0, generation=4)
    # 新一代任务：只传 result["value"]
    w._hd_tile_done({"ok": True, "value": good}, rec, 2.0, generation=5)
    arr = w.view._pix_item.pixmap().toImage()
    assert arr.width() == 300 and arr.height() == 200, \
        "set_image 收到的必须是解包后的数组（尺寸=tile）"
    # 帧已切换：丢弃
    w2, rec2 = _win_with_frame()
    w2._hd_generation = 1
    w2.current = object()      # 模拟切到别的帧
    w2._hd_tile_done({"ok": True, "value": good}, rec, 2.0, generation=1)
    assert w2.view._pix_item.pixmap().toImage().height() != 200


def test_adjust_stage_never_shows_negative_as_positive(monkeypatch):
    """调整阶段 HD 路径：worker 必须经 process_image 取正片，不得直接
    上屏 tr.apply(负片)。用 process_image 桩验证调用契约。"""
    import negative_lab.ui.main_window as mw

    w, rec = _win_with_frame()
    w._set_stage(1)
    # 让缓存正片尺寸与全分辨率不一致：强制走 process_image 分支
    w._positive_full = np.full((7, 9, 3), 0.8, np.float32)
    calls = {}

    class FakePR:
        class result:
            positive_linear = np.full((60, 80, 3), 0.8, np.float32)

    def fake_process_image(rgb, **kw):
        calls["rgb_shape"] = rgb.shape
        calls["keep_rebate"] = kw.get("keep_rebate")
        calls["geometry"] = kw.get("geometry")
        return FakePR()

    monkeypatch.setattr(mw, "process_image", fake_process_image)
    monkeypatch.setattr(w, "_load", lambda r: type(
        "L", (), {"rgb": w.t_source})())
    from negative_lab.pipeline.geometry import OrientTransform

    tr = OrientTransform(w.t_source.shape[:2], 0, False, 0.0)
    out = w._hd_tile_job(rec, tr)
    assert calls.get("keep_rebate") is True
    assert calls.get("geometry") is rec.geometry
    assert out.max() > 0.5, "调整阶段 HD 帧必须是正片结果（高亮），不是负片"
