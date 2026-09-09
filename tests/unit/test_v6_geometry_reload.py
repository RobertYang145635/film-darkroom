"""P0-A 回归（v6.1 §2）：重新载入已有完整 ROI 的帧不再崩溃。

历史缺陷：把 (H,W,3) RGB 数组直接传给只接受 (H,W) 的
GeometryRegions.normalized() -> ValueError: too many values to unpack；
以及 `if ann_source` 对 ndarray 触发 ambiguous truth value。
锁定契约：任何 source 入口（ndarray 或尺寸元组）都按 shape[:2] 处理。
"""

from __future__ import annotations

import os
import time
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtWidgets import QApplication

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module", autouse=True)
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from negative_lab.ui import theme

    app = QApplication.instance() or QApplication([])
    theme.apply(app)
    return app


def _complete_geo(h: int, w: int):   # 返回 GeometryRegions（函数内导入，避免顶层依赖 Qt）
    from negative_lab.library.models import GeometryRegions

    return GeometryRegions(
        output_crop_roi=(20, 30, h - 40, w - 50),
        image_area_roi=(24, 34, h - 44, w - 54),
        base_sample_rois=[(4, 4, 14, w - 4)])


def test_rgb_source_with_complete_geometry_no_unpack_error():
    """完整三类 ROI + (H,W,3) source：不抛 ValueError，ROI 落在 small 内。"""
    from negative_lab.library.models import GeometryRegions
    from negative_lab.ui.main_window import _geometry_for_job

    src = np.zeros((800, 1200, 3), np.float32)
    small = src[::4, ::4, :].copy()
    geo = _complete_geo(*src.shape[:2])
    out = _geometry_for_job(None, small, src, geo)
    assert isinstance(out, GeometryRegions)
    sh, sw = small.shape[:2]
    for name in ("output_crop_roi", "image_area_roi"):
        r = getattr(out, name)
        assert r is not None
        assert 0 <= r[0] < r[2] <= sh and 0 <= r[1] < r[3] <= sw
    assert out.base_sample_rois, "三类 ROI 必须保留"
    for r in out.base_sample_rois:
        assert 0 <= r[0] < r[2] <= sh and 0 <= r[1] < r[3] <= sw


def test_rgb_ann_source_no_truthiness_or_unpack_error():
    """重新载入路径：ann_source 为 (H,W,3) 时不得触发 ndarray 真值判断
    或尺寸解包崩溃；(H,W) 尺寸元组路径保持可用。"""
    from negative_lab.library.models import GeometryRegions
    from negative_lab.ui.main_window import _geometry_for_job

    src = np.zeros((600, 900, 3), np.float32)
    small = src[::3, ::3, :].copy()
    geo = _complete_geo(*src.shape[:2])
    out = _geometry_for_job(geo, small, src, None, ann_source=src)
    assert isinstance(out, GeometryRegions)
    sh, sw = small.shape[:2]
    r = out.image_area_roi
    assert r is not None and 0 <= r[0] < r[2] <= sh and 0 <= r[1] < r[3] <= sw
    # (H,W) 尺寸元组依旧可用（旧调用约定）
    out2 = _geometry_for_job(geo, small, src, None, ann_source=src.shape[:2])
    assert out2.image_area_roi is not None


def test_reopen_frame_with_complete_roi_no_value_error(monkeypatch):
    """构造完整 FrameRecord，走「重新打开已有完整 ROI」路径（_load_frame_job）：
    断言无 ValueError，三类 ROI 数量与 source 坐标语义不变。"""
    from negative_lab.library.models import (FrameRecord, ImportSession)
    from negative_lab.testing.synthetic import make_synthetic_negative
    from negative_lab.ui.main_window import MainWindow

    t = make_synthetic_negative(seed=11)["negative"]
    h, w = t.shape[:2]
    rec = FrameRecord(source_path="synthetic/reload.tif",
                      source_sha256="0" * 64,
                      input_facts={"abs_path": "synthetic/reload.tif",
                                   "kind": "tiff", "width": w, "height": h},
                      geometry=_complete_geo(h, w))
    assert rec.geometry.is_complete()
    geo_before = rec.geometry.to_dict()

    win = MainWindow()
    win.session = ImportSession(files=[rec])
    win.current = rec
    monkeypatch.setattr(MainWindow, "_load",
                        lambda self, r: SimpleNamespace(rgb=t, meta={
                            "kind": "tiff", "sha256": "0" * 64,
                            "width": w, "height": h}))
    d = win._load_frame_job(rec, lambda msg: None)
    assert d["rec"] is rec
    win._frame_done({"ok": True, "value": d}, rec)
    g = rec.geometry
    assert len(g.base_sample_rois) == 1, "重新载入不得改变 ROI 数量"
    assert g.image_area_roi is not None and g.output_crop_roi is not None
    def norm(v):
        if v is None:
            return None
        if isinstance(v, (list, tuple)) and v and isinstance(v[0], (list, tuple)):
            return [list(x) for x in v]
        return list(v) if isinstance(v, tuple) else v

    for name in ("output_crop_roi", "image_area_roi", "base_sample_rois"):
        assert norm(getattr(g, name)) == norm(geo_before[name]), \
            f"{name} 的 SOURCE 坐标语义不得改变"


def test_base_roi_edit_refresh_has_no_error_banner(monkeypatch):
    """片基 ROI 修改触发 _refresh_preview()：worker 完成后无错误 banner，
    草稿仍存在（等待 worker 并处理 Qt 事件）。"""
    from negative_lab.library.models import (FrameRecord, GeometryRegions,
                                             ImportSession)
    from negative_lab.testing.synthetic import make_synthetic_negative
    from negative_lab.ui.main_window import MainWindow
    from negative_lab.ui.preview_view import RoiKind

    t = make_synthetic_negative(seed=5, black_border=12)["negative"]
    rec = FrameRecord(source_path="synthetic/smoke.tif",
                      source_sha256="0" * 64,
                      input_facts={"abs_path": "synthetic/smoke.tif",
                                   "kind": "tiff"},
                      geometry=GeometryRegions())
    win = MainWindow()
    win.show()
    win.session = ImportSession(files=[rec])
    win.current = rec
    win.t_source = t
    win._load_meta = {"kind": "tiff", "sha256": "0" * 64}
    win._prepare_frame_views()

    view = win.view
    view.set_draw_mode(RoiKind.BASE)
    vp = view.viewport()
    vp.resize(900, 700)
    from PySide6.QtTest import QTest

    QTest.mousePress(vp, Qt.LeftButton, Qt.NoModifier, QPoint(30, 260))
    QTest.mouseMove(vp, QPoint(200, 330))
    QTest.mouseRelease(vp, Qt.LeftButton, Qt.NoModifier, QPoint(200, 330))
    assert rec.geometry.base_sample_rois, "片基草稿必须先落到内存"

    deadline = time.time() + 20
    while time.time() < deadline and threading.active_count() > 1:
        QApplication.processEvents()
        time.sleep(0.02)
    QApplication.processEvents()
    assert not win._banner_text.startswith("预览刷新失败"), \
        f"worker 失败被吞或报错：{win._banner_text!r}"
    assert rec.geometry.base_sample_rois, "刷新后草稿必须仍存在"
