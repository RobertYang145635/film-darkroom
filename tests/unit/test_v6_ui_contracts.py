"""v6 重做后的 UI 契约（用户 2026-09-02 确认的 Lightroom 式单工作区 + 深色）。

锁定：
- 单工作区（无模式页切换）：中央常驻正片，右栏是可折叠节面板；
- 裁切 = 右栏小工具节（默认收起），激活进入照片叠加态，不换页；
- 深色主题 token（#1E1E1E/#252526/#D4D4D4/#4C8DFF）；页面零硬编码色；
- 交互即时性：裁切/微调即时合成（信号接线存在且不走管线）；
- 服务失败不阻断画布编辑与草稿保存。
"""

from __future__ import annotations

import os
import re
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtWidgets import (
    QApplication, QPushButton, QStackedWidget, QToolButton, QWidget)

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
    v.set_image(np.zeros((400, 600, 3), np.float32))
    yield v


def _drag(view, p0, p1, button=Qt.LeftButton):
    from PySide6.QtTest import QTest

    vp = view.viewport()
    QTest.mousePress(vp, button, Qt.NoModifier, QPoint(*p0))
    QTest.mouseMove(vp, QPoint(*p1))
    QTest.mouseRelease(vp, button, Qt.NoModifier, QPoint(*p1))


# ------------------------------------------------------- 单工作区结构
def _win():
    from negative_lab.ui.main_window import MainWindow

    return MainWindow()


class TestSingleWorkspace:
    def test_no_mode_stack(self):
        """v6 重做：不存在"裁切/转换"两个页面（用户指示推翻旧规格）。"""
        w = _win()
        assert not hasattr(w, "right_stack"), \
            "右栏应为节面板，不再是模式页堆栈"
        assert not hasattr(w, "btn_mode_crop") and \
            not hasattr(w, "btn_mode_convert"), "不得再有模式切换按钮"
        assert w.page_count() == 1, "工作区只有一个"

    def test_right_panel_is_section_stack(self):
        w = _win()
        rp = w.right
        for name in ("sec_crop", "sec_basic", "sec_cand", "sec_tech"):
            assert hasattr(rp, name), f"右栏缺 {name} 节"
        # 裁切节默认收起（Lightroom 小工具）
        assert rp.sec_crop.btn_head.isChecked() is False

    def test_crop_tool_is_panel_section_not_page(self):
        """裁切是右栏按钮激活的叠加态；激活/退出不换页、照片不消失。"""
        w = _win()
        img_before = w.view._pix_item
        w.right.sec_crop._on_activate(True)
        assert w.view._crop_dim > 0.3, "激活后框外必须压暗"
        w.right.sec_crop._on_activate(False)
        assert w.view._crop_dim == 0.0, "退出后遮罩必须消失"
        assert w.view._pix_item is img_before, "照片控件必须常驻"

    def test_compose_is_instant_no_pipeline(self, monkeypatch):
        """构图合成不得触发管线（即时性契约）。"""
        import negative_lab.ui.main_window as mw

        w = _win()
        calls = []
        monkeypatch.setattr(mw, "process_image",
                            lambda *a, **k: calls.append(1))
        w._compose_view()
        assert not calls, "_compose_view 必须零管线调用"

    def test_base_change_is_only_slow_path(self):
        from negative_lab.library.models import (FrameRecord, GeometryRegions,
                                                 ImportSession)
        from negative_lab.testing.synthetic import make_synthetic_negative
        from negative_lab.ui.preview_view import RoiKind

        w = _win()
        t = make_synthetic_negative(seed=9)["negative"]
        rec = FrameRecord(source_path="synthetic/x.tif",
                          source_sha256="0" * 64,
                          input_facts={"abs_path": "synthetic/x.tif",
                                       "kind": "tiff"},
                          geometry=GeometryRegions())
        w.session = ImportSession(files=[rec])
        w.current = rec
        w.t_source = t
        w._load_meta = {"kind": "tiff", "sha256": "0" * 64}
        w._prepare_frame_views()
        # 输出裁切释放：不发管线刷新请求
        ran = []
        w._refresh_preview = lambda *a, **k: ran.append(1)
        w._on_roi_edited(RoiKind.OUTPUT_CROP)
        assert not ran, "输出裁切变化不得重跑管线"
        # 片基变化：必须走管线
        w._on_roi_edited(RoiKind.BASE)
        import time as _t
        t0 = _t.time()
        while _t.time() - t0 < 5 and not ran:
            QApplication.processEvents()
            _t.sleep(0.02)
        assert ran, "片基变化必须重跑管线"


# ------------------------------------------------------------- 深色主题
class TestTheme:
    def test_tokens_match_spec(self):
        from negative_lab.ui import theme

        assert theme.COLORS["bg"] == "#1E1E1E"
        assert theme.COLORS["panel"] == "#252526"
        assert theme.COLORS["text"] == "#D4D4D4"
        assert theme.COLORS["accent"] == "#4C8DFF"
        assert theme.COLORS["viewer"] == "#141414"
        assert theme.SIZES["left_w"] == 220 and theme.SIZES["right_w"] == 300

    def test_ui_files_do_not_define_their_own_color_tables(self):
        """页面不得硬编码颜色；theme.py 与 preview_view.py（ROI 三色）除外。"""
        ui = ROOT / "src" / "negative_lab" / "ui"
        hexre = re.compile(r"#[0-9A-Fa-f]{6}\b")
        offenders = {}
        for f in ui.glob("*.py"):
            if f.name in ("theme.py", "preview_view.py"):
                continue
            hits = hexre.findall(f.read_text(encoding="utf-8-sig"))
            if hits:
                offenders[f.name] = hits[:5]
        assert not offenders, f"页面各自硬编码颜色：{offenders}"

    def test_stylesheet_has_no_gradient_big_radius_or_purple(self):
        from negative_lab.ui import theme

        qss = theme.qss().lower()
        assert "qlineargradient" not in qss and "gradient(" not in qss
        for m in re.finditer(r"border-radius:\s*(\d+)px", qss):
            assert int(m.group(1)) <= 6, f"圆角超过 6px：{m.group(0)}"
        assert "purple" not in qss and "#800080" not in qss

    def test_text_contrast_on_dark_surfaces(self):
        from negative_lab.ui import theme

        def lum(hexc):
            r, g, b = (int(hexc[i:i + 2], 16) / 255 for i in (1, 3, 5))
            f = lambda c: (c / 12.92 if c <= 0.04045
                           else ((c + 0.055) / 1.055) ** 2.4)
            return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b)

        def ratio(a, b):
            la, lb = sorted((lum(a), lum(b)), reverse=True)
            return (la + 0.05) / (lb + 0.05)

        assert ratio(theme.COLORS["text"], theme.COLORS["bg"]) >= 7.0
        assert ratio(theme.COLORS["text"], theme.COLORS["panel"]) >= 7.0
        assert ratio(theme.COLORS["accent"], theme.COLORS["panel"]) >= 2.5

    def test_main_window_viewer_uses_token(self):
        w = _win()
        assert w._viewer_background == "#141414"


# ------------------------------------------------- 绘制状态机（保留语义）
class TestDrawMode:
    def test_draw_and_replace_and_delete(self, view):
        from negative_lab.ui.preview_view import RoiKind

        view.set_draw_mode(RoiKind.OUTPUT_CROP)
        _drag(view, (40, 40), (260, 180))
        assert view.roi(RoiKind.OUTPUT_CROP) is not None
        _drag(view, (300, 220), (580, 380))
        assert view.roi_count(RoiKind.OUTPUT_CROP) == 1, "单类重复拖动=替换"
        view.set_draw_mode(RoiKind.BASE)
        _drag(view, (10, 10), (100, 40))
        _drag(view, (10, 60), (100, 90))
        assert view.roi_count(RoiKind.BASE) == 2
        view.set_active(RoiKind.BASE, 0)
        from PySide6.QtTest import QTest

        QTest.keyClick(view, Qt.Key_Delete)
        assert view.roi_count(RoiKind.BASE) == 1, "Delete 只删当前对象"
        QTest.keyClick(view, Qt.Key_Escape)
        assert view.draw_mode() is None

    def test_nested_kind_can_be_drawn(self, view):
        from negative_lab.ui.preview_view import RoiKind

        view.set_draw_mode(RoiKind.OUTPUT_CROP)
        _drag(view, (20, 20), (400, 300))
        view.set_draw_mode(RoiKind.IMAGE_AREA)
        _drag(view, (80, 60), (300, 200))
        assert view.roi(RoiKind.IMAGE_AREA) is not None, "嵌套区域必须可画"

    def test_kind_visibility_toggle(self, view):
        from negative_lab.ui.preview_view import RoiKind

        view.set_draw_mode(RoiKind.BASE)
        _drag(view, (10, 10), (100, 40))
        view.set_kind_visible(RoiKind.BASE, False)
        assert view.kind_visible(RoiKind.BASE) is False
        view.set_kind_visible(RoiKind.BASE, True)
        assert view.kind_visible(RoiKind.BASE) is True

    def test_view_transform_keeps_rois(self, view):
        from negative_lab.ui.preview_view import RoiKind

        from PySide6.QtCore import QRectF

        view.set_roi(RoiKind.OUTPUT_CROP, QRectF(10, 10, 100, 80))
        before = view.roi(RoiKind.OUTPUT_CROP)
        view.fit()
        view.zoom_100()
        assert view.roi(RoiKind.OUTPUT_CROP) == before


# ------------------------------------------- 服务失败不阻断（保留契约）
class TestServiceFailureTolerance:
    def test_service_error_does_not_block_canvas_edit_or_draft(
            self, tmp_path, monkeypatch):
        import negative_lab.calibration.annotations as annmod
        import negative_lab.ui.main_window as mw

        monkeypatch.setattr(annmod, "ANNOTATIONS_DIR", tmp_path)
        from negative_lab.library.models import (FrameRecord, GeometryRegions,
                                                 ImportSession)
        from negative_lab.testing.synthetic import make_synthetic_negative
        from negative_lab.ui.main_window import MainWindow
        from negative_lab.ui.preview_view import RoiKind

        w = MainWindow()
        w.show()
        t = make_synthetic_negative(seed=5, black_border=12)["negative"]
        rec = FrameRecord(source_path="synthetic/smoke.tif",
                          source_sha256="0" * 64,
                          input_facts={"abs_path": "synthetic/smoke.tif",
                                       "kind": "tiff"},
                          geometry=GeometryRegions())
        w.session = ImportSession(files=[rec])
        w.current = rec
        w.t_source = t
        w._load_meta = {"kind": "tiff", "sha256": "0" * 64}
        w._prepare_frame_views()

        def boom(*a, **k):
            raise RuntimeError("模拟服务崩溃")
        monkeypatch.setattr(mw, "process_image", boom)
        view = w.view
        view.set_draw_mode(RoiKind.IMAGE_AREA)
        _drag(view, (30, 30), (200, 160))
        assert rec.geometry.image_area_roi is not None, \
            "服务崩溃后用户草稿必须仍然更新"
        # 片基变化走管线 -> 管线崩溃必须给出非阻断提示（而非崩溃/吞掉）
        view.set_draw_mode(RoiKind.BASE)
        _drag(view, (30, 260), (200, 330))
        deadline = time.time() + 5
        while time.time() < deadline and threading.active_count() > 1:
            QApplication.processEvents()
            time.sleep(0.02)
        QApplication.processEvents()
        assert w._banner_text, "必须给出非阻断提示"
