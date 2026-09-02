"""v6 P0-1/P0-2：裁切页 UI 契约（§10.2/§10.3）。

先红后绿：本文件在实现前必须失败。覆盖：
- 绘制状态机（空白拖出新 ROI、替换语义、命中优先级、过小拒绝、Esc/Delete）
- 删除语义（删一个片基不影响其他；删有效画面后不得自动伪造）
- 视图变换（适合窗口/100% 只改变变换，不改变 ROI 数值）
- 主题（token 唯一来源、无渐变/大圆角/紫色、可读对比度、点击目标尺寸）
- 服务失败不阻断画布编辑与草稿更新（§5.2）
"""

from __future__ import annotations

import re
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtWidgets import QPushButton, QToolButton, QWidget

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module", autouse=True)
def qapp():
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


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


def _click(view, p):
    from PySide6.QtTest import QTest

    QTest.mouseClick(view.viewport(), Qt.LeftButton, Qt.NoModifier, QPoint(*p))


def _key(view, key):
    from PySide6.QtTest import QTest

    QTest.keyClick(view, key)


# ------------------------------------------------------------- 绘制状态机
class TestDrawMode:
    def test_draw_output_crop_on_empty_canvas(self, view):
        from negative_lab.ui.preview_view import RoiKind

        created = []
        view.roi_created.connect(lambda k, i, r: created.append((k, i, r)))
        view.set_draw_mode(RoiKind.OUTPUT_CROP)
        _drag(view, (40, 40), (260, 180))
        assert view.roi(RoiKind.OUTPUT_CROP) is not None, "拖动后必须产生输出裁切框"
        assert created and created[-1][0] is RoiKind.OUTPUT_CROP

    def test_draw_image_area_on_empty_canvas(self, view):
        from negative_lab.ui.preview_view import RoiKind

        view.set_draw_mode(RoiKind.IMAGE_AREA)
        _drag(view, (60, 60), (300, 200))
        assert view.roi(RoiKind.IMAGE_AREA) is not None

    def test_draw_two_base_regions(self, view):
        from negative_lab.ui.preview_view import RoiKind

        view.set_draw_mode(RoiKind.BASE)
        _drag(view, (10, 10), (100, 40))
        _drag(view, (10, 60), (100, 90))
        assert view.roi_count(RoiKind.BASE) == 2, "片基区每次拖动应新增一个"
        assert view.roi(RoiKind.BASE, 0) != view.roi(RoiKind.BASE, 1)

    def test_single_kind_redraw_replaces_and_hints(self, view):
        from negative_lab.ui.preview_view import RoiKind

        hints = []
        view.replace_hint.connect(hints.append)
        view.set_draw_mode(RoiKind.OUTPUT_CROP)
        _drag(view, (20, 20), (200, 150))
        first = view.roi(RoiKind.OUTPUT_CROP)
        _drag(view, (260, 200), (580, 380))          # 第二次拖动 -> 替换
        assert view.roi_count(RoiKind.OUTPUT_CROP) == 1, "输出裁切最多一个"
        assert view.roi(RoiKind.OUTPUT_CROP) != first
        assert hints and "替换" in hints[-1]

    def test_press_inside_existing_roi_moves_instead_of_creating(self, view):
        from negative_lab.ui.preview_view import RoiKind

        view.set_draw_mode(RoiKind.OUTPUT_CROP)
        _drag(view, (20, 20), (200, 150))
        r0 = view.roi(RoiKind.OUTPUT_CROP)
        center = (int(r0.center().x()), int(r0.center().y()))
        _drag(view, center, (center[0] + 30, center[1] + 20))
        r1 = view.roi(RoiKind.OUTPUT_CROP)
        assert view.roi_count(RoiKind.OUTPUT_CROP) == 1, \
            "从现有 ROI 内开始拖动必须进入移动，不得误新建"
        assert (r1.left(), r1.top()) != (r0.left(), r0.top()), "必须真的移动了"

    def test_nested_kind_can_be_drawn_inside_output_crop(self, view):
        """有效画面画在输出裁切内部是合法操作：跨类型按下必须新建，
        否则嵌套区域永远画不出来（v6 §5.2 + §5.5）。"""
        from negative_lab.ui.preview_view import RoiKind

        view.set_draw_mode(RoiKind.OUTPUT_CROP)
        _drag(view, (20, 20), (400, 300))
        view.set_draw_mode(RoiKind.IMAGE_AREA)
        _drag(view, (80, 60), (300, 200))          # 起点在 output_crop 内部
        assert view.roi(RoiKind.IMAGE_AREA) is not None, \
            "跨类型按下必须按当前绘制类型新建（嵌套语义）"
        assert view.roi_count(RoiKind.OUTPUT_CROP) == 1

    def test_too_small_drag_is_rejected(self, view):
        from negative_lab.ui.preview_view import RoiKind

        rejected = []
        view.draw_rejected.connect(rejected.append)
        view.set_draw_mode(RoiKind.BASE)
        _drag(view, (10, 10), (13, 12))
        assert view.roi_count(RoiKind.BASE) == 0, "过小矩形不得写入模型"
        assert rejected, "必须发出简短错误提示"

    def test_esc_cancels_draw_mode(self, view):
        from negative_lab.ui.preview_view import RoiKind

        view.set_draw_mode(RoiKind.BASE)
        assert view.draw_mode() is RoiKind.BASE
        _key(view, Qt.Key_Escape)
        assert view.draw_mode() is None, "Esc 必须取消当前绘制"
        _drag(view, (10, 10), (120, 60))
        assert view.roi_count(RoiKind.BASE) == 0, "取消后拖动不得再新建"

    def test_delete_key_removes_active_base_only(self, view):
        from negative_lab.ui.preview_view import RoiKind

        deleted = []
        view.roi_deleted.connect(lambda k, i: deleted.append((k, i)))
        view.set_draw_mode(RoiKind.BASE)
        _drag(view, (10, 10), (100, 40))
        _drag(view, (10, 60), (100, 90))
        view.set_active(RoiKind.BASE, 0)
        _key(view, Qt.Key_Delete)
        assert view.roi_count(RoiKind.BASE) == 1, "只删除当前选中的片基区"
        assert deleted == [(RoiKind.BASE, 0)]


# ------------------------------------------------------- 视图变换与选择
class TestViewTransform:
    def test_fit_and_100_only_change_view_transform(self, view):
        from negative_lab.ui.preview_view import RoiKind

        view.set_roi(RoiKind.OUTPUT_CROP, __import__(
            "PySide6.QtCore", fromlist=["QRectF"]).QRectF(10, 10, 100, 80))
        before = view.roi(RoiKind.OUTPUT_CROP)
        m0 = view.transform().m11()
        view.fit()
        assert view.transform().m11() != m0
        view.zoom_100()
        assert view.roi(RoiKind.OUTPUT_CROP) == before, \
            "视图变换不得改变 ROI 数值"
        _click(view, (int(before.center().x()), int(before.center().y())))
        assert view.active_roi() is not None, "单击边框/内部应选中当前对象"


# ------------------------------------------------------------- 主题契约
class TestTheme:
    def test_tokens_match_spec(self):
        from negative_lab.ui import theme

        assert theme.COLORS == {
            "canvas": "#F2F2F7", "surface": "#FFFFFF",
            "text_primary": "#1C1C1E", "text_secondary": "#6E6E73",
            "separator": "#D1D1D6", "accent": "#007AFF",
            "success": "#34C759", "warning": "#FF9500",
            "danger": "#FF3B30", "viewer": "#1C1C1E"}
        assert theme.SPACING == {"xs": 4, "sm": 8, "md": 12, "lg": 16, "xl": 24}
        assert theme.SIZES == {"toolbar_h": 44, "min_target": 32,
                               "primary_target": 36, "sidebar_w": 240,
                               "inspector_w": 280}

    def test_ui_files_do_not_define_their_own_color_tables(self):
        """§5.8：页面不得重复定义同名颜色；hex 只允许出现在 theme.py
        与 preview_view.py（ROI 三色是数据可视化调色板，不是主题表）。"""
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

        qss = theme.qss()
        assert "qlineargradient" not in qss and "gradient" not in qss.lower()
        for m in re.finditer(r"border-radius:\s*(\d+)px", qss):
            assert int(m.group(1)) <= 8, f"圆角超过 8px：{m.group(0)}"
        low = qss.lower()
        assert "purple" not in low and "#800080" not in low and \
            "#a020f0" not in low

    def test_text_contrast_on_light_surfaces(self):
        from negative_lab.ui import theme

        def lum(hexc):
            r, g, b = (int(hexc[i:i + 2], 16) / 255 for i in (1, 3, 5))
            f = lambda c: (c / 12.92 if c <= 0.04045
                           else ((c + 0.055) / 1.055) ** 2.4)
            return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b)

        def ratio(a, b):
            la, lb = sorted((lum(a), lum(b)), reverse=True)
            return (la + 0.05) / (lb + 0.05)

        assert ratio(theme.COLORS["text_primary"], theme.COLORS["canvas"]) >= 4.5
        assert ratio(theme.COLORS["text_primary"], theme.COLORS["surface"]) >= 4.5
        assert ratio(theme.COLORS["accent"], theme.COLORS["surface"]) >= 3.0

    def test_crop_page_click_targets_meet_minimums(self):
        from negative_lab.ui.crop_page import CropPage

        page = CropPage()
        small = [b for b in page.findChildren(QToolButton)
                 if b.minimumSize().width() < 32 or b.minimumSize().height() < 32]
        assert not small, f"工具按钮点击目标必须 >=32px：{small}"
        for name in ("btn_sync", "btn_reset_kind", "btn_reset_all"):
            btn = page.findChild(QPushButton, name)
            assert btn is not None, f"缺少主操作按钮 {name}"
            assert btn.minimumHeight() >= 36, f"{name} 主操作必须 >=36px"

    def test_main_window_uses_theme_viewer_background(self):
        from PySide6.QtWidgets import QApplication

        QApplication.instance() or QApplication([])
        from negative_lab.ui.main_window import MainWindow

        win = MainWindow()
        assert win._viewer_background in (None, "#1C1C1E")
        assert "#F2F2F7" in (win.styleSheet() or ""), \
            "主窗口背景必须来自 theme.canvas"


# ------------------------------------------------- 服务失败不阻断编辑
class TestServiceFailureTolerance:
    def test_service_error_does_not_block_canvas_edit_or_draft(self, tmp_path,
                                                               monkeypatch):
        """§5.2：service warning/error 不得阻止画布编辑和 ROI 保存。"""
        import negative_lab.calibration.annotations as annmod
        import negative_lab.ui.main_window as mw
        from PySide6.QtWidgets import QApplication

        # 草稿保存必须重定向到临时目录：绝不污染真实 fixtures/
        monkeypatch.setattr(annmod, "ANNOTATIONS_DIR", tmp_path)
        QApplication.instance() or QApplication([])
        from negative_lab.ui.main_window import MainWindow
        from negative_lab.ui.preview_view import RoiKind

        win = MainWindow()
        win.show()
        from negative_lab.testing.synthetic import make_synthetic_negative
        t = make_synthetic_negative(seed=5, black_border=12)["negative"]
        rec = _make_record(t)
        win.session = _session_with(rec)
        win.current = rec
        win.t_source = t
        win._load_meta = {"kind": "tiff", "sha256": "0" * 64}
        win._prepare_frame_views()

        def boom(*a, **k):
            raise RuntimeError("模拟服务崩溃")
        monkeypatch.setattr(mw, "process_image", boom)
        view = win.crop_page.view
        view.set_draw_mode(RoiKind.IMAGE_AREA)
        _drag(view, (30, 30), (200, 160))       # 不得抛异常
        assert rec.geometry.image_area_roi is not None, \
            "服务崩溃后用户草稿必须仍然更新"
        assert rec.status in ("needs_crop_review", "imported"), rec.status
        # 预览刷新在后台线程，banner 由回调发出：等线程结束再断言
        import time
        deadline = time.time() + 5
        while time.time() < deadline and threading.active_count() > 1:
            QApplication.processEvents()
            time.sleep(0.02)
        QApplication.processEvents()
        assert win._banner_text, "必须给出非阻断提示"


# ------------------------------------------------------------ 辅助构造
def _make_record(t):
    from negative_lab.library.models import FrameRecord, GeometryRegions

    return FrameRecord(source_path="synthetic/smoke.tif",
                       source_sha256="0" * 64,
                       input_facts={"abs_path": "synthetic/smoke.tif",
                                    "kind": "tiff"},
                       geometry=GeometryRegions())


def _session_with(rec):
    from negative_lab.library.models import ImportSession

    return ImportSession(files=[rec])
