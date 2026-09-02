"""可交互预览视图：缩放/平移 + 三类独立 ROI 的拖动与八向缩放（v5 §5.2）。

坐标约定：矩形以图像像素坐标存储（QRectF，半开区间语义由上层换算），
视图负责缩放到视口。拖动/缩放不改变源图；释放后由上层重算预览。

v5 要求（本文件逐条落实）：
- `RoiKind.FRAME` 已拆成 `OUTPUT_CROP`（蓝）与 `IMAGE_AREA`（绿）；
  片基取样区 `BASE`（黄/橙）允许**多个**，支持新增/删除/移动/缩放；
- 每个 ROI 有 8 个缩放手柄 + 整体拖动 + 越界钳制；
- 当前操作对象除颜色外还用**标签与线型**区分（可访问性）。
"""

from __future__ import annotations

from enum import Enum

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPixmap, QPen
from PySide6.QtWidgets import QGraphicsItem, QGraphicsScene, QGraphicsView

HANDLE_COUNT = 8
HANDLES = ("nw", "n", "ne", "e", "se", "s", "sw", "w")
_HANDLE_TOL = 8.0         # 命中容差（视口像素）
DEFAULT_MIN_ROI = 16.0


class RoiKind(Enum):
    OUTPUT_CROP = "output_crop"     # 最终构图（可含片边/齿孔）
    IMAGE_AREA = "image_area"       # 有效画面：算法与统计区域
    BASE = "base"                   # 未曝光片基取样区（可多个）


#: 颜色之外再用标签与线型区分（v5 §5.2）
ROI_STYLE: dict[RoiKind, dict] = {
    RoiKind.OUTPUT_CROP: {"color": (80, 160, 255), "label": "输出裁切",
                          "dash": Qt.CustomDashLine, "pattern": [6, 2],
                          "width": 2},
    RoiKind.IMAGE_AREA: {"color": (90, 220, 130), "label": "有效画面",
                         "dash": Qt.SolidLine, "pattern": None, "width": 2},
    RoiKind.BASE: {"color": (255, 200, 60), "label": "片基取样",
                   "dash": Qt.DashLine, "pattern": None, "width": 2},
}
ACTIVE_COLOR = (255, 255, 255)


class _OverlayItem(QGraphicsItem):
    """自绘叠加层：ROI、手柄、标签、蒙版、参考点。"""

    def __init__(self, view: "PreviewView"):
        super().__init__()
        self._view = view
        self.setZValue(10)

    def boundingRect(self) -> QRectF:      # noqa: N802
        # 注意：不得在 paint/boundingRect 里触发视图重算（prepareGeometryChange
        # -> sceneRect -> update 的环会递归爆栈）；直接用场景矩形缓存。
        r = self._view._scene_rect
        return QRectF(r).adjusted(-64, -64, 64, 64)

    def paint(self, painter: QPainter, *args, **kwargs):
        self._view._paint_overlay(painter)


class PreviewView(QGraphicsView):
    """图像预览 + 三类 ROI 叠加层。"""

    roi_changed = Signal(object, int, object)     # RoiKind, index, QRectF
    roi_selected = Signal(object, int)            # RoiKind, index
    point_picked = Signal(str, float, float)      # "bright"|"dark", x, y
    mask_painted = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHints(self.renderHints())
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        # 必须在创建 _OverlayItem 之前就绪：首个 boundingRect() 就会读它
        self._scene_rect = QRectF(0, 0, 1, 1)
        self._pix_item = None
        self._overlay = _OverlayItem(self)
        self._scene.addItem(self._overlay)
        self._rois: dict[RoiKind, list[QRectF]] = {k: [] for k in RoiKind}
        self._active: tuple[RoiKind, int] | None = None
        self._drag_mode: str | None = None          # "move" | "resize"
        self._drag_handle: str | None = None
        self._drag_off = QPointF()
        self._drag_orig: QRectF | None = None
        self._min_roi = DEFAULT_MIN_ROI
        self._pick_mode: str | None = None
        self._points: dict[str, QPointF] = {}
        self._mask = None
        self._paint_mode = False
        self._painting = False
        self._brush_r = 6.0

    # ---------------------------------------------------------------- 图像
    def set_image(self, rgb01, flip_y: bool = True):
        """rgb01: (H,W,3) float in [0,1]。"""
        import numpy as np

        arr = (np.clip(rgb01, 0, 1) * 255.0 + 0.5).astype(np.uint8)
        h, w, _ = arr.shape
        img = QImage(arr.data, w, h, 3 * w, QImage.Format_RGB888)
        img = img.copy()
        if self._pix_item is not None:
            self._scene.removeItem(self._pix_item)
        self._scene_rect = QRectF(0, 0, w, h)
        self._pix_item = self._scene.addPixmap(QPixmap.fromImage(img))
        self._pix_item.setZValue(0)
        self._scene.setSceneRect(self._scene_rect)
        self._overlay.update()

    def fit(self):
        self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)

    def zoom_100(self):
        self.resetTransform()

    # ---------------------------------------------------------------- ROI
    def roi_kind_order(self) -> list[RoiKind]:
        return [RoiKind.OUTPUT_CROP, RoiKind.IMAGE_AREA, RoiKind.BASE]

    def min_roi_size(self) -> float:
        return self._min_roi

    def set_min_roi_size(self, px: float):
        self._min_roi = max(2.0, float(px))

    def set_roi(self, kind: RoiKind, rect: QRectF | None, index: int = 0):
        lst = self._rois.setdefault(kind, [])
        if rect is None:
            if index < len(lst):
                lst.pop(index)
        else:
            r = self._clamped(kind, QRectF(rect))
            while len(lst) <= index:
                lst.append(QRectF(r))
            lst[index] = QRectF(r)
        self._overlay.update()

    def roi(self, kind: RoiKind, index: int = 0) -> QRectF | None:
        lst = self._rois.get(kind) or []
        return lst[index] if 0 <= index < len(lst) else None

    def rois(self, kind: RoiKind) -> list[QRectF]:
        return list(self._rois.get(kind) or [])

    def roi_count(self, kind: RoiKind) -> int:
        return len(self._rois.get(kind) or [])

    def add_roi(self, kind: RoiKind, rect: QRectF) -> int:
        lst = self._rois.setdefault(kind, [])
        if kind is not RoiKind.BASE and lst:
            lst[0] = self._clamped(kind, QRectF(rect))
            self._overlay.update()
            return 0
        lst.append(self._clamped(kind, QRectF(rect)))
        self._active = (kind, len(lst) - 1)
        self._overlay.update()
        return len(lst) - 1

    def remove_roi(self, kind: RoiKind, index: int = 0) -> bool:
        lst = self._rois.get(kind) or []
        if not (0 <= index < len(lst)):
            return False
        lst.pop(index)
        if self._active and self._active[0] is kind:
            self._active = None
        self._overlay.update()
        return True

    def active_roi(self) -> tuple[RoiKind, int] | None:
        return self._active

    def set_active(self, kind: RoiKind, index: int = 0):
        if self.roi(kind, index) is not None:
            self._active = (kind, index)
            self.roi_selected.emit(kind, index)
            self._overlay.update()

    def clear_rois(self, kind: RoiKind | None = None):
        for k in ([kind] if kind else list(RoiKind)):
            self._rois[k] = []
        self._active = None
        self._overlay.update()

    def _clamped(self, kind: RoiKind, r: QRectF) -> QRectF:
        s = self._scene.sceneRect()
        w = max(self._min_roi, min(r.width(), s.width()))
        h = max(self._min_roi, min(r.height(), s.height()))
        x = max(s.left(), min(r.x(), s.right() - w))
        y = max(s.top(), min(r.y(), s.bottom() - h))
        return QRectF(x, y, w, h)

    # ---------------------------------------------------------------- 手柄
    def _handle_points(self, r: QRectF) -> dict[str, QPointF]:
        cx, cy = r.center().x(), r.center().y()
        return {
            "nw": r.topLeft(), "n": QPointF(cx, r.top()), "ne": r.topRight(),
            "e": QPointF(r.right(), cy), "se": r.bottomRight(),
            "s": QPointF(cx, r.bottom()), "sw": r.bottomLeft(),
            "w": QPointF(r.left(), cy)}

    def handle_at(self, kind: RoiKind, pos, index: int = 0) -> str | None:
        """命中测试：返回 "n"/"ne"/... 或 None（pos 为场景坐标或 (x,y)）。"""
        r = self.roi(kind, index)
        if r is None:
            return None
        p = self._as_point(pos)
        tol = self._tolerance()
        best, bestd = None, tol + 1.0
        for name, hp in self._handle_points(r).items():
            d = max(abs(p.x() - hp.x()), abs(p.y() - hp.y()))
            if d <= tol and d < bestd:
                best, bestd = name, d
        return best

    def _tolerance(self) -> float:
        z = self.transform().m11() or 1.0
        return _HANDLE_TOL / max(z, 1e-6)

    def resize_roi(self, kind: RoiKind, handle: str, pos, index: int = 0) -> QRectF | None:
        """按手柄缩放到场景坐标 pos，并做越界与最小尺寸钳制。"""
        r = self.roi(kind, index)
        if r is None or handle not in HANDLES:
            return None
        p = self._as_point(pos)
        s = self._scene.sceneRect()
        left, top, right, bottom = r.left(), r.top(), r.right(), r.bottom()
        if "w" in handle:
            left = min(p.x(), right - self._min_roi)
        if "e" in handle:
            right = max(p.x(), left + self._min_roi)
        if "n" in handle:
            top = min(p.y(), bottom - self._min_roi)
        if "s" in handle:
            bottom = max(p.y(), top + self._min_roi)
        rect = QRectF(left, top, right - left, bottom - top)
        rect = self._clamped(kind, rect)
        self._rois[kind][index] = rect
        self._overlay.update()
        return rect

    def move_roi(self, kind: RoiKind, pos, index: int = 0) -> QRectF | None:
        r = self.roi(kind, index)
        if r is None:
            return None
        p = self._as_point(pos)
        rect = QRectF(p - self._drag_off, r.size())
        rect = self._clamped(kind, rect)
        self._rois[kind][index] = rect
        self._overlay.update()
        return rect

    @staticmethod
    def _as_point(pos):
        if isinstance(pos, QPointF):
            return pos
        return QPointF(float(pos[0]), float(pos[1]))

    # ---------------------------------------------------------------- 取点
    def set_pick_mode(self, mode: str | None):
        self._pick_mode = mode
        self.viewport().setCursor(Qt.CrossCursor if mode else Qt.ArrowCursor)

    def points(self) -> dict[str, tuple[float, float]]:
        return {k: (v.x(), v.y()) for k, v in self._points.items()}

    def clear_points(self):
        self._points.clear()
        self._overlay.update()

    # ---------------------------------------------------------------- 蒙版
    def set_paint_mode(self, on: bool):
        self._paint_mode = bool(on)
        if on:
            self._pick_mode = None
        self.viewport().setCursor(Qt.CrossCursor if on else Qt.ArrowCursor)

    def paint_mode(self) -> bool:
        return self._paint_mode

    def ensure_mask(self, h: int, w: int):
        import numpy as np

        if self._mask is None or self._mask.shape != (h, w):
            self._mask = np.zeros((h, w), bool)

    def mask(self):
        return self._mask

    def set_mask(self, mask):
        self._mask = mask
        self._overlay.update()

    def clear_mask(self):
        import numpy as np

        if self._mask is not None:
            self._mask = np.zeros_like(self._mask)
        self._overlay.update()
        self.mask_painted.emit()

    def _paint_at(self, pos: QPointF):
        import numpy as np

        if self._mask is None:
            return
        h, w = self._mask.shape
        cx, cy = int(round(pos.x())), int(round(pos.y()))
        r = int(self._brush_r)
        y0, y1 = max(0, cy - r), min(h, cy + r + 1)
        x0, x1 = max(0, cx - r), min(w, cx + r + 1)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        self._mask[y0:y1, x0:x1] |= (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r

    # ---------------------------------------------------------------- 绘制
    def _paint_overlay(self, painter: QPainter):
        painter.setRenderHint(QPainter.Antialiasing, True)
        if self._mask is not None and self._mask.any():
            import numpy as np

            h, w = self._mask.shape
            rgba = np.zeros((h, w, 4), np.uint8)
            rgba[self._mask] = (255, 60, 60, 110)
            img = QImage(rgba.data, w, h, 4 * w, QImage.Format_RGBA8888).copy()
            painter.drawPixmap(0, 0, QPixmap.fromImage(img))
        for kind in self.roi_kind_order():
            style = ROI_STYLE[kind]
            for idx, r in enumerate(self.rois(kind)):
                active = self._active == (kind, idx)
                pen = QPen(QColor(*(ACTIVE_COLOR if active else style["color"])))
                pen.setCosmetic(True)
                pen.setWidthF(style["width"] + (1 if active else 0))
                if style["dash"] is Qt.CustomDashLine and style.get("pattern"):
                    pen.setStyle(Qt.CustomDashLine)
                    pen.setDashPattern(style["pattern"])
                else:
                    pen.setStyle(style["dash"])
                painter.setPen(pen)
                painter.setBrush(Qt.NoBrush)
                painter.drawRect(r)
                label = style["label"] + (f" {idx + 1}" if kind is RoiKind.BASE
                                          and len(self.rois(kind)) > 1 else "")
                if active:
                    label += " ●"
                painter.setPen(QPen(QColor(*(ACTIVE_COLOR if active
                                             else style["color"]))))
                painter.drawText(QPointF(r.left(), max(0, r.top() - 4)), label)
                if active or kind is RoiKind.OUTPUT_CROP:
                    painter.setPen(QPen(QColor(255, 255, 255)))
                    for hp in self._handle_points(r).values():
                        painter.drawRect(QRectF(hp.x() - 3, hp.y() - 3, 6, 6))
        for name, p in self._points.items():
            color = QColor(120, 255, 120) if name == "bright" else QColor(180, 120, 255)
            painter.setPen(QPen(color))
            painter.drawEllipse(p, 6, 6)

    # ---------------------------------------------------------------- 交互
    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            pos = self.mapToScene(ev.position().toPoint())
            if self._paint_mode:
                self._painting = True
                self.setDragMode(QGraphicsView.NoDrag)
                self._paint_at(pos)
                self._overlay.update()
                return
            if self._pick_mode:
                self._points[self._pick_mode] = pos
                self.point_picked.emit(self._pick_mode, pos.x(), pos.y())
                self.set_pick_mode(None)
                self._overlay.update()
                return
            for kind in reversed(self.roi_kind_order()):
                for idx in range(self.roi_count(kind) - 1, -1, -1):
                    h = self.handle_at(kind, pos, idx)
                    r = self.roi(kind, idx)
                    if h is not None:
                        self._active = (kind, idx)
                        self._drag_mode = "resize"
                        self._drag_handle = h
                        self._drag_orig = QRectF(r)
                        self.setDragMode(QGraphicsView.NoDrag)
                        self.roi_selected.emit(kind, idx)
                        self._overlay.update()
                        return
                    if r is not None and r.contains(pos):
                        self._active = (kind, idx)
                        self._drag_mode = "move"
                        self._drag_off = pos - r.topLeft()
                        self.setDragMode(QGraphicsView.NoDrag)
                        self.roi_selected.emit(kind, idx)
                        self._overlay.update()
                        return
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if self._painting:
            self._paint_at(self.mapToScene(ev.position().toPoint()))
            self._overlay.update()
            return
        if self._drag_mode and self._active is not None:
            kind, idx = self._active
            pos = self.mapToScene(ev.position().toPoint())
            if self._drag_mode == "move":
                self.move_roi(kind, pos, idx)
            else:
                self.resize_roi(kind, self._drag_handle, pos, idx)
            return
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        if self._painting:
            self._painting = False
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            self.mask_painted.emit()
            return
        if self._drag_mode and self._active is not None:
            kind, idx = self._active
            r = self.roi(kind, idx)
            self._drag_mode = None
            self._drag_handle = None
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            if r is not None:
                self.roi_changed.emit(kind, idx, QRectF(r))
            return
        super().mouseReleaseEvent(ev)

    def wheelEvent(self, ev):
        factor = 1.25 if ev.angleDelta().y() > 0 else 0.8
        self.scale(factor, factor)
        self._overlay.update()
