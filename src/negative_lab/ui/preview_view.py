"""可交互预览视图：缩放/平移 + 可拖动片基框、帧框与中性参考点（§5.1.5）。

坐标约定：矩形以图像像素坐标存储（QRectF），视图负责缩放到视口。
自动检测结果可被用户拖动覆盖（§4.2.2）。
"""

from __future__ import annotations

from enum import Enum

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPixmap, QPen
from PySide6.QtWidgets import QGraphicsScene, QGraphicsView


class RoiKind(Enum):
    FRAME = "frame"
    BASE = "base"


class PreviewView(QGraphicsView):
    """图像预览 + ROI 叠加层。roi_changed(kind, QRectF) 在用户拖动后发出。"""

    roi_changed = Signal(object, object)  # RoiKind, QRectF
    point_picked = Signal(str, float, float)  # "bright"|"dark", x, y（图像坐标）
    mask_painted = Signal()               # 画笔蒙版更新

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHints(self.renderHints())
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self._pix_item = None
        self._rois: dict[RoiKind, QRectF] = {}
        self._dragging: RoiKind | None = None
        self._drag_off = QPointF()
        self._pick_mode: str | None = None  # "bright"/"dark" 点击取点
        self._points: dict[str, QPointF] = {}
        # 除尘画笔蒙版（图像分辨率 bool 数组）
        self._mask = None
        self._mask_item = None
        self._paint_mode = False
        self._painting = False
        self._brush_r = 6.0

    # ---- 图像

    def set_image(self, rgb01, flip_y: bool = True):
        """rgb01: (H,W,3) float in [0,1]。flip_y=False 时数据已是自顶向下顺序。"""
        import numpy as np

        arr = (np.clip(rgb01, 0, 1) * 255.0 + 0.5).astype(np.uint8)
        h, w, _ = arr.shape
        img = QImage(arr.data, w, h, 3 * w, QImage.Format_RGB888)
        img = img.copy()
        self._scene.clear()
        self._pix_item = self._scene.addPixmap(QPixmap.fromImage(img))
        self._scene.setSceneRect(0, 0, w, h)
        self._redraw_overlays()

    def fit(self):
        self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)

    # ---- ROI

    def set_roi(self, kind: RoiKind, rect: QRectF | None):
        if rect is None:
            self._rois.pop(kind, None)
        else:
            self._rois[kind] = QRectF(rect)
        self._redraw_overlays()

    def roi(self, kind: RoiKind) -> QRectF | None:
        return self._rois.get(kind)

    def set_pick_mode(self, mode: str | None):
        self._pick_mode = mode
        self.viewport().setCursor(Qt.CrossCursor if mode else Qt.ArrowCursor)

    # ---- 除尘画笔蒙版

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
        self._redraw_overlays()

    def clear_mask(self):
        import numpy as np

        if self._mask is not None:
            self._mask = np.zeros_like(self._mask)
        self._redraw_overlays()
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

    def points(self) -> dict[str, tuple[float, float]]:
        return {k: (v.x(), v.y()) for k, v in self._points.items()}

    def clear_points(self):
        self._points.clear()
        self._redraw_overlays()

    # ---- 绘制与交互

    def _redraw_overlays(self):
        # 保留 pixmap，删除叠加图元后重画
        for item in list(self._scene.items()):
            if item is not self._pix_item:
                self._scene.removeItem(item)
        self._mask_item = None
        if self._mask is not None and self._mask.any():
            import numpy as np

            h, w = self._mask.shape
            rgba = np.zeros((h, w, 4), np.uint8)
            rgba[self._mask] = (255, 60, 60, 110)
            img = QImage(rgba.data, w, h, 4 * w, QImage.Format_RGBA8888).copy()
            self._mask_item = self._scene.addPixmap(QPixmap.fromImage(img))
        pens = {RoiKind.FRAME: QPen(QColor(80, 160, 255), 0),
                RoiKind.BASE: QPen(QColor(255, 200, 60), 0)}
        for kind, r in self._rois.items():
            pen = pens[kind]
            pen.setCosmetic(True)
            self._scene.addRect(r, pen)
        for name, p in self._points.items():
            color = QColor(120, 255, 120) if name == "bright" else QColor(180, 120, 255)
            self._scene.addEllipse(p, 6, 6, QPen(color))

    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            pos = self.mapToScene(ev.position().toPoint())
            if self._paint_mode:
                self._painting = True
                self.setDragMode(QGraphicsView.NoDrag)
                self._paint_at(pos)
                self._redraw_overlays()
                return
            if self._pick_mode:
                self._points[self._pick_mode] = pos
                self.point_picked.emit(self._pick_mode, pos.x(), pos.y())
                self.set_pick_mode(None)
                self._redraw_overlays()
                return
            for kind, r in self._rois.items():
                if r.contains(pos):
                    self._dragging = kind
                    self._drag_off = pos - r.topLeft()
                    self.setDragMode(QGraphicsView.NoDrag)
                    return
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if self._painting:
            self._paint_at(self.mapToScene(ev.position().toPoint()))
            self._redraw_overlays()
            return
        if self._dragging is not None:
            pos = self.mapToScene(ev.position().toPoint())
            r = self._rois[self._dragging]
            tl = pos - self._drag_off
            tl.setX(max(0.0, min(tl.x(), self._scene.sceneRect().width() - r.width())))
            tl.setY(max(0.0, min(tl.y(), self._scene.sceneRect().height() - r.height())))
            r.moveTo(tl)
            self._redraw_overlays()
            return
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        if self._painting:
            self._painting = False
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            self.mask_painted.emit()
            return
        if self._dragging is not None:
            kind = self._dragging
            self._dragging = None
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            self.roi_changed.emit(kind, QRectF(self._rois[kind]))
            return
        super().mouseReleaseEvent(ev)

    def wheelEvent(self, ev):
        factor = 1.25 if ev.angleDelta().y() > 0 else 0.8
        self.scale(factor, factor)
