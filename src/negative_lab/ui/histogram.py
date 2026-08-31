"""RGB 通道直方图控件（§5.1.2）。"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QPainter, QPolygonF
from PySide6.QtWidgets import QSizePolicy, QWidget


class HistogramWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(260, 130)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self._data: np.ndarray | None = None  # (3, bins) 归一化计数

    def set_image(self, rgb01: np.ndarray, bins: int = 128):
        x = np.clip(rgb01, 0, 1)
        self._data = np.stack([
            np.histogram(x[..., c], bins=bins, range=(0.0, 1.0))[0]
            for c in range(3)
        ]).astype(np.float64)
        m = self._data.max()
        if m > 0:
            self._data /= m
        self.update()

    def paintEvent(self, ev):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(28, 28, 28))
        if self._data is None:
            return
        w, h = self.width(), self.height()
        colors = (QColor(255, 80, 80), QColor(80, 255, 80), QColor(90, 140, 255))
        p.setPen(Qt.NoPen)
        bins = self._data.shape[1]
        for c in range(3):
            p.setBrush(colors[c])
            poly = QPolygonF([QPointF(0, h)])
            for i, v in enumerate(self._data[c]):
                poly.append(QPointF((i + 0.5) / bins * w, h - v * (h - 4)))
            poly.append(QPointF(w, h))
            p.drawPolygon(poly)
