"""通用矢量图标（v5 §5.1：按钮使用熟悉的旋转/镜像/裁切/缩放图标 + tooltip，
不得用长句按钮替代图标）。

不引入外部图标包：用 QPainter 直接绘制标准字形，离线、可冻结、无授权风险。
"""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap, QPolygonF

_SIZE = 32
_COLOR = QColor(225, 225, 225)


def _canvas() -> tuple[QPixmap, QPainter]:
    pm = QPixmap(_SIZE, _SIZE)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing, True)
    pen = QPen(_COLOR)
    pen.setWidthF(2.0)
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    p.setPen(pen)
    return pm, p


def _arrow_head(p: QPainter, c: QPointF, angle_deg: float, size: float = 5.0):
    import math

    a = math.radians(angle_deg)
    for da in (150.0, -150.0):
        b = a + math.radians(da)
        p.drawLine(c, QPointF(c.x() + size * math.cos(b),
                              c.y() + size * math.sin(b)))


def _icon(draw) -> QIcon:
    pm, p = _canvas()
    try:
        draw(p)
    finally:
        p.end()
    return QIcon(pm)


def rotate_cw() -> QIcon:
    def d(p: QPainter):
        p.drawArc(QRectF(6, 6, 20, 20), -60 * 16, 240 * 16)
        _arrow_head(p, QPointF(23.0, 23.0), -30)
    return _icon(d)


def rotate_ccw() -> QIcon:
    def d(p: QPainter):
        p.drawArc(QRectF(6, 6, 20, 20), -120 * 16, 240 * 16)
        _arrow_head(p, QPointF(9.0, 23.0), -150)
    return _icon(d)


def mirror_h() -> QIcon:
    def d(p: QPainter):
        pen = QPen(_COLOR)
        pen.setStyle(Qt.DashLine)
        p.setPen(pen)
        p.drawLine(QPointF(16, 4), QPointF(16, 28))
        p.setPen(QPen(_COLOR))
        p.drawPolygon(QPolygonF([QPointF(6, 10), QPointF(13, 16), QPointF(6, 22)]))
        p.drawPolygon(QPolygonF([QPointF(26, 10), QPointF(19, 16), QPointF(26, 22)]))
    return _icon(d)


def crop() -> QIcon:
    def d(p: QPainter):
        p.drawLine(QPointF(9, 4), QPointF(9, 24))
        p.drawLine(QPointF(9, 24), QPointF(28, 24))
        p.drawLine(QPointF(4, 9), QPointF(24, 9))
        p.drawLine(QPointF(24, 9), QPointF(24, 28))
    return _icon(d)


def target() -> QIcon:
    def d(p: QPainter):
        p.drawEllipse(QRectF(9, 9, 14, 14))
        p.drawLine(QPointF(16, 3), QPointF(16, 9))
        p.drawLine(QPointF(16, 23), QPointF(16, 29))
        p.drawLine(QPointF(3, 16), QPointF(9, 16))
        p.drawLine(QPointF(23, 16), QPointF(29, 16))
    return _icon(d)


def zoom_fit() -> QIcon:
    def d(p: QPainter):
        p.drawRect(QRectF(6, 6, 20, 20))
        p.drawLine(QPointF(11, 11), QPointF(15, 15))
        p.drawLine(QPointF(21, 11), QPointF(17, 15))
        p.drawLine(QPointF(11, 21), QPointF(15, 17))
        p.drawLine(QPointF(21, 21), QPointF(17, 17))
    return _icon(d)


def zoom_100() -> QIcon:
    def d(p: QPainter):
        p.drawEllipse(QRectF(6, 6, 15, 15))
        p.drawLine(QPointF(20, 20), QPointF(27, 27))
        p.drawLine(QPointF(13, 10), QPointF(13, 17))
    return _icon(d)


def plus() -> QIcon:
    def d(p: QPainter):
        p.drawLine(QPointF(16, 7), QPointF(16, 25))
        p.drawLine(QPointF(7, 16), QPointF(25, 16))
    return _icon(d)


def trash() -> QIcon:
    def d(p: QPainter):
        p.drawRect(QRectF(10, 11, 12, 15))
        p.drawLine(QPointF(7, 11), QPointF(25, 11))
        p.drawLine(QPointF(13, 8), QPointF(19, 8))
        p.drawLine(QPointF(14, 15), QPointF(14, 22))
        p.drawLine(QPointF(18, 15), QPointF(18, 22))
    return _icon(d)


def prev() -> QIcon:
    def d(p: QPainter):
        p.drawPolygon(QPolygonF([QPointF(20, 8), QPointF(10, 16), QPointF(20, 24)]))
    return _icon(d)


def next_() -> QIcon:
    def d(p: QPainter):
        p.drawPolygon(QPolygonF([QPointF(12, 8), QPointF(22, 16), QPointF(12, 24)]))
    return _icon(d)


def compare() -> QIcon:
    def d(p: QPainter):
        p.drawRect(QRectF(4, 7, 11, 18))
        p.drawRect(QRectF(17, 7, 11, 18))
        p.drawLine(QPointF(16, 4), QPointF(16, 28))
    return _icon(d)


def group() -> QIcon:
    def d(p: QPainter):
        for i, y in enumerate((7, 15, 23)):
            p.drawRect(QRectF(6, y, 20, 5))
    return _icon(d)


def export() -> QIcon:
    def d(p: QPainter):
        p.drawLine(QPointF(16, 5), QPointF(16, 20))
        _arrow_head(p, QPointF(16, 20), 90, 6)
        p.drawLine(QPointF(7, 24), QPointF(25, 24))
    return _icon(d)


def reset() -> QIcon:
    def d(p: QPainter):
        p.drawArc(QRectF(7, 7, 18, 18), 30 * 16, 270 * 16)
        _arrow_head(p, QPointF(23.5, 12.5), 60)
    return _icon(d)
