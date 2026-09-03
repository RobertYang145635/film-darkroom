import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, "src")

import numpy as np
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from negative_lab.ui.preview_view import PreviewView, RoiKind

app = QApplication([])
view = PreviewView()
view.resize(900, 700)
view.set_image(np.zeros((400, 600, 3), np.float32))


def drag(p0, p1):
    vp = view.viewport()
    QTest.mousePress(vp, Qt.LeftButton, Qt.NoModifier, QPoint(*p0))
    app.processEvents()
    QTest.mouseMove(vp, QPoint(*p1))
    app.processEvents()
    QTest.mouseRelease(vp, Qt.LeftButton, Qt.NoModifier, QPoint(*p1))
    app.processEvents()


view.set_draw_mode(RoiKind.BASE)
drag((10, 10), (100, 40))
print("after 1st:", view.roi_count(RoiKind.BASE), flush=True)
print("box1:", view.roi(RoiKind.BASE, 0), flush=True)
drag((10, 60), (100, 90))
print("after 2nd:", view.roi_count(RoiKind.BASE), flush=True)
if view.roi_count(RoiKind.BASE) > 1:
    print("box2:", view.roi(RoiKind.BASE, 1), flush=True)
print("banner-ish state: active=", view.active_roi(), flush=True)
