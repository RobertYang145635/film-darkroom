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
view.show()
view.set_image(np.zeros((400, 600, 3), np.float32))

orig_press = view.mousePressEvent
orig_move = view.mouseMoveEvent
orig_release = view.mouseReleaseEvent


def press(ev):
    print(f"  PRESS pos={ev.position().toPoint()} draw_kind={view._draw_kind}",
          flush=True)
    orig_press(ev)
    print(f"    -> drag_mode={view._drag_mode} origin={view._draw_origin}",
          flush=True)


def move(ev):
    print(f'  MOVE raw scene={view.mapToScene(ev.position().toPoint())}', flush=True)
    orig_move(ev)
    print(f"  MOVE -> rect={view._draw_rect}", flush=True)


def release(ev):
    print(f"  RELEASE draw_mode={view._drag_mode} rect={view._draw_rect}",
          flush=True)
    orig_release(ev)
    print(f"    -> count={view.roi_count(RoiKind.BASE)}", flush=True)


view.mousePressEvent = press
view.mouseMoveEvent = move
view.mouseReleaseEvent = release


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
print("count:", view.roi_count(RoiKind.BASE), flush=True)
