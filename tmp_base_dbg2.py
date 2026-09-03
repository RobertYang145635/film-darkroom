import os
import sys
import time
import threading

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, "src")

from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from negative_lab.ui import theme
from negative_lab.ui.main_window import MainWindow
from negative_lab.ui.preview_view import RoiKind

app = QApplication([])
theme.apply(app)
win = MainWindow()
win.resize(1440, 900)
win.show()

from pathlib import Path

win.load_folder(Path('20260709_SAMPLE-ID(1)/1298_G200/翻拍'))
t0 = time.time()
while (win._busy._locks or threading.active_count() > 2) and time.time() - t0 < 150:
    app.processEvents(); time.sleep(0.05)

win.session.create_roll('卷', [f.source_path for f in win.session.files])
win.library.set_session(win.session)
win.library.select_roll(win.session.roll_groups[-1].id)
t0 = time.time()
while (win.current is None) and time.time() - t0 < 10:
    app.processEvents(); time.sleep(0.05)
t0 = time.time()
while (win._busy._locks or threading.active_count() > 2) and time.time() - t0 < 240:
    app.processEvents(); time.sleep(0.05)
app.processEvents()

vp = win.view.viewport()
print("viewport size:", vp.width(), vp.height(), flush=True)
print("scene:", win.view._scene_rect.width(), win.view._scene_rect.height(), flush=True)
print("mapToScene(80,620):", win.view.mapToScene(QPoint(80, 620)), flush=True)

win.right.sec_areas.area_action.emit("base")
print("draw mode:", win.view.draw_mode(), flush=True)


def drag(p0, p1):
    QTest.mousePress(vp, Qt.LeftButton, Qt.NoModifier, QPoint(*p0))
    app.processEvents()
    QTest.mouseMove(vp, QPoint(*p1))
    app.processEvents()
    print("  after press+move: drag_mode=", win.view._drag_mode,
          "draw_rect=", None if win.view._draw_rect is None else
          (round(win.view._draw_rect.x()), round(win.view._draw_rect.y()),
           round(win.view._draw_rect.width()), round(win.view._draw_rect.height())),
          flush=True)
    QTest.mouseRelease(vp, Qt.LeftButton, Qt.NoModifier, QPoint(*p1))
    app.processEvents()


drag((80, 620), (200, 700))
print("banner:", win._banner_text, flush=True)
print("base count after draw:", win.view.roi_count(RoiKind.BASE), flush=True)
