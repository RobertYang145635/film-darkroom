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

# 导入即预览验证
print("session files:", len(win.session.files), flush=True)
print("current after import:", None if win.current is None else win.current.source_path, flush=True)

win.session.create_roll('卷', [f.source_path for f in win.session.files])
win.library.set_session(win.session)
win.library.select_roll(win.session.roll_groups[-1].id)   # 应触发 _load_first_frame_of
t0 = time.time()
while (win.current is None) and time.time() - t0 < 10:
    app.processEvents(); time.sleep(0.05)
print("current after select_roll:", None if win.current is None else Path(win.current.source_path).name, flush=True)

t0 = time.time()
while (win._busy._locks or threading.active_count() > 2) and time.time() - t0 < 240:
    app.processEvents(); time.sleep(0.05)
app.processEvents()

# 场景与已有 base
print("scene:", win.view._scene_rect.width(), win.view._scene_rect.height(), flush=True)
print("base count before:", win.view.roi_count(RoiKind.BASE),
      "draft base:", win.current.geometry.base_sample_rois, flush=True)

# 模拟点 [取样] 然后拖一个框
win.right.sec_areas.area_action.emit("base")
print("draw mode:", win.view.draw_mode(), flush=True)


def drag(p0, p1):
    vp = win.view.viewport()
    QTest.mousePress(vp, Qt.LeftButton, Qt.NoModifier, QPoint(*p0))
    QTest.mouseMove(vp, QPoint(*p1))
    QTest.mouseRelease(vp, Qt.LeftButton, Qt.NoModifier, QPoint(*p1))


drag((80, 620), (200, 700))
print("base count after draw:", win.view.roi_count(RoiKind.BASE), flush=True)
print("draft base after draw:", win.current.geometry.base_sample_rois, flush=True)
t0 = time.time()
while (win._busy._locks or threading.active_count() > 2) and time.time() - t0 < 60:
    app.processEvents(); time.sleep(0.05)
app.processEvents()
print("base count after pipeline:", win.view.roi_count(RoiKind.BASE), flush=True)
print("draft base after pipeline:", win.current.geometry.base_sample_rois, flush=True)
