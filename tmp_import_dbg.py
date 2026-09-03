import os
import sys
import time
import threading

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, "src")

from PySide6.QtWidgets import QApplication

from negative_lab.ui import theme
from negative_lab.ui.main_window import MainWindow

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
app.processEvents()

print("A. after import: current=", None if win.current is None else Path(win.current.source_path).name, flush=True)
print("B. roll_groups:", [r.display_name for r in win.session.roll_groups], flush=True)
print("C. library.current_roll:", win.library.current_roll, flush=True)
print("D. library roll rows:", win.library.roll_list.count(),
      "currentRow:", win.library.roll_list.currentRow(), flush=True)

# 若 current 为 None，模拟用户点一次未分卷
if win.current is None:
    win.library.select_roll("__ungrouped__")
    t0 = time.time()
    while win.current is None and time.time() - t0 < 5:
        app.processEvents(); time.sleep(0.05)
    print("E. after manual select ungrouped:", None if win.current is None else Path(win.current.source_path).name, flush=True)
