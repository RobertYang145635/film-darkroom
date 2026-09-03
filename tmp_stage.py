import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, "src")

from PySide6.QtWidgets import QApplication

from negative_lab.ui import theme
from negative_lab.ui.main_window import MainWindow

app = QApplication([])
theme.apply(app)
w = MainWindow()
print("construct OK", flush=True)
w._set_stage(0)
print("stage0 stack", w.right.stack.currentIndex(), flush=True)
w._set_stage(1)
print("stage1 stack", w.right.stack.currentIndex(), flush=True)
w._set_stage(0)
print("back0 stack", w.right.stack.currentIndex(), flush=True)
