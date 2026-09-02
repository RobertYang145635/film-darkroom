"""临时诊断：抓取当前 UI 实际渲染状态（导入后 / 裁切页 / 转换页）。"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from PySide6.QtWidgets import QApplication  # noqa: E402

from negative_lab.ui import theme  # noqa: E402
from negative_lab.ui.main_window import MainWindow  # noqa: E402


def wait_idle(win, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        app.processEvents()
        if not win._busy._locks and not threading.active_count() > 2:
            time.sleep(0.4)
            app.processEvents()
            return True
        time.sleep(0.05)
    return False


import threading  # noqa: E402

app = QApplication([])
theme.apply(app)
win = MainWindow()
win.resize(1440, 900)
win.show()
win.load_folder(ROOT / "20260709_SAMPLE-ID(1)" / "1298_G200" / "翻拍")
ok = wait_idle(win)
print("import idle:", ok, flush=True)
win.session.create_roll("卷 1298", [f.source_path for f in win.session.files])
win.library.set_session(win.session)
win.library.select_roll(win.session.roll_groups[-1].id)
wait_idle(win)
win.load_frame(win.session.files[0].source_path)
ok = wait_idle(win)
print("frame idle:", ok, flush=True)
app.processEvents()
out = ROOT / "outputs" / "v6_ui" / "diag"
out.mkdir(parents=True, exist_ok=True)
win.grab().save(str(out / "diag_crop.png"))

win._show_page(1)
app.processEvents()
time.sleep(0.3)
app.processEvents()
win.grab().save(str(out / "diag_convert.png"))
print("saved:", out / "diag_crop.png", out / "diag_convert.png", flush=True)
