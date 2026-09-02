import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, "src")

from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

import negative_lab.ui.main_window as mw
from negative_lab.testing.synthetic import make_synthetic_negative
from negative_lab.ui.main_window import MainWindow
from negative_lab.ui.preview_view import RoiKind
sys.path.insert(0, "tests/unit")
from test_v6_ui_contracts import _make_record, _session_with

app = QApplication([])
w = MainWindow()
w.show()
t = make_synthetic_negative(seed=5, black_border=12)["negative"]
rec = _make_record(t)
w.session = _session_with(rec)
w.current = rec
w.t_source = t
w._load_meta = {"kind": "tiff", "sha256": "0" * 64}
w._prepare_frame_views()
print("small", w._small.shape, "scene", w.view._scene_rect.width(),
      w.view._scene_rect.height(), flush=True)
print("roi(out)", w.view.roi(RoiKind.OUTPUT_CROP), flush=True)


def drag(p0, p1):
    vp = w.view.viewport()
    QTest.mousePress(vp, Qt.LeftButton, Qt.NoModifier, QPoint(*p0))
    QTest.mouseMove(vp, QPoint(*p1))
    QTest.mouseRelease(vp, Qt.LeftButton, Qt.NoModifier, QPoint(*p1))


w.view.set_draw_mode(RoiKind.IMAGE_AREA)
drag((30, 30), (200, 160))
print("after drag roi(area)", w.view.roi(RoiKind.IMAGE_AREA),
      "rec.area", rec.geometry.image_area_roi, flush=True)
