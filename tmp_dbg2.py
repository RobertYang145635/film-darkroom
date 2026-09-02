import os
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, "src")
sys.path.insert(0, "tests/unit")

from PySide6.QtWidgets import QApplication

from negative_lab.testing.synthetic import make_synthetic_negative
from negative_lab.library.models import FrameRecord, GeometryRegions, ImportSession
from negative_lab.ui.main_window import MainWindow
from negative_lab.ui.preview_view import RoiKind

app = QApplication([])
w = MainWindow()
ran = []
w._refresh_preview = lambda *a, **k: ran.append(1)

t = make_synthetic_negative(seed=5, black_border=12)["negative"]
rec = FrameRecord(source_path="synthetic/smoke.tif", source_sha256="0" * 64,
                  input_facts={"abs_path": "synthetic/smoke.tif", "kind": "tiff"},
                  geometry=GeometryRegions())
w.session = ImportSession(files=[rec])
w.current = rec
w.t_source = t
w._prepare_frame_views()
print("prepare done, pr set:", w.pr is not None, flush=True)
w._refresh_preview = lambda *a, **k: ran.append(1)
w._on_roi_edited(RoiKind.BASE)
time.sleep(0.3)
app.processEvents()
print("ran after base edit:", ran, flush=True)
