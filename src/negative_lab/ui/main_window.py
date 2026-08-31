"""单张工作流主窗口（规格 §5.1、§5.3 手动控制分组）。

布局：左侧预览（原图/正片切换 + 可拖动 ROI），右侧按技术顺序分组的参数，
底部直方图与警告/置信度面板。低置信度项必须可见（§4.5.8）。
大图在预览缩略上实时调整，导出走全分辨率分块（§9）。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PySide6.QtCore import QRectF, Qt
from PySide6.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
    QLabel, QListWidget, QMainWindow, QMessageBox, QPushButton, QScrollArea,
    QSlider, QVBoxLayout, QWidget)

from .. import __version__
from ..io.images import ReadError, load_linear_rgb, save_jpeg8, save_tiff16
from ..pipeline.auto import auto_analyze
from ..pipeline.density import invert, invert_chunked
from ..pipeline.params import InvertParams
from .histogram import HistogramWidget
from .preview_view import PreviewView, RoiKind

MAX_PREVIEW_DIM = 1600  # 实时预览降采样上限（导出用全分辨率）


class SliderRow(QWidget):
    """标签 + 滑块 + 数值框联动行。"""

    def __init__(self, label, lo, hi, val, step, onchange, decimals=3):
        super().__init__()
        self.lo, self.hi, self.onchange = lo, hi, onchange
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(QLabel(label))
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 1000)
        self.spin = QDoubleSpinBox()
        self.spin.setDecimals(decimals)
        self.spin.setRange(lo, hi)
        self.spin.setSingleStep(step)
        self.setValue(val)
        self.slider.valueChanged.connect(self._from_slider)
        self.spin.valueChanged.connect(self._from_spin)
        lay.addWidget(self.slider, 1)
        lay.addWidget(self.spin)
        self._updating = False

    def _to_int(self, v):
        return int(round((v - self.lo) / (self.hi - self.lo) * 1000))

    def setValue(self, v):
        self._updating = True
        v = float(np.clip(v, self.lo, self.hi))
        self.spin.setValue(v)
        self.slider.setValue(self._to_int(v))
        self._updating = False

    def value(self):
        return self.spin.value()

    def _from_slider(self, i):
        if self._updating:
            return
        self._updating = True
        self.spin.setValue(self.lo + i / 1000 * (self.hi - self.lo))
        self._updating = False
        self.onchange(self.spin.value())

    def _from_spin(self, v):
        if self._updating:
            return
        self._updating = True
        self.slider.setValue(self._to_int(v))
        self._updating = False
        self.onchange(v)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"负片去色罩 negative-lab v{__version__}")
        self.resize(1400, 900)
        self.src_path: Path | None = None
        self.export_dir: Path | None = None      # None -> 源文件同目录
        self.t_full: np.ndarray | None = None      # 全分辨率线性透射率
        self.t_prev: np.ndarray | None = None      # 预览分辨率
        self.params = InvertParams()
        self.confidence: dict = {}
        self.rotation = 0
        self.show_original = True

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        # ---- 左：预览 + 切换按钮
        left = QVBoxLayout()
        bar = QHBoxLayout()
        self.btn_open = QPushButton("打开文件…")
        self.btn_auto = QPushButton("自动转换")
        self.btn_auto.setEnabled(False)
        self.btn_orig = QPushButton("原图")
        self.btn_orig.setCheckable(True)
        self.btn_orig.setChecked(True)
        self.btn_prev = QPushButton("正片预览")
        self.btn_prev.setCheckable(True)
        self.btn_export = QPushButton("导出 TIFF16 + JPEG8 + 旁车")
        self.btn_export.setEnabled(False)
        for b in (self.btn_open, self.btn_auto, self.btn_orig, self.btn_prev,
                  self.btn_export):
            bar.addWidget(b)
        left.addLayout(bar)
        self.view = PreviewView()
        left.addWidget(self.view, 1)
        note = QLabel("JPEG 导出固定为 8 位；TIFF 为 16 位；原始文件只读不覆盖。")
        note.setStyleSheet("color:#aaa;")
        left.addWidget(note)
        root.addLayout(left, 2)

        # ---- 右：参数分组（§5.3 顺序）
        panel = QVBoxLayout()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        controls = QWidget()
        form = QVBoxLayout(controls)

        g1 = QGroupBox("1 胶片类型与配置")
        f1 = QFormLayout(g1)
        self.combo_film = QComboBox()
        self.combo_film.addItems(["unknown", "c41_color", "bw", "ecn2_color"])
        f1.addRow("胶片类型", self.combo_film)
        self.btn_profile = QPushButton("加载配置文件…（Phase 3）")
        self.btn_profile.setEnabled(False)
        f1.addRow(self.btn_profile)
        panel.addWidget(g1)

        g2 = QGroupBox("2 胶片区域与片基 Dmin")
        f2 = QVBoxLayout(g2)
        row = QHBoxLayout()
        self.btn_pick_base = QPushButton("片基框：拖动画面矩形可移动")
        self.btn_rebase = QPushButton("用当前片基框重新估计 Dmin")
        row.addWidget(self.btn_pick_base)
        f2.addLayout(row)
        f2.addWidget(self.btn_rebase)
        self.dmin_rows = []
        for name in ("R", "G", "B"):
            r = SliderRow(name, 1e-5, 1.0, 0.5, 0.001,
                          lambda v: self._dmin_changed(name, v), decimals=5)
            self.dmin_rows.append((name, r))
            f2.addWidget(r)
        panel.addWidget(g2)

        g3 = QGroupBox("3 Dmax 与扫描曝光偏置")
        f3 = QVBoxLayout(g3)
        self.s_dmax = SliderRow("Dmax", 0.3, 4.0, 2.0, 0.01, lambda v: self._param("dmax", v))
        self.s_offset = SliderRow("offset", -0.5, 1.0, 0.0, 0.01, lambda v: self._param("offset", v))
        f3.addWidget(self.s_dmax)
        f3.addWidget(self.s_offset)
        panel.addWidget(g3)

        g4 = QGroupBox("4 阴影色偏 / 高光白平衡 / 幂指数")
        f4 = QVBoxLayout(g4)
        rowp = QHBoxLayout()
        self.btn_pick_bright = QPushButton("取亮部中性点")
        self.btn_pick_dark = QPushButton("取暗部中性点")
        self.btn_clear_pts = QPushButton("清除参考点")
        for b in (self.btn_pick_bright, self.btn_pick_dark, self.btn_clear_pts):
            rowp.addWidget(b)
        f4.addLayout(rowp)
        self.wb_low_rows = [
            (n, SliderRow(f"wb_low_{n}", 0.2, 5.0, 1.0, 0.01,
                          lambda v, i=i: self._wb_changed("wb_low", i, v))
             ) for i, n in enumerate("RGB")]
        self.wb_high_rows = [
            (n, SliderRow(f"wb_high_{n}", 0.2, 5.0, 1.0, 0.01,
                          lambda v, i=i: self._wb_changed("wb_high", i, v))
             ) for i, n in enumerate("RGB")]
        for _, r in self.wb_low_rows + self.wb_high_rows:
            f4.addWidget(r)
        self.s_refexp = SliderRow("reference_exponent", 0.5, 2.0, 1.0, 0.01,
                                  lambda v: self._param("reference_exponent", v))
        self.s_redr = SliderRow("red_ratio", 0.5, 2.0, 1.0, 0.005,
                                lambda v: self._param("red_ratio", v))
        self.s_bler = SliderRow("blue_ratio", 0.5, 2.0, 1.0, 0.005,
                                lambda v: self._param("blue_ratio", v))
        for r in (self.s_refexp, self.s_redr, self.s_bler):
            f4.addWidget(r)
        panel.addWidget(g4)

        g5 = QGroupBox("5 相纸黑位 / 伽马 / 高光压缩 / 输出曝光")
        f5 = QVBoxLayout(g5)
        self.s_black = SliderRow("black", 0.0, 1.0, 0.0, 0.01, lambda v: self._param("black", v))
        self.s_gamma = SliderRow("gamma", 0.3, 3.0, 1.0, 0.01, lambda v: self._param("gamma", v))
        self.s_hl = SliderRow("highlight", 0.0, 1.0, 0.0, 0.01,
                              lambda v: self._param("highlight_strength", v))
        self.s_exp = SliderRow("exposure", 0.1, 4.0, 1.0, 0.01, lambda v: self._param("exposure", v))
        for r in (self.s_black, self.s_gamma, self.s_hl, self.s_exp):
            f5.addWidget(r)
        panel.addWidget(g5)

        g6 = QGroupBox("6 旋转")
        f6 = QHBoxLayout(g6)
        self.combo_rot = QComboBox()
        self.combo_rot.addItems(["0°", "90°", "180°", "270°"])
        f6.addWidget(QLabel("旋转（裁切 Phase 2 后续）"))
        f6.addWidget(self.combo_rot)
        panel.addWidget(g6)

        scroll.setWidget(controls)
        panel.addWidget(scroll, 1)

        # ---- 底部：直方图 + 警告
        bottom = QHBoxLayout()
        self.hist = HistogramWidget()
        bottom.addWidget(self.hist)
        self.warn_list = QListWidget()
        self.warn_list.setMinimumHeight(140)
        self.warn_list.setStyleSheet(
            "QListWidget{background:#241d10;color:#e8c87a;border:1px solid #554422;}")
        bottom.addWidget(self.warn_list, 1)
        panel.addLayout(bottom)
        root.addLayout(panel, 1)

        # ---- 信号
        self.btn_open.clicked.connect(self.open_file)
        self.btn_auto.clicked.connect(self.run_auto)
        self.btn_export.clicked.connect(self.export_all)
        self.btn_orig.clicked.connect(lambda: self._toggle(True))
        self.btn_prev.clicked.connect(lambda: self._toggle(False))
        self.btn_rebase.clicked.connect(self.rebase)
        self.btn_pick_bright.clicked.connect(lambda: self.view.set_pick_mode("bright"))
        self.btn_pick_dark.clicked.connect(lambda: self.view.set_pick_mode("dark"))
        self.btn_clear_pts.clicked.connect(self.view.clear_points)
        self.view.roi_changed.connect(self._roi_moved)
        self.view.point_picked.connect(lambda *_: self.run_auto())
        self.combo_rot.currentIndexChanged.connect(self._rotated)

        self.statusBar().showMessage("就绪：打开 RAW / 16 位 TIFF / DNG")

    # ---------------- 文件

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "打开负片", "",
            "支持 (*.NEF *.nef *.TIF *.tif *.TIFF *.tiff *.DNG *.dng *.PNG *.png);;所有文件 (*)")
        if path:
            self.load(Path(path))

    def load(self, path: Path):
        try:
            res = load_linear_rgb(path)
        except ReadError as e:
            QMessageBox.critical(self, "读取失败", str(e))
            return
        self.src_path = path
        self._t_unrot = res.rgb  # 未旋转原始数组（只读语义）
        self.t_full = self._rot(res.rgb)
        scale = max(1, int(np.ceil(max(self.t_full.shape[:2]) / MAX_PREVIEW_DIM)))
        self.t_prev = self.t_full[::scale, ::scale, :].copy()
        self.show_original = True
        self.btn_orig.setChecked(True)
        self.btn_prev.setChecked(False)
        self.btn_auto.setEnabled(True)
        self.btn_export.setEnabled(True)
        self._refresh()
        # 自动检测框（§5.1.2：打开即显示候选框）
        from ..calibration.border import detect_regions
        det = detect_regions(self.t_prev)
        self._det = det
        if det.get("frame"):
            y0, x0, y1, x1 = det["frame"]
            self.view.set_roi(RoiKind.FRAME, QRectF(x0, y0, x1 - x0, y1 - y0))
        if det.get("base"):
            y0, x0, y1, x1 = det["base"]
            self.view.set_roi(RoiKind.BASE, QRectF(x0, y0, x1 - x0, y1 - y0))
        bits = res.meta.get("bits", "?")
        self.statusBar().showMessage(
            f"{path.name} | {self.t_full.shape[1]}x{self.t_full.shape[0]} | {bits} 位 | "
            f"{res.meta.get('camera', '')}")
        self._set_warnings([f"元数据: {k}={v}" for k, v in res.meta.items()
                            if k in ("bits", "camera")] + res.warnings
                           + list(det.get("warnings", [])))

    def _rotated(self, idx):
        if getattr(self, "_t_unrot", None) is None:
            return
        self.rotation = idx * 90
        self.t_full = self._rot(self._t_unrot)
        scale = max(1, int(np.ceil(max(self.t_full.shape[:2]) / MAX_PREVIEW_DIM)))
        self.t_prev = self.t_full[::scale, ::scale, :].copy()
        self._refresh()

    def _rot(self, a: np.ndarray) -> np.ndarray:
        return np.rot90(a, k={0: 0, 90: 1, 180: 2, 270: 3}[self.rotation]).copy() \
            if self.rotation else a

    # ---------------- 自动与手动

    def run_auto(self):
        if self.t_prev is None:
            return
        base_roi = self._base_roi_from_view(self.t_prev)
        pts = self.view.points()
        nb = [pts["bright"]] if "bright" in pts else None
        nd = [pts["dark"]] if "dark" in pts else None
        nb = [(int(y), int(x)) for (x, y) in nb] if nb else None
        nd = [(int(y), int(x)) for (x, y) in nd] if nd else None
        rep = auto_analyze(self.t_prev, base_roi=base_roi,
                           neutral_bright_points=nb, neutral_dark_points=nd)
        self.params = rep.params
        self.confidence = rep.confidence
        self._sync_controls()
        self.show_original = False
        self.btn_orig.setChecked(False)
        self.btn_prev.setChecked(True)
        self._refresh()
        warns = [f"置信度 {k}={v}" for k, v in rep.confidence.items()] + rep.warnings
        self._set_warnings(warns)
        self.statusBar().showMessage("自动转换完成（预览分辨率计算；导出用全分辨率）")

    def rebase(self):
        if self.t_prev is None:
            return
        rect = self.view.roi(RoiKind.BASE)
        if rect is None:
            QMessageBox.information(self, "片基框", "请先在画面上确定黄色片基框")
            return
        x0, y0 = int(rect.x()), int(rect.y())
        x1, y1 = int(rect.right()), int(rect.bottom())
        from ..calibration.base import robust_dmin
        res = robust_dmin(self.t_prev[y0:y1, x0:x1, :])
        self.params.dmin = tuple(float(v) for v in res["dmin"])
        self.confidence["dmin"] = res["confidence"]
        self._sync_controls()
        self._refresh()
        self._set_warnings(res["warnings"] + [f"Dmin 置信度 {res['confidence']}"])

    def _base_roi_from_view(self, arr):
        rect = self.view.roi(RoiKind.BASE)
        if rect is None:
            return None
        x0, y0 = int(rect.x()), int(rect.y())
        x1, y1 = int(rect.right()), int(rect.bottom())
        if x1 - x0 < 4 or y1 - y0 < 4:
            return None
        return arr[y0:y1, x0:x1, :]

    def _roi_moved(self, kind, rect):
        if kind is RoiKind.BASE:
            self.rebase()

    # ---------------- 参数回调

    def _param(self, name, v):
        setattr(self.params, name, float(v))
        self._refresh()

    def _dmin_changed(self, name, v):
        i = "RGB".index(name)
        d = list(self.params.dmin)
        d[i] = float(v)
        self.params.dmin = tuple(d)
        self._refresh()

    def _wb_changed(self, which, i, v):
        cur = list(getattr(self.params, which))
        cur[i] = float(v)
        setattr(self.params, which, tuple(cur))
        self._refresh()

    def _sync_controls(self):
        p = self.params
        for (name, row), v in zip(self.dmin_rows, p.dmin):
            row.setValue(v)
        self.s_dmax.setValue(p.dmax)
        self.s_offset.setValue(p.offset)
        for (_, row), v in zip(self.wb_low_rows, p.wb_low):
            row.setValue(v)
        for (_, row), v in zip(self.wb_high_rows, p.wb_high):
            row.setValue(v)
        self.s_refexp.setValue(p.reference_exponent)
        self.s_redr.setValue(p.red_ratio)
        self.s_bler.setValue(p.blue_ratio)
        self.s_black.setValue(p.black)
        self.s_gamma.setValue(p.gamma)
        self.s_hl.setValue(p.highlight_strength)
        self.s_exp.setValue(p.exposure)

    # ---------------- 预览与导出

    def _refresh(self):
        if self.t_prev is None:
            return
        if self.show_original:
            self.view.set_image(self.t_prev)
            self.hist.set_image(self.t_prev)
        else:
            res = invert(self.t_prev, self.params)
            self.view.set_image(res.preview_srgb)
            self.hist.set_image(res.preview_srgb)

    def _toggle(self, orig: bool):
        self.show_original = orig
        self.btn_orig.setChecked(orig)
        self.btn_prev.setChecked(not orig)
        self._refresh()

    def export_all(self):
        if self.t_full is None or self.src_path is None:
            return
        # 裁切到帧框（P0 §2.1.5）：帧 ROI 以预览坐标存储，按比例映射到全分辨率
        img = self.t_full
        rect = self.view.roi(RoiKind.FRAME)
        if rect is not None and self.t_prev is not None:
            sy = self.t_full.shape[0] / self.t_prev.shape[0]
            sx = self.t_full.shape[1] / self.t_prev.shape[1]
            y0 = max(0, int(rect.y() * sy))
            x0 = max(0, int(rect.x() * sx))
            y1 = min(self.t_full.shape[0], int(rect.bottom() * sy))
            x1 = min(self.t_full.shape[1], int(rect.right() * sx))
            if y1 - y0 > 16 and x1 - x0 > 16:
                img = self.t_full[y0:y1, x0:x1, :]
        res = invert_chunked(img, self.params)
        outdir = self.export_dir or self.src_path.parent
        stem = self.src_path.stem
        tif = self._unique(outdir / f"{stem}-positive.tif")
        jpg = tif.with_suffix(".jpg")
        side = Path(str(tif) + ".negative.json")
        try:
            save_tiff16(tif, res.preview_srgb, sources=[self.src_path])
            save_jpeg8(jpg, res.preview_srgb, sources=[self.src_path])
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "导出失败", f"{type(e).__name__}: {e}\n源文件未改动")
            return
        rect = self.view.roi(RoiKind.BASE)
        roi = {"frame": [0, 0, self.t_full.shape[1], self.t_full.shape[0]],
               "film_base": ([int(rect.x()), int(rect.y()),
                              int(rect.width()), int(rect.height())]
                             if rect is not None else ["auto"])}
        import hashlib
        h = hashlib.sha256()
        with open(self.src_path, "rb") as f:
            for blk in iter(lambda: f.read(1 << 22), b""):
                h.update(blk)
        side.write_text(json.dumps({
            "schema_version": 1, "source_sha256": h.hexdigest(),
            "film_type": self.combo_film.currentText(),
            "device_profile_id": "none", "input_profile": "linear-normalized",
            "roi": roi, "parameters": self.params.to_parameters_dict(),
            "diagnostics": res.diagnostics | {"confidence": self.confidence},
            "software_version": __version__,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        self.statusBar().showMessage(f"导出完成：{tif.name} / {jpg.name}（JPEG 为 8 位）+ 旁车")

    @staticmethod
    def _unique(p: Path) -> Path:
        i = 1
        out = p
        while out.exists():
            out = p.with_name(f"{p.stem}-{i}{p.suffix}")
            i += 1
        return out

    def _set_warnings(self, msgs):
        self.warn_list.clear()
        for m in msgs:
            self.warn_list.addItem(f"• {m}")
