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
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QListWidget, QMainWindow, QMessageBox, QPushButton,
    QScrollArea, QSlider, QVBoxLayout, QWidget)

from .. import __version__
from ..color.management import OutputEncoding
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
        self.btn_batch = QPushButton("批处理…")
        for b in (self.btn_open, self.btn_auto, self.btn_orig, self.btn_prev,
                  self.btn_export, self.btn_batch):
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
        self.btn_profile = QPushButton("加载配置文件…")
        self.btn_profile.clicked.connect(self.load_profile)
        self.lbl_profile = QLabel("当前配置：无")
        f1.addRow(self.btn_profile)
        f1.addRow(self.lbl_profile)
        self.btn_save_regions = QPushButton("保存区域为标注（本图，用户为准）")
        self.btn_save_regions.clicked.connect(self.save_regions)
        f1.addRow(self.btn_save_regions)
        self.btn_calibration = QPushButton("加载卷级校准…")
        self.btn_calibration.clicked.connect(self.load_calibration)
        self.lbl_calibration = QLabel("当前校准：无（彩色负片将 needs_calibration）")
        f1.addRow(self.btn_calibration)
        f1.addRow(self.lbl_calibration)
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

        g6 = QGroupBox("6 旋转 / 裁切 / 除尘划痕（§7 可回退）")
        f6 = QHBoxLayout(g6)
        self.combo_rot = QComboBox()
        self.combo_rot.addItems(["0°", "90°", "180°", "270°"])
        f6.addWidget(QLabel("旋转"))
        f6.addWidget(self.combo_rot)
        lay6 = QVBoxLayout()
        self.chk_dust = QCheckBox("启用除尘修复（关闭即完全回退原图）")
        self.btn_paint = QPushButton("画笔涂蒙版")
        self.btn_paint.setCheckable(True)
        self.btn_dust_auto = QPushButton("自动检测(保守)")
        self.btn_dust_clear = QPushButton("清蒙版")
        row6b = QHBoxLayout()
        for b in (self.btn_paint, self.btn_dust_auto, self.btn_dust_clear):
            row6b.addWidget(b)
        lay6.addWidget(self.chk_dust)
        lay6.addLayout(row6b)
        f6.addLayout(lay6)
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
        self.btn_batch.clicked.connect(self.open_batch)
        self.btn_orig.clicked.connect(lambda: self._toggle(True))
        self.btn_prev.clicked.connect(lambda: self._toggle(False))
        self.btn_rebase.clicked.connect(self.rebase)
        self.btn_pick_bright.clicked.connect(lambda: self.view.set_pick_mode("bright"))
        self.btn_pick_dark.clicked.connect(lambda: self.view.set_pick_mode("dark"))
        self.btn_clear_pts.clicked.connect(self.view.clear_points)
        self.view.roi_changed.connect(self._roi_moved)
        self.view.point_picked.connect(lambda *_: self.run_auto())
        self.combo_rot.currentIndexChanged.connect(self._rotated)
        self.btn_paint.toggled.connect(self.view.set_paint_mode)
        self.btn_dust_auto.clicked.connect(self.dust_auto_detect)
        self.btn_dust_clear.clicked.connect(self.view.clear_mask)
        self.view.mask_painted.connect(self._refresh)
        self.chk_dust.toggled.connect(self._refresh)

        self.statusBar().showMessage("就绪：打开 RAW / 16 位 TIFF / DNG")

    def load_calibration(self):
        """v4 §5.1.2：GUI 卷级校准选择控件（损坏可见错误、状态显示）。"""
        from ..calibration.rollcalib import RollCalibration
        path, _ = QFileDialog.getOpenFileName(
            self, "选择卷级校准", "", "校准 (*.calib.json *.json)")
        if not path:
            return
        try:
            cal = RollCalibration.load(Path(path))
        except Exception as e:  # noqa: BLE001
            self._calibration = None
            self.lbl_calibration.setText("当前校准：无")
            QMessageBox.critical(self, "校准文件损坏/不兼容",
                                 f"{Path(path).name}\n{type(e).__name__}: {e}")
            return
        self._calibration = cal
        self.lbl_calibration.setText(
            f"当前校准：{cal.roll} [{cal.source}] verified={cal.verified} "
            f"({cal.film_type})")
        if self._t_unrot is not None:
            self.run_auto()

    def save_regions(self):
        """用户拖框 -> 保存为 verified 标注（v3 用户裁决：手动区域为主，
        批处理/重开文件都认这份框，算法不再猜）。"""
        if self._t_unrot is None or self.src_path is None:
            return
        frame = self._view_rect_oriented(RoiKind.FRAME)
        base = self._view_rect_oriented(RoiKind.BASE)
        if frame is None or base is None:
            QMessageBox.information(self, "保存区域",
                                    "需要同时框好画面区（绿）与片基区（黄）")
            return
        from ..calibration.annotations import Annotation, save_annotation
        from ..pipeline.geometry import oriented_to_source
        sh, sw = self._t_unrot.shape[:2]
        try:
            rel = self.src_path.resolve().relative_to(
                Path(__file__).resolve().parents[3]).as_posix()
        except ValueError:
            rel = self.src_path.as_posix()
        ann = Annotation(
            file=rel, roll=self.src_path.parent.parent.name,
            rotation_degrees=self.rotation,
            frame_roi=oriented_to_source(frame, sh, sw, self.rotation),
            base_roi=oriented_to_source(base, sh, sw, self.rotation),
            verified=True, verified_by="user",
            notes="GUI 用户框选")
        self._ann = ann
        p = save_annotation(rel, ann)
        self.statusBar().showMessage(f"区域已保存为 verified 标注：{p.name}")

    def load_profile(self):
        """单张配置加载（整改 I-8）：损坏必须可见错误，成功后显示并驱动先验。"""
        from ..profiles.profile import FilmProfile, load_profile as _lp
        path, _ = QFileDialog.getOpenFileName(
            self, "选择配置文件", "", "配置文件 (*.profile.json *.json)")
        if not path:
            return
        try:
            prof = _lp(path)
        except Exception as e:  # noqa: BLE001
            self._profile = None
            self.lbl_profile.setText("当前配置：无")
            QMessageBox.critical(self, "配置文件损坏/不兼容",
                                 f"{Path(path).name}\n{type(e).__name__}: {e}")
            return
        self._profile = prof
        self.lbl_profile.setText(
            f"当前配置：{prof.name} v{prof.version} [{prof.film_type}]")
        self.combo_film.setCurrentText(
            prof.film_type if prof.film_type in
            [self.combo_film.itemText(i) for i in range(self.combo_film.count())]
            else "unknown")
        if self._t_unrot is not None:
            self.run_auto()

    def open_batch(self):
        from PySide6.QtWidgets import QDialog, QDialogButtonBox
        from .batch_panel import BatchPanel

        dlg = QDialog(self)
        dlg.setWindowTitle("批处理队列")
        dlg.resize(900, 560)
        lay = QVBoxLayout(dlg)
        lay.addWidget(BatchPanel(dlg))
        bb = QDialogButtonBox(QDialogButtonBox.Close)
        bb.rejected.connect(dlg.reject)
        bb.accepted.connect(dlg.accept)
        lay.addWidget(bb)
        dlg.exec()

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
            msg = str(e)
            if "tiff_encoding" in msg:
                # 无 ICC/Gamma 的 TIFF：让用户显式声明传递函数假设（P0-B）
                box = QMessageBox(self)
                box.setWindowTitle("TIFF 传递函数未知")
                box.setText(msg + "\n\n请选择该文件的像素编码假设：")
                b_lin = box.addButton("线性 (linear)", QMessageBox.AcceptRole)
                b_srgb = box.addButton("sRGB", QMessageBox.AcceptRole)
                b_adobe = box.addButton("Adobe RGB", QMessageBox.AcceptRole)
                box.addButton("取消", QMessageBox.RejectRole)
                box.exec()
                choice = {b_lin: "linear", b_srgb: "srgb", b_adobe: "adobe"}.get(
                    box.clickedButton())
                if choice:
                    try:
                        res = load_linear_rgb(path, tiff_encoding=choice)
                    except ReadError as e2:
                        QMessageBox.critical(self, "读取失败", str(e2))
                        return
                else:
                    return
            else:
                QMessageBox.critical(self, "读取失败", msg)
                return
        self.src_path = path
        # ---- 换图必须重置全部图像相关状态（整改 I-1/F-1）
        self.rotation = 0
        self.combo_rot.blockSignals(True)
        self.combo_rot.setCurrentIndex(0)
        self.combo_rot.blockSignals(False)
        self.view.set_mask(None)
        self.view.set_roi(RoiKind.FRAME, None)
        self.view.set_roi(RoiKind.BASE, None)
        self.view.clear_points()
        self.params = InvertParams()
        self.confidence = {}
        self._pr = None
        self._t_unrot = res.rgb      # SOURCE 空间（只读语义）
        self._load_meta = res.meta
        # verified 标注驱动初始方向/ROI（整改 F-1：GUI 与批处理同一信息源）
        from ..calibration.annotations import load_annotation
        self._ann = None
        try:
            root = Path(__file__).resolve().parents[3]
            self._ann = load_annotation(path.resolve().relative_to(root).as_posix())
        except (ValueError, OSError):
            pass
        if self._ann is not None:
            self.rotation = int(self._ann.rotation_degrees)
            self.combo_rot.blockSignals(True)
            self.combo_rot.setCurrentIndex(self.rotation // 90)
            self.combo_rot.blockSignals(False)
        self._rebuild_oriented()
        self.show_original = True
        self.btn_orig.setChecked(True)
        self.btn_prev.setChecked(False)
        self.btn_auto.setEnabled(True)
        self.btn_export.setEnabled(True)
        self._refresh()
        bits = res.meta.get("bits", "?")
        self.statusBar().showMessage(
            f"{path.name} | {self.t_full.shape[1]}x{self.t_full.shape[0]} | {bits} 位 | "
            f"{res.meta.get('camera', '')}")
        self._set_warnings([f"元数据: {k}={v}" for k, v in res.meta.items()
                            if k in ("bits", "camera")] + list(res.warnings))

    def _rebuild_oriented(self):
        """旋转 + 预览 + 检测候选框（oriented 全分辨率坐标 -> 预览坐标）。"""
        from ..pipeline.geometry import rotate_cw
        self.t_full = rotate_cw(self._t_unrot, self.rotation)
        self._pscale = max(1, int(np.ceil(max(self.t_full.shape[:2]) / MAX_PREVIEW_DIM)))
        self.t_prev = self.t_full[::self._pscale, ::self._pscale, :].copy()
        from ..calibration.border import detect_regions
        det = detect_regions(self.t_prev)
        self._det = det
        ia = det.get("image_area")
        if ia:
            y0, x0, y1, x1 = ia
            self.view.set_roi(RoiKind.FRAME, QRectF(x0, y0, x1 - x0, y1 - y0))
        else:
            self.view.set_roi(RoiKind.FRAME, None)
        if det.get("base"):
            y0, x0, y1, x1 = det["base"]
            self.view.set_roi(RoiKind.BASE, QRectF(x0, y0, x1 - x0, y1 - y0))
        # verified 标注覆盖自动框（用户仍可拖动改变）
        ann = getattr(self, "_ann", None)
        if ann is not None and ann.verified:
            from ..pipeline.geometry import roi_source_to_oriented
            s = self._pscale
            sh, sw = self._t_unrot.shape[:2]
            for kind, roi in ((RoiKind.FRAME, ann.frame_roi), (RoiKind.BASE, ann.base_roi)):
                if roi:
                    oy0, ox0, oy1, ox1 = roi_source_to_oriented(
                        tuple(roi), sh, sw, self.rotation)
                    self.view.set_roi(kind, QRectF(ox0 // s, oy0 // s,
                                                   (ox1 - ox0) // s, (oy1 - oy0) // s))
        return det

    def _rotated(self, idx):
        if getattr(self, "_t_unrot", None) is None:
            return
        self.rotation = idx * 90   # 顺时针（geometry.rotate_cw 唯一实现）
        # 旋转后旧 ROI/点/蒙版不可靠：清空并重新检测（整改 I-1 允许的两案之一）
        self.view.clear_points()
        self.view.set_mask(None)
        self._pr = None
        det = self._rebuild_oriented()
        self._refresh()
        self._set_warnings(list(det.get("warnings", [])) +
                           ["旋转已应用：ROI/参考点/蒙版已重置，请重新检查"])

    # ---------------- 自动与手动（统一服务）

    def _view_rect_oriented(self, kind):
        rect = self.view.roi(kind)
        if rect is None:
            return None
        s = self._pscale
        x0, y0 = int(rect.x()) * s, int(rect.y()) * s
        x1, y1 = int(rect.right()) * s, int(rect.bottom()) * s
        if x1 - x0 < 16 or y1 - y0 < 16:
            return None
        return (y0, x0, y1, x1)

    def _run_service(self, force_params=None):
        """process_image 唯一入口（F-1：预览与导出同一管线）。"""
        from ..pipeline.service import process_image
        s = self._pscale
        pts = self.view.points()
        nb = [(int(pts["bright"][1]) * s, int(pts["bright"][0]) * s)] \
            if "bright" in pts else None
        nd = [(int(pts["dark"][1]) * s, int(pts["dark"][0]) * s)] \
            if "dark" in pts else None
        dust = None
        m = self.view.mask()
        if m is not None and m.any() and self.chk_dust.isChecked():
            dust = np.kron(m, np.ones((s, s), bool))[:self.t_full.shape[0], :self.t_full.shape[1]]
        prior = self._profile.to_initial_params() if getattr(self, "_profile", None) else None
        self._pr = process_image(
            self._t_unrot, rotation_degrees=self.rotation,
            ann=getattr(self, "_ann", None),
            frame_roi_oriented=self._view_rect_oriented(RoiKind.FRAME),
            base_roi_oriented=self._view_rect_oriented(RoiKind.BASE),
            prior=prior, neutral_bright_points=nb, neutral_dark_points=nd,
            dust_mask=dust, force_params=force_params,
            render_intent=getattr(self, "_intent", None),
            calibration=getattr(self, "_calibration", None),
            film_type=self.combo_film.currentText())
        self.params = self._pr.report.params
        self.confidence = self._pr.report.confidence
        return self._pr

    def run_auto(self):
        if self._t_unrot is None:
            return
        pr = self._run_service()
        self._sync_controls()
        self.show_original = False
        self.btn_orig.setChecked(False)
        self.btn_prev.setChecked(True)
        self._refresh()
        warns = [f"置信度 {k}={v}" for k, v in pr.report.confidence.items()]
        warns += [f"[{pr.gate.status.upper()}] {w}" for w in pr.gate.warnings]
        warns += list(pr.gate.failures)
        self._set_warnings(warns)
        self.statusBar().showMessage(
            f"自动转换完成 quality={pr.gate.status} crop={pr.geometry.crop_oriented}")

    def rebase(self):
        if self._t_unrot is None:
            return
        self.run_auto()   # 统一服务：拖框后重跑（不再本地重算 Dmin 造成双实现）

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
        # v4 §6.4：风格滑杆绑定 RenderIntent，不再改写技术参数
        intent_map = {"gamma": "grade", "exposure": "exposure",
                      "highlight_strength": "highlight_strength"}
        if name in intent_map:
            if getattr(self, "_intent", None) is None:
                from ..pipeline.render import RenderIntent
                self._intent = RenderIntent()
            setattr(self._intent, intent_map[name], float(v))
        else:
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
        intent = getattr(self, "_intent", None)
        from ..pipeline.render import RenderIntent
        it = intent or RenderIntent()
        self.s_gamma.setValue(it.grade)
        self.s_hl.setValue(it.highlight_strength)
        self.s_exp.setValue(it.exposure)

    # ---------------- 除尘（§7）

    def dust_auto_detect(self):
        if self.t_prev is None:
            return
        from ..restoration.dust import detect_dust
        region = None
        rect = self.view.roi(RoiKind.BASE)
        if rect is not None:
            region = np.zeros(self.t_prev.shape[:2], bool)
            y0, x0 = int(rect.y()), int(rect.x())
            y1, x1 = int(rect.bottom()), int(rect.right())
            region[y0:y1, x0:x1] = True
        det = detect_dust(self.t_prev, region=region)
        self.view.set_mask(det["mask"])
        self.chk_dust.setChecked(True)
        scope = "片基条带" if region is not None else "全画面（保守）"
        msg = (f"自动检测[{scope}]：{det['n_spots']} 斑点像素 / "
               f"{det['n_scratches']} 划痕像素（可再手动画笔增删；内容区建议手动）")
        self._set_warnings([msg] + det["warnings"])
        self.statusBar().showMessage(msg)

    def _dust_mask_full(self):
        """预览分辨率蒙版 -> 全分辨率（最近邻）。"""
        m = self.view.mask()
        if m is None or not m.any() or self.t_full is None:
            return None
        H, W = self.t_full.shape[:2]
        iy = (np.arange(H) * m.shape[0] // H).astype(np.int64)
        ix = (np.arange(W) * m.shape[1] // W).astype(np.int64)
        return m[np.ix_(iy, ix)]

    def _apply_dust(self, t: np.ndarray) -> tuple[np.ndarray, list[str]]:
        """启用时修复；失败回退原图并给警告——绝不阻断导出（§7）。"""
        if not self.chk_dust.isChecked():
            return t, []
        from ..restoration.dust import safe_inpaint
        m = self.view.mask()
        if m is None or not m.any():
            return t, []
        if t.shape[:2] != m.shape[:2]:
            H, W = t.shape[:2]
            iy = (np.arange(H) * m.shape[0] // H).astype(np.int64)
            ix = (np.arange(W) * m.shape[1] // W).astype(np.int64)
            m = m[np.ix_(iy, ix)]
        return safe_inpaint(t, m)

    # ---------------- 预览与导出（统一服务）

    def _refresh(self):
        if self.t_prev is None:
            return
        if self.show_original or self._pr is None:
            self.view.set_image(self.t_prev)
            self.hist.set_image(self.t_prev)
        else:
            # v4 §6：技术正片（服务已算，缓存的 result）+ RenderIntent 风格层；
            # 导出重新走全分辨率服务（同一 intent、同一管线）。
            from ..pipeline.render import RenderIntent, apply_render
            it = getattr(self, "_intent", None) or RenderIntent()
            tp = self._pr.result.positive_linear[::self._pscale, ::self._pscale][
                :self.t_prev.shape[0], :self.t_prev.shape[1]]
            view = apply_render(tp, it)
            from ..pipeline.density import encode_srgb
            self.view.set_image(encode_srgb(view))
            self.hist.set_image(encode_srgb(view))

    def _toggle(self, orig: bool):
        self.show_original = orig
        self.btn_orig.setChecked(orig)
        self.btn_prev.setChecked(not orig)
        self._refresh()

    def export_all(self):
        if self._t_unrot is None or self.src_path is None:
            return
        # F-1：导出走同一 process_image（全分辨率），禁止入口自算 crop
        pr = self._run_service(force_params=self.params)
        if pr.gate.status == "fail":
            QMessageBox.critical(
                self, "质量门禁阻断导出",
                "\n".join(pr.gate.failures) + "\n源文件未改动，未生成任何输出")
            return
        outdir = self.export_dir or self.src_path.parent
        stem = self.src_path.stem
        try:
            from ..color.management import icc_sha256, srgb_icc_bytes
            meta = dict(getattr(self, "_load_meta", {}) or {})
            meta["output_icc_sha256"] = icc_sha256(srgb_icc_bytes())
            payload = pr.sidecar(
                source_sha256=meta.get("sha256", ""),
                source_path=self.src_path.as_posix(),
                film_type=self.combo_film.currentText(),
                profile_id=(self._profile.profile_id
                            if getattr(self, "_profile", None) else "none"),
                input_meta=meta)
            from ..io.export import export_triplet
            written = export_triplet(outdir, stem, pr.result.positive_linear,
                                     payload, sources=[self.src_path])
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "导出失败（已回滚，无半套工件）",
                                 f"{type(e).__name__}: {e}\n源文件未改动")
            return
        self._last_export = [str(p) for p in written]
        warns = [f"[{pr.gate.status.upper()}] {w}" for w in pr.gate.warnings]
        if warns:
            self._set_warnings(warns)
        self.statusBar().showMessage(
            f"导出完成（事务提交）：{', '.join(p.name for p in written)} "
            f"quality={pr.gate.status}")

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
