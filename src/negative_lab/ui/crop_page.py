"""裁切页（v5 §5）。

顶部紧凑图标工具栏 / 中央三类 ROI 画布 / 右侧裁切模式与同步范围 /
底部上一张·下一张与简短状态。页面只保留裁切所需控件——色彩与微调在
转换页，技术日志墙不在 P0。
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel, QPushButton, QSlider, QToolButton,
    QVBoxLayout, QWidget)

from ..pipeline.geometry import MAX_FINE_ROTATION
from . import icons
from .preview_view import PreviewView, RoiKind

CROP_MODES = ("仅画面", "保留片边")
SYNC_SCOPES = ("当前帧", "选中帧", "整卷")


class CropPage(QWidget):
    rotate_requested = Signal(int)           # -90 / +90
    mirror_requested = Signal()
    auto_crop_requested = Signal()
    redetect_requested = Signal()
    fit_requested = Signal()
    zoom100_requested = Signal()
    geometry_changed = Signal()               # 任意 ROI/方向变化（重算预览）
    roi_edited = Signal(object, int)          # RoiKind, index
    prev_requested = Signal()
    next_requested = Signal()
    sync_requested = Signal(str)             # 范围

    def __init__(self, parent=None):
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        # ---- 顶部工具栏（图标 + tooltip，不用长句按钮）
        bar = QHBoxLayout()
        self.btn_ccw = QToolButton()
        self.btn_ccw.setIcon(icons.rotate_ccw())
        self.btn_ccw.setToolTip("逆时针旋转 90 度")
        self.btn_cw = QToolButton()
        self.btn_cw.setIcon(icons.rotate_cw())
        self.btn_cw.setToolTip("顺时针旋转 90 度")
        self.btn_mirror = QToolButton()
        self.btn_mirror.setIcon(icons.mirror_h())
        self.btn_mirror.setToolTip("水平镜像")
        self.btn_autocrop = QToolButton()
        self.btn_autocrop.setIcon(icons.crop())
        self.btn_autocrop.setToolTip("自动裁切（三类区域一起重算）")
        self.btn_redetect = QToolButton()
        self.btn_redetect.setIcon(icons.target())
        self.btn_redetect.setToolTip("重新检测")
        self.btn_fit = QToolButton()
        self.btn_fit.setIcon(icons.zoom_fit())
        self.btn_fit.setToolTip("适合窗口")
        self.btn_100 = QToolButton()
        self.btn_100.setIcon(icons.zoom_100())
        self.btn_100.setToolTip("100% 检查")
        self.btn_add_base = QToolButton()
        self.btn_add_base.setIcon(icons.plus())
        self.btn_add_base.setToolTip("新增片基取样区")
        self.btn_del_base = QToolButton()
        self.btn_del_base.setIcon(icons.trash())
        self.btn_del_base.setToolTip("删除当前片基取样区")
        for b in (self.btn_ccw, self.btn_cw, self.btn_mirror, self.btn_autocrop,
                  self.btn_redetect, self.btn_fit, self.btn_100,
                  self.btn_add_base, self.btn_del_base):
            b.setAutoRaise(True)
            bar.addWidget(b)
        bar.addStretch(1)
        root.addLayout(bar)

        body = QHBoxLayout()
        self.view = PreviewView()
        body.addWidget(self.view, 1)

        # ---- 右侧：只保留裁切所需
        side = QVBoxLayout()
        side.setSpacing(6)
        f_small = QFont(self.font())
        f_small.setPointSize(max(7, f_small.pointSize() - 1))

        side.addWidget(_header("裁切模式", f_small))
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(CROP_MODES)
        self.combo_mode.setToolTip("“保留片边”时颜色分析仍只使用绿色有效画面区")
        side.addWidget(self.combo_mode)

        side.addWidget(_header("细微旋转", f_small))
        row = QHBoxLayout()
        self.slider_fine = QSlider(Qt.Horizontal)
        self.slider_fine.setRange(int(-MAX_FINE_ROTATION * 100),
                                  int(MAX_FINE_ROTATION * 100))
        self.slider_fine.setToolTip("小范围拉直（±10 度）")
        self.lbl_fine = QLabel("0.00°")
        self.lbl_fine.setFixedWidth(44)
        row.addWidget(self.slider_fine, 1)
        row.addWidget(self.lbl_fine)
        side.addLayout(row)

        side.addWidget(_header("同步范围", f_small))
        self.combo_sync = QComboBox()
        self.combo_sync.addItems(SYNC_SCOPES)
        self.combo_sync.setToolTip("把方向、镜像、细微旋转与三类 ROI 的"
                                   "归一化相对位置同步到选中帧或整卷")
        side.addWidget(self.combo_sync)
        self.btn_sync = QPushButton("同步到该范围")
        side.addWidget(self.btn_sync)

        self.lbl_roi = QLabel("当前对象：—")
        self.lbl_roi.setFont(f_small)
        side.addWidget(self.lbl_roi)
        side.addStretch(1)
        side_w = QWidget()
        side_w.setLayout(side)
        side_w.setFixedWidth(190)
        body.addWidget(side_w)
        root.addLayout(body, 1)

        # ---- 底部：上一张 / 下一张 / 简短状态
        foot = QHBoxLayout()
        self.btn_prev = QToolButton()
        self.btn_prev.setIcon(icons.prev())
        self.btn_prev.setToolTip("上一张")
        self.btn_next = QToolButton()
        self.btn_next.setIcon(icons.next_())
        self.btn_next.setToolTip("下一张")
        self.btn_prev.setAutoRaise(True)
        self.btn_next.setAutoRaise(True)
        foot.addWidget(self.btn_prev)
        foot.addWidget(self.btn_next)
        self.lbl_status = QLabel("就绪")
        self.lbl_status.setFont(f_small)
        foot.addWidget(self.lbl_status, 1)
        root.addLayout(foot)

        # ---- 信号
        self.btn_ccw.clicked.connect(lambda: self.rotate_requested.emit(-90))
        self.btn_cw.clicked.connect(lambda: self.rotate_requested.emit(90))
        self.btn_mirror.clicked.connect(self.mirror_requested)
        self.btn_autocrop.clicked.connect(self.auto_crop_requested)
        self.btn_redetect.clicked.connect(self.redetect_requested)
        self.btn_fit.clicked.connect(self.fit_requested)
        self.btn_100.clicked.connect(self.zoom100_requested)
        self.btn_add_base.clicked.connect(self._add_base)
        self.btn_del_base.clicked.connect(self._del_base)
        self.btn_prev.clicked.connect(self.prev_requested)
        self.btn_next.clicked.connect(self.next_requested)
        self.btn_sync.clicked.connect(
            lambda: self.sync_requested.emit(self.combo_sync.currentText()))
        self.combo_mode.currentIndexChanged.connect(self._mode_changed)
        self.slider_fine.valueChanged.connect(self._fine_changed)
        self.slider_fine.sliderReleased.connect(self.geometry_changed)
        self.view.roi_changed.connect(self._roi_changed)
        self.view.roi_selected.connect(self._roi_selected)

    # ---------------------------------------------------------------- 辅助
    def _add_base(self):
        r = self.view.roi(RoiKind.OUTPUT_CROP) or self.view.roi(RoiKind.IMAGE_AREA)
        if r is None:
            return
        rect = r.adjusted(4, 4, -max(8.0, r.width() * 0.1), -max(8.0, r.height() * 0.9))
        self.view.add_roi(RoiKind.BASE, rect)
        self.geometry_changed.emit()

    def _del_base(self):
        act = self.view.active_roi()
        idx = act[1] if (act and act[0] is RoiKind.BASE) \
            else max(0, self.view.roi_count(RoiKind.BASE) - 1)
        if self.view.remove_roi(RoiKind.BASE, idx):
            self.geometry_changed.emit()

    def _mode_changed(self, _i):
        self.geometry_changed.emit()

    def _fine_changed(self, v):
        self.lbl_fine.setText(f"{v / 100.0:+.2f}°")

    def _roi_changed(self, kind, index, _rect):
        self.roi_edited.emit(kind, index)
        self.geometry_changed.emit()

    def _roi_selected(self, kind, index):
        from .preview_view import ROI_STYLE
        label = ROI_STYLE[kind]["label"]
        suffix = f" {index + 1}" if kind is RoiKind.BASE else ""
        self.lbl_roi.setText(f"当前对象：{label}{suffix}")

    # ---------------------------------------------------------------- 状态
    def fine_rotation(self) -> float:
        return self.slider_fine.value() / 100.0

    def set_fine_rotation(self, deg: float):
        self.slider_fine.blockSignals(True)
        self.slider_fine.setValue(int(round(deg * 100)))
        self.slider_fine.blockSignals(False)
        self.lbl_fine.setText(f"{deg:+.2f}°")

    def keep_rebate(self) -> bool:
        return self.combo_mode.currentIndex() == 1

    def set_keep_rebate(self, on: bool):
        self.combo_mode.setCurrentIndex(1 if on else 0)

    def set_status(self, text: str):
        self.lbl_status.setText(text)


def _header(text: str, font: QFont) -> QLabel:
    lbl = QLabel(text)
    f = QFont(font)
    f.setBold(True)
    lbl.setFont(f)
    return lbl
