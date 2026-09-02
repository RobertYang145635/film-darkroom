"""裁切页（v6 §5/§6 布局与交互契约）。

三列布局：左图库（主窗口持有）/ 中央原始负片查看区 / 右侧检查器。
右侧按 区域 / 几何 / 同步 三组组织（§5.6），当前绘制对象用紧凑分段选择
（图标 + tooltip，§5.2），创建矩形必须由用户在画布拖出。

视觉 token 全部来自 ui.theme；本文件不得出现十六进制颜色（契约锁定）。
本页只保留裁切所需控件；色彩与微调在转换页。
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
MIN_TARGET = 32
PRIMARY_TARGET = 36


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
    draw_mode_requested = Signal(object)     # RoiKind | None
    reset_kind_requested = Signal(object)    # RoiKind
    reset_all_requested = Signal()
    delete_active_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        from . import theme
        self._theme = theme
        root = QVBoxLayout(self)
        root.setContentsMargins(theme.SPACING["sm"], theme.SPACING["sm"],
                                theme.SPACING["sm"], 0)
        root.setSpacing(theme.SPACING["sm"])

        # ---- 中央查看区 + 右侧检查器
        body = QHBoxLayout()
        body.setSpacing(theme.SPACING["sm"])
        self.view = PreviewView()
        self.view.setObjectName("CropViewer")
        body.addWidget(self.view, 1)
        body.addWidget(self._build_inspector())
        root.addLayout(body, 1)

        # ---- 底部：单行 = 上一张/下一张 | 当前帧状态 | 轻提示（不重叠）
        foot = QHBoxLayout()
        foot.setSpacing(theme.SPACING["sm"])
        foot.setContentsMargins(0, 0, 0, 0)
        self.btn_prev = self._tool(icons.prev(), "上一张")
        self.btn_next = self._tool(icons.next_(), "下一张")
        foot.addWidget(self.btn_prev)
        foot.addWidget(self.btn_next)
        self.lbl_status = QLabel("就绪")
        self.lbl_status.setProperty("caption", True)
        self.lbl_status.setMinimumWidth(240)
        foot.addWidget(self.lbl_status, 0)
        self.banner = QLabel("")
        self.banner.setObjectName("Banner")
        self.banner.setProperty("level", "info")
        self.banner.setVisible(False)
        foot.addWidget(self.banner, 1)     # 提示吃掉剩余宽度，单行省略
        root.addLayout(foot)

        # ---- 信号
        self.btn_ccw.clicked.connect(lambda: self.rotate_requested.emit(-90))
        self.btn_cw.clicked.connect(lambda: self.rotate_requested.emit(90))
        self.btn_mirror.clicked.connect(self.mirror_requested)
        self.btn_autocrop.clicked.connect(self.auto_crop_requested)
        self.btn_redetect.clicked.connect(self.redetect_requested)
        self.btn_fit.clicked.connect(self.fit_requested)
        self.btn_100.clicked.connect(self.zoom100_requested)
        self.btn_del_roi.clicked.connect(self.delete_active_requested)
        self.btn_reset_kind.clicked.connect(self._emit_reset_kind)
        self.btn_reset_all.clicked.connect(self.reset_all_requested)
        self.btn_prev.clicked.connect(self.prev_requested)
        self.btn_next.clicked.connect(self.next_requested)
        self.btn_sync.clicked.connect(
            lambda: self.sync_requested.emit(self.combo_sync.currentText()))
        self.combo_mode.currentIndexChanged.connect(self._mode_changed)
        self.slider_fine.valueChanged.connect(self._fine_changed)
        self.slider_fine.sliderReleased.connect(self.geometry_changed)
        self.view.roi_changed.connect(self._roi_changed)
        self.view.roi_selected.connect(self._roi_selected)
        for kind in (RoiKind.OUTPUT_CROP, RoiKind.IMAGE_AREA, RoiKind.BASE):
            b = self._draw_buttons[kind]
            b.clicked.connect(lambda _c=False, k=kind: self._toggle_draw(k))

    # ------------------------------------------------------------- 检查器
    def _build_inspector(self) -> QWidget:
        t = self._theme
        side = QVBoxLayout()
        side.setContentsMargins(0, 0, 0, 0)
        side.setSpacing(t.SPACING["md"])

        # ---- 区域
        side.addWidget(self._header("区域"))
        row = QHBoxLayout()
        row.setSpacing(t.SPACING["xs"])
        self._draw_buttons = {}
        for kind, icon, tip in (
                (RoiKind.OUTPUT_CROP, icons.crop(), "绘制模式：输出裁切"
                 "（在画布拖出新框；重复拖动替换）"),
                (RoiKind.IMAGE_AREA, icons.target(), "绘制模式：有效画面"
                 "（在画布拖出新框；重复拖动替换）"),
                (RoiKind.BASE, icons.plus(), "绘制模式：片基取样"
                 "（每次拖动新增一个区域）")):
            b = self._tool(icon, tip)
            b.setCheckable(True)
            self._draw_buttons[kind] = b
            row.addWidget(b)
        row.addStretch(1)
        self.btn_del_roi = self._tool(icons.trash(), "删除当前选中对象（Delete）")
        row.addWidget(self.btn_del_roi)
        side.addLayout(row)
        self.lbl_draw = QLabel("选择上方类型后在画布拖出矩形；Esc 取消绘制")
        self.lbl_draw.setProperty("caption", True)
        self.lbl_draw.setWordWrap(True)
        side.addWidget(self.lbl_draw)
        self.lbl_roi = QLabel("当前对象：—")
        self.lbl_roi.setProperty("caption", True)
        side.addWidget(self.lbl_roi)
        side.addWidget(self._sep())
        row_r = QHBoxLayout()
        self.btn_reset_kind = QPushButton("重置当前类")
        self.btn_reset_kind.setObjectName("btn_reset_kind")
        self.btn_reset_kind.setToolTip("清空当前选中类型的区域"
                                       "（不影响其他两类）")
        self.btn_reset_all = QPushButton("重置全部几何")
        self.btn_reset_all.setObjectName("btn_reset_all")
        self.btn_reset_all.setToolTip("清空输出裁切、有效画面和全部片基区")
        for b in (self.btn_reset_kind, self.btn_reset_all):
            b.setMinimumHeight(PRIMARY_TARGET)
        row_r.addWidget(self.btn_reset_kind)
        row_r.addWidget(self.btn_reset_all)
        side.addLayout(row_r)

        # ---- 几何
        side.addWidget(self._sep())
        side.addWidget(self._header("几何"))
        tools = QHBoxLayout()
        tools.setSpacing(t.SPACING["xs"])
        self.btn_ccw = self._tool(icons.rotate_ccw(), "逆时针旋转 90 度")
        self.btn_cw = self._tool(icons.rotate_cw(), "顺时针旋转 90 度")
        self.btn_mirror = self._tool(icons.mirror_h(), "水平镜像")
        self.btn_fit = self._tool(icons.zoom_fit(), "适合窗口")
        self.btn_100 = self._tool(icons.zoom_100(), "100% 检查")
        for b in (self.btn_ccw, self.btn_cw, self.btn_mirror,
                  self.btn_fit, self.btn_100):
            tools.addWidget(b)
        tools.addStretch(1)
        side.addLayout(tools)
        rowf = QHBoxLayout()
        rowf.setSpacing(t.SPACING["sm"])
        self.slider_fine = QSlider(Qt.Horizontal)
        self.slider_fine.setRange(int(-MAX_FINE_ROTATION * 100),
                                  int(MAX_FINE_ROTATION * 100))
        self.slider_fine.setToolTip("小范围拉直（±10 度）")
        self.lbl_fine = QLabel("0.00°")
        self.lbl_fine.setProperty("caption", True)
        self.lbl_fine.setFixedWidth(44)
        rowf.addWidget(QLabel("细微旋转"), 0)
        rowf.addWidget(self.slider_fine, 1)
        rowf.addWidget(self.lbl_fine)
        side.addLayout(rowf)
        rowm = QHBoxLayout()
        rowm.setSpacing(t.SPACING["sm"])
        rowm.addWidget(QLabel("输出构图"))
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(CROP_MODES)
        self.combo_mode.setToolTip("“保留片边”时颜色分析仍只使用有效画面区")
        rowm.addWidget(self.combo_mode, 1)
        side.addLayout(rowm)
        rowa = QHBoxLayout()
        rowa.setSpacing(t.SPACING["xs"])
        rowa.addWidget(QLabel("自动检测"))
        self.btn_autocrop = self._tool(icons.crop(), "自动裁切（三类区域重算）")
        self.btn_redetect = self._tool(icons.target(), "重新检测")
        rowa.addWidget(self.btn_autocrop)
        rowa.addWidget(self.btn_redetect)
        rowa.addStretch(1)
        side.addLayout(rowa)

        # ---- 同步
        side.addWidget(self._sep())
        side.addWidget(self._header("同步"))
        rows = QHBoxLayout()
        rows.setSpacing(t.SPACING["sm"])
        self.combo_sync = QComboBox()
        self.combo_sync.addItems(SYNC_SCOPES)
        self.combo_sync.setToolTip("把方向、镜像、细微旋转与三类 ROI 的"
                                   "归一化相对位置同步到选中帧或整卷")
        rows.addWidget(self.combo_sync, 1)
        self.btn_sync = QPushButton("同步")
        self.btn_sync.setObjectName("btn_sync")
        self.btn_sync.setProperty("accent", True)
        self.btn_sync.setMinimumHeight(PRIMARY_TARGET)
        self.btn_sync.setToolTip("按所选范围同步几何")
        rows.addWidget(self.btn_sync)
        side.addLayout(rows)

        side.addStretch(1)
        inner = QWidget()
        inner.setLayout(side)
        from PySide6.QtWidgets import QScrollArea

        scroll = QScrollArea()
        scroll.setWidget(inner)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setMinimumWidth(self._theme.SIZES["inspector_w"] - 60)
        scroll.setMaximumWidth(self._theme.SIZES["inspector_w"] + 60)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        return scroll

    def set_banner(self, text: str, level: str = "info"):
        """轻量 inline 提示（v6 §5.6.7）：单行、省略，不用模态弹窗。"""
        self.banner.setText(text)
        self.banner.setProperty("level", level)
        self.banner.setStyleSheet("")          # 触发属性样式重算
        self.banner.setVisible(bool(text))

    # ------------------------------------------------------------- 工具
    def _tool(self, icon, tooltip: str) -> QToolButton:
        b = QToolButton()
        b.setIcon(icon)
        b.setToolTip(tooltip)
        b.setMinimumSize(MIN_TARGET, MIN_TARGET)
        b.setIconSize(b.iconSize())
        return b

    def _header(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setProperty("groupTitle", True)
        return lbl

    def _sep(self) -> QWidget:
        line = QWidget()
        line.setFixedHeight(1)
        line.setStyleSheet(
            f"background: {self._theme.COLORS['separator']};")
        return line

    # ------------------------------------------------------------- 绘制
    def _toggle_draw(self, kind: RoiKind):
        if self.view.draw_mode() is kind:
            self.set_draw_mode(None)
        else:
            self.set_draw_mode(kind)

    def set_draw_mode(self, kind: RoiKind | None):
        self.view.set_draw_mode(kind)
        for k, b in self._draw_buttons.items():
            b.blockSignals(True)
            b.setChecked(k is kind)
            b.blockSignals(False)
        if kind is RoiKind.OUTPUT_CROP:
            self.lbl_draw.setText("绘制模式：输出裁切 —— 在画布拖出新框；"
                                  "重复拖动替换；Esc 取消")
        elif kind is RoiKind.IMAGE_AREA:
            self.lbl_draw.setText("绘制模式：有效画面 —— 在画布拖出新框；"
                                  "重复拖动替换；Esc 取消")
        elif kind is RoiKind.BASE:
            self.lbl_draw.setText("绘制模式：片基取样 —— 每次拖动新增一个；"
                                  "Delete 删除选中项；Esc 取消")
        else:
            self.lbl_draw.setText("选择上方类型后在画布拖出矩形；Esc 取消绘制")

    def set_banner(self, text: str, level: str = "info"):
        """轻量 inline 提示（v6 §5.6.7：不用模态弹窗打断框选）。"""
        self.banner.setText(text)
        self.banner.setProperty("level", level)
        self.banner.setStyleSheet("")          # 触发属性样式重算
        self.banner.setVisible(bool(text))

    def clear_banner(self):
        self.set_banner("")

    def _emit_reset_kind(self):
        act = self.view.active_roi() or (
            self.view.draw_mode(), 0) if self.view.draw_mode() else \
            self.view.active_roi()
        kind = act[0] if act else RoiKind.OUTPUT_CROP
        self.reset_kind_requested.emit(kind)

    # ------------------------------------------------------------- 辅助
    def _add_base(self):
        raise NotImplementedError  # v6：新增片基必须由用户拖出（§5.2）

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

    # ------------------------------------------------------------- 状态
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
