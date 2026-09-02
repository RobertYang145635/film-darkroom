"""v6 重做：右栏面板栈（Lightroom「修改照片」式，深色）。

节自上而下：裁切与几何 / 基础调整(白平衡·影调·色彩) / 候选起点 /
技术信息（默认折叠）。全部是同一面板的普通节——没有"模式页"。
裁切节激活 = 照片进入裁切叠加态（由 MainWindow 接线），其余节仍可调。
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFrame, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QSlider, QToolButton, QVBoxLayout, QWidget)

from ..pipeline.geometry import MAX_FINE_ROTATION
from . import icons, theme
from .basic_adjustments import BasicAdjustmentsPanel

MIN_TARGET = 28
RATIO_CHOICES = ("自由", "3:2", "2:3", "1:1", "原比例")

_HEAD_QSS = f"""
QToolButton#SectionHead {{ border: none; text-align: left;
  font-size: 11px; font-weight: 600; letter-spacing: 2px;
  color: {theme.COLORS['text_dim']}; padding: 2px 0;
  background: transparent; }}
QToolButton#SectionHead:hover {{ color: {theme.COLORS['text']}; }}
"""


class Section(QFrame):
    """右栏的一个节：可折叠标题带 + 内容行。"""

    def __init__(self, title: str):
        super().__init__()
        self.setStyleSheet("QFrame { background: transparent; border: none; }")
        col = QVBoxLayout(self)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(6)

        self.btn_head = QToolButton()
        self.btn_head.setObjectName("SectionHead")
        self.btn_head.setText(title)
        self.btn_head.setCheckable(True)
        self.btn_head.setChecked(True)
        self.btn_head.setToolTip(f"展开/收起「{title}」")
        self.btn_head.setStyleSheet(_HEAD_QSS)
        self.btn_head.toggled.connect(self._on_toggle)
        head_row = QHBoxLayout()
        head_row.setSpacing(4)
        head_row.addWidget(self.btn_head)
        head_row.addStretch(1)
        self._head_extra = QHBoxLayout()
        self._head_extra.setSpacing(4)
        head_row.addLayout(self._head_extra)
        col.addLayout(head_row)

        self._body = QVBoxLayout()
        self._body.setContentsMargins(0, 0, 0, 0)
        self._body.setSpacing(6)
        self._body_widget = QWidget()
        self._body_widget.setLayout(self._body)
        col.addWidget(self._body_widget)

        line = QFrame()
        line.setObjectName("SectionLine")
        line.setFixedHeight(1)
        col.addWidget(line)

    def _on_toggle(self, on: bool):
        self._body_widget.setVisible(on)

    def _header_extra(self) -> QHBoxLayout:
        return self._head_extra

    def add(self, w: QWidget) -> QWidget:
        self._body.addWidget(w)
        return w

    def add_row(self, label: str, w: QWidget) -> QWidget:
        row = QHBoxLayout()
        row.setSpacing(8)
        lbl = QLabel(label)
        lbl.setProperty("caption", True)
        lbl.setMinimumWidth(40)
        row.addWidget(lbl)
        row.addWidget(w, 1)
        holder = QWidget()
        holder.setLayout(row)
        return self.add(holder)

    def add_tool(self, icon, tip: str) -> QToolButton:
        b = QToolButton()
        b.setIcon(icon)
        b.setToolTip(tip)
        b.setMinimumSize(MIN_TARGET, MIN_TARGET)
        return b


class CropSection(Section):
    """「裁切与几何」小工具节（默认收起；激活=照片进入裁切叠加态）。"""

    activate_requested = Signal(bool)
    rotate_requested = Signal(int)
    mirror_requested = Signal()
    auto_crop_requested = Signal()
    fine_changed = Signal(float)
    reset_geometry = Signal()
    show_base_toggled = Signal(bool)
    show_image_area_toggled = Signal(bool)

    def __init__(self):
        super().__init__("裁切与几何")
        self.btn_head.setChecked(False)      # 默认收起（Lightroom 式小工具）

        head = self._header_extra()
        self.btn_activate = QPushButton("裁切")
        self.btn_activate.setProperty("flat", True)
        self.btn_activate.setCheckable(True)
        self.btn_activate.setToolTip(
            "进入裁切叠加：照片框外压暗，拖框/八向手柄；完成或 Esc 退出")
        self.btn_activate.setMinimumHeight(24)
        self.btn_activate.clicked.connect(self._on_activate)
        head.addWidget(self.btn_activate)

        row = QHBoxLayout()
        row.setSpacing(4)
        self.btn_ccw = self.add_tool(icons.rotate_ccw(), "逆时针 90°")
        self.btn_cw = self.add_tool(icons.rotate_cw(), "顺时针 90°")
        self.btn_mirror = self.add_tool(icons.mirror_h(), "水平镜像")
        self.btn_auto = self.add_tool(icons.target(), "自动检测三类区域")
        for b in (self.btn_ccw, self.btn_cw, self.btn_mirror, self.btn_auto):
            row.addWidget(b)
        row.addStretch(1)
        hr = QWidget()
        hr.setLayout(row)
        self.add(hr)

        self.combo_ratio = QComboBox()
        self.combo_ratio.addItems(RATIO_CHOICES)
        self.combo_ratio.setToolTip("锁定裁切比例")
        self.add_row("比例", self.combo_ratio)

        self.slider_fine = QSlider(Qt.Horizontal)
        self.slider_fine.setRange(int(-MAX_FINE_ROTATION * 100),
                                  int(MAX_FINE_ROTATION * 100))
        self.slider_fine.setToolTip("细微旋转（拉直）")
        self.lbl_fine = QLabel("0.00°")
        self.lbl_fine.setProperty("caption", True)
        self.lbl_fine.setFixedWidth(40)
        rowf = QHBoxLayout()
        rowf.setSpacing(6)
        rowf.addWidget(self.slider_fine, 1)
        rowf.addWidget(self.lbl_fine)
        hf = QWidget()
        hf.setLayout(rowf)
        self.add_row("拉直", hf)

        self.chk_base = QCheckBox("显示片基取样区")
        self.chk_base.setChecked(True)
        self.add(self.chk_base)
        self.chk_area = QCheckBox("显示有效画面区")
        self.chk_area.setChecked(True)
        self.add(self.chk_area)

        self.lbl_state = QLabel("")
        self.lbl_state.setProperty("caption", True)
        self.lbl_state.setWordWrap(True)
        self.add(self.lbl_state)

        self.btn_ccw.clicked.connect(lambda: self.rotate_requested.emit(-90))
        self.btn_cw.clicked.connect(lambda: self.rotate_requested.emit(90))
        self.btn_mirror.clicked.connect(self.mirror_requested)
        self.btn_auto.clicked.connect(self.auto_crop_requested)
        self.slider_fine.valueChanged.connect(
            lambda v: self.lbl_fine.setText(f"{v / 100.0:+.2f}°"))
        self.slider_fine.sliderReleased.connect(
            lambda: self.fine_changed.emit(self.slider_fine.value() / 100.0))

    def _on_activate(self, on: bool):
        self.btn_activate.setChecked(on)
        self.activate_requested.emit(bool(on))

    def set_active(self, on: bool):
        self.btn_activate.blockSignals(True)
        self.btn_activate.setChecked(on)
        self.btn_activate.blockSignals(False)

    def set_fine_rotation(self, deg: float):
        self.slider_fine.blockSignals(True)
        self.slider_fine.setValue(int(round(deg * 100)))
        self.slider_fine.blockSignals(False)
        self.lbl_fine.setText(f"{deg:+.2f}°")

    def fine_rotation(self) -> float:
        return self.slider_fine.value() / 100.0


class CandidateSection(Section):
    """「候选起点」节：三选一 + 一句说明。"""

    candidate_chosen = Signal(str)

    def __init__(self):
        super().__init__("候选起点")
        row = QHBoxLayout()
        row.setSpacing(4)
        self._btns: dict[str, QPushButton] = {}
        for key, label in (("preserve", "保留氛围"), ("neutral", "中性"),
                           ("warm", "偏暖")):
            b = QPushButton(label)
            b.setCheckable(True)
            b.setMinimumHeight(24)
            b.clicked.connect(lambda _c=False, k=key: self.chosen.emit(k))
            self._btns[key] = b
            row.addWidget(b, 1)
        holder = QWidget()
        holder.setLayout(row)
        self.add(holder)
        self.lbl_rationale = QLabel("")
        self.lbl_rationale.setProperty("caption", True)
        self.lbl_rationale.setWordWrap(True)
        self.add(self.lbl_rationale)
        self.set_current("preserve")

    def set_current(self, name: str):
        for k, b in self._btns.items():
            b.setChecked(k == name)

    def set_confidence(self, name: str, conf: float):
        b = self._btns.get(name)
        if b is not None:
            b.setToolTip(f"{b.text()}：自动置信度 {conf:.2f}")

    def set_info(self, cset):
        cand = cset.get(cset.default_name)
        bits = []
        if cand.rationale:
            bits.append(cand.rationale)
        if cand.fallback_to:
            bits.append(f"已回退到“{cand.fallback_to}”")
        if cand.correction_capped:
            bits.append("修正已触发硬上限")
        self.lbl_rationale.setText("\n".join(bits))


class TechSection(Section):
    """「技术信息」节（默认折叠：片基来源/指纹/门禁/警告）。"""

    def __init__(self):
        super().__init__("技术信息")
        self.btn_head.setChecked(False)
        self.lbl_body = QLabel("—")
        self.lbl_body.setProperty("caption", True)
        self.lbl_body.setWordWrap(True)
        self.lbl_body.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.add(self.lbl_body)

    def set_info(self, *, frame: str, size: str, base_source: str,
                 gate: str, review: list[str], warnings: list[str],
                 fingerprint: str = ""):
        lines = [f"帧 {frame}   {size}", f"片基来源：{base_source}   门禁：{gate}"]
        if fingerprint:
            lines.append(f"采集指纹 {fingerprint[:12]}…")
        if review:
            lines.append("待复核：\n  " + "\n  ".join(review[:4]))
        if warnings:
            lines.append("警告：\n  " + "\n  ".join(warnings[:4]))
        self.lbl_body.setText("\n".join(lines))


class RightPanel(QWidget):
    """右栏：节堆栈（可滚动）。对外只发信号，数据流在 MainWindow。"""

    crop_activate = Signal(bool)
    rotate_requested = Signal(int)
    mirror_requested = Signal()
    auto_crop_requested = Signal()
    fine_changed = Signal(float)
    reset_geometry = Signal()
    show_base_toggled = Signal(bool)
    show_image_area_toggled = Signal(bool)
    candidate_chosen = Signal(str)
    tone_changed = Signal(str, float)
    tone_committed = Signal(str, float)
    sync_tone_requested = Signal(object)
    roi_edited = Signal(object, int)          # 透传（MainWindow 决定快/慢路径）

    def __init__(self, parent=None):
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        inner = QWidget()
        col = QVBoxLayout(inner)
        col.setContentsMargins(10, 8, 10, 8)
        col.setSpacing(12)

        self.sec_crop = CropSection()
        col.addWidget(self.sec_crop)

        self.sec_basic = Section("基础调整")
        self.panel = BasicAdjustmentsPanel()
        self.sec_basic.add(self.panel)
        col.addWidget(self.sec_basic)

        self.sec_cand = CandidateSection()
        col.addWidget(self.sec_cand)

        self.sec_tech = TechSection()
        col.addWidget(self.sec_tech)
        col.addStretch(1)

        scroll.setWidget(inner)
        outer.addWidget(scroll)

        # ---- 接线（面板内 -> 对外信号）
        self.sec_crop.activate_requested.connect(self.crop_activate)
        self.sec_crop.rotate_requested.connect(self.rotate_requested)
        self.sec_crop.mirror_requested.connect(self.mirror_requested)
        self.sec_crop.auto_crop_requested.connect(self.auto_crop_requested)
        self.sec_crop.fine_changed.connect(self.fine_changed)
        self.sec_crop.reset_geometry.connect(self.reset_geometry)
        self.sec_crop.chk_base.toggled.connect(self.show_base_toggled)
        self.sec_crop.chk_area.toggled.connect(self.show_image_area_toggled)
        self.sec_cand.candidate_chosen.connect(self.candidate_chosen)
        self.panel.changed.connect(self.tone_changed)
        self.panel.committed.connect(self.tone_committed)
        self.panel.sync_requested.connect(self.sync_tone_requested)
        self.sec_crop.fine_changed.connect(lambda _d: None)

    # ------------------------------------------------------------ 状态
    def set_crop_active(self, on: bool):
        self.sec_crop.set_active(on)

    def set_active_roi(self, kind, index: int = 0):
        self.sec_crop.set_active(True)

    def set_fine_rotation(self, deg: float):
        self.sec_crop.set_fine_rotation(deg)

    def set_keep_rebate(self, on: bool):
        pass          # 构图裁切已走缓存合成；导出始终按草稿裁切

    def keep_rebate(self) -> bool:
        return True   # 管线恒输出整幅正片；裁切由缓存合成完成

    def set_candidates(self, cset, chosen: str):
        self.sec_cand.set_current(chosen)
        for c in cset:
            self.sec_cand.set_confidence(c.name, c.confidence)
        self.sec_cand.set_info(cset)

    def set_tone_intent(self, intent, silent: bool = True):
        self.panel.set_intent(intent, silent=silent)

    def set_tech_info(self, **kw):
        self.sec_tech.set_info(**kw)
