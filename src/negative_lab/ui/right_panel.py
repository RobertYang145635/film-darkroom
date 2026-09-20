"""v6 竞品对齐：右栏 Stage 面板（分析 | 调整），Lightroom/ryan's darkroom 式。

Stage 模型（用户 2026-09-02 提供竞品视频后确认的方向）：
- 分析（Convert）：在**原负片**上完成三类区域任务——每个区 = 彩点 + 状态
  文本 + 动作按钮（框选/清除）+ 教学微文案；直方图在画面区就绪后显示；
- 调整（Edit）：在**转换结果**上微调——直方图实时、色罩校正（候选起点）、
  白平衡/影调/色彩滑杆、技术信息。

图像位置永远不动；Stage 只切换右栏内容与中央图内容。
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox, QFrame, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QSlider, QStackedWidget, QToolButton, QVBoxLayout,
    QWidget)

from ..pipeline.geometry import MAX_FINE_ROTATION
from . import icons, theme
from .basic_adjustments import BasicAdjustmentsPanel
from .histogram import HistogramWidget

MIN_TARGET = 28

_HEAD_QSS = f"""
QToolButton#SectionHead {{ border: none; text-align: left;
  font-size: 11px; font-weight: 600; letter-spacing: 2px;
  color: {theme.COLORS['text_dim']}; padding: 2px 0;
  background: transparent; }}
QToolButton#SectionHead:hover {{ color: {theme.COLORS['text']}; }}
"""

# 分析区定义：键 -> (点色 token, 标签, 未完成状态, 完成状态, 按钮文案)
ANALYSIS_AREAS = (
    ("image_area", theme.COLORS["area_image"], "有效画面", "未框选",
     "已框选", "框选"),
    ("base", theme.COLORS["area_base"], "片基 Dmin", "未取样",
     "已取样", "取样"),
    ("output_crop", theme.COLORS["area_crop"], "输出裁切", "未裁切",
     "已裁切", "框选"),
)


class Section(QFrame):
    """可折叠节：标题带 + 内容（1px 分隔线收尾）。"""

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
        self.btn_head.setStyleSheet(_HEAD_QSS)
        self.btn_head.toggled.connect(self._on_toggle)
        head = QHBoxLayout()
        head.setSpacing(4)
        head.addWidget(self.btn_head)
        head.addStretch(1)
        self._extra = QHBoxLayout()
        self._extra.setSpacing(4)
        head.addLayout(self._extra)
        col.addLayout(head)
        self._body_widget = QWidget()
        self._body = QVBoxLayout(self._body_widget)
        self._body.setContentsMargins(0, 0, 0, 0)
        self._body.setSpacing(6)
        col.addWidget(self._body_widget)
        line = QFrame()
        line.setObjectName("SectionLine")
        line.setFixedHeight(1)
        col.addWidget(line)

    def _on_toggle(self, on: bool):
        self._body_widget.setVisible(on)

    def _header_extra(self) -> QHBoxLayout:
        return self._extra

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


def _dot(color: str) -> QLabel:
    lbl = QLabel()
    lbl.setFixedSize(8, 8)
    lbl.setStyleSheet(f"background: {color}; border-radius: 4px;")
    return lbl


class _StageSwitch(QWidget):
    """Stage 分段控件（Convert/Edit 式）。"""

    changed = Signal(int)

    def __init__(self):
        super().__init__()
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        self._btns = []
        for i, name in enumerate(("分析", "调整")):
            b = QPushButton(name)
            b.setCheckable(True)
            b.setMinimumHeight(26)
            b.setChecked(i == 0)
            b.clicked.connect(lambda _c=False, k=i: self.changed.emit(k))
            lay.addWidget(b, 1)
            self._btns.append(b)

    def set_current(self, i: int):
        for k, b in enumerate(self._btns):
            b.setChecked(k == i)


class _AnalysisAreaRow(QWidget):
    """一个分析区行：彩点 + 名称 + 状态文本 + [清除][动作] 按钮。"""

    action_clicked = Signal(str)
    clear_clicked = Signal(str)

    def __init__(self, key, dot_color, name, btn_text):
        super().__init__()
        self.key = key
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
        self.dot = _dot(dot_color)
        lay.addWidget(self.dot)
        col = QVBoxLayout()
        col.setSpacing(0)
        lbl = QLabel(name)
        lbl.setStyleSheet("font-size: 12px;")
        self.lbl_state = QLabel("未完成")
        self.lbl_state.setProperty("caption", True)
        col.addWidget(lbl)
        col.addWidget(self.lbl_state)
        lay.addLayout(col, 1)
        self.btn_clear = QPushButton("清除")
        self.btn_clear.setProperty("flat", True)
        self.btn_clear.setVisible(False)
        self.btn_clear.clicked.connect(
            lambda: self.clear_clicked.emit(self.key))
        self.btn_action = QPushButton(btn_text)
        self.btn_action.setMinimumHeight(24)
        self.btn_action.clicked.connect(
            lambda: self.action_clicked.emit(self.key))
        lay.addWidget(self.btn_clear)
        lay.addWidget(self.btn_action)

    def set_state(self, done: bool, state_text: str, *, allow_clear: bool):
        self.lbl_state.setText(state_text)
        self.btn_clear.setVisible(allow_clear)


class AnalysisSection(Section):
    """「分析区」节：三行任务 + 每区说明微文案 + 几何工具 + 去调整按钮。"""

    area_action = Signal(str)          # "image_area"|"base"|"output_crop"
    area_clear = Signal(str)
    rotate_requested = Signal(int)
    mirror_requested = Signal()
    fine_changed = Signal(float)
    go_edit_requested = Signal()

    HELP = {
        "image_area": "框住单帧的画面内容（不含片边文字），"
                      "转换与质量统计只使用框内。",
        "base": "在未曝光的片基上取样（框选一小块均匀区域），"
                "这是去色罩的零点；避开画面、灰尘和文字。",
        "output_crop": "最终导出的构图框；不框则导出整个画面区。",
    }

    def __init__(self):
        super().__init__("分析区")
        self.rows: dict[str, _AnalysisAreaRow] = {}
        for key, color, name, _undone, _done, btn in ANALYSIS_AREAS:
            row = _AnalysisAreaRow(key, color, name, btn)
            row.action_clicked.connect(self.area_action)
            row.clear_clicked.connect(self.area_clear)
            self.rows[key] = row
            self.add(row)
            tip = QLabel(self.HELP[key])
            tip.setProperty("caption", True)
            tip.setWordWrap(True)
            self.add(tip)

        # ---- 方向工具（旋转 90 / 镜像 / 细微旋转拉直）
        rowg = QHBoxLayout()
        rowg.setSpacing(4)
        self.btn_ccw = self.add_tool(icons.rotate_ccw(), "逆时针旋转 90°")
        self.btn_cw = self.add_tool(icons.rotate_cw(), "顺时针旋转 90°")
        self.btn_mirror = self.add_tool(icons.mirror_h(), "水平镜像")
        for b in (self.btn_ccw, self.btn_cw, self.btn_mirror):
            rowg.addWidget(b)
        rowg.addStretch(1)
        hg = QWidget()
        hg.setLayout(rowg)
        self.add(hg)
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
        self.btn_ccw.clicked.connect(lambda: self.rotate_requested.emit(-90))
        self.btn_cw.clicked.connect(lambda: self.rotate_requested.emit(90))
        self.btn_mirror.clicked.connect(self.mirror_requested)
        self.slider_fine.valueChanged.connect(
            lambda v: self.lbl_fine.setText(f"{v / 100.0:+.2f}°"))
        self.slider_fine.sliderReleased.connect(
            lambda: self.fine_changed.emit(self.slider_fine.value() / 100.0))

        self.btn_auto = QPushButton("自动检测全部区域")
        self.btn_auto.setMinimumHeight(26)
        self.add(self.btn_auto)

        # ---- 分析完成 -> 去调整（显式阶段跳转，竞品式 Stage 切换动作）
        self.btn_go_edit = QPushButton("分析完成，去调整 →")
        self.btn_go_edit.setProperty("accent", True)
        self.btn_go_edit.setMinimumHeight(30)
        self.btn_go_edit.setToolTip("切换到调整阶段：查看去色罩结果并微调")
        self.btn_go_edit.clicked.connect(self.go_edit_requested)
        self.add(self.btn_go_edit)

    def set_state(self, key: str, *, has: bool, text: str,
                  allow_clear: bool = False):
        row = self.rows[key]
        row.set_state(has, text, allow_clear=allow_clear)

    def set_fine_rotation(self, deg: float):
        self.slider_fine.blockSignals(True)
        self.slider_fine.setValue(int(round(deg * 100)))
        self.slider_fine.blockSignals(False)
        self.lbl_fine.setText(f"{deg:+.2f}°")


class InputSection(Section):
    """「输入」节：只读元信息（竞品的 Film Type/Decoder 对应物）。"""

    def __init__(self):
        super().__init__("输入")
        self.lbl = QLabel("—")
        self.lbl.setProperty("caption", True)
        self.lbl.setWordWrap(True)
        self.lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.add(self.lbl)

    def set_info(self, *, file: str, camera: str, decoder: str,
                 bits: str, fingerprint: str = ""):
        lines = [f"文件 {file}"]
        if camera:
            lines.append(f"机身 {camera}")
        lines.append(f"解码 {decoder}   {bits}")
        if fingerprint:
            lines.append(f"指纹 {fingerprint[:12]}…")
        self.lbl.setText("\n".join(lines))


class SettingsSection(Section):
    """「转换设置」节：保留片边 + 自动检测。"""

    keep_rebate_toggled = Signal(bool)

    def __init__(self):
        super().__init__("转换设置")
        self.chk_rebate = QCheckBox("保留片边（导出时含胶片边框）")
        self.chk_rebate.setChecked(False)
        self.chk_rebate.toggled.connect(self.keep_rebate_toggled)
        self.add(self.chk_rebate)


class CandidateSection(QWidget):
    """「色罩校正」块（调整阶段顶部）：候选三选一 + 说明。"""

    candidate_chosen = Signal(str)

    def __init__(self):
        super().__init__()
        col = QVBoxLayout(self)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(6)
        row = QHBoxLayout()
        row.setSpacing(4)
        self._btns: dict[str, QPushButton] = {}
        for key, label in (("preserve", "保留氛围"), ("neutral", "中性"),
                           ("warm", "偏暖")):
            b = QPushButton(label)
            b.setCheckable(True)
            b.setMinimumHeight(24)
            b.clicked.connect(lambda _c=False, k=key: self.candidate_chosen.emit(k))
            self._btns[key] = b
            row.addWidget(b, 1)
        holder = QWidget()
        holder.setLayout(row)
        col.addWidget(holder)
        self.lbl_rationale = QLabel("")
        self.lbl_rationale.setProperty("caption", True)
        self.lbl_rationale.setWordWrap(True)
        col.addWidget(self.lbl_rationale)
        self.set_current("preserve")

    def set_current(self, name: str):
        for k, b in self._btns.items():
            b.setChecked(k == name)

    def set_confidence(self, name: str, conf: float):
        b = self._btns.get(name)
        if b is not None:
            b.setToolTip(f"{b.text()}：自动置信度 {conf:.2f}")


class TechSection(Section):
    """「技术信息」节（默认折叠）。"""

    def __init__(self):
        super().__init__("技术信息")
        self.btn_head.setChecked(False)
        self.lbl_body = QLabel("—")
        self.lbl_body.setProperty("caption", True)
        self.lbl_body.setWordWrap(True)
        self.lbl_body.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.add(self.lbl_body)

    def set_info(self, *, base_source: str, gate: str, review: list[str],
                 warnings: list[str], consistency: str = "",
                 evidence: str = ""):
        """`evidence`：门禁/片基估计器量到的**归因原话**（diagnostics.explain）。

        以前这一节只有"片基来源：generic_prior 门禁：fail"，用户看得见状态、
        看不见原因（2026-09-09：74/169 帧回退通用先验，真因是单通道散布；
        21/169 帧被剪裁挡导出，真因是采集端已丢失）。文案由纯函数生成，
        这里只负责贴上去。
        """
        lines = [f"片基来源：{base_source}   门禁：{gate}"]
        if consistency:
            lines.append(consistency)
        if evidence:
            lines.append("证据：\n  " + evidence.replace("\n", "\n  "))
        if review:
            lines.append("待复核：\n  " + "\n  ".join(review[:4]))
        if warnings:
            lines.append("警告：\n  " + "\n  ".join(warnings[:4]))
        self.lbl_body.setText("\n".join(lines))


class RightPanel(QWidget):
    """右栏：Stage 分段 + 两页节栈。对外只发信号，数据流在 MainWindow。"""

    stage_changed = Signal(int)
    area_action = Signal(str)
    area_clear = Signal(str)
    auto_detect_requested = Signal()
    keep_rebate_toggled = Signal(bool)
    pick_neutral_toggled = Signal(bool)      # R5b：中性点救援开/关
    undo_pick_requested = Signal()           # R5b：撤销上次拾点救援
    candidate_chosen = Signal(str)
    tone_changed = Signal(str, float)
    tone_committed = Signal(str, float)
    sync_tone_requested = Signal(object)
    rotate_requested = Signal(int)
    mirror_requested = Signal()
    fine_changed = Signal(float)
    reset_geometry = Signal()
    #: 「全部重置」微调。此前 BasicAdjustmentsPanel 只有 emit，全仓仅有的两处
    #: connect 都在已废弃的 crop_page/crop_panel 里 —— 在用面板上点了没反应
    #: （2026-09-08 死代码审计实证）。现在接到 MainWindow._reset_all_intent。
    intent_reset_requested = Signal()
    go_edit_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ---- Stage 分段（吸顶）
        stage_bar = QWidget()
        stage_lay = QVBoxLayout(stage_bar)
        stage_lay.setContentsMargins(10, 8, 10, 6)
        stage_lay.setSpacing(6)
        row = QHBoxLayout()
        row.addWidget(QLabel("阶段"), 0)
        self.stage = _StageSwitch()
        self.stage.changed.connect(self._on_stage)
        row.addWidget(self.stage, 1)
        stage_lay.addLayout(row)
        outer.addWidget(stage_bar)

        self.stack = QStackedWidget()
        outer.addWidget(self.stack, 1)

        # ===== Stage 0：分析 =====
        w0 = QWidget()
        c0 = QVBoxLayout(w0)
        c0.setContentsMargins(10, 0, 10, 8)
        c0.setSpacing(12)
        self.hist_convert = HistogramWidget()
        self.hist_convert.setMinimumHeight(110)
        c0.addWidget(self.hist_convert)
        self.sec_input = InputSection()
        c0.addWidget(self.sec_input)
        self.sec_areas = AnalysisSection()
        c0.addWidget(self.sec_areas)
        self.sec_settings = SettingsSection()
        c0.addWidget(self.sec_settings)
        c0.addStretch(1)
        scroll0 = QScrollArea()
        scroll0.setWidgetResizable(True)
        scroll0.setFrameShape(QScrollArea.NoFrame)
        scroll0.setWidget(w0)
        self.stack.addWidget(scroll0)

        # ===== Stage 1：调整 =====
        w1 = QWidget()
        c1 = QVBoxLayout(w1)
        c1.setContentsMargins(10, 0, 10, 8)
        c1.setSpacing(12)
        self.hist_edit = HistogramWidget()
        self.hist_edit.setMinimumHeight(110)
        c1.addWidget(self.hist_edit)

        self.sec_cast = Section("色罩校正")
        self.cand = CandidateSection()
        self.sec_cast.add(self.cand)
        # ---- R5b 中性点救援（痛点调研 #2：自动失败时"点一下灰就能救"）
        self.btn_pick_neutral = QPushButton("中性点救援（点画面里该是灰的地方）")
        self.btn_pick_neutral.setCheckable(True)
        self.btn_pick_neutral.setToolTip(
            "进入拾点模式后在画面上点击：该点会被拉向中性灰（立即生效）。\n"
            "点错地方会给出警告；Esc 退出；「撤销拾点」回到之前的状态。")
        self.btn_pick_neutral.setMinimumHeight(28)
        self.sec_cast.add(self.btn_pick_neutral)
        self.btn_undo_pick = QPushButton("撤销拾点")
        self.btn_undo_pick.setProperty("flat", True)
        self.btn_undo_pick.setMinimumHeight(28)
        self.btn_undo_pick.setEnabled(False)
        self.sec_cast.add(self.btn_undo_pick)
        self.lbl_pick_hint = QLabel("")
        self.lbl_pick_hint.setProperty("caption", True)
        self.lbl_pick_hint.setWordWrap(True)
        self.lbl_pick_hint.setVisible(False)
        self.sec_cast.add(self.lbl_pick_hint)
        c1.addWidget(self.sec_cast)

        self.sec_basic = Section("基础调整")
        self.panel = BasicAdjustmentsPanel()
        self.sec_basic.add(self.panel)
        c1.addWidget(self.sec_basic)

        self.sec_tech = TechSection()
        c1.addWidget(self.sec_tech)
        c1.addStretch(1)
        scroll1 = QScrollArea()
        scroll1.setWidgetResizable(True)
        scroll1.setFrameShape(QScrollArea.NoFrame)
        scroll1.setWidget(w1)
        self.stack.addWidget(scroll1)

        # ---- 接线
        self.sec_areas.area_action.connect(self.area_action)
        self.sec_areas.area_clear.connect(self.area_clear)
        self.sec_areas.btn_auto.clicked.connect(self.auto_detect_requested)
        self.sec_areas.rotate_requested.connect(self.rotate_requested)
        self.sec_areas.mirror_requested.connect(self.mirror_requested)
        self.sec_areas.fine_changed.connect(self.fine_changed)
        self.sec_areas.go_edit_requested.connect(self.go_edit_requested)
        self.sec_settings.keep_rebate_toggled.connect(self.keep_rebate_toggled)
        self.btn_pick_neutral.toggled.connect(self.pick_neutral_toggled)
        self.btn_undo_pick.clicked.connect(self.undo_pick_requested)
        self.cand.candidate_chosen.connect(self.candidate_chosen)
        self.panel.changed.connect(self.tone_changed)
        self.panel.committed.connect(self.tone_committed)
        self.panel.sync_requested.connect(self.sync_tone_requested)
        self.panel.reset_all_requested.connect(self.intent_reset_requested)

    def _on_stage(self, i: int):
        self.stack.setCurrentIndex(i)
        self.stage_changed.emit(i)

    def set_stage(self, i: int):
        """程序化切阶段：**不**回发 stage_changed（避免与 MainWindow 互调
        无限递归——2026-09-02 实测栈溢出教训）。"""
        self.stage.blockSignals(True)
        self.stage.set_current(i)
        self.stage.blockSignals(False)
        self.stack.setCurrentIndex(i)

    # ------------------------------------------------------------ 状态
    def set_area_state(self, key: str, *, has: bool, text: str,
                       allow_clear: bool = False):
        self.sec_areas.set_state(key, has=has, text=text,
                                 allow_clear=allow_clear)

    def set_input_info(self, **kw):
        self.sec_input.set_info(**kw)

    def set_tech_info(self, **kw):
        self.sec_tech.set_info(**kw)

    def set_candidates(self, cset, chosen: str):
        self.cand.set_current(chosen)
        for c in cset:
            self.cand.set_confidence(c.name, c.confidence)
        cand = cset.get(chosen)
        bits = []
        if cand is not None and cand.rationale:
            bits.append(cand.rationale)
        if cand is not None and cand.fallback_to:
            bits.append(f"已回退到“{cand.fallback_to}”")
        if cand is not None and cand.correction_capped:
            bits.append("修正已触发硬上限")
        self.cand.lbl_rationale.setText("\n".join(bits))

    def keep_rebate(self) -> bool:
        return self.sec_settings.chk_rebate.isChecked()

    def set_keep_rebate(self, on: bool):
        """v6.1 §4 公开 setter：帧加载/切换后反映 FrameRecord.keep_rebate。

        blockSignals 防回发（程序化恢复，不是用户点击）。
        """
        self.sec_settings.chk_rebate.blockSignals(True)
        self.sec_settings.chk_rebate.setChecked(bool(on))
        self.sec_settings.chk_rebate.blockSignals(False)

    # ------------------------------------------------------------ R5b 拾点救援
    def set_pick_mode_active(self, on: bool):
        """拾点模式开/关（view.pick_mode_changed 回来时同步按钮态）。"""
        self.btn_pick_neutral.blockSignals(True)
        self.btn_pick_neutral.setChecked(bool(on))
        self.btn_pick_neutral.blockSignals(False)

    def set_undo_pick_enabled(self, on: bool):
        self.btn_undo_pick.setEnabled(bool(on))

    def set_pick_hint(self, text: str):
        """拾点结果/警告（一级提示：警告常驻直到下一次拾点/撤销）。"""
        self.lbl_pick_hint.setText(text)
        self.lbl_pick_hint.setVisible(bool(text))
