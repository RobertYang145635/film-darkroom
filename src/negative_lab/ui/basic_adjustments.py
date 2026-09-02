"""Lightroom 式基础微调面板（v5 §2.2 P0 微调面板）。

只实现 P0 控件；交互要求逐条落实：

- 每个控件有滑杆、数值输入和**单项重置**；
- 面板提供**全部重置**；
- 支持**复制/粘贴**微调；
- 支持**只同步用户勾选的项目**到当前卷或当前选中帧；
- 所有微调进入 `RenderIntent`，技术正片不变化；
- 拖动中用预览分辨率实时更新，松开滑杆后触发高质量预览。

密度参考 Lightroom 基础调整区：标题与字号紧凑，不使用大卡片或教学文字。
"""

from __future__ import annotations

import copy

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QAbstractSpinBox, QCheckBox, QComboBox, QDoubleSpinBox, QGridLayout,
    QHBoxLayout, QLabel, QPushButton, QSlider, QToolButton, QVBoxLayout, QWidget)

from ..pipeline.render import P0_FIELDS, RANGES, RenderIntent
from . import icons

#: 分组与显示名（顺序即 §7.2 的应用顺序）
GROUPS: list[tuple[str, list[tuple[str, str]]]] = [
    ("白平衡", [("temperature", "色温"), ("tint", "色调")]),
    ("影调", [("exposure", "曝光"), ("contrast", "对比度"),
              ("highlights", "高光"), ("shadows", "阴影"),
              ("whites", "白色"), ("blacks", "黑色")]),
    ("色彩", [("vibrance", "自然饱和度"), ("saturation", "饱和度")]),
]

_STEPS_PER_UNIT = 1000.0


class AdjustmentRow(QWidget):
    """一行：标签 + 滑杆 + 数值输入 + 单项重置。"""

    value_changed = Signal(str, float)         # 拖动中（预览分辨率）
    value_committed = Signal(str, float)       # 松开（高质量）

    def __init__(self, field: str, label: str, parent=None):
        super().__init__(parent)
        self.field = field
        lo, hi, step = RANGES[field]
        self.lo, self.hi, self.step = lo, hi, step
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        lbl = QLabel(label)
        lbl.setMinimumWidth(58)
        f = QFont(lbl.font())
        f.setPointSize(max(7, f.pointSize() - 1))
        lbl.setFont(f)
        lay.addWidget(lbl)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(int(lo * _STEPS_PER_UNIT), int(hi * _STEPS_PER_UNIT))
        self.slider.setSingleStep(int(step * _STEPS_PER_UNIT))
        self.spin = QDoubleSpinBox()
        self.spin.setDecimals(2 if step < 0.1 else 1)
        self.spin.setRange(lo, hi)
        self.spin.setSingleStep(step)
        self.spin.setKeyboardTracking(False)
        self.spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.spin.setFixedWidth(52)
        self.btn_reset = QToolButton()
        self.btn_reset.setIcon(icons.reset())
        self.btn_reset.setAutoRaise(True)
        self.btn_reset.setToolTip(f"重置{label}")
        lay.addWidget(self.slider, 1)
        lay.addWidget(self.spin)
        lay.addWidget(self.btn_reset)
        self._updating = False
        self.slider.valueChanged.connect(self._from_slider)
        self.slider.sliderReleased.connect(self._released)
        self.spin.valueChanged.connect(self._from_spin)
        self.btn_reset.clicked.connect(self.reset)
        self._pending = 0.0

    # ----
    def _to_int(self, v: float) -> int:
        return int(round(v * _STEPS_PER_UNIT))

    def _to_val(self, i: int) -> float:
        return i / _STEPS_PER_UNIT

    def setValue(self, v: float):              # noqa: N802（Qt 风格）
        self._updating = True
        v = float(max(self.lo, min(self.hi, float(v))))
        self.spin.setValue(v)
        self.slider.setValue(self._to_int(v))
        self._updating = False
        self._pending = v

    def value(self) -> float:
        return float(self.spin.value())

    def _from_slider(self, i: int):
        if self._updating:
            return
        self._updating = True
        v = self._to_val(i)
        self.spin.setValue(v)
        self._updating = False
        self._pending = v
        self.value_changed.emit(self.field, v)

    def _released(self):
        if self._updating:
            return
        self.value_committed.emit(self.field, self._pending)

    def _from_spin(self, v: float):
        if self._updating:
            return
        self._updating = True
        self.slider.setValue(self._to_int(v))
        self._updating = False
        self._pending = float(v)
        self.value_changed.emit(self.field, float(v))
        self.value_committed.emit(self.field, float(v))

    def reset(self):
        self.setValue(0.0)
        self.value_changed.emit(self.field, 0.0)
        self.value_committed.emit(self.field, 0.0)


class BasicAdjustmentsPanel(QWidget):
    """P0 基础微调面板。"""

    changed = Signal(str, float)            # field, value（预览刷新）
    committed = Signal(str, float)          # field, value（高质量刷新）
    reset_all_requested = Signal()
    sync_requested = Signal(object)         # list[str] 勾选的字段

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.setSpacing(6)
        self.rows: dict[str, AdjustmentRow] = {}
        self._sync_checks: dict[str, QCheckBox] = {}

        head = QHBoxLayout()
        title = QLabel("基础调整")
        tf = QFont(title.font())
        tf.setBold(True)
        tf.setPointSize(max(8, tf.pointSize() - 1))
        title.setFont(tf)
        head.addWidget(title)
        head.addStretch(1)
        self.btn_reset_all = QPushButton("全部重置")
        self.btn_copy = QPushButton("复制")
        self.btn_paste = QPushButton("粘贴")
        for b in (self.btn_reset_all, self.btn_copy, self.btn_paste):
            f = QFont(b.font())
            f.setPointSize(max(7, f.pointSize() - 1))
            b.setFont(f)
            head.addWidget(b)
        lay.addLayout(head)
        self.btn_reset_all.setToolTip("把所有微调恢复为恒等（不改变技术正片）")
        self.btn_copy.setToolTip("复制当前微调参数")
        self.btn_paste.setToolTip("粘贴已复制的微调参数")

        sync_row = QHBoxLayout()
        sync_lbl = QLabel("同步勾选项到")
        f = QFont(sync_lbl.font())
        f.setPointSize(max(7, f.pointSize() - 1))
        sync_lbl.setFont(f)
        sync_row.addWidget(sync_lbl)
        self.combo_sync_scope = QComboBox()
        self.combo_sync_scope.addItems(["当前帧", "选中帧", "整卷"])
        sync_row.addWidget(self.combo_sync_scope, 1)
        self.btn_sync = QPushButton("同步")
        sync_row.addWidget(self.btn_sync)
        lay.addLayout(sync_row)

        grid = QGridLayout()
        grid.setHorizontalSpacing(4)
        grid.setVerticalSpacing(1)
        grid.setColumnStretch(1, 1)
        r = 0
        for gname, items in GROUPS:
            gl = QLabel(gname)
            gf = QFont(gl.font())
            gf.setBold(True)
            gf.setPointSize(max(7, gf.pointSize() - 1))
            gl.setFont(gf)
            grid.addWidget(gl, r, 0, 1, 4)
            r += 1
            for field, label in items:
                row = AdjustmentRow(field, label)
                grid.addWidget(row, r, 0, 1, 4)
                chk = QCheckBox("")
                chk.setToolTip(f"勾选后“同步”只复制{label}")
                self._sync_checks[field] = chk
                grid.addWidget(chk, r, 4)
                self.rows[field] = row
                row.value_changed.connect(self.changed)
                row.value_committed.connect(self.committed)
                r += 1
        lay.addLayout(grid)
        lay.addStretch(1)

        self.btn_reset_all.clicked.connect(self._reset_all)
        self.btn_sync.clicked.connect(self._emit_sync)
        self._clipboard: RenderIntent | None = None
        self.btn_copy.clicked.connect(self._copy)
        self.btn_paste.clicked.connect(self._paste)

    # ---- 值读写
    def intent(self) -> RenderIntent:
        it = RenderIntent()
        for f, row in self.rows.items():
            setattr(it, f, row.value())
        return it

    def set_intent(self, it: RenderIntent, *, silent: bool = True):
        for f, row in self.rows.items():
            row.blockSignals(silent)
            row.setValue(float(getattr(it, f, 0.0)))
            row.blockSignals(False)

    def reset_all(self):
        for row in self.rows.values():
            row.setValue(0.0)

    def checked_fields(self) -> list[str]:
        return [f for f, c in self._sync_checks.items() if c.isChecked()]

    def set_all_checked(self, on: bool = True):
        for c in self._sync_checks.values():
            c.setChecked(bool(on))

    # ---- 内部
    def _reset_all(self):
        for row in self.rows.values():
            row.reset()
        self.reset_all_requested.emit()

    def _emit_sync(self):
        fields = self.checked_fields() or list(P0_FIELDS)
        self.sync_requested.emit(fields)

    def _copy(self):
        self._clipboard = copy.deepcopy(self.intent())

    def _paste(self):
        if self._clipboard is None:
            return
        self.set_intent(self._clipboard, silent=False)
        for f, row in self.rows.items():
            self.changed.emit(f, row.value())
            self.committed.emit(f, row.value())
