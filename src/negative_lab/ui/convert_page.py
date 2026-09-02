"""转换页（v5 §7）。

顶部三段候选选择 / 中央大幅正片预览（按住看原片、并排与滑动对比）/
右侧"自动结果 → 白平衡 → 影调 → 色彩"/ 底部卷、帧、置信度与导出状态。
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QPushButton, QSlider, QVBoxLayout, QWidget)

from .basic_adjustments import BasicAdjustmentsPanel
from .preview_view import PreviewView

CANDIDATE_LABELS = (("preserve", "保留氛围"), ("neutral", "中性"),
                    ("warm", "偏暖"))


class _Segment(QWidget):
    """三段选择控件。"""

    chosen = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        self._btns: dict[str, QPushButton] = {}
        for key, label in CANDIDATE_LABELS:
            b = QPushButton(label)
            b.setCheckable(True)
            b.setAutoExclusive(True)
            b.clicked.connect(lambda _c=False, k=key: self.chosen.emit(k))
            lay.addWidget(b)
            self._btns[key] = b
        self._current = "preserve"

    def set_current(self, name: str):
        self._current = name
        for k, b in self._btns.items():
            b.setChecked(k == name)

    def set_confidence(self, name: str, conf: float):
        b = self._btns.get(name)
        if b is not None:
            b.setToolTip(f"{b.text()}：自动置信度 {conf:.2f}")


class ConvertPage(QWidget):
    candidate_chosen = Signal(str)
    prev_requested = Signal()
    next_requested = Signal()
    export_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        top = QHBoxLayout()
        self.segments = _Segment()
        top.addWidget(self.segments)
        top.addStretch(1)
        self.btn_original = QPushButton("按住看原片")
        self.btn_original.setCheckable(True)
        self.btn_compare = QPushButton("并排对比")
        self.btn_compare.setCheckable(True)
        for b in (self.btn_original, self.btn_compare):
            f = QFont(b.font())
            f.setPointSize(max(7, f.pointSize() - 1))
            b.setFont(f)
            top.addWidget(b)
        root.addLayout(top)

        body = QHBoxLayout()
        self.view = PreviewView()
        body.addWidget(self.view, 1)
        right = QVBoxLayout()
        right.setSpacing(6)

        self.lbl_auto = QLabel("自动结果：—")
        f_small = QFont(self.font())
        f_small.setPointSize(max(7, f_small.pointSize() - 1))
        self.lbl_auto.setFont(f_small)
        self.lbl_auto.setWordWrap(True)
        self.lbl_auto.setMaximumWidth(230)
        right.addWidget(self.lbl_auto)
        self.panel = BasicAdjustmentsPanel()
        right.addWidget(self.panel)
        right.addStretch(1)
        rw = QWidget()
        rw.setLayout(right)
        rw.setFixedWidth(240)
        body.addWidget(rw)
        root.addLayout(body, 1)

        self.slider_split = QSlider(Qt.Horizontal)
        self.slider_split.setRange(0, 100)
        self.slider_split.setValue(50)
        self.slider_split.setToolTip("滑动前后对比位置")
        root.addWidget(self.slider_split)

        foot = QHBoxLayout()
        from PySide6.QtWidgets import QToolButton

        from . import icons
        self.btn_prev = QToolButton()
        self.btn_prev.setIcon(icons.prev())
        self.btn_prev.setToolTip("上一张")
        self.btn_prev.setAutoRaise(True)
        self.btn_next = QToolButton()
        self.btn_next.setIcon(icons.next_())
        self.btn_next.setToolTip("下一张")
        self.btn_next.setAutoRaise(True)
        self.btn_export = QPushButton("整卷导出…")
        foot.addWidget(self.btn_prev)
        foot.addWidget(self.btn_next)
        foot.addWidget(self.btn_export, 1)
        self.lbl_state = QLabel("未选择帧")
        self.lbl_state.setFont(f_small)
        foot.addWidget(self.lbl_state, 2)
        root.addLayout(foot)

        self.segments.chosen.connect(self.candidate_chosen)
        self.btn_prev.clicked.connect(self.prev_requested)
        self.btn_next.clicked.connect(self.next_requested)
        self.btn_export.clicked.connect(self.export_requested)
        self._negative: np.ndarray | None = None
        self._positive: np.ndarray | None = None
        self.btn_original.pressed.connect(lambda: self._show_original(True))
        self.btn_original.released.connect(lambda: self._show_original(False))
        self.btn_original.toggled.connect(self._show_original)
        self.slider_split.valueChanged.connect(lambda _: self._repaint())

    # ---------------------------------------------------------------- 显示
    def set_images(self, negative: np.ndarray | None, positive: np.ndarray | None):
        self._negative = negative
        self._positive = positive
        self._repaint()

    def _show_original(self, on: bool):
        if on and self._negative is not None:
            from ..pipeline.density import encode_srgb

            x = np.clip(self._negative, 0, None)
            self.view.set_image(x / max(float(x.max()), 1e-6))
        else:
            self._repaint()

    def _repaint(self):
        if self._positive is None:
            return
        from ..pipeline.density import encode_srgb

        pos = encode_srgb(self._positive)
        if self.btn_compare.isChecked() and self._negative is not None:
            neg = self._negative
            x = np.clip(neg, 0, None)
            neg = x / max(float(x.max()), 1e-6)
            split = int(pos.shape[1] * self.slider_split.value() / 100.0)
            out = neg.copy()
            out[:, :split] = pos[:, :split]
            self.view.set_image(out)
        else:
            self.view.set_image(pos)

    # ---------------------------------------------------------------- 状态
    def set_candidates(self, cset, chosen: str):
        self.segments.set_current(chosen)
        for c in cset:
            self.segments.set_confidence(c.name, c.confidence)
        cand = cset.get(chosen)
        bits = [f"自动结果：{dict(CANDIDATE_LABELS).get(chosen, chosen)}"
                f"（置信度 {cand.confidence:.2f}）"]
        if cand.rationale:
            bits.append(cand.rationale)
        if cand.fallback_to:
            bits.append(f"已回退到“{cand.fallback_to}”")
        if cand.correction_capped:
            bits.append("色温/色调修正已触发硬上限")
        self.lbl_auto.setText("\n".join(bits))

    def set_state(self, roll_name: str, frame_name: str, review: list[str],
                  export_note: str = ""):
        txt = f"卷：{roll_name or '—'}  帧：{frame_name or '—'}"
        if review:
            txt += f"  待复核 {len(review)} 项"
        if export_note:
            txt += f"  {export_note}"
        self.lbl_state.setText(txt)
