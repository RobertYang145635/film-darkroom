"""转换检查器（Lightroom 式右栏；中央大图由 MainWindow 常驻持有）。

v6 布局重排：候选三段、原片/对比按钮与基础微调都在这里；
顶部工具条（候选分段/对比/适合/100%）在 MainWindow，随模式切换显隐。
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QPushButton, QScrollArea, QSlider, QVBoxLayout,
    QWidget)

from . import icons, theme
from .basic_adjustments import BasicAdjustmentsPanel

CANDIDATE_LABELS = (("preserve", "保留氛围"), ("neutral", "中性"),
                    ("warm", "偏暖"))


class _Segment(QWidget):
    """三段选择控件（Lightroom 式分段按钮）。"""

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


class ConvertPanel(QWidget):
    """右栏：候选选择 + 自动结果说明 + 基础微调。"""

    candidate_chosen = Signal(str)
    compare_toggled = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._image_sink = None
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        side = QVBoxLayout()
        side.setContentsMargins(0, 0, 0, 0)
        side.setSpacing(theme.SPACING["md"])

        f_small = QFont(self.font())
        f_small.setPointSize(max(7, f_small.pointSize() - 1))

        side.addWidget(self._header("自动结果"))
        self.segments = _Segment()
        side.addWidget(self.segments)
        self.lbl_auto = QLabel("—")
        self.lbl_auto.setProperty("caption", True)
        self.lbl_auto.setWordWrap(True)
        side.addWidget(self.lbl_auto)
        side.addWidget(self._sep())

        side.addWidget(self._header("基础调整"))
        self.panel = BasicAdjustmentsPanel()
        side.addWidget(self.panel)
        side.addStretch(1)

        inner = QWidget()
        inner.setLayout(side)
        scroll = QScrollArea()
        scroll.setWidget(inner)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        outer.addWidget(scroll)

        # ---- 查看选项（原片 / 并排对比）
        rowv = QHBoxLayout()
        rowv.setSpacing(theme.SPACING["xs"])
        self.btn_original = QPushButton("按住看原片")
        self.btn_original.setCheckable(True)
        self.btn_compare = QPushButton("并排对比")
        self.btn_compare.setCheckable(True)
        for b in (self.btn_original, self.btn_compare):
            b.setFont(f_small)
            rowv.addWidget(b)
        rowv.addStretch(1)
        outer.insertLayout(0, rowv)

        self.slider_split = QSlider(Qt.Horizontal)
        self.slider_split.setRange(0, 100)
        self.slider_split.setValue(50)
        self.slider_split.setToolTip("滑动前后对比位置")
        self.slider_split.setVisible(False)
        outer.insertWidget(1, self.slider_split)

        self.segments.chosen.connect(self.candidate_chosen)
        self.btn_compare.toggled.connect(self.compare_toggled)
        self._negative: np.ndarray | None = None
        self._positive: np.ndarray | None = None
        self.btn_original.pressed.connect(lambda: self._show_original(True))
        self.btn_original.released.connect(lambda: self._show_original(False))
        self.btn_original.toggled.connect(self._show_original)
        self.slider_split.valueChanged.connect(lambda _v: self.repaint())

    # ------------------------------------------------------------ 图像
    def set_images(self, negative: np.ndarray | None, positive: np.ndarray | None,
                   *, push: bool = False):
        """缓存两幅图；只有 push=True 才推给中央 viewer（模式接线）。"""
        self._negative = negative
        self._positive = positive
        if push:
            self.repaint()

    def _show_original(self, on: bool):
        if on and self._negative is not None:
            from ..pipeline.density import encode_srgb

            x = np.clip(self._negative, 0, None)
            self._emit_image(x / max(float(x.max()), 1e-6))
        else:
            self.repaint()

    def _emit_image(self, arr01):
        """把显示图交给中央 viewer（MainWindow 通过 image_sink 接线）。"""
        if self._image_sink is not None:
            self._image_sink(arr01)

    def repaint(self):
        if self._positive is None:
            return
        from ..pipeline.density import encode_srgb

        pos = encode_srgb(self._positive)
        if self.btn_compare.isChecked() and self._negative is not None:
            self.slider_split.setVisible(True)
            x = np.clip(self._negative, 0, None)
            neg = x / max(float(x.max()), 1e-6)
            split = int(pos.shape[1] * self.slider_split.value() / 100.0)
            out = neg.copy()
            out[:, :split] = pos[:, :split]
            self._emit_image(out)
        else:
            self.slider_split.setVisible(False)
            self._emit_image(pos)

    def image_sink(self, sink):
        """注册中央 viewer 的显示回调（MainWindow 接线）。"""
        self._image_sink = sink

    # ------------------------------------------------------------ 状态
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

    def _header(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setProperty("groupTitle", True)
        return lbl

    def _sep(self) -> QWidget:
        line = QWidget()
        line.setFixedHeight(1)
        line.setStyleSheet(f"background: {theme.COLORS['separator']};")
        return line
