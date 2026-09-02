"""卷与缩略图库面板（v5 §2.1.1/§2.1.2、§7.1 左侧）。

- 按卷分组显示缩略图、当前帧与状态；
- 手动建立/重命名/拆分/移动卷分组；
- 采集指纹冲突的帧必须标红并提供"一键拆成不同卷"（v5 §4.4）；
- 缩略图在后台线程生成，不阻塞交互；原始文件永远只读。
"""

from __future__ import annotations

from pathlib import Path
import threading

from PySide6.QtCore import QObject, QSize, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout, QInputDialog, QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QListView, QMessageBox, QPushButton, QVBoxLayout, QWidget)

from ..library.models import FrameStatus

THUMB = 128


class _ThumbSignals(QObject):
    done = Signal(str, object)


class _ThumbTask:
    """后台缩略图加载（daemon 线程，不阻塞程序退出）。

    早期用 QThreadPool.globalInstance()：它持有非 daemon 工作线程，
    程序退出时即使队列清空也可能挂住解释器（GUI 冒烟实测超时）。
    """

    def __init__(self, path: str, signals: _ThumbSignals, assumption: str | None):
        self.path = path
        self.signals = signals
        self.assumption = assumption

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        try:
            from ..io.images import load_thumbnail

            img = load_thumbnail(self.path, max_dim=THUMB,
                                 tiff_encoding=self.assumption)
        except Exception:  # noqa: BLE001 —— 单张失败只显示占位，不中断队列
            img = None
        self.signals.done.emit(self.path, img)


_STATUS_COLOR = {
    FrameStatus.DECODE_ERROR.value: QColor(220, 90, 90),
    FrameStatus.PROCESSING_ERROR.value: QColor(220, 90, 90),
    FrameStatus.NEEDS_GROUP.value: QColor(230, 190, 90),
    FrameStatus.NEEDS_CROP_REVIEW.value: QColor(230, 190, 90),
    FrameStatus.NEEDS_REVIEW.value: QColor(150, 200, 240),
    FrameStatus.EXPORTED.value: QColor(120, 220, 140),
}


class LibraryPanel(QWidget):
    """卷列表 + 当前卷缩略图网格。"""

    frame_selected = Signal(str)
    roll_selected = Signal(str)
    request_split = Signal(str, list)          # roll_id, frame_paths
    grouping_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.session = None
        self.current_roll: str | None = None
        self.current_frame: str | None = None
        self._pending: set[str] = set()
        self._pixmaps: dict[str, QPixmap] = {}
        self._assumption: str | None = None

        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.setSpacing(4)

        head = QHBoxLayout()
        head.setSpacing(4)
        self.btn_new_roll = QPushButton("新建卷")
        self.btn_rename = QPushButton("改名")
        self.btn_split = QPushButton("拆分")
        for b in (self.btn_new_roll, self.btn_rename, self.btn_split):
            head.addWidget(b, 1)
        lay.addLayout(head)

        self.roll_list = QListWidget()
        self.roll_list.setMaximumHeight(96)
        lay.addWidget(self.roll_list)

        self.grid = _IconGrid()
        lay.addWidget(self.grid, 1)

        foot = QHBoxLayout()
        self.lbl_state = QLabel("未导入")
        self.lbl_state.setProperty("caption", True)
        foot.addWidget(self.lbl_state, 1)
        lay.addLayout(foot)

        self.roll_list.currentItemChanged.connect(self._roll_changed)
        self.grid.frame_activated.connect(self.frame_selected)
        self.btn_new_roll.clicked.connect(self._new_roll)
        self.btn_rename.clicked.connect(self._rename_roll)
        self.btn_split.clicked.connect(self._split_roll)
        self._signals = _ThumbSignals()
        self._signals.done.connect(self._thumb_ready)

    # ---------------------------------------------------------------- 数据
    def set_session(self, session, *, tiff_assumption: str | None = None):
        self.session = session
        self._assumption = tiff_assumption
        self._pixmaps.clear()
        self.roll_list.clear()
        self.grid.clear()
        if session is None:
            self.lbl_state.setText("未导入")
            return
        for r in session.roll_groups:
            self._add_roll_item(r.id, r.display_name)
        ungrouped = session.ungrouped()
        if ungrouped:
            self._add_roll_item("__ungrouped__",
                                f"未分卷（{len(ungrouped)}）")
        self.lbl_state.setText(f"{len(session.files)} 个文件 / "
                               f"{len(session.roll_groups)} 卷")

    def _add_roll_item(self, rid: str, name: str):
        it = QListWidgetItem(name)
        it.setData(Qt.UserRole, rid)
        self.roll_list.addItem(it)

    def select_roll(self, rid: str):
        for i in range(self.roll_list.count()):
            if self.roll_list.item(i).data(Qt.UserRole) == rid:
                self.roll_list.setCurrentRow(i)
                return

    def _roll_changed(self, cur, _prev):
        if cur is None:
            return
        self.current_roll = cur.data(Qt.UserRole)
        self._populate_grid()
        self.roll_selected.emit(self.current_roll)

    def _frames_of(self, rid: str) -> list:
        if self.session is None:
            return []
        if rid == "__ungrouped__":
            return self.session.ungrouped()
        r = self.session.roll(rid)
        return [f for f in (self.session.frame(p) for p in (r.frame_ids if r else []))
                if f is not None]

    def _populate_grid(self):
        self.grid.clear()
        for f in self._frames_of(self.current_roll or ""):
            item = QListWidgetItem(self._pixmaps.get(f.source_path) or QIcon(),
                                   Path(f.source_path).stem)
            item.setData(Qt.UserRole, f.source_path)
            item.setToolTip(f"{f.source_path}\n状态：{f.status}")
            c = _STATUS_COLOR.get(f.status)
            if c is not None:
                # 状态只用前景色小圆点语义（文字着色），不整块染色——
                # 深底色块叠在深色负片缩略图上就是一坨不可读色块
                item.setForeground(c)
            self.grid.addItem(item)
            self._request_thumb(f)

    def current_frames(self) -> list[str]:
        return [self.grid.item(i).data(Qt.UserRole)
                for i in range(self.grid.count())]

    def selected_frames(self) -> list[str]:
        sel = [it.data(Qt.UserRole) for it in self.grid.selectedItems()]
        return sel or self.current_frames()

    def set_current_frame(self, path: str):
        self.current_frame = path
        self.grid.set_current(path)

    # ---------------------------------------------------------------- 缩略图
    def _request_thumb(self, frame):
        path = frame.source_path
        if path in self._pixmaps or path in self._pending:
            return
        self._pending.add(path)
        abs_path = self._abs(path)
        _ThumbTask(str(abs_path), self._signals, self._assumption).start()

    def _abs(self, path: str) -> Path:
        p = Path(path)
        if p.is_absolute():
            return p
        root = Path(__file__).resolve().parents[3]
        return root / path

    def _thumb_ready(self, path: str, img):
        self._pending.discard(path)
        rel = path
        if self.session:
            for f in self.session.files:
                if self._abs(f.source_path) == Path(path):
                    rel = f.source_path
                    break
        if img is None:
            pm = QPixmap(THUMB, THUMB)
            pm.fill(QColor(60, 40, 40))
        else:
            pm = QPixmap.fromImage(img)
        self._pixmaps[rel] = pm
        for i in range(self.grid.count()):
            it = self.grid.item(i)
            if it.data(Qt.UserRole) == rel:
                it.setIcon(QIcon(pm))
        return

    # ---------------------------------------------------------------- 分组操作
    def _new_roll(self):
        if self.session is None:
            return
        frames = self.selected_frames()
        if not frames:
            QMessageBox.information(self, "新建卷", "请先在缩略图中选择帧")
            return
        name, ok = QInputDialog.getText(self, "新建卷", "卷名：")
        if not ok or not name.strip():
            return
        self.session.create_roll(name.strip(), frames)
        self.set_session(self.session, tiff_assumption=self._assumption)
        self.select_roll(self.session.roll_groups[-1].id)
        self.grouping_changed.emit()

    def _rename_roll(self):
        if self.session is None or not self.current_roll \
                or self.current_roll == "__ungrouped__":
            return
        r = self.session.roll(self.current_roll)
        name, ok = QInputDialog.getText(self, "重命名卷", "卷名：",
                                        text=r.display_name if r else "")
        if ok and name.strip():
            self.session.rename_roll(self.current_roll, name.strip())
            self.set_session(self.session, tiff_assumption=self._assumption)
            self.select_roll(self.current_roll)
            self.grouping_changed.emit()

    def _split_roll(self):
        if self.session is None or not self.current_roll \
                or self.current_roll == "__ungrouped__":
            return
        frames = self.selected_frames()
        if not frames:
            return
        name, ok = QInputDialog.getText(self, "拆分为新卷", "新卷名：")
        if not ok or not name.strip():
            return
        self.session.split_roll(self.current_roll, frames, name.strip())
        self.set_session(self.session, tiff_assumption=self._assumption)
        self.select_roll(self.session.roll_groups[-1].id)
        self.request_split.emit(self.current_roll, frames)
        self.grouping_changed.emit()

    def mark_conflicts(self, conflicting: list[str]):
        """采集指纹冲突帧标记（v5 §4.4）。"""
        for i in range(self.grid.count()):
            it = self.grid.item(i)
            if it.data(Qt.UserRole) in conflicting:
                it.setForeground(QColor(255, 120, 120))
                it.setToolTip(it.toolTip() + "\n采集指纹与本卷不一致：建议拆卷")


class _IconGrid(QListWidget):
    """缩略图网格（多选 + 双击选中；信息在缩略图本身，不做大卡片）。"""

    frame_activated = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setViewMode(QListWidget.IconMode)
        self.setIconSize(QSize(THUMB, THUMB))
        self.setResizeMode(QListWidget.Adjust)
        self.setMovement(QListView.Static)
        self.setSelectionMode(QListWidget.ExtendedSelection)
        self.setSpacing(4)
        self.setWordWrap(False)              # 文件名单行省略（§5.6.3）
        self.setTextElideMode(Qt.ElideRight)
        self.setUniformItemSizes(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.itemDoubleClicked.connect(
            lambda it: self.frame_activated.emit(it.data(Qt.UserRole)))
        self.itemSelectionChanged.connect(self._on_sel)

    def set_current(self, path: str):
        for i in range(self.count()):
            if self.item(i).data(Qt.UserRole) == path:
                self.setCurrentRow(i)
                return

    def _on_sel(self):
        items = self.selectedItems()
        if len(items) == 1:
            self.frame_activated.emit(items[0].data(Qt.UserRole))
