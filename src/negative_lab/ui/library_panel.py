"""卷与缩略图库面板（v5 §2.1.1/§2.1.2、§7.1 左侧）。

- 按卷分组显示缩略图、当前帧与状态；
- 手动建立/重命名/拆分/移动卷分组；
- 采集指纹冲突的帧必须标红并提供"一键拆成不同卷"（v5 §4.4）；
- 缩略图在后台线程生成，不阻塞交互；原始文件永远只读。
"""

from __future__ import annotations

from .. import storage

from pathlib import Path
import threading

from PySide6.QtCore import QObject, QSize, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout, QInputDialog, QLabel, QListWidget, QListWidgetItem,
    QListView, QMessageBox, QPushButton, QVBoxLayout, QWidget)

from ..library.models import FrameStatus

THUMB = 128           # 胶片条用
LIST_THUMB = 44       # 左栏单列列表用（240px 栏宽下可读）
STRIP_THUMB = 72


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
    # B-9：被门禁**故意挡下** ≠ 坏了。**深橙**——必须区别于错误红 (220,90,90)
    # 与浅琥珀 (230,190,90)：人要一眼分得清"被挡"和"坏了"（B-7 分桶的意义）。
    FrameStatus.GATE_BLOCKED.value: QColor(240, 150, 60),
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
        self.btn_film_type = QPushButton("声明片种")
        for b in (self.btn_new_roll, self.btn_rename, self.btn_split, self.btn_film_type):
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
        self.btn_film_type.clicked.connect(self._declare_film_type)
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
            self._add_roll_item(r.id, self._roll_caption(r))
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

    @staticmethod
    def _roll_caption(roll) -> str:
        """卷名后带上已声明的片种：声明了却在界面上看不出来，用户下一轮只能靠猜——
        参数选择的证据必须在使用它的界面上看得见。"""
        from ..library.film_types import resolve
        ft = resolve(getattr(roll, "film_type", "") or "")
        return f"{roll.display_name}｜{ft.label}" if ft else roll.display_name

    def select_roll(self, rid: str):
        """编程化选卷：显式走 _roll_changed 全流程（setCurrentRow 在
        行号未变时不发 currentItemChanged 信号，会导致导入后不预览——
        2026-09-02 用户实测：导入后还得手动点一次'未分卷'）。"""
        if self.roll_list.count() == 0:
            return False
        if rid is None:
            rid = self.roll_list.item(0).data(Qt.UserRole)   # 第一项（含未分卷）
        for i in range(self.roll_list.count()):
            if self.roll_list.item(i).data(Qt.UserRole) == rid:
                if self.roll_list.currentRow() == i:
                    self._roll_changed(self.roll_list.item(i), None)
                else:
                    self.roll_list.setCurrentRow(i)   # 行号变化自动发信号
                return True
        return False

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
        root = storage.repo_path()
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

    def _declare_film_type(self):
        """卷级片种声明：只决定"这一卷用哪套技术级通道斜率"，不碰几何、不碰电平。

        未声明 = 现产默认（逐比特不变），所以这一栏留空是安全选项，不是缺陷。
        """
        from ..library.film_types import FILM_TYPES, known_keys
        if self.session is None or not self.current_roll \
                or self.current_roll == "__ungrouped__":
            return
        keys = [""] + known_keys()
        labels = ["（不声明，用通用默认）"] + [FILM_TYPES[k].label for k in keys[1:]]
        if len(keys) == 1:
            QMessageBox.information(self, "声明片种",
                                    "还没有已标定的片种（参数表为空）。\n"
                                    "所有卷都按通用默认出图，这是正常状态。")
            return
        r = self.session.roll(self.current_roll)
        if r is None:
            return
        cur = labels[keys.index(r.film_type)] if r.film_type in keys else labels[0]
        choice, ok = QInputDialog.getItem(self, "声明片种",
                                         f"这一卷（{r.display_name}）是什么片种？",
                                         labels, labels.index(cur), False)
        if not ok:
            return
        old, key = r.film_type, keys[labels.index(choice)]
        try:
            self.session.declare_film_type(self.current_roll, key)
        except (KeyError, ValueError) as exc:
            QMessageBox.warning(self, "声明片种", f"声明没有写入：{exc}")
            return
        # 帧级同步：预览/草稿走的是 FrameRecord，不同步的话界面与导出会各说一套。
        # 只改"上一轮由这一卷填进去"的帧——逐帧单独声明过的帧不归卷级管。
        frames = self._frames_of(self.current_roll)
        if old:
            for f in frames:
                if f.film_type == old:
                    f.film_type = key
        r.apply_film_type(frames)
        self.set_session(self.session, tiff_assumption=self._assumption)
        self.select_roll(self.current_roll)
        self.grouping_changed.emit()

    def mark_conflicts(self, conflicting: list[str]):
        """采集指纹冲突帧标记（v5 §4.4）。"""
        for i in range(self.grid.count()):
            it = self.grid.item(i)
            if it.data(Qt.UserRole) in conflicting:
                it.setForeground(QColor(255, 120, 120))
                it.setToolTip(it.toolTip() + "\n采集指纹与本卷不一致：建议拆卷")


class _IconGrid(QListWidget):
    """当前卷的帧列表（Lightroom 左栏式：单列紧凑行，缩略图+文件名）。

    240px 宽的栏里放 128px IconMode 网格会挤成一坨（2026-09-02 用户实测），
    因此改用 ListMode 单列 + 48px 缩略图 + 单行省略文件名。
    """

    frame_activated = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setViewMode(QListWidget.ListMode)
        self.setIconSize(QSize(LIST_THUMB, LIST_THUMB))
        self.setResizeMode(QListWidget.Adjust)
        self.setMovement(QListView.Static)
        self.setSelectionMode(QListWidget.ExtendedSelection)
        self.setSpacing(1)
        self.setWordWrap(False)
        self.setTextElideMode(Qt.ElideMiddle)
        self.setUniformItemSizes(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.itemDoubleClicked.connect(
            lambda it: self.frame_activated.emit(it.data(Qt.UserRole)))
        self.itemClicked.connect(
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


class FilmStrip(QListWidget):
    """底部胶片条（Lightroom 标志性）：当前卷全部帧横向缩略图。

    点选即切换当前帧；缩略图与左栏同源（MainWindow 提供缓存）。
    """

    frame_selected = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setViewMode(QListWidget.IconMode)
        self.setIconSize(QSize(72, 72))
        self.setFlow(QListView.LeftToRight)
        self.setWrapping(False)
        self.setResizeMode(QListWidget.Adjust)
        self.setMovement(QListView.Static)
        self.setSelectionMode(QListWidget.SingleSelection)
        self.setSpacing(4)
        self.setFixedHeight(104)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.itemClicked.connect(
            lambda it: self.frame_selected.emit(it.data(Qt.UserRole)))

    def set_frames(self, frames: list, pixmap_of=None):
        """frames: FrameRecord 列表；pixmap_of(rec) -> QPixmap|None（可缺）。"""
        self.clear()
        for f in frames:
            pm = pixmap_of(f) if pixmap_of else None
            icon = QIcon(pm) if pm is not None else QIcon()
            it = QListWidgetItem(icon, Path(f.source_path).stem)
            it.setData(Qt.UserRole, f.source_path)
            it.setToolTip(f"{f.source_path}\n状态：{f.status}")
            self.addItem(it)

    def select(self, path: str):
        for i in range(self.count()):
            if self.item(i).data(Qt.UserRole) == path:
                self.setCurrentRow(i)
                return
