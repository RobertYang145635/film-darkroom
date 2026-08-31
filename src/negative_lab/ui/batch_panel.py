"""批处理队列面板（规格 §5.2）。

状态列：pending/running/ok/warn/failed/cancelled；失败项可单独重试；
取消在文件边界生效，原子导出保证不留半成品。
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
    QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from ..pipeline.batch import BatchJob, STATUS_ICON, collect_files
from ..profiles.profile import FilmProfile, list_profiles


class _Worker(QThread):
    item_done = Signal(object)
    all_done = Signal(dict)

    def __init__(self, job: BatchJob, parent=None):
        super().__init__(parent)
        self.job = job

    def run(self):
        rep = self.job.run(on_update=self.item_done.emit)
        self.all_done.emit(rep)


class BatchPanel(QWidget):
    """可嵌入主窗口的批处理面板。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)

        row = QHBoxLayout()
        self.edit_dir = QLineEdit()
        self.edit_dir.setPlaceholderText("输入文件夹…")
        btn_dir = QPushButton("选择…")
        btn_dir.clicked.connect(self._pick_dir)
        row.addWidget(QLabel("文件夹"))
        row.addWidget(self.edit_dir, 1)
        row.addWidget(btn_dir)
        self.combo_profile = QComboBox()
        self.combo_profile.addItem("（无配置文件，纯自动）", None)
        row.addWidget(QLabel("配置(先验)"))
        row.addWidget(self.combo_profile)
        btn_refresh = QPushButton("↻")
        btn_refresh.clicked.connect(self.reload_profiles)
        row.addWidget(btn_refresh)
        lay.addLayout(row)

        row2 = QHBoxLayout()
        self.btn_run = QPushButton("开始批处理")
        self.btn_cancel = QPushButton("取消")
        self.btn_cancel.setEnabled(False)
        self.btn_retry = QPushButton("重试失败项")
        self.btn_retry.setEnabled(False)
        for b in (self.btn_run, self.btn_cancel, self.btn_retry):
            row2.addWidget(b)
        self.lbl_summary = QLabel("")
        row2.addWidget(self.lbl_summary, 1)
        lay.addLayout(row2)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["文件", "状态", "置信度", "警告/错误"])
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        lay.addWidget(self.table, 1)

        self.btn_run.clicked.connect(self.start)
        self.btn_cancel.clicked.connect(self._cancel)
        self.btn_retry.clicked.connect(self._retry)
        self._job: BatchJob | None = None
        self._worker: _Worker | None = None
        self.reload_profiles()

    # ----

    def reload_profiles(self):
        cur = self.combo_profile.currentText()
        self.combo_profile.clear()
        self.combo_profile.addItem("（无配置文件，纯自动）", None)
        for p in list_profiles():
            self.combo_profile.addItem(f"{p.name} [{p.film_type}]", p)
        idx = self.combo_profile.findText(cur)
        if idx >= 0:
            self.combo_profile.setCurrentIndex(idx)

    def _pick_dir(self):
        d = QFileDialog.getExistingDirectory(self, "选择负片文件夹")
        if d:
            self.edit_dir.setText(d)

    def start(self):
        d = Path(self.edit_dir.text() or "")
        if not d.is_dir():
            self.lbl_summary.setText("请先选择有效文件夹")
            return
        files = collect_files(d)
        if not files:
            self.lbl_summary.setText("文件夹内无支持的文件")
            return
        prior = self.combo_profile.currentData()
        outdir = d / "negative-lab-out"
        self._run_job(BatchJob(files, outdir, prior=prior))

    def _retry(self):
        if not self._job:
            return
        failed = self._job.failed_items()
        if not failed:
            return
        self._run_job(BatchJob([it.path for it in failed], self._job.outdir,
                               prior=self._job.prior))

    def _run_job(self, job: BatchJob):
        self._job = job
        self.table.setRowCount(0)
        for it in job.items:
            r = self.table.rowCount()
            self.table.insertRow(r)
            self.table.setItem(r, 0, QTableWidgetItem(it.path.name))
            self.table.setItem(r, 1, QTableWidgetItem("○ pending"))
            self.table.setItem(r, 2, QTableWidgetItem(""))
            self.table.setItem(r, 3, QTableWidgetItem(""))
        self.btn_run.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.btn_retry.setEnabled(False)
        self._worker = _Worker(job, self)
        self._worker.item_done.connect(self._on_item)
        self._worker.all_done.connect(self._on_done)
        self._worker.start()

    def _row_of(self, path: Path) -> int:
        for r in range(self.table.rowCount()):
            if self.table.item(r, 0).text() == path.name:
                return r
        return -1

    def _on_item(self, it):
        r = self._row_of(it.path)
        if r < 0:
            return
        self.table.item(r, 1).setText(f"{STATUS_ICON.get(it.status, '?')} {it.status}")
        conf = it.confidence or {}
        self.table.item(r, 2).setText(
            " ".join(f"{k}:{v}" for k, v in conf.items()) if conf else "")
        self.table.item(r, 3).setText(
            it.error or ("; ".join(it.warnings[:2]) if it.warnings else ""))

    def _on_done(self, rep: dict):
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        s = rep.get("summary", {})
        self.lbl_summary.setText(
            "  ".join(f"{k}={v}" for k, v in sorted(s.items()))
            + f"  报告: {self._job.outdir / 'batch-report.json'}")
        self.btn_retry.setEnabled(bool(self._job.failed_items()))

    def _cancel(self):
        if self._job:
            self._job.cancel()
            self.lbl_summary.setText("取消中：当前文件完成后停止…")
