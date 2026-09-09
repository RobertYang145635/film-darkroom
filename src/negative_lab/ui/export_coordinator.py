"""ExportCoordinator：整卷导出的 UI 协调（2026-09-08 MainWindow 拆分）。

只搬不改（交接 §12-(6)）：`_on_export_button / export_roll / _export_job /
cancel_export / _on_export_done` 从 MainWindow 原文搬入本 mixin；共享状态
（session/_intent/btn_export/_export_running/_export_cancel）仍在 MainWindow
实例上，mixin 只承载方法。导出输入冻结与状态回写契约见
pipeline/roll_export.freeze_export_inputs / apply_status_writeback。
"""

from __future__ import annotations

import threading
from pathlib import Path

from PySide6.QtWidgets import QFileDialog, QMessageBox

from ..pipeline.render import RenderIntent


class ExportCoordinatorMixin:
    """整卷导出：开始/取消两态、后台执行、主线程状态回写。"""

    # ---------------------------------------------------------------- 导出
    def _on_export_button(self):
        """同一按钮承担「开始 / 取消」两态。

        导出改到后台线程后必须给出取消入口，否则用户只能干等；
        不加新控件是因为右栏/工具条布局刚定稿，不值得为它改版式。
        """
        if getattr(self, "_export_running", False):
            self.cancel_export()
            return
        self.export_roll()

    def export_roll(self):
        if self.session is None:
            return
        rid = self.library.current_roll
        frames = self._frames_of(rid or "")
        if not frames:
            QMessageBox.information(self, "整卷导出", "当前卷没有帧")
            return
        if getattr(self, "_export_running", False):
            self._set_banner("上一次整卷导出仍在进行，请先取消或等待其完成",
                             "warning")
            return
        out = QFileDialog.getExistingDirectory(self, "选择导出目录")
        if not out:
            return
        # 审查修复（2026-09-08）：整卷导出 = 每帧全管线 + 16bit TIFF + JPEG +
        # 写后重读校验，是分钟级重活。此前直接跑在 GUI 线程上，违反
        # ui/workers.py 自己立的契约「一切 >200ms 的工作都必须在 QThread 中
        # 执行」：窗口「未响应」数分钟；progress 回调因事件循环被阻塞而根本
        # 不重绘；也没有取消入口，导致 roll_export 新增的 cancelled 计数在
        # 主流程里永远不可达。现在走 run_in_thread + cancel 事件。
        from .workers import run_in_thread
        from ..pipeline.roll_export import freeze_export_inputs
        # §4-1（2026-09-08）：主线程一次性冻结整卷导出全部输入（几何深拷贝、
        # 微调、意图查表、批次编码假设、卷统计），worker 全程零读零写活对象；
        # 状态回写由 _on_export_done 在主线程按 items 完成。
        frozen = freeze_export_inputs(
            self.session, rid, frames, default_intent=self._intent)
        self._export_cancel = threading.Event()
        self._export_running = True
        self.btn_export.setText("取消导出")
        self._set_banner(f"整卷导出开始（{len(frames)} 帧）…", "info")
        run_in_thread(
            lambda cb: self._export_job(frozen, Path(out), rid, cb),
            on_done=self._on_export_done,
            on_progress=lambda m: self._set_banner(str(m), "info"))

    def _export_job(self, frozen, outdir, roll_id, progress):
        """后台线程执行整卷导出：只消费冻结输入，不碰控件与活对象。"""
        from ..pipeline.roll_export import export_roll_frames
        return export_roll_frames(
            frozen["records"], outdir, session=None, roll_id=roll_id,
            intent_resolver=lambda rec: frozen["intents"].get(
                rec.source_path) or RenderIntent(),
            tiff_assumption=frozen["tiff_encoding"],
            cancel=self._export_cancel, progress=progress,
            presnapshot=True, frozen_roll_stats=frozen["roll_stats"],
            frozen_batch_assumption=frozen["batch_assumption"])

    def cancel_export(self):
        """取消整卷导出：当前帧完成后停止（已写出的成品不回滚）。"""
        if getattr(self, "_export_running", False):
            self._export_cancel.set()
            self._set_banner("取消中：当前帧完成后停止…", "warning")

    def _on_export_done(self, res):
        """导出结束（主线程）：按 items 回写状态，如实报告四类计数。

        §4-1：worker 线程全程零写共享状态（presnapshot 冻结输入），
        状态回写必须在这里——主线程——完成。
        审查修复：此前标题恒为「整卷导出完成」且不报 cancelled —— 取消掉
        半卷仍被说成成功，正是 v6.2 §6.6 想消灭的静默丢帧，只是修在数据层。
        """
        self._export_running = False
        self.btn_export.setText("整卷导出…")
        ok = isinstance(res, dict) and bool(res.get("ok"))
        if ok:
            from ..pipeline.roll_export import apply_status_writeback
            rep0 = res.get("value") or {}
            if isinstance(rep0, dict) and isinstance(rep0.get("items"), list):
                apply_status_writeback(self.session, rep0["items"])
        if not ok:
            err = res.get("error") if isinstance(res, dict) else str(res)
            QMessageBox.critical(self, "导出失败（已回滚）",
                                 f"{err}\n源文件未改动")
            self._set_banner("整卷导出失败（已回滚，源文件未改动）", "danger")
            return
        rep = res.get("value") or {}
        cancelled = int(rep.get("cancelled") or 0)
        failed = int(rep.get("failed") or 0)
        line = (f"成功 {int(rep.get('ok') or 0)} / "
                f"待复核 {int(rep.get('needs_review') or 0)} / "
                f"失败 {failed}")
        if cancelled:
            line += f" / 已取消 {cancelled}"
        detail = f"报告：{rep.get('report_path', '')}"
        if rep.get("summary_complete") is False:
            detail += "（计数不完整，请查看报告核对逐帧结果）"
        title = ("整卷导出已取消" if cancelled
                 else "整卷导出结束（有失败帧）" if failed
                 else "整卷导出完成")
        if cancelled or failed:
            QMessageBox.warning(self, title, f"{line}\n{detail}")
        else:
            QMessageBox.information(self, title, f"{line}\n{detail}")
        self._set_banner(f"{title}：{line}",
                         "warning" if (cancelled or failed) else "success")
