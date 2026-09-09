"""closeEvent 导出进行中守卫回归（冷读 B 轴发现，2026-09-08）。

历史缺陷：整卷导出跑在 daemon 线程，进程退出直接丢弃它——关窗会
**静默**放弃未完成的帧，没有任何提示。

锁定契约：
- _export_running=False：closeEvent 行为不变（只做草稿 flush）；
- _export_running=True 且用户选择「继续等待」：ev.ignore()，窗口不关；
- 用户选择「取消剩余帧并退出」：置 cancel 事件，允许关闭；
- 确认框逻辑独立成 _confirm_close_during_export，测试直接驱动返回值，
  不依赖真实模态对话框。
"""

from __future__ import annotations

import os

import pytest


@pytest.fixture(scope="module", autouse=True)
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from negative_lab.ui import theme

    app = QApplication.instance() or QApplication([])
    theme.apply(app)
    return app


def _win_with_export_state(running: bool):
    from negative_lab.library.models import FrameRecord, GeometryRegions, \
        ImportSession
    from negative_lab.ui.main_window import MainWindow

    w = MainWindow()
    rec = FrameRecord(source_path="synthetic/ce.tif",
                      source_sha256="0" * 64,
                      input_facts={"abs_path": "synthetic/ce.tif",
                                   "kind": "tiff"},
                      geometry=GeometryRegions())
    w.session = ImportSession(files=[rec])
    w._export_running = running
    w._export_cancel = __import__("threading").Event()
    return w


def test_close_ignored_when_export_running_and_user_waits(qapp, monkeypatch):
    from PySide6.QtGui import QCloseEvent

    w = _win_with_export_state(running=True)
    monkeypatch.setattr(w, "_confirm_close_during_export", lambda: False)
    ev = QCloseEvent()
    w.closeEvent(ev)
    assert not ev.isAccepted(), "用户选择等待：必须忽略关闭事件"
    assert not w._export_cancel.is_set(), "等待路径不得触发取消"


def test_close_allowed_when_user_quits_and_cancel_set(qapp, monkeypatch):
    from PySide6.QtGui import QCloseEvent

    w = _win_with_export_state(running=True)
    monkeypatch.setattr(w, "_confirm_close_during_export", lambda: True)
    ev = QCloseEvent()
    w.closeEvent(ev)
    assert ev.isAccepted(), "用户坚持退出：允许关闭"
    assert w._export_cancel.is_set(), "退出前必须请求取消（当前帧收尾）"


def test_close_unaffected_when_no_export(qapp):
    from PySide6.QtGui import QCloseEvent

    w = _win_with_export_state(running=False)
    called = []
    monkey_dummy = object()
    ev = QCloseEvent()
    # 无确认框路径：_export_running=False 不应触碰确认逻辑
    w.closeEvent(ev)
    assert ev.isAccepted()
    assert not getattr(w, "_export_cancel", monkey_dummy).is_set() \
        if hasattr(w, "_export_cancel") else True
    del called, monkey_dummy
