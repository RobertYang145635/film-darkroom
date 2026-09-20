"""2026-09-09 冷读审查回归锁（导入线程纪律 + 清除区域后分析失效 + 光标态）。

锁定三条契约（每条都对应本轮实测到的真实缺陷，不是假想题）：

1. **「导入文件…」不在 GUI 线程算哈希**（ui/workers.py 的既有契约：
   一切 >200ms 的工作必须离开主线程）。实测本仓库真实样片：
   NEF 55 MB ≈ 80 ms/张、16 位 TIFF 274 MB ≈ 400 ms/张 —— 旧实现选 20 个
   文件就是数秒窗口未响应，且横幅不重绘（`load_folder` 早就在后台跑，
   `load_files` 漏了）。
2. **导入忙碌守卫必须真生效**：旧代码 `self._busy_enter("import")` 的返回值
   被丢弃，守卫是装饰性的，连点两次导入会并发跑两遍、后到的会话无声覆盖。
3. **凡改变统计范围的清除都要让分析失效**（v6.2 F2 同一条规则）：
   旧代码只修了键盘 Delete 路径（`_on_roi_deleted`），右栏「清除」按钮和
   「重置几何」清掉片基取样区/有效画面区后，界面继续挂着按已删除区域算出的
   Dmin 与色罩结果。
"""

from __future__ import annotations

import os
import threading
import time

import numpy as np
import pytest


@pytest.fixture(scope="module", autouse=True)
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from negative_lab.ui import theme

    app = QApplication.instance() or QApplication([])
    theme.apply(app)
    return app


def _wait_until(predicate, qapp, timeout: float = 20.0) -> bool:
    """跑事件循环等条件成立（后台导入经 Qt 信号回主线程）。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        qapp.processEvents()
        if predicate():
            return True
        time.sleep(0.005)
    qapp.processEvents()
    return predicate()


@pytest.fixture()
def fake_tiffs(tmp_path):
    """两个最小 .tif 占位文件：导入路径只做哈希与元数据，不解码。"""
    out = []
    for i in range(2):
        p = tmp_path / "roll" / f"frame{i}.tif"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"II*\x00" + bytes([i]) * 4096)
        out.append(p)
    return out


def _bare_window(monkeypatch):
    """构造 MainWindow，并把帧加载接成空操作（本文件只测导入/清除链路）。

    必须在构造**之前**打补丁：__init__ 里 connect(self.load_frame) 捕获的是
    当时的绑定方法，事后再替换实例属性对已建立的连接无效。
    """
    from negative_lab.ui.main_window import MainWindow

    monkeypatch.setattr(MainWindow, "load_frame", lambda self, path: None)
    return MainWindow()


# ------------------------------------------------- 1/2 导入线程纪律
def test_load_files_does_not_hash_on_gui_thread(qapp, monkeypatch, fake_tiffs):
    """`load_files` 的逐文件 SHA-256 必须发生在非调用线程。"""
    import negative_lab.ui.frame_loader as fl

    main = threading.main_thread()
    seen: list[bool] = []
    real = fl.make_frame_record

    def spy(*a, **kw):
        seen.append(threading.current_thread() is main)
        return real(*a, **kw)

    monkeypatch.setattr(fl, "make_frame_record", spy)
    w = _bare_window(monkeypatch)

    w.load_files(fake_tiffs)
    # 同步阶段绝不允许已经把会话建好（旧实现就是在这里冻结主线程）
    assert w.session is None, "旧实现：哈希+建会话跑在调用线程上"
    assert _wait_until(lambda: w.session is not None, qapp), "后台导入未落地"

    assert seen, "make_frame_record 一次都没被调用"
    assert not any(seen), f"哈希发生在 GUI 线程上（{seen.count(True)} 次）"
    assert len(w.session.files) == 2


def test_load_files_background_false_stays_synchronous(monkeypatch, fake_tiffs):
    """脚本/测试路径保留同步语义：直接拿得到会话。"""
    w = _bare_window(monkeypatch)
    sess = w.load_files(fake_tiffs, background=False)
    assert sess is not None and len(sess.files) == 2
    assert w.session is sess


def test_second_import_is_refused_while_first_runs(qapp, monkeypatch, tmp_path):
    """忙碌守卫必须真拒绝，并且**可见**（不给不成立的承诺）。"""
    import negative_lab.ui.frame_loader as fl

    real_import = fl.import_session        # 先取原件：补丁后 fl.import_session 就是自己
    gate = threading.Event()
    started = threading.Event()

    def slow_import(folder, *, probe=True, **kw):
        started.set()
        assert gate.wait(20.0), "测试死锁：释放门失败"
        return real_import(folder, probe=False)

    monkeypatch.setattr(fl, "import_session", slow_import)
    w = _bare_window(monkeypatch)

    first = w.load_folder(tmp_path)
    assert first is not None and started.wait(5.0)
    second = w.load_folder(tmp_path)          # 第一遍还没跑完
    assert second is None, "忙碌守卫失效：并发跑了两遍导入"
    assert "仍在进行" in w.lbl_banner.text(), "拒绝必须说给用户听"
    gate.set()
    assert _wait_until(lambda: w.session is not None, qapp)
    assert w._busy._locks == set(), "导入结束后忙碌锁未释放"


# --------------------------------------- 3 清除区域后分析必须失效
def _window_with_regions(monkeypatch):
    """手工装配一帧（不经解码），并记录 _refresh_preview 的调用参数。"""
    from PySide6.QtCore import QRectF

    from negative_lab.library.models import FrameRecord, GeometryRegions
    from negative_lab.ui.main_window import MainWindow
    from negative_lab.ui.preview_view import RoiKind

    calls: list[dict] = []
    # 必须在构造前打补丁：__init__ 里 connect 捕获的是当时的绑定方法
    monkeypatch.setattr(MainWindow, "_refresh_preview",
                        lambda self, **kw: calls.append(dict(kw)))
    w = _bare_window(monkeypatch)
    src = np.zeros((600, 800, 3), np.float32)
    rec = FrameRecord(source_path="synthetic/clear.tif", source_sha256="0" * 64,
                      geometry=GeometryRegions())
    w.current = rec
    w.t_source = src
    w._small = src[::4, ::4, :].copy()
    w._small_scale = 4
    w._canvas_rec = rec
    w.view.set_image(w._small)
    # add_roi 收 QRectF（y0,x0 起 + 宽高），不是四角坐标
    for kind, rect in ((RoiKind.OUTPUT_CROP, QRectF(10, 10, 110, 90)),
                       (RoiKind.IMAGE_AREA, QRectF(20, 20, 90, 70)),
                       (RoiKind.BASE, QRectF(2, 2, 58, 10))):
        w.view.add_roi(kind, rect)
    assert w.view.roi_count(RoiKind.BASE) == 1
    calls.clear()
    return w, calls


def test_clear_base_and_area_rerun_analysis(qapp, monkeypatch):
    """片基取样区/有效画面区 = 统计范围：清除后必须重算（v6.2 F2）。

    判别器是「_refresh_preview 有没有被调用」——`_update_draft_from_view`
    只在 rerun=True 时才调它，所以"有调用"就等于"走了重算"。
    """
    w, calls = _window_with_regions(monkeypatch)
    w._on_area_clear("base")
    assert calls, "清除片基取样区后仍挂着按旧区域算出的分析（F2 病灶复发）"

    calls.clear()
    w._on_area_clear("image_area")
    assert calls, "清除有效画面区后统计范围已变，却未重算"


def test_clear_output_crop_does_not_rerun(qapp, monkeypatch):
    """输出裁切只改构图、不改统计：保持零管线调用的快路径。"""
    w, calls = _window_with_regions(monkeypatch)
    w._on_area_clear("output_crop")
    assert not calls, "输出裁切不进管线（v6 即时合成契约）"


def test_reset_all_geometry_reruns_analysis(qapp, monkeypatch):
    """重置几何把三类全清（含统计范围）：同样必须重算。"""
    w, calls = _window_with_regions(monkeypatch)
    w._reset_all_geometry()
    assert calls, "清空全部区域后仍显示按旧区域算出的分析"


# ------------------------------------------------- 4 光标态不被忙碌吞掉
def _cursor_shape(view) -> object:
    """归一化视口光标形状（.shape() 在不同 PySide6 版本可能是 int 或枚举）。"""
    from PySide6.QtCore import Qt

    return Qt.CursorShape(view.viewport().cursor().shape())


def test_busy_does_not_swallow_pick_crosshair(qapp):
    """拾点模式下结束「分析」：十字准星必须回来（模式还在，Esc 才真退出）。"""
    from PySide6.QtCore import Qt

    from negative_lab.ui.preview_view import PreviewView

    v = PreviewView()
    v.set_image(np.zeros((80, 80, 3), np.float32))
    v.set_pick_mode("neutral")
    assert _cursor_shape(v) == Qt.CursorShape.CrossCursor
    v.set_busy(True)
    v.set_busy(False)
    assert _cursor_shape(v) == Qt.CursorShape.CrossCursor, \
        "旧实现：退出忙碌把光标变回箭头，用户以为拾点已被退出"
    v.set_pick_mode(None)
    v.set_busy(True)
    v.set_busy(False)
    assert _cursor_shape(v) == Qt.CursorShape.ArrowCursor
