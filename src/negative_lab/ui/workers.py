"""后台工作线程（v6 §15.7 响应性修复）。

教训（2026-09-02 用户实测"页面多次未响应"）：主线程上的重活会让 UI
完全卡死——

- 导入 12 张 NEF 逐一探针+哈希：20s
- 首次选帧 NEF 解码+卷分析+预览：16s
- 每次拖完 ROI 的预览刷新：2.8s

因此**一切 >200ms 的工作都必须离开主线程执行**，主线程只做
界面更新。所有 worker 通过 Qt 信号回主线程，**绝不直接触碰控件**
（QListWidget item、QLabel、QMessageBox 都不行：跨线程操作控件是未定义
行为，事故样本见 main_window._analyze_roll_frames 的 2026-09-08 修复）。

实现注意：run_in_thread() 用的是 threading.Thread(daemon=True)，不是
QThread（本节沿用旧说法）。daemon 线程在进程退出时被直接丢弃，因此
worker 里只做计算与读取；任何落盘都必须走 io.atomic，否则硬退出会留下
半截文件。
"""

from __future__ import annotations

import threading
import traceback
from dataclasses import dataclass, field

from PySide6.QtCore import QObject, Signal


class _JobSignals(QObject):
    """worker -> 主线程（Qt 自动排队，线程安全）。

    生命周期注意：信号载体必须保活到事件送达。曾把 QThread 局部变量
    交给 GC，排队事件随 C++ 对象销毁而丢失（banner 静默消失），
    因此这里用模块级注册表持引用，回调后再释放。
    """

    finished = Signal(object)          # 结果 dict
    progress = Signal(str)


_LIVE_JOBS: set[_JobSignals] = set()


def run_in_thread(fn, *, on_done, on_progress=None, daemon: bool = True):
    """把阻塞函数放进 daemon 线程；回调经 Qt 信号回主线程。

    fn(progress_cb) -> 结果对象；异常变成 {"error": str, "traceback": str}。
    返回 threading.Thread（测试可 join 等待）。
    """
    sig = _JobSignals()
    _LIVE_JOBS.add(sig)                # 保活：防止排队事件随对象销毁丢失

    def _cleanup(*_a):
        _LIVE_JOBS.discard(sig)

    sig.finished.connect(on_done)
    sig.finished.connect(_cleanup)
    if on_progress is not None:
        sig.progress.connect(on_progress)

    def _cb(msg: str):
        sig.progress.emit(str(msg))

    def _run():
        try:
            result = {"ok": True, "value": fn(_cb)}
        except Exception as e:  # noqa: BLE001 —— 后台错误必须可见，不崩 UI
            result = {"ok": False, "error": f"{type(e).__name__}: {e}",
                      "traceback": traceback.format_exc()}
        sig.finished.emit(result)

    th = threading.Thread(target=_run, daemon=daemon)
    th.start()
    return th


@dataclass
class BusyState:
    """轻量忙碌状态：某类工作进行中就不重复排队（防抖）。"""

    _locks: set = field(default_factory=set)

    def try_enter(self, key: str) -> bool:
        if key in self._locks:
            return False
        self._locks.add(key)
        return True

    def leave(self, key: str):
        self._locks.discard(key)
