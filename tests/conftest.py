"""全局测试护栏（v6.2 §3 阶段 A + 审查 §3.1）。

两件事，都是根因级的：

1. **写入物理隔离**：测试会话一开始就把标注目录与 GUI 草稿目录整体改道到
   临时根（`storage.isolated_storage`，走环境变量，子进程同样继承）。
   真实 `fixtures/annotations/` 里的既有用户标注**复制**进临时根，因此
   regression 测试照样读得到，但任何写入都落在临时目录。
   会话结束时核对真实目录的 (size, mtime_ns, sha256) 清单与开始时完全一致；
   不一致就让会话失败——被污染的测试不算通过。

   背景（实锤）：`outputs/drafts/1298_G200__G200-1.json` 曾被 GUI 冒烟
   画进真实样片的工作稿（v6.1 审计 F6）。

2. **共享 QApplication**：`QApplication.instance() or QApplication([])` 在
   各测试模块里各写一遍，进程内只有一个实例，但该实例只被模块 fixture 的
   局部变量引用；模块结束时可能被回收，后续模块再 `theme.apply(app)` 就会
   踩到已销毁的 C++ 对象——实测表现为
   `Windows fatal exception: access violation` at `theme.py:191 apply`
   （全量第 81%，同命令第二次运行却 355 全绿）。
   这里用会话级 fixture 持住强引用，实例整个会话不被回收。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from negative_lab import storage  # noqa: E402

#: 会话级强引用：防止 QApplication 在模块之间被回收（见模块 docstring）
_QAPP = None


def _real_dirs() -> list[Path]:
    return [storage.DEFAULTS["annotations"], storage.DEFAULTS["drafts"]]


@pytest.fixture(scope="session", autouse=True)
def isolated_writable_storage():
    """全会话写入隔离 + 真实目录未被写入的硬校验。"""
    before = storage.inventory(_real_dirs())
    with storage.isolated_storage() as st:
        # 复制既有用户标注（只读语义）：regression 测试依赖真实标注内容
        src = storage.DEFAULTS["annotations"]
        if src.exists():
            for f in sorted(src.glob("*.json")):
                try:
                    (st.annotations / f.name).write_bytes(f.read_bytes())
                except OSError:
                    pass
        yield st
    after = storage.inventory(_real_dirs())
    if after != before:
        changed = [k for k in set(before) | set(after)
                   if before.get(k) != after.get(k)]
        pytest.fail(
            "测试会话污染了真实可写目录（v6.2 §3 阶段 A）。改动项："
            + "; ".join(sorted(changed)[:8])
            + f"；真实目录={_real_dirs()}")


@pytest.fixture(scope="session", autouse=True)
def shared_qapp():
    """进程内唯一 QApplication，整个会话持有，避免跨模块回收导致崩溃。"""
    global _QAPP
    from PySide6.QtWidgets import QApplication

    _QAPP = QApplication.instance() or QApplication([])
    yield _QAPP
