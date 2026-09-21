"""B-9 锁：`gate_blocked` 是**记录性状态**，且**颜色表必须覆盖每个状态**。

**已锁住**（本文件 + `test_v5_contracts.py` 里那条）：
- `is_blocking("gate_blocked") is False`（钉死"不进 BLOCKING_STATUSES"的决定）；
- **每个"示警态/终态"都有配色**（`imported / ready_to_convert / converted` 属流程平常态，
  走默认字色，**故意无颜色**）；漏配会让 UI **静默不画**（`.get()` 返回 `None`，实测过）；
- **"被挡"的颜色 ≠ "坏了"的颜色**（B-7 分桶的全部意义：人能一眼分清）。

**诚实声明覆盖缺口（别当成已全覆盖）**：
"真实整卷导出被门禁挡下后，帧状态确实变成 `gate_blocked`"这一条**本文件没有锁**——
走到 `roll_export` 那个分支需要 session + 整卷夹具，本轮上下文不足没有硬凑。
现有的 `test_service.py::test_gate_fail_blocks_batch_export` 走的是 **batch 层**，
**拿不到 `rec.status`**，所以它**不能**替代这条。**下一轮补上，别把缺口读成已覆盖。**
"""

from __future__ import annotations

import os

import pytest

qt = pytest.importorskip("PySide6.QtWidgets")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QColor  # noqa: E402

from negative_lab.library.models import FrameStatus  # noqa: E402
from negative_lab.ui import library_panel  # noqa: E402


def _qapp():
    return qt.QApplication.instance() or qt.QApplication([])


#: 必须配色的状态：会**示警**的、或**终态**（人要靠颜色扫视定位）。
#: **不含** `imported / ready_to_convert / converted`——它们是流程中的平常态，
#: 走默认字色，**无颜色是设计**而非漏配（我第一版把"全覆盖"当成规则，
#: 是**期望写错**，不是实现的错；收窄的是期望，**没有**动任何安全属性）。
_MUST_HAVE_COLOUR = (
    FrameStatus.DECODE_ERROR, FrameStatus.PROCESSING_ERROR,
    FrameStatus.NEEDS_GROUP, FrameStatus.NEEDS_CROP_REVIEW,
    FrameStatus.NEEDS_REVIEW, FrameStatus.EXPORTED,
    FrameStatus.GATE_BLOCKED,
)


def test_attention_and_terminal_statuses_have_a_colour():
    """示警态与终态**必须**有配色：漏一个就会**静默没有颜色标记**。

    这条锁的由来正是本类缺陷的温床——`_STATUS_COLOR.get(status)` 对未知/漏配
    状态返回 `None`，UI 什么都不画，**不报错**（前置实验实测过）。
    新增枚举成员时，只要它属于上面这组却忘了配色，本锁就会红。
    """
    _qapp()
    missing = [s.value for s in _MUST_HAVE_COLOUR
               if s.value not in library_panel._STATUS_COLOR]
    assert not missing, (
        f"这些状态没有配色，列表里会**没有颜色标记**（静默失效）：{missing}")


def test_blocked_is_not_coloured_like_broken():
    """"被挡"不得与"坏了"同色——否则分桶在视觉上又合回去了（B-7）。"""
    _qapp()
    m = library_panel._STATUS_COLOR
    err = m[FrameStatus.PROCESSING_ERROR.value]
    blocked = m[FrameStatus.GATE_BLOCKED.value]
    assert isinstance(err, QColor) and isinstance(blocked, QColor)
    assert (err.red(), err.green(), err.blue()) != (
        blocked.red(), blocked.green(), blocked.blue()), (
        "被挡和坏了同色：用户无法区分'故意不导'与'导不出'")


def test_status_value_is_stable_wire_string():
    """状态字符串是**写进 session 文件**的，改名会让老 session 读不懂。"""
    assert FrameStatus.GATE_BLOCKED.value == "gate_blocked"
    assert not FrameStatus.is_blocking(FrameStatus.GATE_BLOCKED)
