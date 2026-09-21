"""B-7 回归锁：门禁阻断不再冒充"导出失败"。

上一轮实测：语料里 **21 帧**被门禁**故意**挡住（且 100% 继承自采集端），
但 `roll_export` 把它们和读图/写盘错误一起计进 `failed`，
GUI 标题就成了"整卷导出结束（有失败帧）"——用户会去**重试导出**，
而正确处置是重扫/重拍或裁决 B-6。四类原因修法完全不同，混一个数就是误导。

锁的设计照 B-7 登记时写好的四条：
① 分桶真生效（blocked 不进 failed）；② 未知状态不许消失；
③ 帧数守恒；④ `GateBlocked` 仍是 `ExportError` 子类（老调用方不破）。
"""

from __future__ import annotations

import pytest

from negative_lab.io.export import ExportError
from negative_lab.pipeline.roll_export import (
    EXPORT_STATUS_BUCKETS, GateBlocked, bucket_of, guard_gate_blocked)


class _Gate:
    def __init__(self, blocks, failures=(), attribution=None):
        self.blocks_export = blocks
        self.failures = list(failures)
        self.clip_attribution = attribution or {}


INHERITED = {"B": {"high_total": 0.0597, "high_inherited": 0.0597,
                   "high_introduced": 0.0}}


def test_blocked_is_its_own_exception_and_still_an_export_error():
    assert issubclass(GateBlocked, ExportError), (
        "老调用方写的是 except ExportError，子类关系破了它们就会漏抓")


def test_guard_raises_blocked_with_numeric_attribution():
    with pytest.raises(GateBlocked) as exc:
        guard_gate_blocked(_Gate(True, ["B 通道高端剪裁 5.97%"], INHERITED))
    msg = str(exc.value)
    assert msg.startswith("质量门禁阻断："), "前缀是既有契约，不能改"
    assert "采集端就已丢失" in msg, "归因缺席，用户又得猜是谁的责任"
    assert "软件新引入 0.00%" in msg


def test_guard_lets_a_passing_gate_through():
    guard_gate_blocked(_Gate(False))   # 不该抛


def test_blocked_counts_separately_from_failed():
    """① 分桶必须真生效：blocked 不许再落进 failed。"""
    assert bucket_of("blocked") == "blocked"
    assert bucket_of("failed") == "failed"


def test_unknown_status_never_disappears():
    """② 未知状态一律算 failed——凭空消失就是吞帧。"""
    for weird in ("", "OK", "skipped", "blocked_by_gate", None):
        assert bucket_of(weird) == "failed", f"{weird!r} 没有归宿"


def test_buckets_are_conserved():
    """③ 守恒：逐帧状态归类后总数不许变。"""
    statuses = ["ok", "ok", "needs_review", "blocked", "blocked", "failed",
                "cancelled", "外星状态"]
    counts = {}
    for s in statuses:
        k = bucket_of(s)
        counts[k] = counts.get(k, 0) + 1
    assert sum(counts.values()) == len(statuses)
    assert counts == {"ok": 2, "needs_review": 1, "blocked": 2,
                      "failed": 2, "cancelled": 1}
    assert set(EXPORT_STATUS_BUCKETS) == set(counts), "桶集合与计数集合必须一致"
