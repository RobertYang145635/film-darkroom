"""不变式：**"哪些帧挡住导出"必须恰好等于"哪些帧门禁 FAIL"**（实测语料核对）。

2026-09 实测：`gate == "fail"` 的帧数 **21**，`blocks_export` 为真的帧数也是 **21**，
且**两边是同一批帧**（不是凑数的相等）。这条关系之所以值得钉住：

- 门禁结论和"挡不挡导出"由**两处代码**决定，一旦各走各路就会出现
  **门是绿的却挡导出**（用户莫名被拦）或**门是红的却放行**（假绿，本项目的头号敌人）；
- 本会话已因"分桶改动没同步到退出码"自伤过一次（全挡却 `exit=0`），
  所以**凡是"两处必须一致"的关系都要有锁**，而不是靠人记住。

只读核对，不改任何判定；`uid` 用生产代码的 `uid_of`，避免用手搓键。
"""

from __future__ import annotations

import json
import pathlib

from negative_lab.diagnostics.verdicts import uid_of

ROOT = pathlib.Path(__file__).resolve().parents[2]
METRICS = ROOT / "fixtures" / "verdicts" / "metrics.json"


def _frames() -> list:
    m = json.loads(METRICS.read_text(encoding="utf-8"))
    fr = m.get("frames")
    if isinstance(fr, dict):
        return list(fr.values())
    return list(fr or [])


def test_blocked_export_set_equals_gate_fail_set():
    fr = _frames()
    assert fr, "metrics 读不出帧，锁本身失效（不是被测代码的问题）"

    def key(f):
        return uid_of(f) or f"{f.get('roll')}/{f.get('id')}/{f.get('file')}"

    blocked = {key(f) for f in fr if bool(f.get("blocks_export"))}
    failed = {key(f) for f in fr if str(f.get("gate")) == "fail"}

    assert blocked == failed, (
        f"挡导出的帧与门禁 FAIL 的帧**不是同一批**：\n"
        f"  只被挡（门不是 fail）：{sorted(blocked - failed)[:5]}\n"
        f"  只 FAIL（却没挡）：  {sorted(failed - blocked)[:5]}\n"
        "这两处判定必须一致，否则用户会被莫名拦住或被告知没事却仍被拦")


def test_gate_fail_is_not_vacuously_empty():
    """防"空集合相等"：两边都空也算相等，那种绿毫无意义。"""
    fr = _frames()
    failed = [f for f in fr if str(f.get("gate")) == "fail"]
    assert len(failed) > 0, (
        "语料里没有任何 FAIL 帧，上面那条锁退化成空判断——"
        "要么补语料，要么删锁并说明，别留假防护")


def test_blocked_count_matches_documented_figure():
    """文档写"导出被挡的 **21 帧**"，这里从工件反算核对（防手抄漂移）。"""
    doc = (ROOT / "docs" / "visual-acceptance.md").read_text(encoding="utf-8")
    n = sum(1 for f in _frames() if bool(f.get("blocks_export")))
    assert n > 0
    assert f"导出被挡的 **{n} 帧**" in doc, (
        f"实测被挡 {n} 帧，文档措辞已对不上（漂移）")
