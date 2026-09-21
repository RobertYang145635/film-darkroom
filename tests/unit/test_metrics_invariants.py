"""数据不变量锁：给入库的 `fixtures/verdicts/metrics.json` 上一道闸。

只锁**恒等式与结构**，不锁具体数值（数值会随重测变，恒等式不该变）。
本会话在 metrics 上吃过两次亏：一次是 id 不唯一导致按 id 合并把 169 帧压成 132，
一次是归因统计用中文字串而不是字段——所以这两件事都用恒等式钉住。
"""

from __future__ import annotations

import json
import pathlib

import pytest

METRICS = (pathlib.Path(__file__).resolve().parents[2]
           / "fixtures" / "verdicts" / "metrics.json")


@pytest.fixture(scope="module")
def frames():
    assert METRICS.exists(), f"缺少 {METRICS}（判定与归因都依赖它）"
    doc = json.loads(METRICS.read_text(encoding="utf-8"))
    rows = [f for f in doc["frames"] if f.get("status") == "ok"]
    assert len(rows) > 40, f"语料太少（{len(rows)}），这条锁就没有意义了"
    return rows


def test_uid_is_unique_and_derived_from_id(frames):
    uids = [f["uid"] for f in frames]
    assert len(set(uids)) == len(uids), (
        "uid 撞车：判定/归因按 uid 合并会再次把两拍合成一帧（Round 11 的坑）")
    for f in frames:
        assert str(f["uid"]).startswith(f"{f['id']}#"), (
            f"uid 不是 id#sha8 形状：{f['uid']}")


def test_clip_attribution_partitions_total(frames):
    """继承 + 新引入 必须等于总剪裁：对不上就是归因算错或在偷改口径。

    容差不是 0，也不是拍脑袋：本会话实测 169 帧里**最大偏差恰好 1e-6、
    没有一个超过 1.5e-6**（`metrics.json` 写盘时把分数舍入到 6 位小数，
    两个舍入值相加最坏就差 1.5e-6）。所以 1.5e-6 以下是舍入，
    **超过就是真错账**——那时这条锁必须响，别去调宽容差。
    """
    tol = 1.5e-6
    worst, wname, bad = 0.0, None, []
    for f in frames:
        for ch, v in (f.get("clip_attribution") or {}).items():
            if not isinstance(v, dict):
                continue
            t, i, r = (float(v.get(k) or 0.0) for k in
                       ("high_total", "high_inherited", "high_introduced"))
            if i < 0 or r < 0:
                bad.append(f"{f['uid']}/{ch} 负数")
                continue
            d = abs((i + r) - t)
            if d > worst:
                worst, wname = d, f"{f['uid']}/{ch}"
            if d > tol:
                bad.append(f"{f['uid']}/{ch} {i}+{r}≠{t}（差 {d:g}）")
    assert not bad, f"归因不守恒（样例 {bad[:3]}）；当前最大偏差 {worst:g} @ {wname}"


def test_low_side_refuses_to_claim_attribution(frames):
    """低端始终声明"不可归因"：本软件只声称能归因高端（物理对应唯一）。"""
    for f in frames:
        for v in (f.get("clip_attribution") or {}).values():
            if isinstance(v, dict):
                assert v.get("low") is None
                assert "不假装" in str(v.get("low_reason")), (
                    f"{f['uid']} 的低端理由文案丢了：{v.get('low_reason')}")


def test_coverage_block_matches_frame_count(frames):
    cov = json.loads(METRICS.read_text(encoding="utf-8"))["coverage"]
    assert cov["frames"] >= len(frames), (
        "coverage 报的帧数比可用帧还少，多半是又发生了静默覆盖")
    assert cov["unique_keys"] == cov["frames"], (
        "唯一键少于帧数：有帧在 uid 层面互相覆盖")
