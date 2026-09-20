"""回归锁：局部测量**不许**把整份语料覆盖掉（本会话真实踩过的坑）。

2026-09-09 实测：为了验证 `--measure` 新加的字段，跑了一次
`build_verdict_kit.py --measure --roll 1298_G200`，
`fixtures/verdicts/metrics.json` 从 169 帧被**整份覆盖成 12 帧**。
下游（分层抽样、判定条覆盖率、`verify_verdicts.py`）读的都是这个文件：
语料缩水不报错，只会让"覆盖率"看起来变好 —— 又一个假绿形态。
"""

from __future__ import annotations

from negative_lab.diagnostics.verdicts import merge_metrics


def _corpus():
    """5 卷 × 若干帧的假语料。"""
    frames = [{"id": f"r{r}::{i}", "roll": f"r{r}", "base_source": "frame"}
              for r in range(5) for i in range(12)]
    return {"schema_version": 3, "frames": frames}


def test_partial_run_must_not_shrink_the_corpus():
    existing = _corpus()
    fresh = {"schema_version": 3,
             "frames": [{"id": "r0::1", "roll": "r0", "base_source": "generic_prior"},
                        {"id": "r0::2", "roll": "r0", "base_source": "frame"}]}
    doc = merge_metrics(existing, fresh, partial=True)
    assert len(doc["frames"]) == 60, "局部跑把 60 帧变成 2 帧就是本会话踩的那个坑"
    assert doc["coverage"]["partial_run"] is True
    assert doc["coverage"]["this_run_frames"] == 2
    assert doc["coverage"]["kept_from_previous"] == 58
    by_id = {f["id"]: f for f in doc["frames"]}
    assert by_id["r0::1"]["base_source"] == "generic_prior", "同 id 必须用本次的新值"
    assert by_id["r4::11"]["roll"] == "r4", "没跑到的卷必须原样留着"


def test_frames_stay_sorted_and_deduplicated():
    existing = {"frames": [{"id": "a::2", "roll": "a"}, {"id": "a::1", "roll": "a"}]}
    fresh = {"frames": [{"id": "a::1", "roll": "a", "gate": "warn"}]}
    doc = merge_metrics(existing, fresh, partial=True)
    assert [f["id"] for f in doc["frames"]] == ["a::1", "a::2"]
    assert doc["coverage"]["rolls"] == 1


def test_full_run_replaces_but_reports_the_shrinkage():
    existing = _corpus()
    fresh = {"frames": [{"id": "r0::1", "roll": "r0"}]}
    doc = merge_metrics(existing, fresh, partial=False)
    assert len(doc["frames"]) == 1
    assert doc["coverage"]["dropped_keys"] == 59, (
        "全量跑可以整份替换，但消失的键必须写出来，不能静默")


def test_full_run_with_same_ids_reports_no_loss():
    existing = _corpus()
    fresh = {"frames": [dict(f, gate="ok") for f in existing["frames"]]}
    doc = merge_metrics(existing, fresh, partial=False)
    assert doc["coverage"]["dropped_keys"] == 0
    assert all(f.get("gate") == "ok" for f in doc["frames"])


def test_no_previous_file_is_not_a_partial_merge():
    fresh = {"frames": [{"id": "z::1", "roll": "z"}]}
    doc = merge_metrics(None, fresh, partial=True)
    assert len(doc["frames"]) == 1
    assert doc["coverage"]["partial_run"] is False, (
        "没有旧语料时无可并入，标 partial 会误导下游"
        "（覆盖率报告会说'已并入'，其实只有这一次）")
