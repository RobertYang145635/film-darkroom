"""锁：判定条目上的分层必须能和**最新测量**对得上，且比较**不依赖存储形状**。

实测（2026-09-09）：当前 48 条与最新 `metrics.json` **零漂移**——
但这份一致纯属侥幸：以后只重跑 `--measure` 不重跑 `--select`，
`decisions.json` 里的分层会悄悄停在旧测量上，验收报告照读旧分层。

另一个坑是真的存在过：`decisions` 里 `strata` 是**列表**、
`metrics` 里是**字典**，直接 `==` 会全判成漂移（或全判成一致，取决于写法）。
"""

from __future__ import annotations

from negative_lab.diagnostics.verdicts import strata_drift


def test_list_shape_matches_dict_shape():
    """同一条分层信息，两种形状**不许**报成漂移。"""
    entries = [{"uid": "R::1#a", "id": "R::1",
                "strata": ["highlight_clip", "low_base_confidence"]}]
    metrics = [{"uid": "R::1#a", "id": "R::1",
                "strata": {"highlight_clip": True, "low_base_confidence": True,
                           "night_like": False}}]
    got = strata_drift(entries, metrics)
    assert got == {"checked": 1, "drifted": 0, "samples": [], "unmatched": 0}, got


def test_real_drift_is_reported_with_both_directions():
    entries = [{"uid": "R::1#a", "id": "R::1", "strata": ["highlight_clip"]}]
    metrics = [{"uid": "R::1#a", "id": "R::1",
                "strata": {"highlight_clip": True, "night_like": True}}]
    got = strata_drift(entries, metrics)
    assert got["drifted"] == 1
    assert "night_like" in got["samples"][0], got["samples"]
    assert "新测多" in got["samples"][0] and "条目多" in got["samples"][0]


def test_entries_missing_from_metrics_are_counted_not_guessed():
    got = strata_drift([{"uid": "R::9#z", "id": "R::9", "strata": []}],
                       [{"uid": "R::1#a", "id": "R::1", "strata": {}}])
    assert got["checked"] == 0 and got["unmatched"] == 1
    assert got["drifted"] == 0, "查不到新测不等于漂移，不许瞎报"


def test_empty_inputs_are_not_errors():
    assert strata_drift([], [])["checked"] == 0
