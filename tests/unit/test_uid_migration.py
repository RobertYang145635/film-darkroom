"""回归锁：换了键（id → id#sha8）也不许弄丢用户已经签下的判定。

2026-09-09 实测：把判定文件的合并键从 `id` 改成 `uid_of()` 之后，
第一次 `--select` 让旧 48 条**全部**落进 `retired_ids` ——
因为旧条目的键是 `CN400-1`，新条目的键是 `CN400-1#84c88a2a`。
当时侥幸 0 签；如果用户已经签了 20 帧，重跑一次工具就等于**把签名擦了**。
所以改键这件事本身必须有锁：

1. 旧条目（无 uid）在**该 id 只对应一个新帧**时必须认亲到新键，判定原样保留；
2. 该 id 对应**多个**新帧（同 id 两次拍摄）时**不许猜**：
   签名条目原样留着，两个新帧各建空白条目，宁多不认错；
3. 只是换了键**不算退休**，`retired_ids` 不许因此暴涨。
"""

from __future__ import annotations

import json
import pathlib

from negative_lab.diagnostics.verdicts import merge_decisions, uid_of

SHA_A = "aaaa111122223333"
SHA_B = "bbbb999988887777"


def _signed_legacy():
    return {"frames": [{"id": "r::1", "roll": "r", "file": "A/1.NEF",
                        "verdict": "usable", "tweak_steps": 0, "issue_tags": [],
                        "reference_product": "满意", "note": "肤色对",
                        "signed_by": "YZH"}]}


def test_signed_entry_follows_the_new_key_when_the_id_is_unambiguous():
    selected = [{"id": "r::1", "roll": "r", "file": "A/1.NEF",
                 "source_sha256": SHA_A, "strata": {}}]
    doc = merge_decisions(_signed_legacy(), selected)
    assert len(doc["frames"]) == 1, (
        f"认亲失败会多出一条空白判定：{[f['id'] for f in doc['frames']]}")
    entry = doc["frames"][0]
    assert entry["verdict"] == "usable" and entry["signed_by"] == "YZH", (
        "换键把用户签下的判定弄丢了")
    assert entry["uid"] == uid_of(selected[0]) == f"r::1#{SHA_A[:8]}"
    assert doc["retired_ids"] == [], (
        "只是换了键，不算退休；retired_ids 暴涨会让人以为语料被砍")


def test_ambiguous_id_never_guesses_whose_signature_it_is():
    selected = [{"id": "r::1", "roll": "r", "file": "A/1.NEF",
                 "source_sha256": SHA_A, "strata": {}},
                {"id": "r::1", "roll": "r", "file": "B/1.NEF",
                 "source_sha256": SHA_B, "strata": {}}]
    doc = merge_decisions(_signed_legacy(), selected)
    by_uid = {f.get("uid") or f["id"]: f for f in doc["frames"]}
    assert len(doc["frames"]) == 3, (
        "同 id 两次拍摄 + 一条旧签名：签名条目与两个新帧都要在，"
        "把签名硬塞给其中任何一个都是造假")
    legacy = by_uid["r::1"]
    assert legacy["signed_by"] == "YZH" and legacy["file"] == "A/1.NEF"
    for sha in (SHA_A, SHA_B):
        assert not by_uid[f"r::1#{sha[:8]}"]["verdict"], (
            "新捕获必须是空白条目（blank_verdict 以 None 表示未签），等用户自己判")
    assert doc["retired_ids"] == []


def test_unsigned_legacy_entry_may_retire_quietly():
    """没签的旧条目被新键取代是正常回收，但必须在 retired_ids 留痕。"""
    existing = {"frames": [{"id": "r::9", "roll": "r", "verdict": "",
                            "signed_by": ""}]}
    doc = merge_decisions(existing, [{"id": "r::1", "roll": "r",
                                      "source_sha256": SHA_A, "strata": {}}])
    assert [f["uid"] for f in doc["frames"]] == [f"r::1#{SHA_A[:8]}"]
    assert doc["retired_ids"] == ["r::9"], "未签条目退掉也要留痕，不能无声消失"


def _load_kit():
    import importlib.util
    import sys
    script = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "build_verdict_kit.py"
    sys.path.insert(0, str(script.parent))
    spec = importlib.util.spec_from_file_location("build_verdict_kit_under_test", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_render_sheets_refuses_to_call_zero_sheets_a_success(tmp_path, monkeypatch):
    """`--sheets` 曾经一张都不渲染却退出 0（索引键不一致 -> 每帧 continue）。

    静默的"渲染 0 张"是假绿：下一步 `--verify` 拿着空目录照样跑完。
    这里用一对故意对不上的 decisions/metrics 触发它，必须**非零退出**。
    """
    import pytest

    bk = _load_kit()
    decisions = tmp_path / "decisions.json"
    metrics = tmp_path / "metrics.json"
    decisions.write_text(json.dumps({"frames": [
        {"id": "nope::1", "roll": "nope", "uid": "nope::1#deadbeef",
         "verdict": None, "signed_by": ""}]}, ensure_ascii=False), encoding="utf-8")
    metrics.write_text(json.dumps({"frames": []}), encoding="utf-8")
    monkeypatch.setattr(bk, "DECISIONS", decisions)
    monkeypatch.setattr(bk, "METRICS", metrics)
    with pytest.raises(SystemExit) as exc:
        bk.render_sheets()
    assert "一张判定条都没渲染" in str(exc.value)
