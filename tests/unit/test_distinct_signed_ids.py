"""回归锁：帧数门槛不许被"同一张照片的两拍"灌水（至少要让看的人知道）。

2026-09-09 实测：`fixtures/acceptance/manifest.json` 的 169 个 required 帧
只有 **132 个 id**（同卷 `翻拍/` 与 `翻拍2/翻拍/` 同名，`source_sha256` 不同）。
判定条目现在按 `uid` 存，所以同 id 的两拍**可以各自占一个名额**——
"40–60 帧"的门槛就可能被同一场景的两拍顶着，覆盖率看着达标，实际样本更薄。

判定归用户，脚本一律不改；但 `verify_verdicts.py` 的报告必须摆出
`signed_distinct_ids`，并在 detail 里点名"有 N 条是同 id 的多次拍摄"。
"""

from __future__ import annotations

from negative_lab.diagnostics.verdicts import summarize_verdicts


def _entry(uid, fid, verdict="usable"):
    return {"uid": uid, "id": fid, "roll": "R1", "file": f"{fid}.NEF",
            "strata": {"highlight_clip": True}, "verdict": verdict,
            "tweak_steps": None, "issue_tags": [], "reference_product": True,
            "note": "", "signed_by": "YZH"}


def _doc(frames):
    return {"frames": frames}


def test_duplicate_captures_are_counted_once_as_distinct_ids():
    frames = [_entry("R1::1#aaaa1111", "R1::1"), _entry("R1::1#bbbb2222", "R1::1")]
    s = summarize_verdicts(_doc(frames))
    assert s["signed"] == 2
    assert s["signed_distinct_ids"] == 1, (
        "两拍同名必须暴露成 1 个不同帧 id，否则覆盖率是灌出来的")
    detail = next(c["detail"] for c in s["checks"] if "已签帧数" in c["label"])
    assert "去重后只有 1" in detail and "多次拍摄" in detail, (
        f"报告里看不到重复样本的提示：{detail}")


def test_no_warning_when_every_signed_entry_is_a_different_frame():
    frames = [_entry("R1::1#aaaa1111", "R1::1"), _entry("R1::2#cccc3333", "R1::2")]
    s = summarize_verdicts(_doc(frames))
    assert s["signed_distinct_ids"] == 2
    detail = next(c["detail"] for c in s["checks"] if "已签帧数" in c["label"])
    assert "⚠" not in detail
