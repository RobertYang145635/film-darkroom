"""回归锁：验收环节必须能看出「判定条目与判定条图片脱钩」。

2026-09-09 实测：改键后 `--select` 生成的 48 条 `sheet` 全空，
而 `--sheets` 当时**一张都没渲染却退出码 0**。
那种状态下即便签了字，`verify_verdicts.py` 也只读 JSON 里的判定字段，
报告照样"通过"——**没人能复核这条判的到底是哪张图**。
"""

from __future__ import annotations

import copy

from negative_lab.diagnostics.verdicts import sheet_integrity


def _entry(tmp_path, name="G200-1.jpg", signed=False, uid="r::1#aaaa1111"):
    (tmp_path / name).write_bytes(b"fake jpg")
    return {"id": "r::1", "uid": uid, "sheet": f"{tmp_path / name}",
            "verdict": "usable" if signed else None,
            "signed_by": "YZH" if signed else ""}


def test_existing_sheet_is_not_flagged(tmp_path):
    got = sheet_integrity([_entry(tmp_path)], root=tmp_path)
    assert got["entries"] == 1
    assert got["missing_sheet"] == 0 and got["signed_missing_sheet"] == 0


def test_missing_image_is_counted_and_quoted(tmp_path):
    e = _entry(tmp_path)
    e["sheet"] = f"{tmp_path / 'gone.jpg'}"
    got = sheet_integrity([e], root=tmp_path)
    assert got["missing_sheet"] == 1
    assert got["samples"] and "图片不存在" in got["samples"][0]


def test_signed_entry_without_image_is_called_out_separately(tmp_path):
    """已签 + 无图 = 最危险的一种：看起来有证据，其实复核不了。"""
    e = _entry(tmp_path, signed=True)
    e["sheet"] = ""
    got = sheet_integrity([e], root=tmp_path)
    assert got["missing_sheet"] == 1 and got["signed_missing_sheet"] == 1
    assert "sheet 为空" in got["samples"][0]


def test_check_is_read_only(tmp_path):
    """体检不许动判定：本项目里只有用户能写判定字段（v5 §10.2）。"""
    entries = [_entry(tmp_path, signed=True)]
    before = copy.deepcopy(entries)
    sheet_integrity(entries, root=tmp_path)
    assert entries == before


def test_no_entries_is_not_an_error():
    # 原意：空输入**不许崩**，也不许把"没判定"报成错误。
    # B-10 给返回值加了 3 个**只读报告字段**（陈旧/未登记判定条），
    # 这里**同步补全期望值而不是改成子集比较**——形状仍然全等卡死，
    # 将来谁再偷偷加字段，这条会照样拦住。
    for empty in ([], None):
        got = sheet_integrity(empty)
        assert got == {"entries": 0, "missing_sheet": 0,
                       "signed_missing_sheet": 0, "samples": [],
                       "stale_sheets": 0, "untracked_sheets": 0,
                       "orphan_samples": []}
