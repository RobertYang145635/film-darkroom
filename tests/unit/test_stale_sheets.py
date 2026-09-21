"""B-10 回归锁：判定条目录里的**陈旧图**必须被说出来（且不许动判定）。

上一轮实测：`outputs/verdicts/sheets/` 有 **96 张图对应 48 条判定**——
一半是 uid 迁移前的旧命名，而 `sheet_integrity()` 只**从条目往文件查**，
于是 `missing 0` 一脸绿；可**签判定的人是按文件名挑图的**，
完全可能对着一张陈旧渲染下判。锁的设计对应 B-10 修法里的锁①②：
① 同时验"多 1 个陈旧"和"缺 0"两头（只验一头都能骗过自己）；
② 陈旧图存在时，**判定统计一字不变**（报告项不许变成挡导出/翻判定）。
清理逻辑（会删图）**本轮刻意未实现**，故路径边界锁③④一并留到实现那轮。
"""

from __future__ import annotations

import copy

from negative_lab.diagnostics.verdicts import sheet_integrity


def _mk(tmp_path, names):
    for n in names:
        p = tmp_path / n
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")


def test_stale_and_missing_are_both_reported(tmp_path):
    """① 两头一起验：当前图齐全（missing=0），但目录里多一张带标记的陈旧图。"""
    _mk(tmp_path, ["A__aaaaaaaa.jpg", "B__bbbbbbbb.jpg", "C__cccccccc.jpg"])
    entries = [{"uid": "A#aaaaaaaa", "sheet": "A__aaaaaaaa.jpg"},
               {"uid": "B#bbbbbbbb", "sheet": "B__bbbbbbbb.jpg"}]
    r = sheet_integrity(entries, root=tmp_path)
    assert r["missing_sheet"] == 0, "当前引用的图都在，这半边要绿"
    assert r["stale_sheets"] == 1, (
        f"带标记却没被引用的图必须报出来，实得 {r}（这就是 B-10 的洞）")
    assert "C__cccccccc.jpg" in r["orphan_samples"]


def test_bare_named_files_are_untracked_not_stale(tmp_path):
    """裸命名（迁移前旧命名 **或** 用户自己的图）只报数，**不猜**是哪一种。"""
    _mk(tmp_path, ["A__aaaaaaaa.jpg", "G200-3.jpg"])
    entries = [{"uid": "A#aaaaaaaa", "sheet": "A__aaaaaaaa.jpg"}]
    r = sheet_integrity(entries, root=tmp_path)
    assert r["stale_sheets"] == 0
    assert r["untracked_sheets"] == 1
    assert "G200-3.jpg" in r["orphan_samples"]


def test_reporting_orphans_never_touches_the_verdicts(tmp_path):
    """② 只读保证：扫完目录，判定条目对象**一个字段都不许变**。"""
    _mk(tmp_path, ["A__aaaaaaaa.jpg", "old__deadbeef.jpg"])
    entries = [{"uid": "A#aaaaaaaa", "sheet": "A__aaaaaaaa.jpg",
                "verdict": None, "signed_by": ""}]
    before = copy.deepcopy(entries)
    sheet_integrity(entries, root=tmp_path)
    assert entries == before, "体检把判定改了——那就不再是『只报告』"


def test_no_root_means_no_directory_scan(tmp_path):
    """不传 root 就不碰文件系统（旧调用方语义不变，也不许凭空报错）。"""
    entries = [{"uid": "A", "sheet": "whatever.jpg"}]
    r = sheet_integrity(entries)
    assert r["stale_sheets"] == 0 and r["untracked_sheets"] == 0
