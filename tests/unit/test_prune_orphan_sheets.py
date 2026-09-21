"""B-10 清理半的**边界锁**：这是一个会删图的函数，锁先于功能。

`prune_orphan_sheets` 的默认值必须是"只列不删"，越界/裸命名/被引用三类
**一个都不许动**。这里每条锁都对应 `verdicts.py` 里写死的一条边界；
少一条都不许放开 `remove`。
"""

from __future__ import annotations

import pathlib

from negative_lab.diagnostics.verdicts import prune_orphan_sheets


def _mk(d: pathlib.Path, name: str) -> pathlib.Path:
    d.mkdir(parents=True, exist_ok=True)
    p = d / name
    p.write_bytes(b"x")
    return p


def _entries(names):
    return [{"uid": n, "sheet": n} for n in names]


def test_default_is_dry_run_and_deletes_nothing(tmp_path):
    """① 默认**不许删**：忘了带开关也不能把图删了。"""
    _mk(tmp_path, "A__aaaaaaaa.jpg")            # 无人引用 → 候选
    r = prune_orphan_sheets(_entries([]), root=tmp_path)
    assert r["candidates"] == 1
    assert r["removed"] == 0
    assert (tmp_path / "A__aaaaaaaa.jpg").exists(), "默认调用竟删了文件"


def test_referenced_files_are_never_deleted(tmp_path):
    """④ 守恒：被引用的一个不少。"""
    _mk(tmp_path, "A__aaaaaaaa.jpg")
    _mk(tmp_path, "B__bbbbbbbb.jpg")
    _mk(tmp_path, "C__cccccccc.jpg")            # 孤儿
    r = prune_orphan_sheets(_entries(["A__aaaaaaaa.jpg", "B__bbbbbbbb.jpg"]),
                            root=tmp_path, remove=True)
    assert r["referenced_kept"] == 2
    assert r["candidates"] == 1 and r["removed"] == 1
    assert (tmp_path / "A__aaaaaaaa.jpg").exists()
    assert (tmp_path / "B__bbbbbbbb.jpg").exists()
    assert not (tmp_path / "C__cccccccc.jpg").exists()


def test_bare_named_files_are_never_touched(tmp_path):
    """③ 裸命名（用户自己的照片 / 迁移前旧图）**永不删**，只交回给人看。"""
    _mk(tmp_path, "holiday.jpg")                 # 像用户自己的图
    _mk(tmp_path, "G200-3.jpg")                  # 迁移前旧命名
    _mk(tmp_path, "A__aaaaaaaa.jpg")
    r = prune_orphan_sheets(_entries(["A__aaaaaaaa.jpg"]), root=tmp_path,
                            remove=True)
    assert (tmp_path / "holiday.jpg").exists(), "删了用户的图，不可挽回"
    assert (tmp_path / "G200-3.jpg").exists(), "裸命名只能由人判断，工具不许猜"
    assert r["bare_kept"] == 2, r
    # 本用例里**没有**"我们造的孤儿"（A 被引用），所以候选与删除都必须是 0。
    # 我第一版期望 `removed == 1`，是**期望写错**而不是实现的错——
    # 当时若为了绿去松实现，就把一条安全行为弯掉了。
    assert r["candidates"] == 0 and r["removed"] == 0, r


def test_non_image_files_survive(tmp_path):
    _mk(tmp_path, "notes.txt")
    _mk(tmp_path, "A__aaaaaaaa.jpg")
    prune_orphan_sheets(_entries([]), root=tmp_path, remove=True)
    assert (tmp_path / "notes.txt").exists(), "只该碰图片，别的文件一律不动"


def test_nothing_outside_the_root_is_touched(tmp_path):
    """② 路径边界：`root` 之外的文件**碰都不碰**（清理脚本跑错目录时的最后防线）。"""
    outside = _mk(tmp_path.parent, f"outside_{tmp_path.parent.name}__deadbeef.jpg")
    inside = _mk(tmp_path, "A__aaaaaaaa.jpg")
    r = prune_orphan_sheets(_entries([]), root=tmp_path, remove=True)
    assert outside.exists(), "越界删除——这是最严重的一类事故"
    assert inside.exists() is False or r["removed"] == 1
    assert r["refused_outside"] == 0, "根外文件不该出现在扫描里，只该被挡住"


def test_missing_root_is_not_an_error(tmp_path):
    r = prune_orphan_sheets(_entries([]), root=tmp_path / "nope", remove=True)
    assert r["candidates"] == 0 and r["removed"] == 0
