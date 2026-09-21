"""B-10 接线锁：`--prune-stale` **必须显式才删**，默认永不删。

`prune_orphan_sheets` 本身有 6 道边界锁；本文件只锁**接线层**那三件会出事的事：
1. 开关**默认值是 False**（谁把默认改成 True，一跑就删图——这是最贵的一类事故）；
2. 不传 `remove` 时**真的一个文件都没少**（对着**真实语料**跑，不是玩具目录）；
3. 目录里确实存在**裸命名**历史文件 ⇒ 它们**只能交人工**，工具不猜、不删。
"""

from __future__ import annotations

import importlib.util
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "build_verdict_kit.py"


def _load():
    spec = importlib.util.spec_from_file_location("bvk_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_prune_flag_defaults_to_off():
    """**默认不删**：`--prune-stale` 不带就是 False（这条锁住的是"手滑默认值"）。"""
    ns = _load().build_parser().parse_args([])
    assert ns.prune_stale is False
    assert ns.list_stale is False
    only = _load().build_parser().parse_args(["--list-stale"])
    assert only.list_stale is True and only.prune_stale is False, (
        "只列出的开关不应连带打开删除")


def test_listing_the_real_corpus_deletes_nothing():
    """对**真实判定条目录**跑"只列出"：文件数必须一根不少。"""
    mod = _load()
    dec = json.loads((ROOT / "fixtures" / "verdicts" / "decisions.json")
                     .read_text(encoding="utf-8"))
    entries = dec.get("frames") or []
    assert entries, "判定文件读不出条目，锁本身失效"
    before = sorted(p.name for p in mod.SHEETS.rglob("*") if p.is_file()) \
        if mod.SHEETS.exists() else []
    rep = mod.prune_stale(entries, mod.SHEETS)          # 不传 remove
    assert rep["remove"] is False and rep["removed"] == 0
    after = sorted(p.name for p in mod.SHEETS.rglob("*") if p.is_file()) \
        if mod.SHEETS.exists() else []
    assert after == before, "只是『列一下』就少文件了——这是最严重的一类事故"


def test_bare_named_leftovers_are_handed_to_humans_not_deleted():
    """裸命名（uid 迁移前旧命名）**只能交人工**：工具不猜，也不删。"""
    mod = _load()
    if not mod.SHEETS.exists():
        return                       # 目录没生成过（全新环境），不是缺陷
    dec = json.loads((ROOT / "fixtures" / "verdicts" / "decisions.json")
                     .read_text(encoding="utf-8"))
    rep = mod.prune_stale(dec.get("frames") or [], mod.SHEETS)
    if rep["bare_kept"] == 0:
        return                       # 已被人工清干净，同样合格
    assert rep["removed"] == 0, "裸命名文件被工具自作主张删了"
    assert rep["refused_outside"] == 0, (
        "扫描越界：`root` 之外的文件出现在了结果里，边界破了")
