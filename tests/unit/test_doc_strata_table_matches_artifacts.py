"""锁：`docs/visual-acceptance.md` 第三节**分层表的每一格**都必须等于工件实算值。

**由来（实测）**：2026-09-21 逐格重算发现 **6 格已过期**
（`few_neutral_surfaces` 116/41→25/9、`hue_concentrated` 36/23→2/2、
`saturated_dominant` 23/17→2/2、`bright_uniform_top` 选中 28→32、
`low_result_contrast` 选中 20→25），**另有整行漏写**（`no_nlp_reference` 37/7）。
根因是**分层阈值改过而文档没跟着重算**——`saturated_dominant` 那行的备注
本身就写着"阈值 0.6 会全命中，故用 0.85"，说明阈值确实动过。

本锁把"**文档数字 = 工件数字**"变成机器判定：**会变的是阈值，能守住的是锁**。
"""

from __future__ import annotations

import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[2]
DOC = ROOT / "docs" / "visual-acceptance.md"

#: 容忍 Markdown 加粗：`night_like` 那行写成 `` **`night_like` 夜景** | **1** | ``，
#: 第一版正则只认不加粗的行 ⇒ **完整性锁当场抓到漏读**（这就是它该有的样子）。
ROW = re.compile(r'^\|\s*\**\s*`([a-z_]+)`[^|]*?\|\s*\**(\d+)\**\s*\|\s*\**(\d+)\**\s*\|')


def _frames(path: pathlib.Path) -> list[dict]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    fr = obj["frames"] if isinstance(obj, dict) else obj
    return fr if isinstance(fr, list) else list(fr.values())


def _flags(f: dict) -> set[str]:
    """metrics 里 `strata` 是 dict，decisions 里是 **list**（实测形状不同）。"""
    s = f.get("strata")
    if isinstance(s, dict):
        return {k for k, v in s.items() if v}
    if isinstance(s, list):
        out = set()
        for x in s:
            if isinstance(x, str):
                out.add(x)
            elif isinstance(x, dict):
                k = x.get("key") or x.get("name")
                if k and (x.get("hit", True)):
                    out.add(k)
        return out
    return set()


def _counts(rows: list[dict]) -> dict[str, int]:
    c: dict[str, int] = {}
    for f in rows:
        for k in _flags(f):
            c[k] = c.get(k, 0) + 1
    return c


def _table(text: str) -> dict[str, tuple[int, int]]:
    sec = text.split("## 三、")[1].split("## 四")[0]
    got = {}
    for line in sec.splitlines():
        m = ROW.match(line)
        if m:
            got[m.group(1)] = (int(m.group(2)), int(m.group(3)))
    return got


def _diff(got: dict[str, tuple[int, int]], corpus: dict[str, int],
          picked: dict[str, int]) -> list[str]:
    errs = []
    for k, (a, b) in sorted(got.items()):
        if a != corpus.get(k, 0):
            errs.append(f"{k} 全语料：文档 {a} ≠ 实算 {corpus.get(k, 0)}")
        if b != picked.get(k, 0):
            errs.append(f"{k} 选中：文档 {b} ≠ 实算 {picked.get(k, 0)}")
    return errs


def test_table_is_not_empty_of_rows():
    """反空断言①：解析器**必须真读到行**，否则下面的比较是空对空。"""
    got = _table(DOC.read_text(encoding="utf-8"))
    assert len(got) >= 9, f"只解析到 {len(got)} 行，表结构或正则已失效"


def test_every_cell_matches_artifacts():
    corpus = _counts(_frames(ROOT / "fixtures" / "verdicts" / "metrics.json"))
    picked = _counts(_frames(ROOT / "fixtures" / "verdicts" / "decisions.json"))
    errs = _diff(_table(DOC.read_text(encoding="utf-8")), corpus, picked)
    assert not errs, "分层表与工件不符：\n  " + "\n  ".join(errs)


def test_no_nonzero_stratum_is_silently_omitted():
    """完整性：实算**非零**的分层都必须有行（新增分层漏写文档会被抓）。"""
    corpus = _counts(_frames(ROOT / "fixtures" / "verdicts" / "metrics.json"))
    got = _table(DOC.read_text(encoding="utf-8"))
    missing = [k for k, n in corpus.items() if n > 0 and k not in got]
    assert not missing, f"这些分层在语料里非零却没写进表：{missing}"


def test_documented_zero_row_is_really_zero():
    """`shadow_clip` 是**文档化的零**（语料里确实没有），不能悄悄变成非零。"""
    corpus = _counts(_frames(ROOT / "fixtures" / "verdicts" / "metrics.json"))
    got = _table(DOC.read_text(encoding="utf-8"))
    assert "shadow_clip" in got, "`shadow_clip` 那行没了（它是有意记录的语料缺口）"
    assert got["shadow_clip"][0] == corpus.get("shadow_clip", 0) == 0


def test_lock_has_teeth_wrong_number_is_caught():
    """反空断言②：把文档里任一数字改错一格，本锁**必须**报错。

    （对照实验**只在内存里做**，绝不写文件。）
    """
    text = DOC.read_text(encoding="utf-8")
    corpus = _counts(_frames(ROOT / "fixtures" / "verdicts" / "metrics.json"))
    picked = _counts(_frames(ROOT / "fixtures" / "verdicts" / "decisions.json"))
    got = _table(text)
    key = next(k for k, (a, b) in got.items() if a > 0)
    a, b = got[key]
    bad_text = text.replace(f"`{key}`", f"`{key}` 篡改哨兵", 1)
    bad = {key: (a + 1, b)}
    assert _diff(bad, corpus, picked), "改错数字却没报错——本锁是空锁"
    assert _table(bad_text).get(key) == (a, b), "行内容变了，解析应仍读到原值"


def test_per_roll_line_matches_artifacts():
    """**逐卷那行**也纳入同一把锁：`卷 选中/总数`，一格不符即红。

    2026-09-21 实测：原文档**总数全对、但 5 卷里 4 卷的"选中"是错的**
    （6/9/18/6/9 合计 48，实测 6/8/20/7/7 合计也是 48）——
    **合计对不代表分布对**，只看合计的锁抓不到这种漂移。
    """
    corpus = _frames(ROOT / "fixtures" / "verdicts" / "metrics.json")
    picked = _frames(ROOT / "fixtures" / "verdicts" / "decisions.json")
    tot, pk = {}, {}
    for f in corpus:
        tot[f["roll"]] = tot.get(f["roll"], 0) + 1
    for f in picked:
        pk[f["roll"]] = pk.get(f["roll"], 0) + 1
    text = DOC.read_text(encoding="utf-8")
    line = next((ln for ln in text.splitlines() if ln.startswith("逐卷")), "")
    assert line, "找不到『逐卷』那行，锁本身失效"
    got = {m.group(1): (int(m.group(2)), int(m.group(3)))
           for m in re.finditer(r"`([^`]+)`\s*(\d+)/(\d+)", line)}
    assert len(got) == len(tot), (
        f"逐卷行只覆盖 {len(got)} 卷，语料有 {len(tot)} 卷（漏卷=静默失真）")
    bad = [f"{r}: 文档 {got.get(r)} ≠ 实算 ({pk.get(r, 0)}/{tot[r]})"
           for r in tot if got.get(r) != (pk.get(r, 0), tot[r])]
    assert not bad, "逐卷数字与工件不符：\n  " + "\n  ".join(bad)
