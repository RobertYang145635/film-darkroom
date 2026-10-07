"""文档骨架锁：防"编辑时把整节标题覆盖掉"。

**为什么有这条**：本会话我在 `docs/blockers.md` 和 `docs/visual-acceptance.md`
里**连犯两次同一个错**——用整节标题当 `edit` 锚点做替换，新串里忘了带回调题，
结果**整节标题被吞**（正文成了孤儿）。两次都是事后 grep 才发现，
第二次我改成"提交前跑结构自检"，但那只救了一次提交，**没救代码库**。

这条锁把检查搬进测试：标题集合一旦缺项/重复就红。
**新增一节是允许的**——同步更新下面的期望集合即可，
那道"必须显式改测试"的摩擦正是本意（顺手删标题不再可能蒙混过关）。
"""

from __future__ import annotations

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _h2(name: str) -> list[str]:
    text = (ROOT / "docs" / name).read_text(encoding="utf-8")
    return re.findall(r"(?m)^## +(.+)$", text)


def test_blockers_has_exactly_one_of_each_b_number():
    heads = _h2("blockers.md")
    ids = [h.split("（")[0].strip() for h in heads if h.startswith("B-")]
    dup = [i for i in set(ids) if ids.count(i) > 1]
    assert not dup, f"同一个 B-n 出现多次（多半是插入时把旧条目切开了）：{dup}"
    missing = [f"B-{n}" for n in range(1, len(ids) + 1) if f"B-{n}" not in ids]
    assert not missing, f"缺号：{missing}；现有：{sorted(ids)}"


def test_acceptance_doc_keeps_its_sections():
    heads = _h2("visual-acceptance.md")
    # 只比**前缀**（标题里有中文引号，整串比对容易被引号形态绊倒）；
    # 新增/改名章节时**显式**改这里。
    expected = ("一、怎么签", "一·a、导出被挡", "三·a、判之前先读",
                "二、判据", "三、基准集覆盖了什么", "三·b、本批判定条",
                "四、历史")
    missing = [e for e in expected if not any(h.startswith(e) for h in heads)]
    assert not missing, (
        f"验收文档少了小节（被覆盖/误删？）：{missing}\n现有标题：{heads}")
    dup = [h for h in set(heads) if heads.count(h) > 1]
    assert not dup, f"标题重复：{dup}"


def test_negative_control_catches_a_swallowed_heading():
    """负控制：模拟"标题被编辑吞掉"，检测机制**必须响**。

    上一轮这条锁交付时我自己承认"没有负控制"——那按本会话的标准就是次优。
    不响的锁比没锁更糟：它给人"已经防住了"的错觉。
    """
    real = (ROOT / "docs" / "visual-acceptance.md").read_text(encoding="utf-8")
    broken = real.replace("## 二、判据", "判据", 1)   # 正文还在，标题没了
    assert broken != real, "锚文本没匹配上，负控制等于没跑"
    heads = re.findall(r"(?m)^## +(.+)$", broken)
    assert not any(h.startswith("二、判据") for h in heads), (
        "被吞的标题仍被识别为章节——检测逻辑不可信")


CHANGELOG_ANCHORS = ("B-7：门禁阻断不再冒充", "订正：我用中文字串当判据")


def _changelog_heads() -> list[str]:
    # CHANGELOG.md 在仓库根，不在 docs/ —— 用 _h2() 会读错目录
    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    return re.findall(r"(?m)^## +(.+)$", text)


def test_changelog_keeps_both_retro_sections():
    """变更日志的两条"账"必须各存在一次。

    补记 B-7 时我是**在别的条目中间插入**的——正是本会话两次吞标题的手法。
    这条锁盯住被插入点相邻的两条既有标题没被吃掉（少一条就说明出过手）。
    """
    heads = _changelog_heads()
    for a in CHANGELOG_ANCHORS:
        n = sum(1 for h in heads if h.startswith(a))
        assert n == 1, f"锚点 {a!r} 出现 {n} 次（应为 1）：日志条目可能被切开或重复"


def test_negative_control_catches_changelog_entry_split():
    """负控制：模拟"插入时把相邻条目标题吞掉"，锁必须响。"""
    real = (ROOT / "docs" / ".." / "CHANGELOG.md").read_text(encoding="utf-8")
    for anchor in CHANGELOG_ANCHORS:
        broken = real.replace("## " + anchor, anchor, 1)   # 标题降级成正文
        assert broken != real, f"锚文本没匹配上，负控制空跑：{anchor}"
        heads = re.findall(r"(?m)^## +(.+)$", broken)
        assert not any(h.startswith(anchor) for h in heads), (
            f"{anchor} 被降级仍算章节——检测不可信")


#: 历史快照里**给未来读者的指令**（"期望 N passed"）与**必须的限定**（以实测为准）。
STALE_INSTRUCTION = re.compile(r"期望\s*\**\s*\d+\s*passed")
MEASURED_NOTICE = re.compile(r"实测为准")
#: 声明必须是**文件开头的告示**，不能是正文里偶然出现的一句话。
HEAD_LINES = 5


def _dated_snapshots() -> list[tuple[str, str]]:
    """仓库根 + docs/ 下所有**文件名带日期**的 Markdown（历史快照）。"""
    out = []
    for p in sorted(list((ROOT / "docs").glob("*.md")) + list(ROOT.glob("*.md"))):
        if re.search(r"_20\d{6}", p.name):
            out.append((p.name, p.read_text(encoding="utf-8", errors="replace")))
    return out


def _offenders(rows: list[tuple[str, str]]) -> list[str]:
    bad = []
    for name, text in rows:
        if not STALE_INSTRUCTION.search(text):
            continue                                    # 没下指令，无需声明
        head = "\n".join(text.splitlines()[:HEAD_LINES])
        if not MEASURED_NOTICE.search(head):
            bad.append(name)
    return bad


def test_dated_docs_do_not_state_stale_expectations():
    """带日期的历史快照里，**「期望 N passed」这类指令**必须配"以实测为准"的声明。

    2026-10-07 实测：`交接_20260908.md` / `冷读与整改任务书_20260908.md` 里写着
    「跑测试…期望 **417 passed**」——417 是 2026-09-08 的真实值，**不该改写历史**，
    但**"期望"是给未来读者的指令**，不加限定就会让人以为环境坏了。
    所以**不删数字，只在文件头加声明**，并锁住这个要求。
    （本条**故意不写"当前 N passed"**：交付当轮的条数会被下一轮新增用例变成历史值，
    那正是本锁要防的东西——数字只能来自**当场打印**。）
    """
    bad = _offenders(_dated_snapshots())
    assert not bad, (
        "这些历史快照给了『期望 N passed』的指令，文件头却没有『以当场实测为准』的声明："
        f"{bad}")


def test_anti_vacuity_lock_actually_sees_snapshots_with_instructions():
    """反空断言：真实库里**必须至少 1 个**带指令的快照被判定到。

    正则一旦与文档措辞脱节（改字、改加粗），本锁会安静地扫不到任何文件而"全绿"。
    """
    seen = [name for name, text in _dated_snapshots() if STALE_INSTRUCTION.search(text)]
    assert seen, "没有任何历史快照带『期望 N passed』指令——正则已失效，本锁是空锁"
    assert not _offenders([(n, t) for n, t in _dated_snapshots()
                           if STALE_INSTRUCTION.search(t)]), (
        "带指令的快照必须全过判定")


def test_lock_has_teeth_dropping_the_notice_is_caught():
    """反空断言：把某个真实快照开头的声明抹掉，本锁**必须**点名它。

    （对照实验**只在内存里做**，绝不写文件。）
    """
    rows = [(n, t) for n, t in _dated_snapshots() if STALE_INSTRUCTION.search(t)]
    assert rows, "找不到带指令的快照，负控制无对象"
    name, text = rows[0]
    stripped = "\n".join(ln for ln in text.splitlines()
                         if MEASURED_NOTICE.search(ln) is None)
    assert stripped != text, f"锚文本没匹配上，负控制等于没跑：{name}"
    assert _offenders([(name, stripped)]) == [name], (
        f"抹掉『{name}』的声明却没报错——本锁是空锁")
