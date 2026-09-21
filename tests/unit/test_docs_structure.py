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
