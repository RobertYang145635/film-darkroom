"""锁：`clip_note`（给人看的句子）与 `clip_tag`（判定条标题）不许互相打架。

Round 16 的真实教训就是**同一份数据被两套口径解释**：
标签看文案、统计看子串，最后文案里一个"就"字让结论翻转。
现在两处都从 `clip_attribution` 取数，但**仍然是两个函数**——
以后谁改了其中一个的措辞或口径，这条锁要立刻响。

规则（方向一致，不锁具体措辞）：
- 句子说"采集端" ⇒ 标签必须是 `采集端已丢失` 或 `混合`；
- 句子说"本软件能改" ⇒ 标签必须是 `软件引入` 或 `混合`；
- 句子非空（剪裁过告警线）时，标签**也必须非空**。
"""

from __future__ import annotations

from negative_lab.diagnostics.explain import clip_note, clip_tag


def _attr(total, inherited, introduced):
    one = {"high_total": total, "high_inherited": inherited,
           "high_introduced": introduced}
    zero = {"high_total": 0.0, "high_inherited": 0.0, "high_introduced": 0.0}
    return {"R": one, "G": zero, "B": zero}


CASES = [
    ("纯继承", _attr(0.02, 0.02, 0.0)),
    ("纯引入", _attr(0.02, 0.0, 0.02)),
    ("混合", _attr(0.02, 0.015, 0.005)),
]


def test_note_and_tag_never_contradict_each_other():
    """注意：本测试的桩件必须让 `clip_note` **真的吐出句子**，否则两个
    方向断言一个都不会执行（我第一版就是这么空跑的，已加自证）。"""
    saw_capture_side = saw_software_side = False
    for name, attr in CASES:
        note, tag = clip_note({"clip_attribution": attr}), clip_tag(attr)
        assert tag, f"{name}：剪裁过告警线，标签却不许为空（note={note[:20]}）"
        assert note, f"{name}：`clip_note` 空返回，说明桩件不合它的口径"
        if "采集端" in note:
            saw_capture_side = True
            assert "采集端" in tag or "混合" in tag, \
                f"{name}：句子说采集端，标签却是 {tag}"
        if "本软件能改" in note:
            saw_software_side = True
            assert "软件引入" in tag or "混合" in tag, \
                f"{name}：句子说软件能改，标签却是 {tag}"
    assert saw_capture_side and saw_software_side, (
        "两种文案都没出现 -> 方向断言等于没跑（0/0 不算通过，v5 §10.2）")


def test_directions_are_distinguishable():
    """三档必须能分出至少三种标签——全糊成同一个等于没有归因。"""
    tags = {clip_tag(a) for _, a in CASES}
    assert len(tags) == 3, f"归因档位糊成一团：{tags}"
