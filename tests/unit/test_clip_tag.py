"""回归锁：高光归因标签必须**由数值决定**，不能由文案决定。

两条本会话真实踩过的坑（第 1 条我的初稿说过头了，此处按实测订正）：

1. 判定条标签原来写成
   `if note_clip: short += ("[高光:采集端已丢失]" if "采集端" in note_clip else "[高光:软件引入]")`。
   实测复核：`clip_note()` 在剪裁 ≤ 告警线（0.1%）时返回空串，旧逻辑因此**根本不打标签**
   ——是"**该说的没说**"，不是"标错"（本语料没有纯软件引入的帧，
   所以历史上没有任何判定条被误标成软件引入；我最初的措辞夸大了后果，已改）。
   真正的改进是：**低于告警线的继承剪裁现在也有正确的归因**，且多了"混合"一档。
2. 同一天我用子串 `采集端已丢失` 去数条目，真实文案是
   "采集端**就**已丢失"，一个都没匹配上，
   于是写下"12 条全是软件引入、0 条继承"的**假结论**（已订正）。
   数值在手，就不要拿中文字串当判据。
"""

from __future__ import annotations

from negative_lab.diagnostics.explain import clip_note, clip_tag


def _attr(total, inherited, introduced):
    return {"R": {"high_total": total, "high_inherited": inherited,
                  "high_introduced": introduced, "low": None, "low_reason": "…"},
            "G": {"high_total": 0.0, "high_inherited": 0.0,
                  "high_introduced": 0.0, "low": None, "low_reason": "…"},
            "B": {"high_total": 0.0, "high_inherited": 0.0,
                  "high_introduced": 0.0, "low": None, "low_reason": "…"}}


def test_inherited_only_is_labelled_capture_side_even_below_the_warning_line():
    """这就是那个 bug：低于告警线 -> clip_note 返回空串 -> 旧逻辑误标软件引入。"""
    attr = _attr(0.0005, 0.0005, 0.0)          # 0.05% < CLIP_WARN_FRAC 0.1%
    assert clip_note(_GateStub(attr)) == "", "前提：文案确实是空的（低于告警线）"
    assert clip_tag(attr) == " [高光:采集端已丢失]", (
        "文案为空不等于软件引入；100% 继承必须标采集端")


def test_introduced_only_is_labelled_software():
    assert clip_tag(_attr(0.00029, 0.0, 0.00029)) == " [高光:软件引入]"


def test_mixed_shares_the_inherited_percentage_instead_of_choosing_a_side():
    got = clip_tag(_attr(0.02, 0.015, 0.005))
    assert got.startswith(" [高光:混合") and "75%" in got, got


def test_no_clip_gets_no_tag():
    assert clip_tag(_attr(0.0, 0.0, 0.0)) == ""
    assert clip_tag(None) == ""
    assert clip_tag({}) == ""


class _GateStub:
    """clip_note 接受 QualityGate 或 dict；这里只喂它需要的那部分。"""

    def __init__(self, attribution):
        self.clip_attribution = attribution
        self.clip = {"low": [0, 0, 0], "high": [0.0005, 0, 0], "n": 100}
        self.status = "needs_review"

    def to_dict(self):
        return {"clip": self.clip, "clip_attribution": self.clip_attribution,
                "status": self.status}
