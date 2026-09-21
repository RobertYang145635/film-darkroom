"""锁：验收报告正文必须带上「判定与证据图片脱钩」那一节。

Round 14 只把缺图告警打到 **stderr**——那东西在终端一闪，
而 `outputs/verdicts/verdict-report.md` 才是人会存档、日后回看的那份。
事后翻报告时"当时签的 20 帧其实一张图都没有"必须看得见。
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

from negative_lab.diagnostics.verdicts import summarize_verdicts

SCRIPT = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "verify_verdicts.py"


def _load():
    sys.path.insert(0, str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("verify_verdicts_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _summary(integrity):
    doc = {"frames": [
        {"uid": "R::1#a", "id": "R::1", "roll": "R", "strata": {},
         "verdict": "usable", "tweak_steps": None, "issue_tags": [],
         "reference_product": True, "note": "", "signed_by": "YZH"},
        {"uid": "R::2#b", "id": "R::2", "roll": "R", "strata": {},
         "verdict": "unusable", "tweak_steps": None, "issue_tags": [],
         "reference_product": False, "note": "死黑", "signed_by": "YZH"},
    ]}
    s = summarize_verdicts(doc)
    assert s["signed"] == 2, "前提塌了：样本本身没被判到，测的就不是渲染"
    s["sheet_integrity"] = integrity
    return s


def test_missing_sheets_appear_in_the_report_body():
    md = _load().render_md(_summary(
        {"entries": 2, "missing_sheet": 1, "signed_missing_sheet": 1,
         "samples": ["R::1#a：图片不存在 out/x.jpg"]}), signed_by=["YZH"])
    assert "判定与证据图片脱钩" in md
    assert "已签 1 条" in md, "已签却无图是重点，必须在正文里点名"
    assert "R::1#a" in md, "样例要一起进报告，否则不知道是哪条"


def test_healthy_integrity_does_not_add_the_warning():
    md = _load().render_md(_summary(
        {"entries": 2, "missing_sheet": 0, "signed_missing_sheet": 0,
         "samples": []}), signed_by=["YZH"])
    assert "判定与证据图片脱钩" not in md
    assert "## 签署人" in md, "报告结构本身要还在（别把正常报告改坏）"


# ---- B-10：陈旧判定条要进正文，但**没有陈旧时一个字都不许多** -------------

def test_stale_sheets_appear_in_the_report_body():
    """有陈旧图 → 正文必须出现，且**带张数与文件名**（只说"有问题"没用）。"""
    md = _load().render_md(_summary(
        {"entries": 2, "missing_sheet": 0, "signed_missing_sheet": 0,
         "samples": [], "stale_sheets": 3, "untracked_sheets": 0,
         "orphan_samples": ["G200-3__bfebd2f4.jpg"]}), signed_by=["YZH"])
    assert "陈旧" in md, "B-10 修好了却没进报告——用户依然看不到"
    assert "**3**" in md, "张数要写出来，才知道目录被污染到什么程度"
    assert "G200-3__bfebd2f4.jpg" in md, "要点名到文件，否则人还是只能按名字猜"
    assert "别按文件名挑图" in md, "这条警告的落点就是别按文件名挑图"


def test_zero_stale_adds_no_new_section():
    """陈旧为 0（哪怕字段存在）→ **不得**出现新小节，保护既有报告形态。"""
    md = _load().render_md(_summary(
        {"entries": 2, "missing_sheet": 0, "signed_missing_sheet": 0,
         "samples": [], "stale_sheets": 0, "untracked_sheets": 0,
         "orphan_samples": []}), signed_by=["YZH"])
    assert "陈旧" not in md
    assert "裸命名" not in md
    assert "## 签署人" in md


def test_untracked_only_is_reported_as_needing_human_judgement():
    """裸命名只报数并交给人：工具**不猜**它是不是垃圾，也**不许删**。"""
    md = _load().render_md(_summary(
        {"entries": 2, "missing_sheet": 0, "signed_missing_sheet": 0,
         "samples": [], "stale_sheets": 0, "untracked_sheets": 2,
         "orphan_samples": ["holiday.jpg"]}), signed_by=["YZH"])
    assert "裸命名" in md
    assert "**2**" in md
    assert "工具不删" in md, "必须写明不删，否则用户以为报告会自动清"
