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
