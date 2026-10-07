"""锁：**权威文档里点名的规格/报告文件必须真实存在**。

**由来（2026-10-07 实测）**：工作区里躺着 7 个「已删除、未提交」的旧规格与审查报告，
其中 `负片裁切与去色罩_AI执行规格_v5.md` 被 `README.md`「生效规格」明文写成
**仍然生效的产品范围基线**，全仓 `v5 §` 引用 **296 处**、涉及 **63 个源文件**
（`library/models.py:28` 那句"只有解码/处理错误阻断导出（v5 §3.3/§8.2）"就是 B-9
判定 `GATE_BLOCKED` 不进 `BLOCKING_STATUSES` 的**规范依据**），
`fixtures/acceptance/manifest.json` 的 `spec` 字段也指着它。
**而全量测试 577 项照样全绿**——**"绿"和"规范文档还在不在"之间没有任何连接**，
这正是本仓库反复记录过的那类失败：**破坏了东西而门禁看不见**。

本锁只盯**权威来源**（README 生效规格清单、CHANGELOG 的「规格：」条目、
验收 manifest 的 `spec` 字段）点名的**仓库内文档**，
不碰 `outputs/**`（被 .gitignore，全新克隆本来就没有）和生成物文件名。
"""

from __future__ import annotations

import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[2]

#: 只认**不带目录**的裸文件名；带 `/` 的是生成物路径（outputs/…），不归本锁管。
NAME = re.compile(r"`([^`\s/]+\.md)`")


def _readme_spec_names() -> list[str]:
    """README「## 生效规格」整节里反引号点名的文档（生效基线 + 历史资料清单）。"""
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    sec = text.split("## 生效规格")[1].split("\n## ")[0]
    return NAME.findall(sec)


def _changelog_spec_names() -> list[str]:
    """CHANGELOG 每条记录开头的「规格：`….md`」/「规格基线：`….md`」。"""
    names = []
    for line in (ROOT / "CHANGELOG.md").read_text(encoding="utf-8").splitlines():
        m = re.match(r"\s*\*?\*?规格(?:基线)?\*?\*?：\s*`([^`\s/]+\.md)`", line)
        if m:
            names.append(m.group(1))
    return names


def _manifest_spec_names() -> list[str]:
    """验收 manifest 的 `spec` 字段：`负片…_v5.md §10.1` → 取文件名部分。"""
    doc = json.loads((ROOT / "fixtures" / "acceptance" / "manifest.json")
                     .read_text(encoding="utf-8"))
    spec = str(doc.get("spec", ""))
    head = spec.split("§")[0].strip()
    return [head] if head.endswith(".md") and "/" not in head else []


def _referenced_names() -> list[str]:
    return (_readme_spec_names() + _changelog_spec_names() + _manifest_spec_names())


def _missing(names: list[str]) -> list[str]:
    """点名的文档在仓库根和 docs/ 都找不到 → 记为缺失。"""
    return [n for n in names
            if not ((ROOT / n).is_file() or (ROOT / "docs" / n).is_file())]


def test_named_spec_documents_exist():
    missing = sorted(set(_missing(_referenced_names())))
    assert not missing, (
        "这些文档被权威来源点名，但仓库里已经不存在（引用它们的 src 注释会变成"
        "指向空气的规范依据）：\n  " + "\n  ".join(missing))


def test_lock_sees_a_real_number_of_references():
    """反空断言①：**必须真解析到足够多的引用**，否则空集合也会"全绿"。

    2026-10-07 实测：README 7 条 + CHANGELOG 4 条 + manifest 1 条 = 12 条引用。
    阈值取 8（留余量），低于它说明锚点或小节标题已改，**本锁失效**。
    """
    names = _referenced_names()
    assert len(names) >= 8, f"只解析到 {len(names)} 条引用，解析锚点已失效：{names}"
    assert set(_readme_spec_names()) >= {"负片裁切与去色罩_AI执行规格_v5.md"}, (
        "README 不再点名 v5，说明生效基线的表述被改动了，请同步本锁的意图")


def test_lock_has_teeth_a_fabricated_name_is_caught():
    """反空断言②：喂一个**不存在**的文件名，缺失判定**必须**报出来。

    （只在内存里构造，绝不往仓库写假文件。）
    """
    ghost = "这份规格并不存在.md"
    assert _missing([ghost]) == [ghost], "凭空名字没被判缺失——本锁是空锁"
    real = _readme_spec_names()[:1]
    assert not _missing(real), "真实引用被判缺失，说明解析结果不可用"
    assert _missing(real + [ghost]) == [ghost], "混入假名后必须只报假名"
