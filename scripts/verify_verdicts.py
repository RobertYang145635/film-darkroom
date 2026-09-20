"""汇总人工视觉判定，按 20260905 调研 §6.1 的三条判据给出**通过/不通过**。

判据（写死在 `negative_lab.diagnostics.verdicts.ACCEPTANCE_CRITERIA`）：
① 直接可用 ≥ 70%；③ 不可用 ≤ 5%；② 微调可救里 ≥ 80% 能在 2 步内救回；
另加三条本仓库自定的硬条件：**判定必须签满**（不得用部分样本冒充整体）、
基准集 ≥ 40 帧、满意参考成品 ≥ 20 张，以及分层与逐卷覆盖必须齐。

纪律：
- 判定只能由用户本人填写（v5 §10.2）。本脚本**只读**判定，绝不写判定字段；
- **没签满就判不通过**，并且把"未签 N 帧"顶在第一行——
  不许出现"已验收 12/48 → 通过"这种假绿；
- 输出报告写进 `outputs/verdicts/verdict-report.md`，退出码 0/1 供 CI/人判断。

运行：.venv\\Scripts\\python.exe scripts/verify_verdicts.py [--json]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from negative_lab.diagnostics.verdicts import summarize_verdicts  # noqa: E402
from negative_lab.io.sidecar import git_commit  # noqa: E402

DECISIONS = ROOT / "fixtures" / "verdicts" / "decisions.json"
REPORT = ROOT / "outputs" / "verdicts" / "verdict-report.md"


def render_md(s: dict, *, signed_by: list) -> str:
    lines = [
        "# 视觉验收汇总（人工判定基准集）",
        "",
        f"- 生成：{time.strftime('%Y-%m-%d %H:%M:%S')} · git `{git_commit()}`",
        "- 判定文件：`fixtures/verdicts/decisions.json`",
        f"- **结论：{'✅ 通过' if s['passed'] else '❌ 不通过'}**",
        "",
        "## 判据逐项",
        "",
        "图例：✅ 达标 · ❌ 不达标 · ⚪ 本轮无法判定（分母为 0，**不算通过**）。",
        "",
        "| 结果 | 判据 | 实测 |",
        "|---|---|---|",
    ]
    for c in s["checks"]:
        mark = "⚪" if c.get("na") else ("✅" if c["ok"] else "❌")
        lines.append(f"| {mark} | {c['label']} | {c['detail']} |")
    lines += [
        "",
        "## 判定分布",
        "",
        f"- 帧数：{s['total']}（已签 {s['signed']}，未签 {s['unsigned']}）",
        f"- ①直接可用 {s['usable']} · ②微调可救 {s['tweakable']} · "
        f"③不可用 {s['unusable']}",
        f"- ②中 {s['rescued_in_steps']} 帧在 2 步内救回；满意参考成品 "
        f"{s['reference_products']} 张",
        "",
        "## 分层覆盖（已签 / 该分层帧数）",
        "",
        "| 分层 | 已签 | 总数 |",
        "|---|---|---|",
    ]
    for k, d in sorted((s["coverage"].get("strata") or {}).items()):
        lines.append(f"| {k} | {d['signed']} | {d['frames']} |")
    lines += ["", "## 逐卷覆盖", "", "| 卷 | 已签 | 总数 |", "|---|---|---|"]
    for r, d in sorted((s["coverage"].get("rolls") or {}).items()):
        lines.append(f"| {r} | {d['signed']} | {d['frames']} |")
    lines += ["", "## 签署人", ""]
    lines += [f"- {who}" for who in (signed_by or ["（无）"])]
    lines += ["", "> 判定字段只能由用户本人填写；本脚本不写任何判定（v5 §10.2）。", ""]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", action="store_true", help="只打 JSON")
    a = ap.parse_args()
    if not DECISIONS.exists():
        print(f"❌ 找不到 {DECISIONS}\n先跑 scripts/build_verdict_kit.py --all 生成判定模板。",
              file=sys.stderr)
        return 2
    doc = json.loads(DECISIONS.read_text(encoding="utf-8"))
    s = summarize_verdicts(doc)
    if not s["signed"]:
        print("❌ 判定一条都没签 —— 视觉门禁**未通过**（不是「待办」，是没有证据）。\n"
              f"   请打开 {DECISIONS}，对照 outputs/verdicts/sheets/ 里的判定条逐帧填写。",
              file=sys.stderr)
    if not a.json:
        signed_by = sorted({str(f.get("signed_by")) for f in doc.get("frames") or []
                            if str(f.get("signed_by") or "").strip()})
        print(render_md(s, signed_by=signed_by))
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text(render_md(s, signed_by=signed_by), encoding="utf-8")
        print(f"报告已写入 {REPORT}")
    else:
        print(json.dumps(s, ensure_ascii=False, indent=2))
    return 0 if s["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
