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

from negative_lab.diagnostics.verdicts import (  # noqa: E402
    sheet_integrity, strata_drift, summarize_verdicts)
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
    # 报告是要留下的那份东西：判定与图片脱钩只写在 stderr 里，
    # 人翻历史时看不见（Round 14 只做了告警，没进正文）。
    integ = s.get("sheet_integrity") or {}
    if integ.get("missing_sheet"):
        lines += ["", "## ⚠ 判定与证据图片脱钩", "",
                  f"- {integ['missing_sheet']}/{integ.get('entries', '?')} 条缺判定条图片，"
                  f"其中**已签 {integ.get('signed_missing_sheet', 0)} 条**（签了却无从复核）",
                  "- 先跑 `scripts/build_verdict_kit.py --sheets` 补齐，再签收"]
        lines += [f"  - {x}" for x in (integ.get("samples") or [])]
    # B-10：目录里的**陈旧判定条**必须进正文。签字的人是**按文件名挑图**的，
    # 只在 stderr 提醒等于没提醒（同 Round 14 那条教训的同一个坑）。
    stale = int(integ.get("stale_sheets") or 0)
    untracked = int(integ.get("untracked_sheets") or 0)
    if stale or untracked:
        lines += ["", "## ⚠ 目录里有陈旧/未登记判定条", ""]
        if stale:
            lines.append(
                f"- **{stale}** 张带本工具标记（`__<8hex>`）却**已无人引用**："
                "**别按文件名挑图签收**，先跑 `--sheets` 重渲染或按本清单核对")
        if untracked:
            lines.append(
                f"- **{untracked}** 张**裸命名**图片分不清归属（迁移前旧命名或用户自己的文件），"
                "**工具不删**，请人工确认")
        lines += [f"  - {x}" for x in (integ.get("orphan_samples") or [])]
    dr = s.get("strata_drift") or {}
    if dr.get("drifted"):
        lines += ["", "## ⚠ 分层标记已过期", "",
                  f"- {dr['drifted']}/{dr['checked']} 条的分层与最新 metrics 不一致"
                  "（多半是只重跑了 `--measure`，没重跑 `--select`）",
                  "- 判完的条目不会被动；但**分层是抽样依据**，过期请按新分层补抽",
                  ""]
        lines += [f"  - {x}" for x in (dr.get("samples") or [])]
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
    # 判定与证据图片必须能对上：本会话实测过 48 条 sheet 全空 + `--sheets`
    # 渲染 0 张却退出 0 的组合，那时若签了字，验收报告只看 JSON 一样"通过"，
    # 但已经没人能复核"这条判的是哪张图"。
    integ = sheet_integrity(doc.get("frames") or [], root=ROOT)
    s["sheet_integrity"] = integ
    metrics_path = ROOT / "fixtures" / "verdicts" / "metrics.json"
    if metrics_path.exists():
        s["strata_drift"] = strata_drift(
            doc.get("frames") or [],
            json.loads(metrics_path.read_text(encoding="utf-8")).get("frames") or [])
    if integ["missing_sheet"]:
        print(f"⚠ {integ['missing_sheet']}/{integ['entries']} 条判定**缺判定条图片**"
              f"（其中已签 {integ['signed_missing_sheet']} 条）：判定与证据图片已脱钩，"
              "先跑 scripts/build_verdict_kit.py --sheets 补齐再签收。样例：\n   "
              + "\n   ".join(integ["samples"]), file=sys.stderr)
    dr = s.get("strata_drift") or {}
    if dr.get("drifted"):
        print(f"⚠ {dr['drifted']}/{dr['checked']} 条判定的**分层标记已过期**"
              "（只重跑了 --measure 没重跑 --select，判定条上还写着旧分层）：\n   "
              + "\n   ".join(dr["samples"]) + "\n   重跑 --select 与 --sheets 后再签收。",
              file=sys.stderr)
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
