"""B-8 取证脚本：连跑 N 轮测试，**每轮现场都归档**，然后报崩溃率。

为什么要有这个脚本（别再靠手敲连跑）：
本会话抓到 3 次 `0xC0000005` 原生崩溃，全都没留下现场——一次被我的清理
`Remove-Item` 删掉，一次因为传了 pytest 不认的参数，三次全是 `exit=4` 用法错误。
**崩溃时 stdout 只有半行进度假线、没有任何 FAILED 行**，所以"看有没有红字"
根本抓不住它；唯一的办法是把每轮的 stdout/stderr/junitxml **原样落盘**。

用法：
    .venv\\Scripts\\python.exe scripts/soak.py --rounds 6
    .venv\\Scripts\\python.exe scripts/soak.py --rounds 1 -k 某个用例 -v

输出每轮退出码与归类，末尾给崩溃率估计（样本小时**置信度很低**，别当结论）。
"""

from __future__ import annotations

import argparse
import datetime as _dt
import pathlib
import subprocess
import sys

# Windows 原生崩溃的退出码（有符号 32 位表示）
AV_SIGNED = -1073741819           # 0xC0000005 STATUS_ACCESS_VIOLATION


def classify_exit(rc: int) -> str:
    """把退出码翻译成人话。

    崩溃**不会**打印 FAILED，只体现在退出码上——所以这一步不能省。
    """
    if rc == 0:
        return "ok"
    if rc == AV_SIGNED:
        return "access-violation(0xC0000005) ← 原生崩溃，非测试失败"
    if rc == 4:
        return "usage-error ← 参数用错了，本轮对偶发性零信息量"
    if rc == 130:
        return "interrupted(Ctrl-C) ← 人为中断，不计入样本"
    if rc == 1:
        return "tests-failed ← 真失败，有 FAILED 可看"
    if rc == 2:
        return "interrupted/no-tests ← 看 stderr"
    return f"exit={rc}"


def main() -> int:
    ap = argparse.ArgumentParser(description="连跑多轮 pytest 并归档每轮现场")
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--outdir", default="outputs/soak")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("pytest_args", nargs=argparse.REMAINDER,
                    help="透传给 pytest 的其余参数（如 -k 表达式 / 指定文件）")
    a = ap.parse_args()

    args = [x for x in a.pytest_args if x != "-v"]
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    root = pathlib.Path(a.outdir) / stamp
    root.mkdir(parents=True, exist_ok=True)

    rows, crashes = [], 0
    for i in range(1, a.rounds + 1):
        cmd = [sys.executable, "-m", "pytest", "-q", "--tb=short",
               f"--junitxml={root / f'r{i}.xml'}"] + args
        with open(root / f"r{i}.stdout.txt", "wb") as so, \
                open(root / f"r{i}.stderr.txt", "wb") as se:
            rc = subprocess.call(cmd, stdout=so, stderr=se)
        kind = classify_exit(rc)
        tail = ""
        try:
            last = (root / f"r{i}.stdout.txt").read_text(
                encoding="utf-8", errors="replace").strip().splitlines()
            tail = last[-1][:60] if last else "(stdout 空)"
        except OSError:
            pass
        rows.append((i, rc, kind, tail))
        print(f"  第{i}轮: exit={rc} → {kind} | {tail}", flush=True)
        if rc != 0:
            crashes += 1
            if "access-violation" in kind or "usage-error" in kind:
                # 现场留着，并提示最该看的那一份
                print(f"    现场已归档：{root / f'r{i}.stderr.txt'}", flush=True)

    print(f"\n归档目录：{root}")
    bad = [r for r in rows if r[1] != 0]
    usage = [r for r in rows if "usage-error" in r[2]]
    valid = len(rows) - len(usage)
    real = [r for r in bad if "usage-error" not in r[2]]
    print(f"轮数 {len(rows)}｜非零 {len(bad)}（其中用法错误 {len(usage)}，"
          f"有效轮 {valid}）")
    if valid:
        print(f"异常率（不含用法错误）：{len(real)}/{valid} ≈ "
              f"{100.0 * len(real) / valid:.0f}%")
        print("样本这么小的置信度极低——它只说明『有没有』，不说明『多少次』。")
    return 1 if real else 0


if __name__ == "__main__":
    raise SystemExit(main())
