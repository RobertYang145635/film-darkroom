"""机械执行 v5 §10.2：**脚本永不代签**。

"判定字段只能由用户本人填写"一直只是**散文里的规矩**。
这条锁把它变成机器规矩：扫遍生产代码，只要出现
给 `verdict` / `signed_by` 赋**非空字面量**的写法（`"usable"`、某个名字），
测试立刻红——比 code review 可靠，也比"我记得我没写"可靠。

`blank_verdict()` 生成模板时写 `None` / `""` 是**允许**的（那是空条目），
所以正则只抓非空字符串；`test_negative_control_catches_a_forged_signature`
保证这条锁**不是空跑**（本项目吃过"测试通过但断言没执行"的亏）。
"""

from __future__ import annotations

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[2]

SCAN = [ROOT / "src" / "negative_lab", ROOT / "scripts"]

FORGE = re.compile(
    r"""(                       # 1) 字典字面量里赋值  "verdict": "usable"
        (?:"verdict"|"signed_by")\s*:\s*"(?!\s")[^"]+"
      |                        # 2) 下标赋值  entry["verdict"] = "usable"
        \[\s*["'](?:verdict|signed_by)["']\s*\]\s*=\s*"(?!\s")[^"]+"
      |                        # 3) setdefault / .update() 里的非空值
        (?:setdefault|update)\s*\(\s*["'](?:verdict|signed_by)["']\s*,\s*"(?!\s")[^"]+"
    )""",
    re.VERBOSE,
)


def _hits():
    out = []
    for base in SCAN:
        for path in sorted(base.rglob("*.py")):
            for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if FORGE.search(line):
                    out.append(f"{path.relative_to(ROOT)}:{n}: {line.strip()[:90]}")
    return out


def test_scan_actually_covers_the_codebase():
    """防"空扫"：路径写错时命中数永远是 0，第一条锁就成了假通过。"""
    files = [p for base in SCAN for p in base.rglob("*.py")]
    lines = sum(len(p.read_text(encoding="utf-8").splitlines()) for p in files)
    assert len(files) >= 20, f"只扫到 {len(files)} 个文件，SCAN 路径八成写错了"
    assert lines >= 4000, f"只扫到 {lines} 行，覆盖面不足以构成保证"


def test_no_production_code_forges_a_verdict_or_signature():
    hits = _hits()
    assert not hits, (
        "脚本正在写判定/署名（v5 §10.2：只有用户能签）：\n  " + "\n  ".join(hits[:8]))


def test_negative_control_catches_a_forged_signature():
    """锁必须能响：喂它一句伪造签名的代码，它要抓到（否则等于没锁）。"""
    forged = ['                "verdict": "usable",',
              '        entry["signed_by"] = "YZH",',
              '        d.setdefault("verdict", "tweakable")']
    for line in forged:
        assert FORGE.search(line), f"伪造写法没被抓到：{line}"
    # 合法写法不许误报：模板空值、以及**读**判定的比较
    legal = ['"verdict": None,', '"signed_by": "",',
             'if f.get("verdict") == "usable":', 'str(f["signed_by"])']
    for line in legal:
        assert not FORGE.search(line), f"合法写法被误伤：{line}"
