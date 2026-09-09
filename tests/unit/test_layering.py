"""分层门禁：禁止下层反向依赖上层（2026-09-08 结构整改的长效机制）。

背景：本轮已把 3/5 处反向依赖消除（GENERIC_PRIOR 下沉 constants.py、
channel_clip_ratios 下沉 metrics.py、以及各处仓库根解析统一）。剩下的两处
需要跨层签名改动，登记在下面的 ALLOW 里 —— 关键是**新增违规会直接红**，
而不是靠下一轮人肉审查再发现一遍。

双向自证：本测试的扫描器先把所有违规打印出来；若 ALLOW 清空后扫不到任何东西，
说明扫描器本身失效（测试假绿），那才是真正要修的 bug。
"""

from __future__ import annotations

import ast
import pathlib

SRC = pathlib.Path("src/negative_lab")

#: 允许反向依赖的 (模块, 被依赖的上层包)。**每一项都是待办，不是豁免证**；
#: 修好后必须删掉对应条目，否则 test_scanner_actually_detects_the_known_debts 会红。
#:
#: 清点说明：既往三轮审查都记「5 处反向依赖」，但那是按 grep 关键词数出来的。
#: 本测试用 AST 全量扫描后实测至少 7 处（按符号）/ 以下 6 个模块（按文件）仍依赖上层，
#: 其中 4 处从未被任何一轮审查列出 —— 这正是把「清点」变成「门禁」的价值。
#: 已消除的 2 处（GENERIC_PRIOR、channel_clip_ratios）不在此列：扫描确认它们已不违规。
ALLOW = {
    # 待办：sidecar 用 pipeline.render.RenderIntent 做 legacy schema 迁移
    ("io/sidecar.py", "pipeline"),
    # 待办：rollcalib 调 pipeline.backends.fit_decrosstalk_supervised -> 注入拟合器
    ("calibration/rollcalib.py", "pipeline"),
    # 待办（本轮新发现）：neutral_stats.py 同上
    ("calibration/neutral_stats.py", "pipeline"),
    # 半误报：profile -> pipeline.params，而 params.py 只 import dataclasses，
    # 属「放错目录」而非依赖倒置；要根治得连 params.py 一起搬家（高风险零收益）。
    # 登记在此是为了让扫描器口径统一，勿据此去改 profile.py 的逻辑。
    ("profiles/profile.py", "pipeline"),
}

LAYERS = {"io", "calibration", "diagnostics", "library", "profiles", "restoration"}
UPPER = {"pipeline", "ui"}


def _upper_imports(path: pathlib.Path):
    """列出一个模块里所有「下层 -> 上层」的导入目标包名。"""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    except SyntaxError as e:                 # 解析失败必须指名道姓，不能只报红
        raise AssertionError(f"分层扫描器无法解析 {path}: {e}") from None
    rel_parts = path.relative_to(SRC).parts
    own = rel_parts[0] if rel_parts and rel_parts[0] in LAYERS else None
    if own is None:
        return []
    found = []
    for node in ast.walk(tree):
        mod = None
        if isinstance(node, ast.ImportFrom):
            # 相对导入：level=2 且 module=pipeline.x -> 指向 negative_lab.pipeline
            if node.level and node.module:
                if node.level >= 2 and own != "ui":
                    mod = node.module.split(".")[0]
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name.startswith("negative_lab."):
                    mod = a.name.split(".")[1]
        if mod in UPPER and mod != own:
            found.append(mod)
    return found


def test_no_new_reverse_dependencies():
    violations = set()
    for f in sorted(SRC.rglob("*.py")):
        key = f.relative_to(SRC).as_posix()
        for up in _upper_imports(f):
            violations.add((key, up))
    unexpected = violations - ALLOW
    if unexpected:                       # 失败信息要能指路，不能只说"不对"
        raise AssertionError(
            "出现新的反向依赖（下层 import 上层）："
            + "; ".join(f"{m} -> {u}" for m, u in sorted(unexpected))
            + "。请把值下沉到叶子模块（参考 constants.py / metrics.py），"
              "或把上层能力作为参数注入。")


def test_scanner_actually_detects_the_known_debts():
    """自证扫描器有效：ALLOW 里登记的每一项，必须**确实**存在。

    若某项已经修好却还留在 ALLOW，本测试失败 -> 强制回删 ALLOW，
    避免豁免清单变成长期藏污纳垢的黑名单。
    """
    found = set()
    for f in sorted(SRC.rglob("*.py")):
        key = f.relative_to(SRC).as_posix()
        for up in _upper_imports(f):
            found.add((key, up))
    stale = ALLOW - found
    assert not stale, (
        "ALLOW 中这些项已不存在（要么已修复、要么扫描器坏了）："
        + "; ".join(f"{m} -> {u}" for m, u in sorted(stale))
        + "。已修复就从 ALLOW 删除；若是扫描器漏检，那才是真问题。")
