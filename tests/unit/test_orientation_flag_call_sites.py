"""锁：**面向用户的调用路径必须显式表态"这帧的方向决定过没有"**（B-12）。

为什么需要这条锁而不是只改代码：`orientation_decided` 是三态，
`None`（没人说过）与 `False`（明确没决定过）行为不同——`None` 不判定。
所以"忘记传参"的默认后果是**静默**，正是本规则要消灭的东西。
把它变成 AST 检查：界面与验收套具里任何一处 `process_image` /
`ui_process_image` / `FrameRecord` 少了这个关键字实参，就直接红。

负控制：故意构造一个缺关键字的 AST，检查器必须报出来；
否则这条锁可能只是在遍历空集合。
"""

from __future__ import annotations

import ast
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
CALLED = ("process_image", "ui_process_image", "FrameRecord")
FILES = [ROOT / "src" / "negative_lab" / "ui" / n for n in
         ("frame_loader.py", "geometry_bridge.py", "main_window.py",
          "state_store.py")] + [
    ROOT / "src" / "negative_lab" / "pipeline" / "service.py",
    ROOT / "scripts" / "build_visual_contact_sheet.py",
    ROOT / "scripts" / "build_verdict_kit.py"]


def _missing(src: str, path: pathlib.Path) -> list[str]:
    """返回 `文件:行号 调用了谁` 列表：这些调用点没给 orientation_decided。"""
    tree = ast.parse(src, filename=str(path))
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        name = f.id if isinstance(f, ast.Name) else (
            f.attr if isinstance(f, ast.Attribute) else "")
        if name not in CALLED:
            continue
        if any(kw.arg == "orientation_decided" for kw in node.keywords):
            continue
        # 纯转发包装（`f(*args, **kwargs)`）不需要点名关键字，它什么都传
        if (any(isinstance(a, ast.Starred) for a in node.args)
                or any(kw.arg is None for kw in node.keywords)):
            continue
        out.append(f"{path.name}:{node.lineno} {name}")
    return out


def test_user_facing_call_sites_state_orientation():
    missing: list[str] = []
    checked = 0
    for p in FILES:
        src = p.read_text(encoding="utf-8")
        tree = ast.parse(src, filename=str(p))
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and ((isinstance(n.func, ast.Name) and n.func.id in CALLED)
                      or (isinstance(n.func, ast.Attribute)
                          and n.func.attr in CALLED))]
        checked += len(calls)
        missing += _missing(src, p)
    assert checked >= 8, f"只扫到 {checked} 个调用点，检查器可能在空转"
    assert not missing, ("这些面向用户/套具的调用点没有表态方向是否决定过："
                         f"{missing}")


def test_the_checker_is_not_vacuous():
    """**负控制**：缺关键字的调用必须被抓到，带关键字的必须放过。"""
    bad = "pr = process_image(img, geometry=g)\nrec = FrameRecord(source_path='x')\n"
    good = ("pr = process_image(img, geometry=g, orientation_decided=False)\n"
            "rec = FrameRecord(source_path='x', orientation_decided=True)\n")
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = pathlib.Path(td) / "sample.py"
        p.write_text(bad, encoding="utf-8")
        assert len(_missing(bad, p)) == 2, "检查器没抓到两处缺参调用"
        assert _missing(good, p) == [], "检查器误报了带参调用"
