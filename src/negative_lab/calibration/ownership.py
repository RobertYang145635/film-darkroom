"""标注/草稿归属校验（叶子模块，只依赖 stdlib）。

命名方案 annotations.ann_path_for 用相对路径的 parts[1] 当"卷名"，因此
浅层与深层路径都会**撞名**——实测本仓库自己的两个样片目录
    20260506_SAMPLE-ID/G200-1.nef      -> misc__G200-1.json
    20260709_SAMPLE-ID(1)/G200-1.nef   -> misc__G200-1.json
落到同一个文件。所以"文件名命中"绝不等于"这帧的标注"：读取必须校验
ann.file 归属，写入必须拒绝覆盖别人的标注，否则 B 帧会静默套用（并覆盖）
A 帧的签字基线。2026-09-08 冷读审查 L2-④ 实证。
"""

from __future__ import annotations


def norm(path) -> str:
    return str(path or "").replace("\\", "/")


def same_source(recorded, want) -> bool:
    """标注里记的 file 是否就是请求的那一帧（缺 recorded 视为未知，不容许套用）。"""
    a, b = norm(recorded), norm(want)
    return bool(a) and a == b
