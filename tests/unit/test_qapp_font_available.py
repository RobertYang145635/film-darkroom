"""证明**测试会话里真的有字体可用**（防"补字体"静默空转）。

上一轮实测：offscreen 下 `QFontDatabase().families() == 0`，同机原生平台 **406** ⇒
UI 测试全跑在零字体数据库上。本轮在 `conftest.shared_qapp` 里补了字体，
**但"补了"不等于"补上了"**——若候选路径都不存在，或加载失败被 `except` 吞掉，
测试会**一脸绿地继续跑在零字体上**。这条锁就是堵这个的：
加载失败要么**抓出来**，要么**大声 skip**，绝不允许算作通过。
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest
from conftest import _FONT_CANDIDATES, _load_test_font


def _families() -> list:
    from PySide6.QtGui import QFontDatabase
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return list(QFontDatabase().families())


def _candidates_present() -> bool:
    return any(Path(c).exists() for c in _FONT_CANDIDATES)


def test_loader_finds_a_font_when_candidates_exist():
    """候选存在就必须真的加载成功；一个候选都没有则**明确 skip**，不许静默通过。"""
    if not _candidates_present():
        pytest.skip(
            "本机没有任何候选字体 → UI 文本测试处于**零字体退化环境**；"
            "这是环境事实，不是测试通过")
    got = _load_test_font()
    assert got is not None, (
        "候选字体文件明明存在却加载失败——多半被 conftest 的 except 吞了，"
        "这正是本锁要抓的情形")
    assert Path(got).exists()


def test_session_font_database_is_not_empty(shared_qapp):
    """会话内 Qt 字体库**非空**（`shared_qapp` 是 session+autouse，由它补的字体）。"""
    fams = _families()
    if not fams:
        pytest.skip(
            "字体库仍为空：本机无候选字体。**不要**把这当成通过——"
            "它意味着 UI 文本相关断言仍跑在失真环境里")
    assert len(fams) >= 1, f"预期非空字体库，实得 {fams!r}"
