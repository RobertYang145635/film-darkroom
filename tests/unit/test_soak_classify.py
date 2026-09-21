"""`scripts/soak.py` 的退出码归类锁。

**为什么单独锁这个**：B-8 的崩溃**不打印任何 FAILED**，只体现在退出码上
（`0xC0000005` 有符号形式 `-1073741819`）。如果归类把它错当成"测试失败"，
我们就会去找根本不存在的红字；如果错当成"通过"，崩溃就彻底隐身。
所以这个映射表必须有锁，而且锁要能**认得出那个具体数字**。
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "soak", ROOT / "scripts" / "soak.py")
soak = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("soak", soak)
_spec.loader.exec_module(soak)          # type: ignore[union-attr]


def test_access_violation_is_named_not_blurred():
    kind = soak.classify_exit(soak.AV_SIGNED)
    assert "access-violation" in kind and "0xC0000005" in kind, kind
    assert kind != "tests-failed", "崩溃被当成测试失败 = 去找不存在的红字"


def test_clean_and_common_codes():
    assert soak.classify_exit(0) == "ok"
    # 只比前缀：文案后半截给人看的解释会变，**类别**才是契约
    assert soak.classify_exit(4).startswith("usage-error"), "用法错误必须单独分出来"
    assert soak.classify_exit(1).startswith("tests-failed")
    assert soak.classify_exit(130).startswith("interrupted")


def test_unknown_code_is_reported_verbatim():
    """未知码不许猜成任何一种已知类别——原样带出数字才不被误读。"""
    assert soak.classify_exit(-1073741635) == "exit=-1073741635"


def test_negative_control_av_code_is_not_zero():
    """负控制：若哪天常量被写成 0，`ok` 也会"没崩溃"，整条取证就废了。"""
    assert soak.AV_SIGNED != 0
    assert soak.classify_exit(soak.AV_SIGNED) != soak.classify_exit(0)
