"""产品依赖"**自动换行**"来保证文字显示，本文件锁住这个**前提真的成立**。

**为什么需要**：实测产品里有 **7 处**依赖文字布局
（`ui/right_panel.py` **5×** `setWordWrap(True)` + `1× font-size: 12px`、
`ui/library_panel.py` `setWordWrap(False)`），而**全仓库没有一处 `elidedText`**——
也就是说**文字不被裁掉全靠自动换行**。而这些控件过去**全部在零字体数据库下渲染**
（offscreen `families()==0`），**测试却全绿** ⇒ 这个前提**从未被验证过**。

**诚实声明**：本文件验证的是**换行机制本身**（Qt 层面 + 字体存在前提），
**不是**那 5 个具体标签各自不裁字——**后者仍未覆盖**，见
`docs/visual-acceptance.md` 的覆盖边界小节，别把这里读成已全部覆盖。
"""

from __future__ import annotations

import warnings

import pytest

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

qt = pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtGui import QFontDatabase, QFontMetrics  # noqa: E402
from PySide6.QtWidgets import QLabel  # noqa: E402

LONG = ("这一段的文字故意写得很长，用来验证窄宽度下会自动换行而不是被裁掉；"
        "如果换行没有发生，右侧内容就会消失，用户在界面上看不到完整提示。")


def _require_fonts():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fams = list(QFontDatabase().families())
    if not fams:
        pytest.skip(
            "字体库为空（offscreen 固有）⇒ 本用例**无法验证换行**；"
            "这是**环境退化**，不是通过——见 docs/visual-acceptance.md 覆盖边界")
    return fams


def test_word_wrap_actually_wraps_at_narrow_width():
    """**换行必须真的发生**：同一段长文本，窄宽度下换行版应明显更高。

    若 Qt 换行没生效（或字体缺失导致没有可用的换行度量），两者高度会一样——
    那意味着产品那 5 处 `setWordWrap(True)` 的提示语会被**静默裁掉**。
    """
    _require_fonts()
    plain = QLabel(LONG)
    wrapped = QLabel(LONG)
    wrapped.setWordWrap(True)
    for w in (plain, wrapped):
        w.setFixedWidth(90)
        w.adjustSize()
    h_plain, h_wrapped = plain.height(), wrapped.height()
    assert h_wrapped > h_plain, (
        f"换行没有发生（窄宽度下两者同高 {h_plain}）："
        "产品里靠 setWordWrap(True) 兜住文字的提示会被裁掉")


def test_font_size_in_stylesheet_yields_sane_metrics():
    """`font-size: 12px` 那处：**只验量级**（8–32px），**故意不锁死像素**。

    锁死具体像素会把**平台/字体差异当成回归**——那是假的严格。
    """
    _require_fonts()
    lbl = QLabel("中性点")
    lbl.setStyleSheet("font-size: 12px;")
    h = QFontMetrics(lbl.font()).height()
    assert 8 <= h <= 32, f"字号度量异常（{h}px），文本渲染不可信"
