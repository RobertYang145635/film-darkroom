"""v6 重做：全局视觉 token（深色，Codex 式克制；用户 2026-09-02 确认）。

唯一颜色/间距/尺寸/样式来源。设计基调：
- 深色（看片软件事实标准）：背景近黑、面板深灰、文字浅灰；
- 无渐变、无彩色大底、无嵌套卡片；分组靠 1px 分隔线 + 节标题；
- 全部控件自定义样式（Windows 默认按钮是"丑"的主要来源）；
- 唯一强调色；成功/警告/错误只用于状态语义；
- 字号两级（12/11px），圆角 <=4px，扁平。
"""

from __future__ import annotations

#: 颜色 token（深色；用户确认 2026-09-02）
COLORS = {
    "bg": "#1E1E1E",            # 主窗口背景（近黑）
    "panel": "#252526",         # 左右栏面板
    "panel_header": "#2D2D2D",  # 面板节标题带
    "text": "#D4D4D4",          # 正文
    "text_dim": "#8A8A8A",      # 辅助
    "separator": "#333333",     # 分隔线
    "accent": "#4C8DFF",        # 唯一强调色（当前/焦点/主按钮）
    "success": "#4EC9B0",
    "warning": "#D7BA7D",
    "danger": "#F48771",
    "viewer": "#141414",        # 照片查看区（比面板更深）
}

#: 8pt 间距基线
SPACING = {"xs": 4, "sm": 8, "md": 12, "lg": 16, "xl": 24}

#: 结构尺寸
SIZES = {"topbar_h": 44, "min_target": 28, "primary_target": 32,
         "left_w": 220, "right_w": 300, "filmstrip_h": 96}

RADIUS = 4

FONT_STACK = ["Segoe UI Variable", "Segoe UI", "Microsoft YaHei UI"]
MAC_FONT_STACK = ["SF Pro Text", "SF Pro Display", "PingFang SC"]

FONT_PX = {"body": 12, "caption": 11, "title": 13}


def font_family() -> str:
    import sys

    from PySide6.QtGui import QFontDatabase

    stack = MAC_FONT_STACK if sys.platform == "darwin" else FONT_STACK
    try:
        fams = set(QFontDatabase.families())
    except Exception:  # noqa: BLE001
        return stack[1]
    for name in stack:
        if name in fams:
            return name
    return stack[1]


def qss() -> str:
    c = COLORS
    r = RADIUS
    return f"""
* {{ outline: none; }}
QMainWindow, QWidget#Root {{ background: {c['bg']}; }}
QWidget {{ color: {c['text']}; font-size: {FONT_PX['body']}px; }}

QLabel {{ background: transparent; }}
QLabel[caption="true"] {{ color: {c['text_dim']};
    font-size: {FONT_PX['caption']}px; }}
QLabel[groupTitle="true"] {{ color: {c['text']};
    font-size: {FONT_PX['caption']}px; font-weight: 600; letter-spacing: 2px; }}
QLabel#SectionLine {{ background: {c['separator']}; max-height: 1px;
    border: none; }}

/* ---- 按钮：扁平深色，hover 提亮，focus 描边（不用系统默认） ---- */
QPushButton {{ background: #333333; color: {c['text']};
    border: 1px solid #3F3F46; border-radius: {r}px;
    padding: 4px 12px; font-size: {FONT_PX['body']}px; }}
QPushButton:hover {{ background: #3E3E42; border-color: #5A5A64; }}
QPushButton:pressed {{ background: #2A2A2E; }}
QPushButton:disabled {{ color: #6B6B70; background: {c['panel']};
    border-color: {c['separator']}; }}
QPushButton:focus {{ border-color: {c['accent']}; }}
QPushButton[accent="true"] {{ background: {c['accent']}; color: #FFFFFF;
    border-color: {c['accent']}; font-weight: 600; }}
QPushButton[accent="true"]:hover {{ background: #5E99FF; }}
QPushButton[accent="true"]:pressed {{ background: #3A76D9; }}
QPushButton[flat="true"] {{ background: transparent;
    border: 1px solid transparent; color: {c['text_dim']}; }}
QPushButton[flat="true"]:hover {{ background: #333333; color: {c['text']}; }}

QToolButton {{ background: transparent; border: 1px solid transparent;
    border-radius: {r}px; padding: 2px; }}
QToolButton:hover {{ background: #333333; }}
QToolButton:pressed {{ background: #2A2A2E; }}
QToolButton:checked {{ background: #3A3A3E;
    border-color: {c['accent']}; }}
QToolButton:disabled {{ opacity: 0.4; }}
QToolButton:focus {{ border-color: {c['accent']}; }}

/* ---- 输入类 ---- */
QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox {{
    background: {c['panel']}; color: {c['text']};
    border: 1px solid {c['separator']}; border-radius: {r}px;
    padding: 3px 8px; font-size: {FONT_PX['body']}px; }}
QComboBox:hover, QLineEdit:hover {{ border-color: #5A5A64; }}
QComboBox:focus, QLineEdit:focus {{ border-color: {c['accent']}; }}
QComboBox::drop-down {{ border: none; width: 20px; }}
QComboBox QAbstractItemView {{ background: {c['panel']};
    color: {c['text']}; border: 1px solid {c['separator']};
    selection-background-color: #3A3A3E; }}

QCheckBox {{ background: transparent; spacing: 6px; }}
QCheckBox::indicator {{ width: 14px; height: 14px;
    border: 1px solid #5A5A64; border-radius: 3px;
    background: {c['panel']}; }}
QCheckBox::indicator:hover {{ border-color: {c['accent']}; }}
QCheckBox::indicator:checked {{ background: {c['accent']};
    border-color: {c['accent']}; }}

/* ---- 滑杆：细轨小手柄 ---- */
QSlider {{ background: transparent; min-height: 20px; }}
QSlider::groove:horizontal {{ height: 3px; background: #3F3F46;
    border-radius: 1px; }}
QSlider::handle:horizontal {{ width: 12px; height: 12px; margin: -5px 0;
    border-radius: 6px; background: {c['text']}; border: none; }}
QSlider::handle:horizontal:hover {{ background: {c['accent']}; }}
QSlider::sub-page:horizontal {{ background: {c['accent']};
    border-radius: 1px; }}

/* ---- 列表（卷/帧/胶片条）---- */
QListWidget {{ background: {c['panel']}; color: {c['text']};
    border: none; outline: 0; font-size: {FONT_PX['body']}px; }}
QListWidget::item {{ padding: 3px 6px; border-radius: {r}px;
    color: {c['text_dim']}; }}
QListWidget::item:hover {{ background: #2D2D30; color: {c['text']}; }}
QListWidget::item:selected {{ background: #3A3A3E; color: {c['text']}; }}

QScrollBar:vertical {{ background: transparent; width: 8px; margin: 1px; }}
QScrollBar::handle:vertical {{ background: #3F3F46; border-radius: 3px;
    min-height: 24px; }}
QScrollBar::handle:vertical:hover {{ background: #5A5A64; }}
QScrollBar:horizontal {{ background: transparent; height: 8px; margin: 1px; }}
QScrollBar::handle:horizontal {{ background: #3F3F46; border-radius: 3px;
    min-width: 24px; }}
QScrollBar::handle:horizontal:hover {{ background: #5A5A64; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

QSplitter::handle {{ background: {c['separator']}; }}
QSplitter::handle:horizontal {{ width: 1px; }}
QSplitter::handle:vertical {{ height: 1px; }}

QScrollArea {{ background: transparent; border: none; }}
QStatusBar {{ background: {c['panel']}; color: {c['text_dim']};
    border-top: 1px solid {c['separator']}; }}
QToolTip {{ background: #2D2D30; color: {c['text']};
    border: 1px solid {c['separator']}; padding: 4px 6px;
    font-size: {FONT_PX['caption']}px; }}
"""


def apply(app) -> None:
    """把主题应用到 QApplication：字体 + 样式表 + 深色调色板。"""
    from PySide6.QtGui import QColor, QFont, QPalette

    f = QFont(font_family())
    f.setPixelSize(FONT_PX["body"])
    app.setFont(f)
    c = COLORS
    pal = app.palette()
    pal.setColor(QPalette.Window, QColor(c["bg"]))
    pal.setColor(QPalette.Base, QColor(c["panel"]))
    pal.setColor(QPalette.WindowText, QColor(c["text"]))
    pal.setColor(QPalette.Text, QColor(c["text"]))
    pal.setColor(QPalette.Button, QColor("#333333"))
    pal.setColor(QPalette.ButtonText, QColor(c["text"]))
    pal.setColor(QPalette.Highlight, QColor(c["accent"]))
    pal.setColor(QPalette.HighlightedText, QColor("#FFFFFF"))
    pal.setColor(QPalette.ToolTipBase, QColor("#2D2D30"))
    pal.setColor(QPalette.ToolTipText, QColor(c["text"]))
    pal.setColor(QPalette.PlaceholderText, QColor("#6B6B70"))
    pal.setColor(QPalette.Disabled, QPalette.Text, QColor("#6B6B70"))
    pal.setColor(QPalette.Disabled, QPalette.ButtonText, QColor("#6B6B70"))
    app.setPalette(pal)
    app.setStyleSheet(qss())
