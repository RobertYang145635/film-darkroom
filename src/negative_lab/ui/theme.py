"""v6 §2.C/§5.8：全局视觉 token（iOS/macOS 简洁层，Qt 原生实现）。

唯一颜色/间距/尺寸来源：MainWindow、CropPage、PreviewView、ConvertPage
都从这里读取，禁止页面各自硬编码另一套颜色和间距（契约测试锁定）。

"iOS 风格"指系统化的留白、层级、控件状态与克制动效：浅色分组背景、
白色内容表面、细分隔线、低强度阴影；禁止渐变、玻璃拟态、装饰光晕、
嵌套卡片和超过 8px 的圆角。不引入 WebView/QML/CSS 框架/网络字体。
"""

from __future__ import annotations

#: 颜色 token（v6 §2.C.5，值与规格逐字一致）
COLORS = {
    "canvas": "#F2F2F7", "surface": "#FFFFFF",
    "text_primary": "#1C1C1E", "text_secondary": "#6E6E73",
    "separator": "#D1D1D6", "accent": "#007AFF",
    "success": "#34C759", "warning": "#FF9500",
    "danger": "#FF3B30", "viewer": "#1C1C1E",
}

#: 8pt 间距基线（§2.C.2）
SPACING = {"xs": 4, "sm": 8, "md": 12, "lg": 16, "xl": 24}

#: 结构尺寸（§5.8）
SIZES = {"toolbar_h": 44, "min_target": 32, "primary_target": 36,
         "sidebar_w": 240, "inspector_w": 280}

#: 控件圆角（契约：<= 8px）
RADIUS = 6

#: 字体栈：Windows 优先；macOS 由调用方替换为 SF Pro（§2.C.3）
FONT_STACK = ["Segoe UI Variable", "Segoe UI", "Microsoft YaHei UI"]
MAC_FONT_STACK = ["SF Pro Text", "SF Pro Display", "PingFang SC"]

#: 正式字号（§2.C.4，不随窗口宽度缩放）
FONT_PX = {"body": 13, "group_title": 13, "page_title": 17, "caption": 11}


def font_family() -> str:
    """返回系统实际可用的第一个字体；都不可用时回退默认（布局不变）。"""
    import sys

    from PySide6.QtGui import QFontDatabase

    stack = MAC_FONT_STACK if sys.platform == "darwin" else FONT_STACK
    try:
        fams = set(QFontDatabase.families())
    except Exception:  # noqa: BLE001 —— 字体枚举失败不得改变布局
        return stack[1]
    for name in stack:
        if name in fams:
            return name
    return stack[1]


def qss() -> str:
    """全局样式表（由 token 生成；测试扫描渐变/圆角/紫色）。"""
    c = COLORS
    r = RADIUS
    return f"""
* {{ outline: none; }}
QMainWindow, QWidget#Root {{ background: {c['canvas']}; }}
QLabel {{ color: {c['text_primary']}; font-size: {FONT_PX['body']}px; }}
QLabel[caption="true"] {{ color: {c['text_secondary']};
    font-size: {FONT_PX['caption']}px; }}
QLabel[groupTitle="true"] {{ color: {c['text_primary']};
    font-size: {FONT_PX['group_title']}px; font-weight: 600; }}
QLabel[pageTitle="true"] {{ color: {c['text_primary']};
    font-size: {FONT_PX['page_title']}px; font-weight: 600; }}
QLabel#Banner {{ border-radius: {r}px; padding: 6px 10px;
    font-size: {FONT_PX['caption']}px; }}
QLabel#Banner[level="warning"] {{ background: {c['warning']};
    color: {c['surface']}; }}
QLabel#Banner[level="success"] {{ background: {c['success']};
    color: {c['surface']}; }}
QLabel#Banner[level="danger"] {{ background: {c['danger']};
    color: {c['surface']}; }}
QLabel#Banner[level="info"] {{ background: {c['surface']};
    color: {c['text_secondary']}; border: 1px solid {c['separator']}; }}

QGroupBox {{ background: {c['surface']}; border: 1px solid {c['separator']};
    border-radius: {r}px; margin-top: 10px; padding: 8px 8px 8px 8px;
    font-size: {FONT_PX['group_title']}px; font-weight: 600; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; top: -2px; }}

QPushButton {{ background: {c['surface']}; color: {c['text_primary']};
    border: 1px solid {c['separator']}; border-radius: {r}px;
    padding: 5px 12px; font-size: {FONT_PX['body']}px; }}
QPushButton:hover {{ background: {c['canvas']}; }}
QPushButton:pressed {{ background: {c['separator']}; }}
QPushButton:disabled {{ color: {c['text_secondary']};
    background: {c['canvas']}; }}
QPushButton:focus {{ border: 2px solid {c['accent']}; }}
QPushButton[accent="true"] {{ background: {c['accent']};
    color: {c['surface']}; border: 1px solid {c['accent']}; }}
QPushButton[accent="true"]:hover {{ background: #2B93FF; }}
QPushButton[accent="true"]:pressed {{ background: #0062CC; }}
QPushButton[danger="true"] {{ color: {c['danger']}; }}

QToolButton {{ background: transparent; border: none; border-radius: {r}px;
    color: {c['text_primary']}; padding: 4px; }}
QToolButton:hover {{ background: {c['separator']}; }}
QToolButton:pressed {{ background: {c['text_secondary']}; }}
QToolButton:checked {{ background: {c['accent']}; color: {c['surface']}; }}
QToolButton:disabled {{ color: {c['text_secondary']}; }}
QToolButton:focus {{ border: 2px solid {c['accent']}; }}

QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox {{
    background: {c['surface']}; color: {c['text_primary']};
    border: 1px solid {c['separator']}; border-radius: {r}px;
    padding: 4px 8px; font-size: {FONT_PX['body']}px; }}
QComboBox:hover, QLineEdit:hover {{ border-color: {c['text_secondary']}; }}
QComboBox:focus, QLineEdit:focus {{ border: 2px solid {c['accent']}; }}
QComboBox::drop-down {{ border: none; width: 22px; }}

QCheckBox {{ color: {c['text_primary']}; font-size: {FONT_PX['body']}px; }}
QCheckBox::indicator {{ width: 16px; height: 16px;
    border: 1px solid {c['separator']}; border-radius: 4px;
    background: {c['surface']}; }}
QCheckBox::indicator:checked {{ background: {c['accent']};
    border-color: {c['accent']}; }}

QSlider::groove:horizontal {{ height: 4px; background: {c['separator']};
    border-radius: 2px; }}
QSlider::handle:horizontal {{ width: 16px; height: 16px; margin: -6px 0;
    border-radius: 8px; background: {c['surface']};
    border: 1px solid {c['separator']}; }}
QSlider::handle:horizontal:hover {{ border-color: {c['accent']}; }}
QSlider::sub-page:horizontal {{ background: {c['accent']};
    border-radius: 2px; }}

QListWidget {{ background: {c['surface']}; color: {c['text_primary']};
    border: 1px solid {c['separator']}; border-radius: {r}px;
    font-size: {FONT_PX['body']}px; outline: 0; }}
QListWidget::item {{ padding: 4px 6px; border-radius: 4px; }}
QListWidget::item:selected {{ background: #D8E9FF;
    color: {c['text_primary']}; }}
QListWidget::item:hover:!selected {{ background: {c['canvas']}; }}

QStatusBar {{ background: {c['surface']}; color: {c['text_secondary']};
    border-top: 1px solid {c['separator']}; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {c['separator']};
    border-radius: 4px; min-height: 24px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QToolTip {{ background: {c['surface']}; color: {c['text_primary']};
    border: 1px solid {c['separator']}; padding: 4px 6px; }}
"""


def apply(app) -> None:
    """把主题应用到 QApplication：字体 + 样式表 + 基础调色板。

    尊重系统减少动效：这里只定义 120-180ms 的悬停/按压瞬态（由 Qt 原生
    渲染），不引入任何持续动画或计时器。
    """
    from PySide6.QtGui import QFont, QPalette
    from PySide6.QtGui import QColor

    f = QFont(font_family())
    f.setPixelSize(FONT_PX["body"])
    app.setFont(f)
    pal = app.palette()
    c = COLORS
    pal.setColor(QPalette.Window, QColor(c["canvas"]))
    pal.setColor(QPalette.Base, QColor(c["surface"]))
    pal.setColor(QPalette.WindowText, QColor(c["text_primary"]))
    pal.setColor(QPalette.Text, QColor(c["text_primary"]))
    pal.setColor(QPalette.Button, QColor(c["surface"]))
    pal.setColor(QPalette.ButtonText, QColor(c["text_primary"]))
    pal.setColor(QPalette.Highlight, QColor(c["accent"]))
    pal.setColor(QPalette.HighlightedText, QColor(c["surface"]))
    pal.setColor(QPalette.ToolTipBase, QColor(c["surface"]))
    pal.setColor(QPalette.ToolTipText, QColor(c["text_primary"]))
    pal.setColor(QPalette.PlaceholderText, QColor(c["text_secondary"]))
    app.setPalette(pal)
    app.setStyleSheet(qss())
