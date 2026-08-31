"""程序入口（Phase 2）。默认离线，不发起网络请求（§9）。"""

from __future__ import annotations

import sys


def main() -> int:
    from PySide6.QtWidgets import QApplication

    from .ui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("negative-lab")
    win = MainWindow()
    win.show()
    if len(sys.argv) > 1:
        from pathlib import Path
        win.load(Path(sys.argv[1]))
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
