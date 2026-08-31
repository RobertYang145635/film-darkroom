"""打包入口（PyInstaller 用；开发时直接 python -m negative_lab.app）。"""

import sys
from pathlib import Path

# 冻结环境下确保包可导入（spec 已加 pathex，这里双保险）
if getattr(sys, "frozen", False):
    base = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    if str(base) not in sys.path:
        sys.path.insert(0, str(base))

from negative_lab.app import main

if __name__ == "__main__":
    raise SystemExit(main())
