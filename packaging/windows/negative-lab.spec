# -*- mode: python ; coding: utf-8 -*-
# PyInstaller 打包配置（规格 §10 Phase 5）。
#
# 关键决策（docs/dependency-licenses.md）：
# - onedir 而非 onefile：LGPL 合规要求 Qt/rawpy DLL 对用户可见、可替换
#   （onefile 把 DLL 封进自解压归档，妨碍重新链接权）。
# - console=False：GUI 应用；CLI 用法从源码运行（面向高级用户）。
# - 不打包任何用户样片。
#
# 构建：.venv\Scripts\pyinstaller packaging\windows\negative-lab.spec --noconfirm

import os
from pathlib import Path

ROOT = Path(SPECPATH).parent.parent

a = Analysis(
    [str(ROOT / "packaging" / "windows" / "entry.py")],
    pathex=[str(ROOT / "src")],
    binaries=[],
    datas=[
        # 许可证与声明随包分发（LGPL/HPND/BSD 要求）
        (str(ROOT / "docs" / "dependency-licenses.md"), "docs"),
        (str(ROOT / "docs" / "user-quickstart.md"), "docs"),
        (str(ROOT / "CHANGELOG.md"), "."),
    ],
    hiddenimports=[
        "negative_lab.app",
        "negative_lab.ui.main_window",
        "negative_lab.ui.batch_panel",
        "tifffile.tifffile",
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=["PySide6.QtWebEngineCore", "PySide6.QtQml", "PySide6.QtQuick"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="negative-lab",
    debug=False,
    strip=False,
    upx=False,
    console=False,
    icon=None,
    version=str(ROOT / "packaging" / "windows" / "version_info.txt"),
    disable_windowed_traceback=False,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="negative-lab",
)
