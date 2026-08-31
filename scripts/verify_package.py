"""Phase 5 打包验证：产物完整性 + LGPL 合规检查 + 启动冒烟 + SHA-256。

运行：.venv\\Scripts\\python.exe scripts/verify_package.py
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist" / "negative-lab"
EXE = DIST / "negative-lab.exe"


def check(cond: bool, label: str, fails: list[str]):
    print(("✓" if cond else "✗") + " " + label)
    if not cond:
        fails.append(label)


def main() -> int:
    fails: list[str] = []
    check(EXE.exists(), "exe 存在", fails)
    if not EXE.exists():
        return 1

    # 1) onedir：Qt/rawpy DLL 必须可见可替换（LGPL 重新链接权）
    qt_dlls = list(DIST.glob("**/Qt6*.dll")) + list(DIST.glob("**/Qt*.dll"))
    check(len(qt_dlls) > 5, f"Qt DLL 独立可见（{len(qt_dlls)} 个，可替换）", fails)
    rawpy = list(DIST.glob("**/*rawpy*"))
    check(bool(rawpy), "rawpy/LibRaw 扩展在包内", fails)

    # 2) 许可证与文档随包（PyInstaller 6 onedir：datas 在 _internal/）
    internal = DIST / "_internal"
    check((internal / "docs" / "dependency-licenses.md").exists(), "许可证清单随包", fails)
    check((internal / "docs" / "user-quickstart.md").exists(), "快速指南随包", fails)
    check((internal / "CHANGELOG.md").exists(), "变更日志随包", fails)
    lic = [p for p in DIST.rglob("*") if p.name.lower() in
           ("license", "license.txt", "licence.txt", "lgpl-3.0.txt")]
    check(bool(lic), "存在第三方许可证文本文件", fails)

    # 3) 不打包样片/用户数据
    big = [p for p in DIST.rglob("*.NEF")]
    check(not big, "包内无用户样片", fails)

    # 4) 启动冒烟（offscreen，8s 后终止；窗口程序无控制台）
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    proc = subprocess.Popen([str(EXE)], env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(8)
    alive = proc.poll() is None
    if alive:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
    check(alive, "exe 可启动并保持运行（offscreen 8s）", fails)

    # 4b) 功能自检：冻结运行时内完整管线（合成 -> 自动 -> 导出）
    st_out = ROOT / "dist" / "selftest-result.txt"
    if st_out.exists():
        st_out.unlink()
    rc = subprocess.run([str(EXE), "--selftest", "--selftest-out", str(st_out)],
                        env=env, timeout=180,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode
    msg = st_out.read_text(encoding="utf-8") if st_out.exists() else "<no output>"
    check(rc == 0 and msg.startswith("SELFTEST OK"),
          f"冻结运行时功能自检：{msg}", fails)

    # 5) 体积与文件数
    n = sum(1 for _ in DIST.rglob("*") if _.is_file())
    size = sum(f.stat().st_size for f in DIST.rglob("*") if f.is_file()) / 1e6
    print(f"  文件数 {n}，总大小 {size:.0f} MB")

    # 6) zip + SHA-256（交付物）
    zip_path = ROOT / "dist" / "negative-lab-0.1.0-win-x64.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in DIST.rglob("*"):
            if f.is_file():
                z.write(f, f.relative_to(DIST.parent))
    h = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    (ROOT / "dist" / "negative-lab-0.1.0-win-x64.zip.sha256").write_text(
        f"{h}  negative-lab-0.1.0-win-x64.zip\n", encoding="utf-8")
    print(f"  ZIP: {zip_path.name}  SHA-256: {h[:16]}…")

    print("\n结果：", "全部通过" if not fails else f"失败 {len(fails)} 项: {fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
