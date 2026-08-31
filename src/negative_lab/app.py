"""程序入口（Phase 2）。默认离线，不发起网络请求（§9）。"""

from __future__ import annotations

import sys


def selftest() -> int:
    """冻结运行时功能自检：合成负片 -> 自动分析 -> 反相 -> 导出三件套。

    窗口化 exe 无 stdout：结果写入 --selftest-out 指定文件，并以退出码返回。
    """
    import tempfile
    from pathlib import Path

    out_path = Path("negative-lab-selftest.txt")
    if "--selftest-out" in sys.argv:
        out_path = Path(sys.argv[sys.argv.index("--selftest-out") + 1])

    def report(msg: str, code: int) -> int:
        try:
            out_path.write_text(msg, encoding="utf-8")
        except OSError:
            pass
        return code

    try:
        import numpy as np

        from .io.images import save_jpeg8, save_tiff16
        from .pipeline.auto import auto_analyze
        from .pipeline.density import invert
        from .testing.synthetic import make_synthetic_negative

        s = make_synthetic_negative(black_border=16, seed=1)
        rep = auto_analyze(s["negative"])
        res = invert(s["negative"], rep.params)
        assert np.isfinite(res.positive_linear).all()
        assert rep.confidence["dmin"] > 0.5, rep.confidence
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "in.tif"
            import tifffile
            tifffile.imwrite(src, (s["negative"] * 65535).astype(np.uint16),
                             photometric="rgb")
            out = Path(td) / "out.tif"
            save_tiff16(out, res.preview_srgb, sources=[src])
            save_jpeg8(out.with_suffix(".jpg"), res.preview_srgb, sources=[src])
            assert out.exists() and out.with_suffix(".jpg").exists()
        try:
            import rawpy  # noqa: F401
            rawpy_ok = True
        except ImportError:
            rawpy_ok = False
    except Exception as e:  # noqa: BLE001
        return report(f"SELFTEST FAIL: {type(e).__name__}: {e}", 1)
    return report(f"SELFTEST OK (rawpy={'yes' if rawpy_ok else 'MISSING'})", 0)


def main() -> int:
    if "--selftest" in sys.argv:
        return selftest()
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
