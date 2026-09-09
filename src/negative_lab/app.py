"""程序入口（v5）。默认离线，不发起网络请求（§0.2.12：禁止云端上传原图）。"""

from __future__ import annotations

import sys


def selftest() -> int:
    """冻结运行时功能自检：合成负片 -> 三类区域 -> 统一服务 -> 三候选 -> 导出。

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

        from .color.management import OutputEncoding
        from .io.images import save_jpeg8, save_tiff16
        from .pipeline.auto import auto_analyze
        from .pipeline.density import invert
        from .pipeline.render import RenderIntent
        from .pipeline.service import candidate_positive, process_image
        from .testing.spectral import make_spectral_negative

        s = make_spectral_negative(seed=3)
        rep = auto_analyze(s["negative"])
        res = invert(s["negative"], rep.params)
        assert np.isfinite(res.positive_linear).all()
        assert rep.confidence["dmin"] > 0.5, rep.confidence

        t = s["negative"]
        bx0, bx1 = s["truth"]["base_cols"]
        area = (0, 0, t.shape[0], bx0)
        pr = process_image(t, output_crop_source=area, image_area_source=area,
                           base_sample_source=[(0, bx0, t.shape[0], bx1)])
        assert pr.candidates is not None and pr.gate.blocks_export is False
        for name in ("preserve", "neutral", "warm"):
            y = candidate_positive(pr, name)
            assert np.isfinite(y).all() and y.shape == pr.result.positive_linear.shape
        assert np.array_equal(candidate_positive(pr, "preserve", RenderIntent()),
                              pr.result.positive_linear)
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "in.tif"
            import tifffile
            tifffile.imwrite(src, (t * 65535).astype(np.uint16), photometric="rgb")
            out = Path(td) / "out.tif"
            save_tiff16(out, pr.rendered_positive_linear, OutputEncoding.SRGB,
                        sources=[src])
            save_jpeg8(out.with_suffix(".jpg"), pr.rendered_positive_linear,
                       sources=[src])
            assert out.exists() and out.with_suffix(".jpg").exists()
        try:
            import rawpy  # noqa: F401
            rawpy_ok = True
        except ImportError:
            rawpy_ok = False
    except Exception as e:  # noqa: BLE001
        import traceback
        return report(f"SELFTEST FAIL: {type(e).__name__}: {e}\n"
                      f"{traceback.format_exc()}", 1)
    return report(f"SELFTEST OK (rawpy={'yes' if rawpy_ok else 'MISSING'})", 0)


def main() -> int:
    if "--selftest" in sys.argv:
        return selftest()
    from .ui.main_window import main as gui_main

    return gui_main()


if __name__ == "__main__":
    raise SystemExit(main())
