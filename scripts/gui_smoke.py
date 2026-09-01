"""离屏 GUI 冒烟测试（Phase 2 门禁证据）。

QT_QPA_PLATFORM=offscreen 下：加载真实 NEF -> 自动转换 -> 导出三件套 ->
抓取窗口截图供人工核验。

运行：.venv\\Scripts\\python.exe scripts/gui_smoke.py [NEF路径]
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402
from negative_lab.ui.main_window import MainWindow  # noqa: E402
from negative_lab.ui.preview_view import RoiKind  # noqa: E402


def main() -> int:
    nef = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        ROOT / "20260709_SAMPLE-ID(1)/1298_G200/翻拍/G200-1.NEF")
    if not nef.exists():
        print(f"样片不存在：{nef}")
        return 1
    app = QApplication([])
    win = MainWindow()
    win.show()
    win.load(nef)
    assert win.t_full is not None and win.t_full.shape[0] > 1000
    print("loaded:", win.t_full.shape, "det warnings:", getattr(win, "_det", {}).get("warnings"))
    print("auto base rect:", win.view.roi(RoiKind.BASE))
    win.run_auto()
    p = win.params
    print(f"auto params: dmin={np.round(p.dmin, 4).tolist()} dmax={p.dmax:.2f} "
          f"offset={p.offset:.2f} conf={win.confidence}")
    assert np.isfinite(p.dmin).all() and p.dmax > 0
    # 除尘路径：自动检测 + 画笔 + 开关回退
    win.dust_auto_detect()
    assert win.view.mask() is not None
    win.view.ensure_mask(*win.t_prev.shape[:2])
    win.view._paint_at(__import__("PySide6.QtCore", fromlist=["QPointF"]).QPointF(300, 300))
    win.chk_dust.setChecked(True)
    win._refresh()
    win.chk_dust.setChecked(False)   # 关闭必须完全回退
    win._refresh()
    win.chk_dust.setChecked(True)    # 导出走带除尘路径
    import tempfile
    _tmp = tempfile.TemporaryDirectory(prefix="nl-gui-smoke-")
    out_dir = Path(_tmp.name)
    win.export_dir = out_dir  # v3 P0-I：临时目录，幂等且不污染 outputs/
    win.export_all()
    ev_dir = ROOT / "outputs" / "gui_smoke"
    ev_dir.mkdir(parents=True, exist_ok=True)
    win.grab().save(str(ev_dir / "gui_window.png"))   # 截图证据（不参与计数断言）
    exports = sorted(out_dir.glob("G200-1-positive*"))
    print("exports:", [f.name for f in exports])
    assert len(exports) == 3, "应导出 TIFF/JPEG/旁车 三件（本次 run 临时目录）"
    side = json.loads(Path([f for f in exports if f.suffix == ".json"][0])
                     .read_text(encoding="utf-8"))
    # v4 §10.6：断言 rotation/geometry/quality/ICC + 诚实质量指标（不再假绿）
    assert side["schema_version"] == 3 and len(side["source_sha256"]) == 64
    assert side["rotation_degrees"] == win.rotation
    assert side["geometry"]["crop_oriented"] is not None, "GUI 导出必须裁到画面"
    assert side["geometry"]["roi_source_space"]["coordinate_space"] == "source"
    assert side["quality"]["status"] in ("ok", "warn", "needs_calibration")
    assert side["output"]["icc_sha256"], "旁车必须记录输出 ICC"
    import tifffile
    tif_path = out_dir / "G200-1-positive.tif"
    with tifffile.TiffFile(str(tif_path)) as tf:
        assert tf.pages[0].tags.get(34675) is not None, "TIFF 必须嵌 ICC"
        pos = tf.pages[0].asarray().astype(np.float32) / 65535.0
    # 输出质量指标（sRGB 显示空间）
    from negative_lab.color.management import linear_to_srgb
    s = np.clip(linear_to_srgb(np.clip(pos, 0, 1)), 0, 1)
    lum = pos @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    sat = (s.max(2) - s.min(2)) / np.maximum(s.mean(2), 1e-6)
    clip_lo = (pos <= 1e-6).mean(axis=(0, 1))
    clip_hi = (pos >= 1 - 1e-6).mean(axis=(0, 1))
    metrics = {"lum_p50": round(float(np.quantile(lum, .5)), 4),
               "sat_p50_p90": [round(float(np.quantile(sat, q)), 4)
                               for q in (.5, .9)],
               "clip_low": [round(float(v), 4) for v in clip_lo],
               "clip_high": [round(float(v), 4) for v in clip_hi]}
    print("quality metrics:", metrics)
    if max(metrics["clip_low"]) > 0.01 or max(metrics["clip_high"]) > 0.01:
        print("[SMOKE-QUALITY-FAIL] 输出剪裁 >1%（v4 §10.6：已知偏青冲淡图"
              "不得打印 GUI SMOKE OK）")
        return 3
    print("GUI SMOKE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
