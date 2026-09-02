"""离屏 GUI 冒烟测试（v5 §12 P0-6/P0-7 门禁证据）。

QT_QPA_PLATFORM=offscreen 下：
  导入混合文件夹 -> 两个一级页面 -> 裁切页三类 ROI -> 转换页三候选 ->
  微调不改变技术正片 -> 整卷导出三件套 -> 截图证据。

运行：.venv\\Scripts\\python.exe scripts/gui_smoke.py [文件夹]
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
from PySide6.QtCore import QRectF  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from negative_lab.io.images import load_linear_rgb  # noqa: E402
from negative_lab.ui.main_window import MainWindow  # noqa: E402
from negative_lab.ui.preview_view import RoiKind  # noqa: E402


def main() -> int:
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        ROOT / "20260709_SAMPLE-ID(1)/1298_G200/翻拍")
    if not folder.exists():
        print(f"样片不存在：{folder}")
        return 1
    app = QApplication([])
    win = MainWindow()
    win.show()

    # ---- 两级页面（v5 §5.1/§7.1）
    assert win.page_count() == 2 and win.page_title(0) == "裁切" \
        and win.page_title(1) == "转换", (win.page_count(), [win.page_title(i)
                                                             for i in range(win.page_count())])

    # ---- 导入 + 手动分卷（v5 §2.1.1/2.1.2）
    win.load_folder(folder)
    assert win.session is not None and win.session.files
    paths = [f.source_path for f in win.session.files[:3]]
    win.session.create_roll("冒烟卷", paths)
    win.library.set_session(win.session)
    win.library.select_roll(win.session.roll_groups[-1].id)
    win.load_frame(paths[0])
    assert win.pr is not None, "服务未产出结果"
    print("loaded:", win.t_source.shape,
          "rois:", {k: len(win.pr.geometry.roi_source_space.get(k) or [])
                    for k in ("output_crop_roi", "image_area_roi",
                              "base_sample_rois")})

    # ---- 裁切页：三类 ROI 同时存在（v5 §5.2）
    view = win.crop_page.view
    assert view.roi(RoiKind.OUTPUT_CROP) is not None, "缺输出裁切框"
    assert view.roi(RoiKind.IMAGE_AREA) is not None, "缺有效画面框"
    assert view.roi_count(RoiKind.BASE) >= 1, "缺片基取样框"

    # ---- 转换页：三候选 + 微调不改技术正片（v5 §6.5/§7.2）
    win._show_page(1)
    cs = win.pr.candidates
    assert cs is not None and {c.name for c in cs} == \
        {"preserve", "neutral", "warm"}
    tech_before = win.pr.result.positive_linear.copy()
    for name in ("temperature", "exposure", "contrast", "saturation"):
        win.convert_page.panel.rows[name].setValue(
            40.0 if name != "exposure" else 0.5)
    win._intent_changed("temperature", 40.0)
    win._refresh_convert_view()
    assert np.array_equal(win.pr.result.positive_linear, tech_before), \
        "微调改变了技术正片（v5 §7.2 禁止）"
    pos = win.convert_page._positive
    assert pos is not None and np.isfinite(pos).all()
    win.convert_page.panel.reset_all()

    # ---- 整卷导出（事务三件套 + sidecar v4 + 批次报告）
    out_dir = Path(tempfile.mkdtemp(prefix="nl-gui-smoke-"))
    from negative_lab.library.models import FrameRecord
    from negative_lab.pipeline.roll_export import export_roll_frames
    frames = [rec for rec in (win.session.frame(p) for p in paths)
              if rec is not None]
    rep = export_roll_frames(frames, out_dir, session=win.session,
                             roll_id=win.session.roll_groups[-1].id,
                             progress=lambda m: print("  ", m))
    print("export summary:", {k: v for k, v in rep.items()
                              if k in ("ok", "needs_review", "failed")})
    assert rep["failed"] == 0, rep["items"]
    assert rep["ok"] + rep["needs_review"] == len(frames)
    exports = sorted(out_dir.glob("G200-1-positive*"))
    assert len(exports) == 3, [f.name for f in exports]
    side = json.loads(Path([f for f in exports if f.suffix == ".json"][0])
                      .read_text(encoding="utf-8"))
    assert side["schema_version"] == 4 and len(side["source_sha256"]) == 64
    assert side["geometry"]["roi_source_space"]["coordinate_space"] == "source"
    for key in ("output_crop_roi", "image_area_roi", "base_sample_rois"):
        assert key in side["geometry"]["roi_source_space"], key
    assert side["candidates"] is not None and "chosen_candidate" in side
    assert side["capture_fingerprint"]["editable"] is False
    assert side["base"]["base_source"] in ("frame", "roll", "roll_partial",
                                           "frame+roll", "generic_prior")
    assert side["color"]["output_icc_sha256"], "旁车必须记录输出 ICC"
    import tifffile
    with tifffile.TiffFile(str(out_dir / "G200-1-positive.tif")) as tf:
        assert tf.pages[0].tags.get(34675) is not None, "TIFF 必须嵌 ICC"
        pos16 = tf.pages[0].asarray()
    assert pos16.dtype == np.uint16 and pos16.ndim == 3
    # 输出质量（sRGB 显示域）：剪裁 <=1%（v5 §8.2）
    x = pos16.astype(np.float32) / 65535.0
    clip_lo = (x <= 1e-6).mean(axis=(0, 1))
    clip_hi = (x >= 1 - 1e-6).mean(axis=(0, 1))
    print("output clip:", {"low": np.round(clip_lo, 4).tolist(),
                           "high": np.round(clip_hi, 4).tolist()})
    if max(clip_lo) > 0.01 or max(clip_hi) > 0.01:
        print("[SMOKE-QUALITY-FAIL] 输出剪裁 >1%（v5 §8.2）")
        return 3

    # ---- 源文件只读校验
    import hashlib
    for p in paths:
        rec = win.session.frame(p)
        h = hashlib.sha256(_abs(p, rec).read_bytes()).hexdigest()
        assert h == rec.source_sha256, f"源文件被改动：{p}"

    ev_dir = ROOT / "outputs" / "gui_smoke"
    ev_dir.mkdir(parents=True, exist_ok=True)
    win.grab().save(str(ev_dir / "gui_window.png"))
    print("GUI SMOKE OK")
    return 0


def _abs(path: str, rec=None) -> Path:
    if rec is not None:
        abs_path = (rec.input_facts or {}).get("abs_path")
        if abs_path and Path(abs_path).exists():
            return Path(abs_path)
    p = Path(path)
    if p.is_absolute():
        return p
    return ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
