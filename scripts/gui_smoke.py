"""离屏 GUI 冒烟测试（v6 Stage 模型；v6.1 §6 同步）。

QT_QPA_PLATFORM=offscreen 下：
  导入混合文件夹 -> 单工作区 Stage（分析/调整）-> 三类 ROI -> 三候选 ->
  微调不改变技术正片 -> 整卷导出三件套 -> 截图证据。

运行：.venv\\Scripts\\python.exe scripts/gui_smoke.py [文件夹]
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from negative_lab.ui.main_window import MainWindow  # noqa: E402
from negative_lab.ui.preview_view import RoiKind  # noqa: E402


def _wait_idle(app, win, timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        app.processEvents()
        if not win._busy._locks and threading.active_count() <= 2:
            time.sleep(0.3)
            app.processEvents()
            return True
        time.sleep(0.05)
    return False


def main() -> int:
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        ROOT / "20260709_SAMPLE-ID(1)/1298_G200/翻拍")
    if not folder.exists():
        print(f"样片不存在：{folder}")
        return 1
    # v6.2 §3 阶段 A：冒烟期间的标注与草稿全部改道临时根，
    # 绝不写真实 fixtures/annotations 与 outputs/drafts。
    from negative_lab import storage

    # copy_annotations=True：读得到真实用户标定（冒烟才有代表性），
    # 写入仍在临时根——实测隔离后第一次丢失标注导致裁切/高光裁切统计变化。
    _storage_ctx = storage.isolated_storage(copy_annotations=True)
    _storage_ctx.__enter__()
    app = QApplication([])
    from negative_lab.ui import theme

    theme.apply(app)
    win = MainWindow()
    win.show()

    # ---- v6 单工作区 Stage 模型（无模式页）
    assert not hasattr(win, "right_stack") and not hasattr(win, "crop_page") \
        and not hasattr(win, "convert_page"), "v6 不得再有裁切/转换页"
    assert win.page_count() == 1 and win.page_title(0) == "工作区"
    assert hasattr(win, "_set_stage") and hasattr(win.right, "sec_areas")

    # ---- 导入 + 手动分卷
    win.load_folder(folder)
    assert _wait_idle(app, win), "导入超时"
    assert win.session is not None and win.session.files
    paths = [f.source_path for f in win.session.files[:3]]
    win.session.create_roll("冒烟卷", paths)
    win.library.set_session(win.session)
    win.library.select_roll(win.session.roll_groups[-1].id)
    assert _wait_idle(app, win), "选卷加载超时"
    win.load_frame(paths[0])
    assert _wait_idle(app, win), "首帧加载超时"
    assert win.pr is not None, "服务未产出结果"
    if win._banner_text.startswith("帧处理失败"):
        print("BANNER:", win._banner_text)
        return 2
    print("loaded:", win.t_source.shape,
          "rois:", {k: len(win.pr.geometry.roi_source_space.get(k) or [])
                    for k in ("output_crop_roi", "image_area_roi",
                              "base_sample_rois")})

    # ---- 分析阶段（Stage 0）：三类 ROI 同时存在
    win._set_stage(0)
    assert win.view.roi(RoiKind.OUTPUT_CROP) is not None, "缺输出裁切框"
    assert win.view.roi(RoiKind.IMAGE_AREA) is not None, "缺有效画面框"
    assert win.view.roi_count(RoiKind.BASE) >= 1, "缺片基取样框"

    # ---- 调整阶段（Stage 1）：三候选 + 微调不改技术正片
    win._set_stage(1)
    cs = win.pr.candidates
    assert cs is not None and {c.name for c in cs} == \
        {"preserve", "neutral", "warm"}
    tech_before = win.pr.result.positive_linear.copy()
    win._intent_changed("temperature", 40.0)
    app.processEvents()
    assert np.array_equal(win.pr.result.positive_linear, tech_before), \
        "微调改变了技术正片（v5 §7.2 禁止）"
    win._intent_changed("temperature", 0.0)
    app.processEvents()

    # ---- 整卷导出（事务三件套 + sidecar v4 + 批次报告）
    out_dir = Path(tempfile.mkdtemp(prefix="nl-gui-smoke-"))
    from negative_lab.pipeline.roll_export import export_roll_frames
    frames = [rec for rec in (win.session.frame(p) for p in paths)
              if rec is not None]
    rep = export_roll_frames(frames, out_dir, session=win.session,
                             roll_id=win.session.roll_groups[-1].id,
                             progress=lambda m: None)
    from negative_lab.pipeline.roll_export import EXPORT_STATUS_BUCKETS
    # 守恒按**生产定义的桶集合**算，不再手抄桶名。
    # B-7 给导出加了 `blocked` 桶之后，这里手抄的四桶会把"被门禁挡住"的帧
    # 当成凭空消失的帧来报假红；在不阻断的语料上则悄悄失去检查力。
    print("export summary:", {k: rep.get(k) for k in EXPORT_STATUS_BUCKETS})
    assert rep["failed"] == 0, rep["items"]
    # B-7 拆桶后，被门禁挡住的帧不再进 failed；冒烟若只盯 failed，
    # 一卷"全被挡住"的导出也能绿——那正是假绿。这里显式要求零阻断。
    assert int(rep.get("blocked") or 0) == 0, (
        f"有帧被质量门禁挡住，冒烟不该当没事发生：{rep.get('blocked')}")
    # 计数必须覆盖全部帧（v6.2 §6.6：取消/失败/阻断都不得静默消失）
    assert sum(int(rep.get(k) or 0) for k in EXPORT_STATUS_BUCKETS) == len(frames), rep
    stem = Path(paths[0]).stem
    exports = sorted(out_dir.glob(f"{stem}-positive*"))
    assert len(exports) == 3, [f.name for f in exports]
    side = json.loads(Path([f for f in exports if f.suffix == ".json"][0])
                      .read_text(encoding="utf-8"))
    assert side["schema_version"] == 4 and len(side["source_sha256"]) == 64
    assert side["geometry"]["roi_source_space"]["coordinate_space"] == "source"
    for key in ("output_crop_roi", "image_area_roi", "base_sample_rois"):
        assert key in side["geometry"]["roi_source_space"], key
    assert side["capture_fingerprint"]["editable"] is False
    assert side["base"]["base_source"] in ("frame", "roll", "roll_partial",
                                           "frame+roll", "generic_prior")
    assert side["color"]["output_icc_sha256"], "旁车必须记录输出 ICC"
    import tifffile
    with tifffile.TiffFile(str(out_dir / f"{stem}-positive.tif")) as tf:
        assert tf.pages[0].tags.get(34675) is not None, "TIFF 必须嵌 ICC"
        pos16 = tf.pages[0].asarray()
    assert pos16.dtype == np.uint16 and pos16.ndim == 3
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
