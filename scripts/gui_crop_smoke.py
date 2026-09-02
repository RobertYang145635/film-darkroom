"""v6 裁切页冒烟（§10.1：只验页面与 ROI 标定，不做整卷颜色导出）。

offscreen 下：导入样片文件夹 -> 加载一帧 -> 用鼠标事件在画布上**拖出**
三类 ROI -> 保存确认标注 -> 关闭重开校验一致 -> 双尺寸截图。

超时纪律（§15.7）：本脚本由调用方限时（120 秒门禁在 test_v6_ui_contracts
与人工运行中执行），内部不设无限等待。

运行：$env:QT_QPA_PLATFORM="offscreen"; .venv\Scripts\python.exe scripts\gui_crop_smoke.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
from PySide6.QtCore import QPoint, Qt, QRectF  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from negative_lab.io.sidecar import git_commit  # noqa: E402
from negative_lab.ui.main_window import MainWindow  # noqa: E402
from negative_lab.ui.preview_view import RoiKind  # noqa: E402

EVIDENCE = ROOT / "outputs" / "v6_ui"
DEFAULT_FOLDER = ROOT / "20260709_SAMPLE-ID(1)" / "1298_G200" / "翻拍"


def _wait_idle(app, win, timeout=180):
    import time as _t

    t0 = _t.time()
    while _t.time() - t0 < timeout:
        app.processEvents()
        if not win._busy._locks and threading.active_count() <= 2:
            _t.sleep(0.3)
            app.processEvents()
            return True
        _t.sleep(0.05)
    return False


def _drag(view, p0, p1):
    """p0/p1 为场景比例坐标（0-1），映射到 viewport 像素后拖动。"""
    vp = view.viewport()
    from PySide6.QtCore import QPointF

    def to_vp(fx, fy):
        # 先用变换矩阵把场景点映射到 viewport 像素（fit() 之后 1:1 失效）
        sp = view.mapFromScene(QPointF(fx * view._scene_rect.width(),
                                       fy * view._scene_rect.height()))
        return QPoint(sp.x(), sp.y())

    QTest.mousePress(vp, Qt.LeftButton, Qt.NoModifier, to_vp(*p0))
    QTest.mouseMove(vp, to_vp(*p1))
    QTest.mouseRelease(vp, Qt.LeftButton, Qt.NoModifier, to_vp(*p1))


def main() -> int:
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_FOLDER
    if not folder.exists():
        print(f"样片不存在：{folder}")
        return 1
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    # 标注重定向到临时目录：冒烟中的“用户确认标注”不能污染真实 fixtures，
    # 更不能替真实样片产生未经用户签字的 verified 记录
    import negative_lab.calibration.annotations as annmod

    ann_tmp = tempfile.TemporaryDirectory(prefix="v6-smoke-ann-")
    annmod.ANNOTATIONS_DIR = Path(ann_tmp.name)
    app = QApplication([])
    from negative_lab.ui import theme

    theme.apply(app)
    win = MainWindow()
    win.show()
    win.load_folder(folder)
    _wait_idle(app, win)
    assert win.session is not None, "导入未完成"
    paths = [f.source_path for f in win.session.files[:1]]
    assert paths, "导入后没有可用文件"
    win.session.create_roll("冒烟卷", paths)
    win.library.set_session(win.session)
    win.library.select_roll(win.session.roll_groups[-1].id)
    win.load_frame(paths[0])
    _wait_idle(app, win)
    assert win.pr is not None, "预览服务未产出结果"
    view = win.view

    # ---- 用户直接拖出三类 ROI（不是自动检测，不是默认矩形）
    # 冒烟要验证"从空白画布创建"：断开预览重种（自动检测会在编辑后重新
    # 种回三类框，按 §5.2 从已有框内按下会进入移动——那是规格行为，
    # 不属于本冒烟要覆盖的路径）。
    win._reset_all_geometry()
    _prev = win._refresh_preview
    win._refresh_preview = lambda *a, **k: None
    win.clear_banner()
    view.set_draw_mode(RoiKind.OUTPUT_CROP)
    _drag(view, (0.05, 0.06), (0.95, 0.55))
    view.set_draw_mode(RoiKind.IMAGE_AREA)
    _drag(view, (0.08, 0.10), (0.92, 0.50))
    view.set_draw_mode(RoiKind.BASE)
    _drag(view, (0.06, 0.60), (0.42, 0.68))
    _drag(view, (0.56, 0.72), (0.94, 0.80))
    win._update_draft_from_view()

    counts = {"output_crop": view.roi_count(RoiKind.OUTPUT_CROP),
              "image_area": view.roi_count(RoiKind.IMAGE_AREA),
              "base": view.roi_count(RoiKind.BASE)}
    assert counts == {"output_crop": 1, "image_area": 1, "base": 2}, counts
    rec = win.current
    assert rec.geometry.is_complete(), rec.geometry.completeness_errors()
    win._refresh_preview = _prev          # 恢复正常预览刷新
    win._refresh_preview()

    # ---- 保存为用户确认标注 + 重开校验一致
    win.save_regions_as_annotation()
    assert "已保存为用户确认标注" in win._banner_text or \
        "确认标注" in win._banner_text, win._banner_text
    from negative_lab.calibration.annotations import load_annotation
    ann = load_annotation(rec.source_path)
    assert ann is not None and ann.is_user_verified()
    saved = {
        "output_crop_roi": list(ann.output_crop_roi),
        "image_area_roi": list(ann.image_area_roi),
        "base_sample_rois": [list(r) for r in ann.base_sample_rois],
        "rotation_degrees": ann.rotation_degrees,
        "mirror_horizontal": bool(getattr(ann, "mirror_horizontal", False)),
        "coordinate_space": "source",
    }

    # 重开：重新加载同一帧，三类 ROI 恢复
    win.load_frame(rec.source_path)
    _wait_idle(app, win)
    reopened = {
        "output_crop_roi": _to_source(win.current.geometry.output_crop_roi),
        "image_area_roi": _to_source(win.current.geometry.image_area_roi),
        "base_sample_rois": [_to_source(r) for r in
                             win.current.geometry.base_sample_rois],
        "rotation_degrees": win.current.geometry.rotation_degrees,
        "mirror_horizontal": win.current.geometry.mirror_horizontal,
        "coordinate_space": "source",
    }
    drift = _max_drift(saved, reopened)
    assert drift <= 1, f"重开漂移 {drift} 源像素 > 1：{saved} vs {reopened}"

    # ---- 双尺寸截图（布局证据，不替代契约断言）
    shots = []
    for w, h in ((1280, 780), (1440, 900)):
        win.resize(w, h)
        app.processEvents()
        p = EVIDENCE / f"ui_smoke_{w}x{h}.png"
        win.grab().save(str(p))
        shots.append(p.as_posix())

    report = {
        "git_commit": git_commit(),
        "folder": folder.as_posix(),
        "frame": rec.source_path,
        "frame_size": list(win.t_source.shape[:2]),
        "roi_counts": counts,
        "saved_annotation": saved,
        "reopened_geometry": reopened,
        "max_roundtrip_drift_px": drift,
        "screenshots": shots,
        "banner": win._banner_text,
        "note": "本冒烟只验裁切页与 ROI 标定；不执行整卷颜色导出（v6 §10.1）",
    }
    (EVIDENCE / "ui_interaction.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in
                      ("frame", "frame_size", "roi_counts",
                       "max_roundtrip_drift_px", "screenshots")},
                     ensure_ascii=False, indent=2))
    print("GUI CROP SMOKE OK")
    return 0


def _to_source(roi):
    return list(roi) if roi else None


def _max_drift(a: dict, b: dict) -> int:
    worst = 0
    for key in ("output_crop_roi", "image_area_roi"):
        d = _roi_drift(a[key], b[key])
        worst = max(worst, d)
    for ra, rb in zip(a["base_sample_rois"], b["base_sample_rois"]):
        worst = max(worst, _roi_drift(ra, rb))
    if a["rotation_degrees"] != b["rotation_degrees"]:
        worst = max(worst, 10 ** 6)
    return worst


def _roi_drift(a, b) -> int:
    if a is None or b is None:
        return 10 ** 6 if (a is None) != (b is None) else 0
    return max(abs(x - y) for x, y in zip(a, b))


if __name__ == "__main__":
    raise SystemExit(main())
