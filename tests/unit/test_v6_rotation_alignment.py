"""P0-A2 回归（v6.1 §2.1）：旋转/镜像后底图与 ROI 必须同一坐标系。

历史缺陷：_set_stage()/_frame_done()/_show_frame()/_hold_original() 直接
set_image(self._small)（SOURCE 方向），而 ROI 已用 OrientTransform 映射到
ORIENTED 坐标——rotation_degrees 非 0 或 mirror 时底图与框错位。
锁定契约：唯一的「分析阶段显示图」函数对 small 应用与 FrameRecord.geometry
完全相同的 OrientTransform；90/270 度宽高交换由 OrientTransform 保证。
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module", autouse=True)
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from negative_lab.ui import theme

    app = QApplication.instance() or QApplication([])
    theme.apply(app)
    return app


def _marked_negative(h: int = 240, w: int = 320) -> np.ndarray:
    """非方形合成负片，四个象限亮度不同 + 左上角高亮点（方向标记）。"""
    t = np.full((h, w, 3), 0.25, np.float32)
    t[: h // 2, : w // 2] = 0.5          # 左上
    t[: h // 2, w // 2:] = 0.35          # 右上
    t[h // 2:, : w // 2] = 0.15          # 左下
    t[:40, :40] = 0.9                    # 左上角方向标记
    return t


def _make_win(monkeypatch, rot: int, mirror: bool):
    from negative_lab.library.models import (FrameRecord, GeometryRegions,
                                             ImportSession)
    from negative_lab.ui.main_window import MainWindow

    t = _marked_negative()
    h, w = t.shape[:2]
    geo = GeometryRegions(
        rotation_degrees=rot, mirror_horizontal=mirror,
        output_crop_roi=(20, 30, h - 40, w - 50),
        image_area_roi=(24, 34, h - 44, w - 54),
        base_sample_rois=[(4, w - 60, 16, w - 4)])
    rec = FrameRecord(source_path="synthetic/rot.tif",
                      source_sha256="0" * 64,
                      input_facts={"abs_path": "synthetic/rot.tif",
                                   "kind": "tiff", "width": w, "height": h},
                      geometry=geo)
    win = MainWindow()
    win.session = ImportSession(files=[rec])
    win.current = rec
    monkeypatch.setattr(MainWindow, "_load",
                        lambda self, r: SimpleNamespace(rgb=t, meta={
                            "kind": "tiff", "sha256": "0" * 64,
                            "width": w, "height": h}))
    d = win._load_frame_job(rec, lambda msg: None)
    win._frame_done({"ok": True, "value": d}, rec)
    return win, rec, t


@pytest.mark.parametrize("rot,mirror", [
    (0, False), (90, False), (180, False), (270, False), (0, True)])
def test_analysis_base_image_and_rois_share_coordinate_system(
        monkeypatch, rot, mirror):
    """底图方向正确（象限亮度断言）、ROI 覆盖同一源区域、宽高交换正确。"""
    from negative_lab.pipeline.geometry import OrientTransform

    win, rec, t = _make_win(monkeypatch, rot, mirror)
    small = win._small
    s = win._small_scale
    tr = OrientTransform(win.t_source.shape[:2],
                         rec.geometry.rotation_degrees,
                         rec.geometry.mirror_horizontal, 0.0)
    oh, ow = tr.oriented_size
    sh, sw = oh // s, ow // s

    # 1) 底图必须是 oriented 方向：走唯一显示函数
    img = win._analysis_image()
    assert img is not None
    assert img.shape[:2] == (sh, sw), \
        f"底图尺寸必须等于 oriented small：{img.shape[:2]} != {(sh, sw)}"

    # 2) 方向正确性：oriented 左上象限应等于对应源象限的降采样
    exp = tr.apply(small)
    assert img.shape == exp.shape
    assert np.allclose(img, exp), "底图必须与 OrientTransform.apply(small) 一致"
    # 象限亮度语义：源左上高亮标记应出现在变换后的对应角
    # （角点位置由变换本身推导，不硬编码）
    from negative_lab.pipeline.geometry import _int_source_to_oriented

    oy, ox = _int_source_to_oriented((0.0, 0.0), *win.t_source.shape[:2],
                                     rot, mirror)
    top = oy == 0.0
    left = ox == 0.0
    q = img.shape[0] // 4
    qs = img.shape[1] // 4
    ys = slice(0, q) if top else slice(img.shape[0] - q, img.shape[0])
    xs = slice(0, qs) if left else slice(img.shape[1] - qs, img.shape[1])
    marker_region = img[ys, xs]
    assert marker_region.mean() > img.mean() + 0.05, \
        f"rot={rot} mirror={mirror}：源左上标记必须出现在 oriented 对应角"

    # 3) ROI 覆盖同一源区域：view 里的 oriented 坐标 -> SOURCE 应还原
    from negative_lab.ui.preview_view import RoiKind

    r = win.view.roi(RoiKind.IMAGE_AREA)
    assert r is not None
    o = (int(r.y()) * s, int(r.x()) * s,
         int(r.bottom()) * s, int(r.right()) * s)
    back = tr.bbox_oriented_to_source(o)
    src_roi = rec.geometry.image_area_roi
    for a, b in zip(back, src_roi):
        assert abs(a - b) <= s, \
            f"ROI 映射回 SOURCE 偏差超过 1 个 small 像素：{back} vs {src_roi}"


@pytest.mark.parametrize("rot,mirror", [(90, False), (270, True)])
def test_save_and_reopen_geometry_within_one_source_pixel(monkeypatch,
                                                          rot, mirror):
    """保存草稿再重开：SOURCE 坐标语义不变（<=1 源像素）。"""
    win, rec, t = _make_win(monkeypatch, rot, mirror)
    geo_before = rec.geometry.to_dict()
    # 重开（同一 rec 再走一次加载路径，模拟关闭后重开同一帧）
    d = win._load_frame_job(rec, lambda msg: None)
    win._frame_done({"ok": True, "value": d}, rec)
    g = rec.geometry

    def norm(v):
        if v is None:
            return None
        if isinstance(v, (list, tuple)) and v and isinstance(v[0], (list, tuple)):
            return [list(x) for x in v]
        return list(v) if isinstance(v, tuple) else v

    for name in ("rotation_degrees", "mirror_horizontal",
                 "output_crop_roi", "image_area_roi", "base_sample_rois"):
        assert norm(getattr(g, name)) == norm(geo_before[name]), \
            f"重开后 {name} 漂移超过容差"
