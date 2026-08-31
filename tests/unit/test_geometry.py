"""P0-C：几何统一测试。"""

from __future__ import annotations

import numpy as np
import pytest

from negative_lab.pipeline.geometry import (
    apply_geometry, roi_source_to_oriented, rotate_cw)


def _label_grid():
    """2x3 标记网格：a(0,0) b(0,1) c(0,2) / d(1,0) e(1,1) f(1,2)。"""
    g = np.zeros((2, 3, 3), np.float32)
    for (y, x), v in [((0, 0), 1), ((0, 1), 2), ((0, 2), 3),
                      ((1, 0), 4), ((1, 1), 5), ((1, 2), 6)]:
        g[y, x, :] = v
    return g


def test_rotate_cw_semantics():
    g = _label_grid()
    r = rotate_cw(g, 90)
    assert r.shape[:2] == (3, 2)          # 宽高互换
    # CW90：顶行 a b c 变成右列
    assert r[0, 1, 0] == 1 and r[2, 1, 0] == 3
    assert r[0, 0, 0] == 4 and r[2, 0, 0] == 6


def test_rotate_360_identity():
    g = _label_grid()
    assert np.array_equal(rotate_cw(rotate_cw(rotate_cw(
        rotate_cw(g, 90), 90), 90), 90), g)


def test_roi_transform_matches_pixels():
    """ROI 变换必须与像素旋转一致：源左列 (0,0,2,1) CW90 后 = 顶行 (0,0,1,2)。"""
    out = roi_source_to_oriented((0, 0, 2, 1), src_h=2, src_w=3, deg=90)
    assert out == (0, 0, 1, 2)
    out = roi_source_to_oriented((0, 0, 2, 3), 2, 3, 180)
    assert out == (0, 0, 2, 3)
    out = roi_source_to_oriented((0, 0, 1, 3), 2, 3, 270)   # 顶行 -> 左列（CCW90）
    assert out == (0, 0, 3, 1)


def test_invalid_rotation_rejected():
    with pytest.raises(ValueError):
        rotate_cw(np.zeros((4, 4, 3), np.float32), 45)


def test_apply_geometry_crop_and_record():
    g = np.random.default_rng(0).random((100, 200, 3)).astype(np.float32)
    # CW90 后尺寸 (200,100)：裁切窗口必须在 oriented 尺寸内
    out, rec = apply_geometry(g, rotation_degrees=90,
                              crop_oriented=(10, 10, 80, 90))
    assert out.shape == (70, 80, 3)
    assert rec.rotation_degrees == 90
    assert rec.source_size == (100, 200)
    assert rec.output_size == (70, 80)
    d = rec.to_dict()
    assert d["crop_oriented"] == [10, 10, 80, 90]


def test_keep_rebate_disables_crop():
    g = np.zeros((50, 60, 3), np.float32)
    out, rec = apply_geometry(g, rotation_degrees=0,
                              crop_oriented=(5, 5, 45, 55), keep_rebate=True)
    assert out.shape == (50, 60, 3)
    assert rec.crop_oriented is None


def test_degenerate_crop_ignored():
    g = np.zeros((50, 60, 3), np.float32)
    out, rec = apply_geometry(g, rotation_degrees=0, crop_oriented=(0, 0, 4, 4))
    assert out.shape == (50, 60, 3)      # 过小裁切被拒绝而非静默产出退化图
    assert rec.crop_oriented is None
