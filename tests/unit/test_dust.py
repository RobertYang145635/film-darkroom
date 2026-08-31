"""§7 除尘/划痕测试：保守检测、可回退修复、修复失败不阻断导出。"""

import numpy as np
import pytest

from negative_lab.pipeline.density import invert
from negative_lab.pipeline.params import InvertParams
from negative_lab.restoration.dust import (detect_dust, inpaint, merge_masks)
from negative_lab.testing.synthetic import make_synthetic_negative


def _inject_dots(t, dots, size=3, depth=0.35):
    t = t.copy()
    for (y, x) in dots:
        t[y - size:y + size + 1, x - size:x + size + 1] *= depth
    return t


class TestDetectDust:
    """尘点测试放在片基条带上：背景均匀，且这正是真实尘点位置。"""

    def _base(self, seed):
        s = make_synthetic_negative(dust=0, noise=0.0005, seed=seed)
        return s["truth"]["base_roi"]

    def test_finds_isolated_spots(self):
        t = self._base(81)
        dots = [(60, 20), (100, 40), (180, 30)]
        t2 = _inject_dots(t, dots)
        det = detect_dust(t2, use_scratches=False)
        for (y, x) in dots:
            assert det["mask"][y - 2:y + 3, x - 2:x + 3].any(), f"漏检 ({y},{x})"
        assert det["n_spots"] >= 3

    def test_not_fooled_by_edges_or_ramp(self):
        """高反差边缘与平滑斜坡不得被当作尘点（误检率控制）。"""
        h, w = 200, 400
        t = np.zeros((h, w, 3), np.float32)
        t[:, :, :] = np.linspace(0.02, 0.6, w)[None, :, None]  # 平滑斜坡
        t[:, w // 2:] = 0.6   # 一条硬边缘
        t[h // 2:, :w // 2] = 0.02  # 另一条硬边缘
        det = detect_dust(t, use_scratches=True)
        assert det["mask"].mean() < 0.005, f"边缘误检 {det['mask'].mean():.3%}"

    def test_scratch_detection(self):
        t = self._base(82).copy()
        t[30:180, 30:32] *= 0.3   # 细长划痕（片基条带上的均匀背景）
        det = detect_dust(t, use_scratches=True)
        assert det["n_scratches"] >= 1
        # 检测应精确覆盖划痕本身（而非扩大到邻域）
        assert det["mask"][30:180, 30:32].mean() > 0.9
        assert det["mask"].sum() < 150 * 6  # 不应大面积误标


class TestInpaint:
    def test_restores_transmittance(self):
        s = make_synthetic_negative(dust=0, noise=0.0002, seed=83)
        t = s["truth"]["base_roi"]
        dots = [(60, 20)]
        t2 = _inject_dots(t, dots, size=2)
        det = detect_dust(t2, use_scratches=False)
        fixed = inpaint(t2, det["mask"], radius=4)
        truth = t[60, 20]
        err_before = abs(t2[60, 20].mean() - truth.mean()) / truth.mean()
        err_after = abs(fixed[60, 20].mean() - truth.mean()) / truth.mean()
        assert err_after < err_before * 0.5, (err_before, err_after)
        assert np.isfinite(fixed).all()

    def test_empty_mask_is_identity(self):
        t = np.random.default_rng(0).random((20, 20, 3), ).astype(np.float32)
        assert np.array_equal(inpaint(t, np.zeros((20, 20), bool)), t)

    def test_toggle_revertible(self):
        """可回退：不启用修复时输出与未修复逐比特一致。"""
        s = make_synthetic_negative(dust=0, noise=0.0005, seed=84)
        t = _inject_dots(s["negative"], [(60, 60)])
        p = InvertParams(dmin=s["truth"]["dmin"], dmax=2.0)
        off = invert(t, p).positive_linear
        on = invert(inpaint(t, np.zeros(t.shape[:2], bool)), p).positive_linear
        assert np.array_equal(off, on)


class TestPipelineSafety:
    def test_repair_failure_does_not_block_export(self, monkeypatch):
        """修复算法抛异常时，核心去色罩仍必须完成（§7 末条）。"""
        import negative_lab.restoration.dust as dust_mod
        s = make_synthetic_negative(dust=0, seed=85)
        t = s["negative"]
        mask = np.zeros(t.shape[:2], bool)
        mask[50, 50] = True

        def boom(*a, **k):
            raise RuntimeError("inpaint crashed")

        monkeypatch.setattr(dust_mod, "inpaint", boom)
        # 安全包装函数必须吞掉修复异常并回退到未修复数据
        from negative_lab.restoration.dust import safe_inpaint  # 待实现则失败
        out, err = safe_inpaint(t, mask)
        assert err is not None and np.array_equal(out, t)


    def test_region_restricted_skips_content_boundary(self):
        """region=片基条带：内容/片基交界暗带不得被标为划痕（真实数据教训）。"""
        s = make_synthetic_negative(black_border=16, dust=30, seed=86)
        t = s["negative"]
        H, W = t.shape[:2]
        bx0, bx1 = s["truth"]["base_cols"]
        region = np.zeros((H, W), bool)
        region[:, bx0:bx1] = True
        det = detect_dust(t, region=region)
        # 交界列（bx0 附近）不得进入蒙版
        assert det["mask"][:, max(0, bx0 - 2):bx0 + 8].sum() == 0
        # 蒙版整体限制在 region 内
        assert (det["mask"] & ~region).sum() == 0


def test_merge_masks():
    a = np.zeros((5, 5), bool)
    a[0, 0] = True
    b = np.zeros((5, 5), bool)
    b[4, 4] = True
    m = merge_masks(a, b, None)
    assert m[0, 0] and m[4, 4] and m.sum() == 2
