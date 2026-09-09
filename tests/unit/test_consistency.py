"""R1 卷内一致性度量（去色罩算法研究第一轮）。

锁定契约：
- frame_consistency：两帧技术正片在有效画面内的 Lab 直方图一致性
  （mean/median ΔE + a*/b* 偏移向量）；
- roll_consistency：卷内两两 ΔE 矩阵 + 汇总（mean/p95/最不一致帧对）；
- a/b 偏移（色偏方向）与 L（明暗）分开报告——卷内"一致"主要指色偏一致，
  明暗差属于曝光语义；
- 离群帧判定：与卷中位色偏的马氏距离式偏差超过阈值。
"""

from __future__ import annotations

import numpy as np


def _img(h: int, w: int, base=(0.5, 0.5, 0.5), cast=(0.0, 0.0, 0.0),
         seed: int = 0) -> np.ndarray:
    """构造 (H,W,3) sRGB 域测试图：基色 + 均匀色偏 + 轻噪声。"""
    rng = np.random.default_rng(seed)
    img = np.array(base, np.float32) + np.array(cast, np.float32) \
        + rng.normal(0, 0.003, (h, w, 3)).astype(np.float32)
    return np.clip(img, 0, 1).astype(np.float32)


class TestFrameConsistency:
    def test_identical_images_near_zero(self):
        from negative_lab.diagnostics.consistency import frame_consistency

        a = _img(64, 64, seed=1)
        r = frame_consistency(a, a)
        assert r["de_mean"] < 0.5
        assert abs(r["ab_shift"][0]) < 0.5 and abs(r["ab_shift"][1]) < 0.5
        assert r["n"] > 0

    def test_blue_cast_detected_in_b_minus(self):
        """偏蓝 = b* 下降：shift[1] 应显著为负，且 ΔE 抬高。"""
        from negative_lab.diagnostics.consistency import frame_consistency

        a = _img(64, 64, seed=2)
        b = _img(64, 64, cast=(0.0, 0.0, 0.12), seed=2)
        r = frame_consistency(a, b)
        assert r["de_mean"] > 2.0, r
        assert r["ab_shift"][1] < -2.0, r["ab_shift"]

    def test_l_shift_reported_separately(self):
        """明暗差异进 l_shift，不算进色偏方向。"""
        from negative_lab.diagnostics.consistency import frame_consistency

        a = _img(64, 64, seed=3)
        dark = np.clip(a * 0.6, 0, 1).astype(np.float32)
        r = frame_consistency(a, dark)
        assert r["l_shift"] < -5.0
        assert abs(r["ab_shift"][0]) < 2.0 and abs(r["ab_shift"][1]) < 2.0

    def test_mask_limits_region(self):
        from negative_lab.diagnostics.consistency import frame_consistency

        a = _img(64, 64, seed=4)
        b = a.copy()
        b[:, :32] = np.array([0.2, 0.3, 0.6], np.float32)   # 左半偏蓝
        m = np.zeros((64, 64), bool)
        m[:, 32:] = True                                    # 只比右半
        r = frame_consistency(a, b, mask=m)
        assert r["de_mean"] < 0.5


class TestRollConsistency:
    def test_uniform_roll_is_consistent(self):
        from negative_lab.diagnostics.consistency import roll_consistency

        frames = [_img(48, 48, seed=i) for i in range(4)]
        r = roll_consistency(frames)
        # 帧间仅独立噪声（sigma=0.003）：ΔE 噪声地板 ~0.7-1
        assert r["de_mean"] < 1.5, r
        assert r["outliers"] == []
        assert r["n_frames"] == 4 and r["n_pairs"] == 6

    def test_single_outlier_detected(self):
        """三帧正常 + 一帧强蓝偏：离群帧必须被点名。"""
        from negative_lab.diagnostics.consistency import roll_consistency

        frames = [_img(48, 48, seed=i) for i in range(3)]
        bad = _img(48, 48, cast=(0.0, 0.0, 0.15), seed=9)
        r = roll_consistency(frames + [bad])
        assert 3 in r["outliers"], r
        assert r["de_p95"] > r["de_mean"]

    def test_ab_matrix_shape_and_symmetry(self):
        from negative_lab.diagnostics.consistency import roll_consistency

        frames = [_img(40, 40, cast=(0.02 * i, 0, 0), seed=i) for i in range(5)]
        r = roll_consistency(frames)
        M = np.asarray(r["de_matrix"])
        assert M.shape == (5, 5)
        assert np.allclose(M, M.T, atol=1e-6)
        assert (np.diag(M) == 0).all()

    def test_empty_and_single_frame(self):
        from negative_lab.diagnostics.consistency import roll_consistency

        assert roll_consistency([])["n_frames"] == 0
        r = roll_consistency([_img(32, 32)])
        assert r["n_frames"] == 1 and r["n_pairs"] == 0


class TestOutlierLogic:
    def test_outlier_threshold_monotone(self):
        """偏移越大越容易判离群（阈值行为单调）。"""
        from negative_lab.diagnostics.consistency import roll_consistency

        frames = [_img(48, 48, seed=7) for _ in range(4)]
        mild = _img(48, 48, cast=(0, 0, 0.05), seed=7)
        strong = _img(48, 48, cast=(0, 0, 0.18), seed=7)
        r_mild = roll_consistency(frames + [mild])
        r_strong = roll_consistency(frames + [strong])
        assert 4 in r_strong["outliers"]
        assert (4 in r_mild["outliers"]) is False or \
            r_mild["outlier_score"][4] < r_strong["outlier_score"][4]
