"""v3 P0-E：校准层级对象测试。"""

from __future__ import annotations

import numpy as np
import pytest

from negative_lab.calibration.rollcalib import (RollCalibration,
                                                build_from_neutral_samples)


def test_roundtrip_json(tmp_path):
    cal = RollCalibration(roll="1298_G200", film_type="Kodak Gold 200",
                          source="user-points", verified=True,
                          dmin=(0.28, 0.11, 0.03), dmax=1.9,
                          red_ratio=0.85, blue_ratio=0.71,
                          confidence={"neutral": 0.8})
    p = cal.save(tmp_path)
    got = RollCalibration.load(p)
    assert got.dmin == (0.28, 0.11, 0.03)
    assert got.usable_for_color
    assert got.roll == "1298_G200"


def test_schema_mismatch_rejected(tmp_path):
    p = tmp_path / "x.calib.json"
    p.write_text('{"schema_version": 99, "roll": "a"}', encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        RollCalibration.load(p)


def test_base_only_not_usable_for_color():
    cal = RollCalibration(roll="r", source="base-only")
    assert not cal.usable_for_color


def test_neutral_pair_builds_user_points_calibration():
    # 两密度中性组：z 通道比 1:1:1.2（模拟蓝层更密）
    lo = np.array([-2.0, -2.0, -2.4])
    hi = np.array([-0.5, -0.5, -0.6])
    Z = np.vstack([np.tile(lo, (20, 1)) + np.random.default_rng(1).normal(0, .01, (20, 3)),
                   np.tile(hi, (20, 1)) + np.random.default_rng(2).normal(0, .01, (20, 3))])
    cal = build_from_neutral_samples("test", "C-41", Z)
    assert cal.source == "user-points"
    assert cal.verified
    assert abs(cal.blue_ratio - 1 / 1.2) < 0.05


def test_single_density_neutral_stays_base_only():
    Z = np.tile(np.array([-1.0, -1.0, -1.0]), (30, 1))
    cal = build_from_neutral_samples("test", "C-41", Z)
    assert cal.source == "base-only"
    assert not cal.usable_for_color


def test_supervised_chart_builds_matrix_calibration():
    C = np.array([[0.92, 0.09, 0.05], [0.06, 0.90, 0.07], [0.04, 0.08, 0.94]])
    rng = np.random.default_rng(4)
    D = rng.uniform(0.1, 2.4, (40, 3))
    Z = -(D @ C.T)
    cal = build_from_neutral_samples("test", "C-41", Z, target_density=-D,
                                     provenance={"samples": "chart-24"})
    assert cal.source == "chart-calibration"
    assert cal.density_matrix is not None
    assert cal.usable_for_color
