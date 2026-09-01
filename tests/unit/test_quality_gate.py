"""P0-C：质量门禁测试（F-4 修复的机器化验证）。"""

from __future__ import annotations

import numpy as np

from negative_lab.diagnostics.quality_gate import QualityGate


def _img_with(clip_low_frac=0.0, clip_high_frac=0.0, shape=(100, 100)):
    a = np.full((*shape, 3), 0.5, np.float32)
    n = int(clip_low_frac * shape[0] * shape[1])
    a.reshape(-1, 3)[:n, 0] = 0.0          # R 通道低端剪裁
    m = int(clip_high_frac * shape[0] * shape[1])
    a.reshape(-1, 3)[:m, 2] = 1.0          # B 通道高端剪裁
    return a


def test_clean_image_ok():
    img = _img_with()
    g = QualityGate.evaluate(t_in=img, positive_linear=img)
    assert g.status == "ok" and not g.warnings


def test_over_1pct_clip_fails_with_channel_direction_ratio():
    """v4 §7.2：>1% 剪裁 = fail（通道/方向/比例必须在 failure 文本里）。"""
    img = _img_with(clip_low_frac=0.03, clip_high_frac=0.02)
    g = QualityGate.evaluate(t_in=img, positive_linear=img)
    assert g.status == "fail"
    txt = " ".join(g.failures)
    assert "R" in txt and "暗部" in txt and "3.00%" in txt
    assert "B" in txt and "高光" in txt and "2.00%" in txt
    assert g.clip_exceeded


def test_over_01pct_warns_below_1pct():
    """v4 §7.2：0.1%-1% 之间 = warn。"""
    img = _img_with(clip_low_frac=0.005)
    g = QualityGate.evaluate(t_in=img, positive_linear=img)
    assert g.status == "warn"
    assert any("0.50%" in w for w in g.warnings)


def test_nan_output_fails():
    img = _img_with()
    bad = img.copy()
    bad[0, 0, 0] = np.nan
    g = QualityGate.evaluate(t_in=img, positive_linear=bad)
    assert g.status == "fail"
    assert any("NaN/Inf" in f for f in g.failures)


def test_status_is_structured_enum():
    g = QualityGate.evaluate(t_in=_img_with(clip_low_frac=0.05),
                             positive_linear=_img_with(clip_low_frac=0.05))
    assert g.status in ("ok", "warn", "needs_regions", "needs_calibration", "fail")
    d = g.to_dict()
    assert d["status"] == g.status and "output_clip" in d
    assert d["thresholds"]["clip_fail_frac"] == 0.01   # v4 §7.2：阈值入 sidecar


def test_mask_validation_errors():
    """v4 §3.3：mask 形状不符/空/全 False 必须报错。"""
    img = _img_with()
    import pytest
    with pytest.raises(ValueError, match="全 False"):
        QualityGate.evaluate(t_in=img, positive_linear=img,
                             valid_mask=np.zeros((100, 100), bool))
    with pytest.raises(ValueError, match="不符"):
        QualityGate.evaluate(t_in=img, positive_linear=img,
                             valid_mask=np.ones((5, 5), bool))


def test_needs_calibration_unconditional():
    """v4 §1.2.5/§7.1：彩色负片缺 verified 标定 -> 无条件 needs_calibration，
    confidence=0.9 也不豁免；fail 优先。"""
    img = _img_with()
    g = QualityGate.evaluate(t_in=img, positive_linear=img,
                             calibration={"level": None, "confidence": 0.9},
                             film_type="C-41")
    assert g.status == "needs_calibration"
    g2 = QualityGate.evaluate(t_in=img, positive_linear=img,
                              calibration={"level": None, "confidence": 0.1},
                              film_type="C-41")
    assert g2.status == "needs_calibration"
    bad = img.copy()
    bad[0, 0, 0] = np.nan
    g3 = QualityGate.evaluate(t_in=img, positive_linear=bad,
                              calibration={"level": None, "confidence": 0.9},
                              film_type="C-41")
    assert g3.status == "fail"     # fail 优先


def test_bw_skips_color_calibration():
    """v4 §7.1：黑白可跳过颜色标定，但剪裁/有限值门禁仍执行。"""
    img = _img_with()
    g = QualityGate.evaluate(t_in=img, positive_linear=img,
                             calibration={"level": None, "confidence": 0.0},
                             film_type="b&w")
    assert g.status == "ok"


def test_needs_regions_priority():
    g = QualityGate.evaluate(t_in=_img_with(), positive_linear=_img_with(),
                             calibration={"level": "chart"}, film_type="C-41",
                             regions_ok=False)
    assert g.status == "needs_regions"
