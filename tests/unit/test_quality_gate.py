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


def test_over_1pct_clip_warns_with_channel_direction_ratio():
    img = _img_with(clip_low_frac=0.03, clip_high_frac=0.02)
    g = QualityGate.evaluate(t_in=img, positive_linear=img)
    assert g.status == "warn"
    txt = " ".join(g.warnings)
    assert "R" in txt and "暗部" in txt and "3.00%" in txt
    assert "B" in txt and "高光" in txt and "2.00%" in txt
    assert g.clip_exceeded


def test_under_1pct_not_flagged():
    img = _img_with(clip_low_frac=0.005)
    g = QualityGate.evaluate(t_in=img, positive_linear=img)
    assert not g.clip_exceeded


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
    assert g.status in ("ok", "warn", "fail")
    d = g.to_dict()
    assert d["status"] == g.status and "output_clip" in d
