"""质量门禁测试（v5 §8 / §11.2）。

v4 的 needs_calibration / needs_regions 产品状态已被 v5 §8.1 删除，
对应测试改写为"缺色卡/缺标定不阻断、低置信度只 needs_review"。
"""

from __future__ import annotations

import numpy as np
import pytest

from negative_lab.diagnostics.quality_gate import VALID_STATUS, QualityGate


def _img_with(clip_low_frac=0.0, clip_high_frac=0.0, shape=(100, 100)):
    a = np.full((*shape, 3), 0.5, np.float32)
    n = int(clip_low_frac * shape[0] * shape[1])
    a.reshape(-1, 3)[:n, 0] = 0.0          # R 通道低端剪裁
    m = int(clip_high_frac * shape[0] * shape[1])
    a.reshape(-1, 3)[:m, 2] = 1.0          # B 通道高端剪裁
    return a


def _preclip(low=(0.0, 0.0, 0.0), high=(0.0, 0.0, 0.0), n=10000):
    return {"low": list(low), "high": list(high), "n": n}


def test_clean_image_ok():
    img = _img_with()
    g = QualityGate.evaluate(t_in=img, positive_linear=img,
                             technical_preclip=_preclip())
    assert g.status == "ok" and not g.warnings
    assert g.blocks_export is False


def test_over_1pct_technical_preclip_fails_with_channel_direction_ratio():
    """v5 §8.2：技术正片 clamp 前 >1% = fail（通道/方向/比例可见）。"""
    img = _img_with()
    g = QualityGate.evaluate(
        t_in=img, positive_linear=img,
        technical_preclip=_preclip(low=(0.03, 0.0, 0.0), high=(0.0, 0.0, 0.02)))
    assert g.status == "fail"
    assert g.blocks_export is True
    txt = " ".join(g.failures)
    assert "R" in txt and "暗部" in txt and "3.00%" in txt
    assert "B" in txt and "高光" in txt and "2.00%" in txt
    assert g.clip_exceeded


def test_over_01pct_warns_below_1pct():
    img = _img_with()
    g = QualityGate.evaluate(t_in=img, positive_linear=img,
                             technical_preclip=_preclip(low=(0.005, 0.0, 0.0)))
    assert g.status == "warn"
    assert any("0.50%" in w for w in g.warnings)
    assert g.blocks_export is False


def test_input_side_clipping_is_warning_only():
    """v5 §8.2 末段：输入端已发生的剪裁属于采集事实，不阻断导出。"""
    img = _img_with(clip_low_frac=0.05)
    g = QualityGate.evaluate(t_in=img, positive_linear=_img_with(),
                             technical_preclip=_preclip())
    assert g.status == "warn"
    assert g.blocks_export is False
    assert any("重扫" in w for w in g.warnings)


def test_nan_output_fails():
    img = _img_with()
    bad = img.copy()
    bad[0, 0, 0] = np.nan
    g = QualityGate.evaluate(t_in=img, positive_linear=bad)
    assert g.status == "fail"
    assert any("NaN/Inf" in f for f in g.failures)


def test_status_is_structured_enum():
    img = _img_with(clip_low_frac=0.05)
    g = QualityGate.evaluate(t_in=img, positive_linear=img,
                             technical_preclip=_preclip(low=(0.05, 0, 0)))
    assert g.status in VALID_STATUS
    d = g.to_dict()
    assert d["status"] == g.status and "technical_preclip" in d
    assert d["thresholds"]["clip_fail_frac"] == 0.01
    assert d["thresholds"]["measurement"].startswith("逐通道")


def test_mask_validation_errors():
    img = _img_with()
    with pytest.raises(ValueError, match="全 False"):
        QualityGate.evaluate(t_in=img, positive_linear=img,
                             valid_mask=np.zeros((100, 100), bool))
    with pytest.raises(ValueError, match="不符"):
        QualityGate.evaluate(t_in=img, positive_linear=img,
                             valid_mask=np.ones((5, 5), bool))


# ------------------------------------------------------- v5 §8.1 删除项
def test_no_needs_calibration_status_exists():
    assert "needs_calibration" not in VALID_STATUS
    assert "needs_regions" not in VALID_STATUS
    with pytest.raises(TypeError):
        QualityGate.evaluate(t_in=_img_with(), positive_linear=_img_with(),
                             calibration={"level": None})


def test_missing_chart_or_film_type_never_blocks():
    """未知胶片型号 + 通用先验片基：只 needs_review，仍可导出。"""
    img = _img_with()
    g = QualityGate.evaluate(t_in=img, positive_linear=img,
                             technical_preclip=_preclip(),
                             base_source="generic_prior")
    assert g.status == "needs_review"
    assert g.blocks_export is False
    assert any("通用彩色负片先验" in r for r in g.review_reasons)


def test_low_confidence_reasons_explain_the_fallback():
    """v5 §8.3：每条提示必须说明软件实际采用了什么回退。"""
    img = _img_with()
    g = QualityGate.evaluate(
        t_in=img, positive_linear=img, technical_preclip=_preclip(),
        base_source="roll", candidate_confidence=0.2, crop_confidence=0.1,
        scene_flags={"night": True})
    joined = " ".join(g.review_reasons)
    assert "卷级" in joined and "保留氛围" in joined and "裁切" in joined
    assert "夜景" in joined
    assert g.status == "needs_review"


def test_rendered_clip_warns_without_touching_technical_gate():
    """v5 §8.2：微调产生的剪裁单独提示，不得改写技术门禁。"""
    img = _img_with()
    rendered = np.full((100, 100, 3), 0.5, np.float32)
    rendered.reshape(-1, 3)[:600, 2] = 1.0        # B 通道 6% 高端剪裁
    g = QualityGate.evaluate(t_in=img, positive_linear=img,
                             technical_preclip=_preclip(),
                             rendered_positive=rendered)
    assert g.status == "warn"
    assert g.blocks_export is False
    assert any("微调后输出剪裁" in w for w in g.warnings)
