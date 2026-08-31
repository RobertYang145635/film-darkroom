"""§8.1.5：ROI 被污染、没有片基、没有中性点时给出明确低置信度警告。"""

import numpy as np

from negative_lab.calibration.base import (base_roi_sanity_check, estimate_dmax_offset,
                                           robust_dmin)
from negative_lab.pipeline.auto import auto_analyze
from negative_lab.testing.synthetic import make_synthetic_negative


def test_clean_base_high_confidence():
    s = make_synthetic_negative(dust=0, noise=0.0005, seed=51)
    res = robust_dmin(s["truth"]["base_roi"])
    err = np.max(np.abs(res["dmin"] - np.array(s["truth"]["dmin"], np.float32))
                 / np.array(s["truth"]["dmin"], np.float32))
    assert err < 0.02, f"干净片基 Dmin 误差应 <2%，实际 {err:.3f}"
    assert res["confidence"] >= 0.7, res


def test_contaminated_base_robust_and_lower_confidence():
    clean = make_synthetic_negative(dust=0, noise=0.0005, seed=52)
    dirty = make_synthetic_negative(dust=400, noise=0.0005, seed=52)
    rc = robust_dmin(clean["truth"]["base_roi"])
    rd = robust_dmin(dirty["truth"]["base_roi"])
    err = np.max(np.abs(rd["dmin"] - np.array(dirty["truth"]["dmin"], np.float32))
                 / np.array(dirty["truth"]["dmin"], np.float32))
    assert err < 0.05, f"尘点污染下 Dmin 仍应稳健（<5%），实际 {err:.3f}"
    assert rd["confidence"] <= rc["confidence"]
    assert any("污染" in w or "置信度" in w for w in rd["warnings"])


def test_roi_on_content_flagged():
    """片基框放在画面内容上：必须给出可解释的警告。"""
    s = make_synthetic_negative(seed=53)
    warns = base_roi_sanity_check(s["negative"], s["truth"]["film_roi"])
    assert warns and "不在未曝光片基" in warns[0]


def test_no_base_strip_low_confidence():
    """ROI 过小/无有效片基：置信度 0 并警告，使用回退值而非崩溃。"""
    roi = np.full((3, 3, 3), 0.02, np.float32)  # 有效像素 <16
    res = robust_dmin(roi)
    assert res["confidence"] == 0.0
    assert res["warnings"]
    assert np.isfinite(res["dmin"]).all()


def test_auto_without_neutral_warns_and_outputs():
    """§4.6.1 + 整改升级链：无用户取点时走统计中性或兜底，但警告必须可见。"""
    s = make_synthetic_negative(seed=54)
    rep = auto_analyze(s["negative"], base_roi=s["truth"]["base_roi"])
    assert any("统计中性" in w or "无中性参考" in w for w in rep.warnings), rep.warnings
    assert rep.confidence.get("neutral_statistical", rep.confidence.get("neutral", 0)) <= 0.4
    from negative_lab.pipeline.density import invert
    out = invert(s["negative"], rep.params).positive_linear
    assert np.isfinite(out).all()


def test_auto_full_pipeline_accuracy():
    """自动链路端到端：Dmin/Dmax 接近真值，输出有限，中性点色偏小。"""
    s = make_synthetic_negative(dust=100, noise=0.001, seed=55)
    neg, truth = s["negative"], s["truth"]
    rep = auto_analyze(neg, base_roi=truth["base_roi"], film_roi=truth["film_roi"])
    dmin_err = np.max(np.abs(np.array(rep.params.dmin) - np.array(truth["dmin"]))) / 0.62
    assert dmin_err < 0.05, rep.params.dmin
    assert abs(rep.params.dmax - truth["dmax"]) / truth["dmax"] < 0.15, rep.params.dmax
    from negative_lab.pipeline.density import invert
    out = invert(neg, rep.params).positive_linear
    assert np.isfinite(out).all()
    r, c = truth["neutral_bright"]
    patch = out[r - 2:r + 3, c - 2:c + 3, :].reshape(-1, 3).mean(axis=0)
    assert (patch.max() - patch.min()) / patch.mean() < 0.02
