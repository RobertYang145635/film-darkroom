"""帧/片基自动检测测试（Phase 2，真实翻拍结构：黑边->片基->内容）。"""

import numpy as np

from negative_lab.calibration.border import detect_regions
from negative_lab.pipeline.auto import auto_analyze
from negative_lab.testing.synthetic import ORANGE_BASE, make_synthetic_negative


def test_detect_base_with_black_border():
    s = make_synthetic_negative(black_border=24, dust=80, seed=61)
    neg, truth = s["negative"], s["truth"]
    det = detect_regions(neg)
    assert det["base"] is not None, det["warnings"]
    assert det["edge"] == "right", det
    y0, x0, y1, x1 = det["base"]
    bx0, bx1 = truth["base_cols"]
    # 检测到的条带必须落在真实片基内（允许降采样一步误差）
    step = det["step"]
    assert x0 >= bx0 - step and x1 <= bx1 + step, (det["base"], truth["base_cols"])
    assert det["confidence"] > 0.5


def test_auto_analyze_without_roi_finds_base():
    """端到端：不给 ROI，自动检测应给出接近真值的 Dmin。"""
    s = make_synthetic_negative(black_border=24, dust=80, seed=62)
    neg, truth = s["negative"], s["truth"]
    rep = auto_analyze(neg)
    err = np.max(np.abs(np.array(rep.params.dmin) - np.array(truth["dmin"]))
                 / np.array(truth["dmin"]))
    assert err < 0.05, (rep.params.dmin, truth["dmin"])
    assert rep.confidence["dmin"] > 0.5
    # Dmax 不应被黑边抬高（黑边 t≈0.0004 会被当作超高密度）
    assert abs(rep.params.dmax - truth["dmax"]) / truth["dmax"] < 0.15, rep.params.dmax


def test_no_base_edge_warns():
    """无未曝光片基（边缘带与中心亮度无差别）：必须警告而不是硬猜。"""
    rng = np.random.default_rng(5)
    dmin = np.array(ORANGE_BASE, np.float32)
    h, w = 200, 400
    # 全画面都是内容：密度在 [0.05, 1]*dmax 随机，无亮条带
    u = rng.uniform(0.05, 1.0, size=(h, w)).astype(np.float32)
    content = dmin[None, None, :] * np.power(10.0, -2.0 * u[..., None])
    neg = np.clip(content, 0, 1).astype(np.float32)
    rep = auto_analyze(neg)
    assert any("片基" in w for w in rep.warnings), rep.warnings
    assert rep.confidence["dmin"] < 0.5 or rep.confidence.get("dmax", 1) < 0.5
