"""回归锁：候选可信度必须判在**校正后**的画面（2026-09-09 缺陷）。

缺陷原文（`pipeline/candidates.py` 修复前）：
`trusted = conf >= NEUTRAL_MIN_CONFIDENCE and ...`，而 `conf` 来自
`scene_confidence(ctx.technical_positive, ...)` —— 喂的是**还没去色罩**的技术正片。
橙色片基色罩自己就把 `sat_pen` 与 `neutral_pen`（最多 ×0.325）打满，
于是"色罩这个症状"成了"不该去色罩"的证据。实测：
`scripts/experiment_trust_domain.py` 跨 5 卷 20 帧 —— 同一帧 conf
0.163→0.952、0.131→0.955；trusted 4/20 → 14/20；
全语料 169 帧里默认候选 `preserve` 占 162 帧（95.9%），
另有 4 帧 `trusted=True` 仍被第二道 `conf >= 0.72` 的坎判回 preserve。

这里锁四件事：
1. 真色罩 + 真有中性面的画面：必须判可信、默认必须是 neutral；
   且**校正前后的统计都留在 diagnostics 里**（换了判据不许藏旧证据）；
2. `_choose_default` 不再在同一把尺子上放第二道门槛；
3. 保留的那道防线仍然有效：估计器分歧过大 -> 仍然 preserve；
4. `scene` 键的含义变了就必须在 diagnostics 里自述（`confidence_measured_on`），
   否则下游读到的数字会被误认为旧口径。
"""

from __future__ import annotations

import numpy as np

from negative_lab.pipeline.candidates import (
    MAX_ESTIMATOR_SPREAD, Candidate, CandidateContext, _choose_default,
    build_candidates)

MASK_GAIN = np.array([1.0, 0.55, 0.35], np.float32)   # 典型 C-41 橙色片基


def _cast_neutral_scene(seed: int = 3) -> np.ndarray:
    """一个**本来就中性**的画面（灰阶 + 少量亮暗块）被片基色罩污染后的技术正片。

    这就是"去色罩"要处理的标准情形：画面里大量像素本该是中性面，
    色罩一盖，校正前看起来"高饱和、几乎没有中性面"。
    """
    rng = np.random.default_rng(seed)
    h, w = 96, 128
    grey = rng.uniform(0.12, 0.95, (h, w, 1)).astype(np.float32)
    img = np.repeat(grey, 3, axis=2)
    img[:20] = 0.03            # 暗块（黑布/阴影）
    img[-14:, 30:80] = 0.97    # 亮块（白墙）
    cast = np.clip(img * MASK_GAIN[None, None, :], 0.0, 1.0)
    return cast


def _ctx(pos: np.ndarray) -> CandidateContext:
    return CandidateContext(technical_positive=pos,
                            valid_mask=np.ones(pos.shape[:2], bool),
                            roll_confidence=0.0, base_source="frame")


def test_cast_with_neutral_surfaces_is_now_trusted():
    pos = _cast_neutral_scene()
    cs = build_candidates(_ctx(pos))
    assert cs is not None
    assert cs.default_name == "neutral", (
        "本卷典型场景：画面有中性面、偏色来自片基色罩，"
        "旧判据会永远退回 preserve（这就是用户说的色罩没除干净）")
    diag = cs.diagnostics
    # 旧证据没被藏起来，且新口径写在数字旁边
    assert "scene_before_correction" in diag and "scene" in diag
    assert diag["confidence_measured_on"].startswith("corrected_positive")
    assert diag["scene"]["sat_frac"] < diag["scene_before_correction"]["sat_frac"]
    assert diag["scene"]["neutral_frac"] > diag["scene_before_correction"]["neutral_frac"]


def test_old_domain_would_have_rejected_it():
    """反向对照：按**旧判据**（判在校正前）这一帧确实会被否掉。

    没有这条对照，上面那条测试可能只是因为合成样张"太好做"而通过。
    """
    from negative_lab.pipeline.candidates import (NEUTRAL_MIN_CONFIDENCE,
                                                  estimate_illuminants,
                                                  scene_confidence)
    pos = _cast_neutral_scene()
    mask = np.ones(pos.shape[:2], bool)
    est = estimate_illuminants(pos, mask, None)
    conf_before, stats_before = scene_confidence(pos, mask, est)
    assert conf_before < NEUTRAL_MIN_CONFIDENCE, (
        "校正前的置信度居然够高 —— 说明这条回归锁的对照前提不成立，"
        "得重挑样张而不是改断言")
    assert stats_before["neutral_frac"] < 0.12   # gray-world 前提"看起来"不成立


def test_no_second_threshold_on_the_same_ruler():
    """`trusted` 即 neutral：0.72 那道坎（同尺子的第二道门槛）必须已经拆掉。"""
    keep = Candidate(name="preserve", intent={}, confidence=1.0)
    ctx = _ctx(np.zeros((4, 4, 3), np.float32))
    for conf in (0.50, 0.55, 0.71):     # 全部低于旧的第二道 0.72
        assert _choose_default(keep, keep, keep, conf, True, ctx) == "neutral", (
            f"conf={conf} 且 trusted=True 仍被退回 preserve：旧的第二道门槛还在")
    assert _choose_default(keep, keep, keep, 0.99, False, ctx) == "preserve"


def test_estimator_disagreement_still_forces_preserve():
    """保留的那道防线不能被顺手拆掉：估计器分歧过大 -> 不自动中和。

    构造：画面本身中性，但 valid_mask 只露出一小块色相杂乱的像素，
    使各估计器给出方向相反的解。断言只看结论 —— 具体分歧多大是实现细节。
    """
    rng = np.random.default_rng(11)
    pos = rng.uniform(0.0, 1.0, (64, 64, 3)).astype(np.float32)   # 各通道独立噪声
    ctx = _ctx(pos)
    ctx.valid_mask = np.zeros((64, 64), bool)
    ctx.valid_mask[:6, :6] = True        # 只剩 36 个随机像素：统计互相打架
    cs = build_candidates(ctx)
    if cs is None or cs.neutral is None:            # 有效像素不足 -> 全 preserve
        assert cs is None or cs.default_name == "preserve"
        return
    spread = float((cs.diagnostics or {}).get("estimator_spread_rad") or 0.0)
    if spread > MAX_ESTIMATOR_SPREAD:
        assert cs.default_name == "preserve", (
            f"估计器分歧 spread={spread:.3f} 超过 {MAX_ESTIMATOR_SPREAD} "
            "却没有退回 preserve，防线被拆了")


def test_diagnostics_self_describe_the_confidence_domain():
    """`scene` 这个键的含义改过（校正前 -> 校正后），必须自带说明。

    下游（sidecar / UI / 判定条）读的是 `diagnostics["scene"]`：
    口径换了却不标注，历史报告就会被当成同一件事比较。
    """
    cs = build_candidates(_ctx(_cast_neutral_scene()))
    diag = cs.diagnostics
    assert diag.get("confidence_measured_on"), (
        "diagnostics['scene'] 的判据口径必须自述，否则新旧数字会被混着比较")
    assert diag.get("neutral_min_confidence") is not None
