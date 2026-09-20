"""回归锁：片基区被拒绝时，必须说得出**为什么**被拒。

实测背景（2026-09-09，169 帧）：74 帧退回通用先验，报表上清一色
`contamination=1.000 / base_confidence=0.000` —— 看着像"片基全脏"，
实际是自动找到的取样区**量到了东西**（中位透射率、污染 25%、上万像素），
却因为置信度低于 `min_base_confidence=0.45` 被整区拒绝；
而拒绝后的汇总把真实数字丢掉、写死 1.0/0.0。
"没量到"和"量到了但不敢用"在 sidecar、UI、判定条上长成同一个样子。

锁四件事：
1. 拒绝路径**判定不改**：`dmin` 仍是 None、聚合 `contamination` 仍是 1.0
   （卷级模型 `analyze_roll` 靠 contamination>0.80 排除不可靠帧，
   改成真实值会悄悄把帧放回卷模型 —— 那是产品决策，不是修报表的顺手福利）；
2. 但要带出 `best_rejected_region`：真实 contamination / kept_frac / rel_spread /
   中位透射率 / 三项分项分数，并明确标注"未参与任何计算"；
3. `binding_score` 必须等于该区三项分数里的最小项（不硬编码阈值，防漂移）；
4. 一个 ROI 都没有时**不许**伪造 `best_rejected_region`。
"""

from __future__ import annotations

import numpy as np
import pytest

from negative_lab.pipeline.roll_analysis import frame_base_stats

H, W = 120, 200
ROI = [(20, 10, 60, 190)]           # 一个足够大的取样区


def _clean_base(seed: int = 1) -> np.ndarray:
    """干净片基：均匀透射 + 轻微颗粒（密度域散布很小）。"""
    rng = np.random.default_rng(seed)
    t = np.full((H, W, 3), 0.30, np.float32)
    noise = rng.normal(0.0, 0.004, (H, W, 1))          # 三通道同一份：纯亮度颗粒
    t = t * (10 ** noise).astype(np.float32)
    return np.clip(t, 0.0, 1.0)


def _grainy_base(seed: int = 7) -> np.ndarray:
    """颗粒/背光明显偏大的片基：密度域 MAD 大到 spread 分数掉光。

    故意让**通道均值仍一致**（不是颜色问题），并让污染比例保持很低，
    这样被拒的原因只可能落在散布上。
    """
    rng = np.random.default_rng(seed)
    t = np.full((H, W, 3), 0.30, np.float32)
    grain = rng.normal(0.0, 0.16, (H, W, 1))            # 约 0.11 decade MAD
    t = t * (10 ** grain).astype(np.float32)
    return np.clip(t, 0.0, 1.0)


def test_clean_roi_is_accepted_so_the_fixture_is_not_vacuous():
    st = frame_base_stats(_clean_base(), ROI, None)
    assert st["n_regions_used"] == 1 and st["dmin"] is not None
    assert "best_rejected_region" not in st


def test_rejected_roi_keeps_the_decision_but_reports_the_truth():
    st = frame_base_stats(_grainy_base(), ROI, None)
    assert st["n_regions_used"] == 0 and st["dmin"] is None
    # 判定与 load-bearing 字段一个字不改
    assert st["contamination"] == 1.0, (
        "聚合 contamination 是卷级模型排除帧的依据，改成真实值会"
        "悄悄把被拒帧放回卷模型 —— 那是产品决策，不许搭报表修复的车")
    best = st["best_rejected_region"]
    assert best is not None, "量到了东西却被拒，却不报真实数字，用户只能猜"
    assert 0.0 < best["contamination"] < 0.5, (
        f"区本身的污染是 {best['contamination']}，不该在报表里变成 1.0")
    assert best["kept_frac"] > 0.5
    assert best["rel_spread"] is not None
    assert len(best["median_transmittance"]) == 3
    assert "未参与" in best["note"]
    assert any("置信度过低" in w for w in st["warnings"])


def test_binding_score_is_the_smallest_of_the_three():
    """不硬编码阈值：binding 必须等于该区自己三项分数的最小项（防实现漂移）。"""
    for img in (_grainy_base(7), _grainy_base(13), _clean_base(3)):
        st = frame_base_stats(img, ROI, None)
        region = st["regions"][0]
        assert set(region["scores"]) == {"size", "purity", "spread"}
        assert region["confidence"] == pytest.approx(min(region["scores"].values()),
                                                     abs=1e-3)
        best = st.get("best_rejected_region")
        if best is None:
            continue
        assert best["binding_score"] == min(best["scores"], key=best["scores"].get)


def test_no_roi_at_all_does_not_invent_a_rejected_region():
    st = frame_base_stats(_clean_base(), [], None)
    assert st["n_regions_used"] == 0 and st["dmin"] is None
    assert "best_rejected_region" not in st, (
        "根本没有取样区时不许凭空造一个'被拒的区'，"
        "否则'没框到片基'和'框到了但不敢用'又混成一个了")
    assert st["warnings"]
