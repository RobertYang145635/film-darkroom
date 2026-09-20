"""回归锁：证据文案必须**说得出是哪个分项、哪个通道**。

背景：门禁与片基估计器刚补上了证据（`clip_attribution` /
`best_rejected_region` / 分项 `scores`），但右栏、判定条、sidecar 三处
都要把证据写成一句话。没有共同出口的文案一定会各自漂移，
所以文案集中在 `diagnostics/explain.py`，这里锁它的语义。
"""

from __future__ import annotations

import numpy as np

from negative_lab.diagnostics.explain import base_fallback_note, clip_note
from negative_lab.diagnostics.quality_gate import QualityGate

_REJECTED = {
    "roi": (958, 658, 1050, 767), "confidence": 0.298, "contamination": 0.2496,
    "kept_frac": 0.7504, "rel_spread": 0.0761,
    "mad": [0.0179, 0.0482, 0.0761],
    "scores": {"size": 1.0, "purity": 0.917, "spread": 0.298},
    "binding_score": "spread",
    "median_transmittance": [0.244077, 0.09264, 0.025872],
    "note": "该区未通过置信度门槛，**未参与**任何计算；数字仅作诊断",
}


def test_rejected_base_names_the_binding_score_and_the_noisy_channel():
    txt = base_fallback_note({"best_rejected_region": _REJECTED,
                              "regions": [{}]}, "generic_prior")
    assert "颗粒/背光散布" in txt, "必须说出是哪一项分数把片基否了"
    assert "一个通道差就作废整帧测量" in txt
    assert "B 0.076" in txt, "不点出 B 通道的 MAD，就等于把结论藏回聚合值里"
    assert "通用先验" in txt and "不是实测" in txt
    assert "1.000" not in txt, "不许再出现那个把'没量到'和'全脏'混在一起的 1.000"


def test_no_region_is_worded_as_no_region():
    txt = base_fallback_note({"regions": []}, "generic_prior")
    assert "没框到" in txt
    assert "污染" not in txt, "根本没有取样区，却报'污染'是编造"
    assert "best_rejected_region" not in txt


def test_silence_when_the_base_was_measured():
    assert base_fallback_note({"dmin": [0.2, 0.1, 0.03]}, "frame") == ""
    assert base_fallback_note(None, "frame+roll") == ""
    assert base_fallback_note({}, "") == ""


def test_clip_note_matches_the_gate_failure_wording():
    """文案和门禁失败信息必须同调：同通道、同一个采集/软件结论。"""
    pos = np.full((60, 60, 3), 0.5, np.float32)
    t = np.full((60, 60, 3), 0.5, np.float32)
    flat_pos, flat_t = pos.reshape(-1, 3), t.reshape(-1, 3)
    flat_pos[:1080, 2] = 1.0            # B 高端剪裁 30%
    flat_t[:1080, 2] = 0.0              # 同一批像素输入端已死
    g = QualityGate.evaluate(
        t_in=t, positive_linear=pos,
        technical_preclip={"low": [0, 0, 0], "high": [0, 0, 0.30], "n": 3600})
    note = clip_note(g)
    assert "B 30.00%" in note and "采集端就已丢失" in note
    failure = " ".join(g.failures)
    assert "B" in failure and "采集端就已丢失" in failure
    assert clip_note(g.to_dict()) == note, "实例与 to_dict 两种读法必须同文"


def test_clip_note_points_at_the_software_when_input_was_healthy():
    pos = np.full((40, 40, 3), 0.5, np.float32)
    pos.reshape(-1, 3)[:480, 2] = 1.0
    g = QualityGate.evaluate(
        t_in=np.full((40, 40, 3), 0.5, np.float32), positive_linear=pos,
        technical_preclip={"low": [0, 0, 0], "high": [0, 0, 0.30], "n": 1600})
    assert "本软件能改" in clip_note(g)


def test_clip_note_stays_quiet_below_the_warning_line():
    assert clip_note({"clip_attribution": {
        "B": {"high_total": 0.0005, "high_inherited": 0.0005,
              "high_introduced": 0.0}}}) == ""
    assert clip_note({}) == ""
    assert clip_note(None) == ""
