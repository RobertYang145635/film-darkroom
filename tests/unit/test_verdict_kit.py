"""人工判定基准集内核的回归锁（diagnostics/verdicts.py）。

这里锁的不是"颜色好不好看"（那只能由用户判），而是**基准集这件事本身
不许骗人**的四条性质：

1. 分层标记只从证据来：缺数字就不判，绝不"默认False以外的东西"；
2. 抽样必须确定且真的覆盖稀有场景 —— 否则"每次复测同一批帧"是空话；
3. 脚本永远不产生判定，且**重建模板绝不覆盖/丢弃用户已签的判定**
   （旧实现把判定写进 outputs/<run-id>/ 里，重跑换目录即丢）；
4. 没签满就判不通过 —— 不许拿部分样本冒充整体绿。
"""

from __future__ import annotations

import numpy as np
import pytest

from negative_lab.diagnostics.quality_gate import CLIP_WARN_FRAC
from negative_lab.diagnostics.verdicts import (
    STRATA_KEYS, VALID_VERDICTS, blank_verdict, frame_strata, merge_decisions,
    select_stratified, summarize_verdicts)

USER = "用户本人"


def _scene(**kw) -> dict:
    base = {"sat_frac": 0.5, "dark_frac": 0.1, "neutral_frac": 0.5,
            "hue_concentration": 0.3}
    base.update(kw)
    return base


def _flat(value: float, *, shape=(60, 80)) -> np.ndarray:
    return np.full((shape[0], shape[1], 3), value, dtype=np.float32)


# ------------------------------------------------------------------ 分层
def test_frame_strata_uses_pipeline_scene_numbers():
    """分层直接取候选层数字：阈值两侧必须翻转。"""
    on = frame_strata(_flat(0.5), scene=_scene(neutral_frac=0.02))["few_neutral_surfaces"]
    off = frame_strata(_flat(0.5), scene=_scene(neutral_frac=0.50))["few_neutral_surfaces"]
    assert on is True and off is False
    night = frame_strata(_flat(0.5), scene=_scene(dark_frac=0.60))["night_like"]
    assert night is True
    assert frame_strata(_flat(0.5), scene=_scene(dark_frac=0.4))["night_like"] is False


def test_frame_strata_without_scene_evidence_does_not_invent_it():
    """`scene` 缺键 -> 对应分层 False：缺证据不当真。"""
    out = frame_strata(_flat(0.5), scene={})
    for k in ("few_neutral_surfaces", "hue_concentrated", "saturated_dominant",
              "night_like"):
        assert out[k] is False, f"{k} 在没有数字时被凭空判真了"
    # 连 scene 都不传也必须一样，且不许抛
    assert frame_strata(_flat(0.5))["night_like"] is False


def test_frame_strata_clip_threshold_is_the_gate_threshold():
    """剪裁分层必须复用质量门禁的警告线，不能另立口径。"""
    just_over = {"high": [0.0, 0.0, CLIP_WARN_FRAC * 2]}
    just_under = {"high": [0.0, 0.0, CLIP_WARN_FRAC / 2]}
    assert frame_strata(_flat(0.5), technical_preclip=just_over)["highlight_clip"] is True
    assert frame_strata(_flat(0.5), technical_preclip=just_under)["highlight_clip"] is False
    # 门禁的真实形状是 {"low": [..], "high": [..]}；映射形状也要能吃
    assert frame_strata(_flat(0.5), technical_preclip={"low": {"B": 0.5}})["shadow_clip"] is True
    assert frame_strata(_flat(0.5))["highlight_clip"] is False  # 没数字不判


def test_frame_strata_pixel_proxies_discriminate_smooth_bright_top():
    """平均亮度相同、只有平滑度不同的两帧：只有平滑的那帧算天空。

    踩过的两个坑（都是这条代理指标的适用条件，写在这里而不是藏在实现里）：
    1. `frame_strata` 先 clip 到 [0,1] 再算 std，拿 0.85±0.3 做噪声时超过 1.0 的
       部分被压平，实测 std 只有 ~0.14 < 0.15，"花"的那帧也被判成天空 ——
       基准亮度必须留在 clip 以下（0.65±0.30 → std≈0.17）；
    2. 噪声要加在**亮度**上（三通道同一份噪声）。三通道各加独立噪声时，
       luma 是三者加权平均，std 被 √3 摊薄到 0.10，仍然"平滑" ——
       这条分层量的就是亮度平滑度，色度噪声不该算纹理。
    """
    rng = np.random.default_rng(7)
    sky = np.zeros((90, 80, 3), np.float32)
    sky[:30] = 0.65                       # 上 1/3 高亮且完全平滑
    sky[30:] = 0.35
    noisy = sky.copy()
    grain = rng.uniform(-0.30, 0.30, (30, 80, 1)).astype(np.float32)   # 亮度纹理
    noisy[:30] += grain
    assert float(sky[:30].mean()) > 0.55 and float(noisy[:30].mean()) > 0.55
    assert abs(float(sky[:30].mean()) - float(noisy[:30].mean())) < 0.05  # 平均一样
    assert float(noisy[:30].std()) > 0.15                                 # 只差平滑度
    assert frame_strata(sky)["bright_uniform_top"] is True
    assert frame_strata(noisy)["bright_uniform_top"] is False


def test_frame_strata_low_result_contrast_only_when_flat():
    flat = _flat(0.5) + np.linspace(0, 0.02, 60 * 80, dtype=np.float32).reshape(60, 80, 1)
    wide = np.concatenate([_flat(0.02, shape=(30, 80)), _flat(0.98, shape=(30, 80))])
    assert frame_strata(np.clip(flat, 0, 1))["low_result_contrast"] is True
    assert frame_strata(wide)["low_result_contrast"] is False


def test_strata_vocabulary_is_frozen():
    """判定文件里存的是分层键名字：改名 = 历史判定读不懂，必须显式改 schema。"""
    assert STRATA_KEYS == (
        "few_neutral_surfaces", "hue_concentrated", "saturated_dominant", "night_like",
        "bright_uniform_top", "low_result_contrast", "highlight_clip", "shadow_clip",
        "low_base_confidence", "no_nlp_reference")


# ------------------------------------------------------------------ 抽样
def _cand(i: int, *, roll: str = "r0", **strata) -> dict:
    s = {k: False for k in STRATA_KEYS}
    s.update(strata)
    return {"id": f"{roll}/f{i:03d}", "roll": roll, "strata": s}


def test_selection_is_deterministic():
    cands = [_cand(i, roll=f"r{i % 5}", night_like=(i % 7 == 0)) for i in range(40)]
    a, _ = select_stratified(cands, target=12)
    b, _ = select_stratified(list(reversed(cands)), target=12)
    assert [f["id"] for f in a] == [f["id"] for f in b]  # 输入顺序不许影响结果
    assert len(a) == 12


def test_selection_actually_pulls_in_the_rare_stratum():
    """40 帧里只有 2 帧是夜景：抽样必须把它们捞进来，而不是按 id 取前 8 个。"""
    cands = [_cand(i, roll=f"r{i % 4}", night_like=(i in (37, 38))) for i in range(40)]
    chosen, rep = select_stratified(cands, target=8)
    ids = {f["id"] for f in chosen}
    assert {"r1/f037", "r2/f038"} <= ids
    assert rep["strata"]["night_like"]["available"] == 2
    assert rep["strata"]["night_like"]["selected"] == 2


def test_selection_covers_every_roll():
    cands = [_cand(i, roll=f"r{i % 6}") for i in range(60)]
    chosen, rep = select_stratified(cands, target=12)
    assert set(rep["rolls"]) == {f"r{k}" for k in range(6)}, rep["rolls"]
    assert all(v >= 1 for v in rep["rolls"].values())


def test_selection_flags_strata_with_no_discrimination():
    """全语料命中的分层不许假装"覆盖到了"，必须点名说它没区分度。"""
    cands = [_cand(i, roll="r0", few_neutral_surfaces=True) for i in range(20)]
    for i in range(3):
        cands[i]["strata"]["night_like"] = True
    _, rep = select_stratified(cands, target=10)
    assert "few_neutral_surfaces" in rep["strata_without_discrimination"]
    assert "night_like" not in rep["strata_without_discrimination"]


def test_selection_empty_input_is_not_a_crash():
    chosen, rep = select_stratified([], target=10)
    assert chosen == [] and rep["selected"] == 0


# ------------------------------------------------------------------ 判定文件
@pytest.fixture()
def selected():
    return [_cand(1, roll="rA", night_like=True),
            _cand(2, roll="rA"),
            _cand(3, roll="rB", hue_concentrated=True),
            _cand(4, roll="rB", no_nlp_reference=True)]


def test_template_never_contains_a_verdict(selected):
    """脚本产出的模板里，判定字段必须全是空 —— 执行 agent 不得代签。"""
    doc = merge_decisions(None, selected)
    for f in doc["frames"]:
        assert f["verdict"] is None
        assert f["signed_by"] is None
        assert f["tweak_steps"] is None
        assert f["reference_product"] is None
        assert f["issue_tags"] == []
    assert doc["frames"][0]["strata"] == ["night_like"]  # dict -> 列表，只留 True 的


def test_signed_verdicts_survive_a_rebuild(selected):
    """重跑（哪怕这次抽样为空）绝不丢已签判定 —— 旧实现换 run-id 目录就丢。"""
    doc = merge_decisions(None, selected)
    doc["frames"][0]["verdict"] = "tweakable"
    doc["frames"][0]["tweak_steps"] = 1
    doc["frames"][0]["issue_tags"] = ["色罩"]
    doc["frames"][0]["signed_by"] = USER
    again = merge_decisions(doc, [])          # 这次一帧都没抽中
    kept = {f["id"]: f for f in again["frames"]}
    assert kept[doc["frames"][0]["id"]]["verdict"] == "tweakable"
    assert kept[doc["frames"][0]["id"]]["signed_by"] == USER
    assert kept[doc["frames"][0]["id"]]["issue_tags"] == ["色罩"]
    # 未签且没被抽中的才退休，且退休是留痕的
    assert doc["frames"][1]["id"] in again["retired_ids"]


def test_rebuild_refreshes_metadata_but_never_human_answers(selected):
    doc = merge_decisions(None, selected)
    fid = doc["frames"][0]["id"]
    doc["frames"][0].update(verdict="usable", reference_product=True, signed_by=USER)
    doc["frames"][0]["note"] = "肤色可接受"
    fresh = [{**selected[0], "sheet": "outputs/verdicts/sheets/rA/x.jpg",
              "strata": {**selected[0]["strata"], "bright_uniform_top": True}}]
    again = merge_decisions(doc, fresh)
    row = {f["id"]: f for f in again["frames"]}[fid]
    assert row["verdict"] == "usable" and row["note"] == "肤色可接受"
    assert row["sheet"] == "outputs/verdicts/sheets/rA/x.jpg"   # 元数据刷新
    assert "bright_uniform_top" in row["strata"]


def _sign(entry: dict, verdict: str, *, steps=None, ref=None) -> dict:
    entry["verdict"] = verdict
    entry["tweak_steps"] = steps
    entry["reference_product"] = ref
    entry["signed_by"] = USER
    return entry


def _filled(n_per_roll: int = 20, rolls: tuple = ("rA", "rB")) -> dict:
    frames = []
    for r in rolls:
        for i in range(n_per_roll):
            f = blank_verdict({"id": f"{r}/f{i:03d}", "file": f"{r}/f{i:03d}.nef",
                               "roll": r, "strata": {"night_like": i % 2 == 0}})
            frames.append(_sign(f, "usable", ref=True))
    return {"schema_version": 1, "frames": frames, "retired_ids": []}


def test_summarize_passes_on_a_healthy_signed_set():
    doc = _filled()
    s = summarize_verdicts(doc)
    assert s["passed"] is True, [c for c in s["checks"] if not c["ok"]]
    assert s["usable_fraction"] == 1.0


def test_vacuous_criterion_is_marked_na_not_green():
    """全批没有 ② 帧时，"② 里 ≥80% 能救回"这条**不许打勾**：0/0 是假绿。"""
    doc = _filled()                       # 40 帧全签 usable，0 帧 ②
    s = summarize_verdicts(doc)
    rescue = [c for c in s["checks"] if "救回" in c["label"]][0]
    assert rescue["na"] is True
    assert s["rescued_fraction"] is None, "分母为 0 时不许给出比率数字"
    # na 不阻断（usable 全绿的批次本来就没有可救对象），但报告里必须看得见它
    assert s["passed"] is True


def test_summarize_refuses_an_unsigned_set():
    """核心：49/50 全签且全部"可用"，只要还有一帧没签，就是**不通过**。"""
    doc = _filled()
    doc["frames"][0]["verdict"] = None
    doc["frames"][0]["signed_by"] = None
    s = summarize_verdicts(doc)
    assert s["passed"] is False
    labels = [c["label"] for c in s["checks"] if not c["ok"]]
    assert any("签满" in x for x in labels), labels


def test_summarize_ignores_verdicts_without_a_signature():
    """填了判定但没有 signed_by = 不算数（防"agent 顺手填了没署名"混进结论）。"""
    doc = _filled()
    for f in doc["frames"]:
        f["signed_by"] = ""
    s = summarize_verdicts(doc)
    assert s["signed"] == 0 and s["passed"] is False


def test_summarize_fails_below_seventy_percent_usable():
    doc = _filled()
    frames = doc["frames"]
    for f in frames[:15]:              # 40 帧里 15 帧降为"微调可救" -> 62.5%
        _sign(f, "tweakable", steps=1)
    s = summarize_verdicts(doc)
    assert s["usable_fraction"] == pytest.approx(0.625)
    assert s["passed"] is False


def test_summarize_counts_unrescuable_tweaks_against_the_gate():
    doc = _filled()
    for f in doc["frames"][:10]:       # ②但救回来花了 5 步，不算"2 步内可救"
        _sign(f, "tweakable", steps=5)
    s = summarize_verdicts(doc)
    assert s["rescued_in_steps"] == 0
    assert s["passed"] is False


def test_summarize_fails_when_too_many_unusable():
    doc = _filled()
    for f in doc["frames"][:3]:        # 3/40 = 7.5% 不可用 > 5%
        _sign(f, "unusable")
    s = summarize_verdicts(doc)
    assert s["not_unusable_fraction"] == pytest.approx(0.925)
    assert s["passed"] is False


def test_summarize_requires_per_stratum_and_per_roll_coverage():
    doc = _filled(rolls=("rA", "rB"))
    for f in doc["frames"]:
        if f["roll"] == "rB":          # 整卷没签 + 夜景分层没签
            f["verdict"] = None
            f["signed_by"] = None
    s = summarize_verdicts(doc)
    detail = " ".join(c["detail"] for c in s["checks"] if not c["ok"])
    assert "rB" in detail
    assert s["passed"] is False


def test_valid_verdicts_constant_is_the_three_tiers():
    assert set(VALID_VERDICTS) == {"usable", "tweakable", "unusable"}
