"""锁：自动打印曝光（B-11 落地的第一级——中灰放到相纸 aim）。

**为什么这些锁该存在**（全部是 2026-10-07 实测出来的）：
- 自动出片整体偏亮：画面阴影抬到显示域 p05=0.71，用户手工成品是 0.35；
- 后端公式 `paper=(1+black)−10^q` **不能**同时满足"黑点锚 p5"和"中间调印到 aim"，
  两式联立把曲线钉死：要么暗部整段压碎（p05→0.03、饱和虚高 0.95），
  要么高光只到 0.27 ⇒ 所以这一级必须落在**美学层**，技术正片不动；
- `AIM=0.18` 与"用几何平均"都不是拟合量，EV 是闭式解，
  跨卷跨工艺实测掉得回去（holdout Δ+0.013、CN400 Δ+0.005、ECN2 Δ−0.020/+0.005）。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from negative_lab.library.models import Candidate, CandidateSet
from negative_lab.library.render import RenderIntent, apply_render
from negative_lab.pipeline.print_exposure import (EV_LIMIT, PRINT_AIM,
                                                  aim_exposure_ev,
                                                  apply_print_exposure)


def _flat(v: float, shape=(8, 8)) -> np.ndarray:
    return np.full((*shape, 3), v, np.float32)


def _gm(lin: np.ndarray) -> float:
    x = np.clip(np.asarray(lin, np.float64)[..., :3], 1e-9, None)
    lum = 0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]
    return float(np.exp(np.mean(np.log(lum))))


# ---------------------------------------------------------------- 闭式正确性
def test_ev_places_geometric_mean_on_aim():
    """EV 施加后，几何平均亮度必须**落在 aim 上**（这是本模块唯一的承诺）。"""
    for v in (0.05, 0.2, 0.5, 0.85):
        lin = _flat(v)
        ev = aim_exposure_ev(lin)
        out = apply_render(lin, RenderIntent(exposure=ev))
        assert abs(_gm(out) - PRINT_AIM) < 1e-3, (
            f"输入 {v}：EV={ev:.3f} 之后几何平均={_gm(out):.4f}，没落到 aim")


def test_direction_bright_input_darkens_and_dark_input_lightens():
    """**负控制**：符号写反时这两条必然同时红——
    亮图要压暗、暗图要提亮；搞反就是"把雾推得更亮"，正是原缺陷方向。"""
    assert aim_exposure_ev(_flat(0.85)) < 0.0, "亮图应当压暗"
    assert aim_exposure_ev(_flat(0.02)) > 0.0, "暗图应当提亮"
    near = aim_exposure_ev(_flat(PRINT_AIM))
    assert abs(near) < 1e-6, f"已经在 aim 上却给了 {near:.4f} EV"


def test_uses_geometric_not_arithmetic_mean():
    """少数极亮像素不该把整帧推亮：算术平均会，几何平均不会。

    这是"为什么用几何平均"的可执行理由——原缺陷就是高光把中间调顶上去。
    """
    base = np.full((64, 64, 3), 0.05, np.float32)
    hot = base.copy()
    hot[:2, :2, :] = 1.0                     # 4 个高光像素（占 0.1%）
    ev_plain = aim_exposure_ev(base)
    ev_with_hot = aim_exposure_ev(hot)
    assert ev_with_hot == pytest.approx(ev_plain, abs=0.05), (
        f"0.1% 高光把曝光挪了 {ev_with_hot - ev_plain:+.3f} EV——"
        "说明用的是算术平均，高调物体会带歪整卷")


# ---------------------------------------------------------------- 边界与降级
@pytest.mark.parametrize("bad", [
    np.zeros((0, 0, 3), np.float32),          # 空图
    np.full((4, 4, 3), np.nan, np.float32),   # 全 NaN
    np.zeros((4, 4, 3), np.float32),          # 全黑
    np.zeros((4, 4), np.float32),             # 形状不对
])
def test_degenerate_inputs_return_zero_shift(bad):
    """退化输入必须**不动曝光**而不是抛错或产出 NaN——
    预览与导出共用这条路径，NaN 会一路传到 sidecar。"""
    ev = aim_exposure_ev(bad)
    assert ev == 0.0 and math.isfinite(ev), f"退化输入给出了 EV={ev}"


def test_ev_is_clamped_to_the_control_range():
    """EV 不能越过 `RenderIntent.exposure` 的合法范围（±5 档）。"""
    ev = aim_exposure_ev(_flat(1.0))
    assert -EV_LIMIT <= ev <= EV_LIMIT
    extreme = aim_exposure_ev(np.full((8, 8, 3), 1e-4, np.float32))
    assert extreme == pytest.approx(EV_LIMIT, abs=1e-9), (
        f"极暗图应被夹到 +{EV_LIMIT}，实得 {extreme}")


# ---------------------------------------------------------------- 候选层不变式
def _cset() -> CandidateSet:
    return CandidateSet(
        preserve=Candidate(name="preserve", intent=RenderIntent().to_dict()),
        neutral=Candidate(name="neutral",
                          intent=RenderIntent(temperature=12.0).to_dict()),
        warm=Candidate(name="warm",
                       intent=RenderIntent(temperature=34.0).to_dict()),
        default_name="neutral")


def test_all_three_candidates_get_the_same_exposure():
    """三候选必须加**同一个** EV：v5 §6.5 要求候选只在色温/色调与
    受限影调起点上不同，若曝光各加各的，"保留氛围/中性/偏暖"就不再可比。"""
    cs = _cset()
    ev = apply_print_exposure(cs, _flat(0.6))
    assert ev != 0.0
    got = [c.intent["exposure"] for c in cs]
    assert got[0] == pytest.approx(ev) and got[1] == pytest.approx(ev) \
        and got[2] == pytest.approx(ev), f"三候选曝光不一致：{got}"


def test_candidate_colour_fields_untouched():
    """打印曝光**不得**顺手改色温/色调——那是候选之间唯一的区别所在。"""
    cs = _cset()
    before = {c.name: (c.intent.get("temperature"), c.intent.get("tint"))
              for c in cs}
    apply_print_exposure(cs, _flat(0.6))
    after = {c.name: (c.intent.get("temperature"), c.intent.get("tint"))
             for c in cs}
    assert before == after, f"色温/色调被打印曝光带跑了：{before} -> {after}"


def test_input_array_is_not_mutated():
    """技术正片必须逐比特不变：本模块只写 intent，不碰像素。
    动了数组就等于改了技术层，"预览=导出"与门禁口径全部失效。"""
    lin = _flat(0.6)
    snapshot = lin.copy()
    cs = _cset()
    apply_print_exposure(cs, lin)
    assert np.array_equal(lin, snapshot), "输入数组被就地改了"


def test_none_candidates_is_a_no_op():
    assert apply_print_exposure(None, _flat(0.6)) == 0.0


def test_existing_candidate_intent_is_composed_not_overwritten():
    """候选已有的曝光起点必须**叠加**而不是被覆盖：
    覆盖会静默丢掉候选自己的影调决定。"""
    cs = CandidateSet(
        preserve=Candidate(name="preserve",
                           intent=RenderIntent(exposure=0.5).to_dict()),
        default_name="preserve")
    ev = apply_print_exposure(cs, _flat(0.6))
    assert cs.preserve.intent["exposure"] == pytest.approx(0.5 + ev), (
        f"起点 0.5 EV 被覆盖成了 {cs.preserve.intent['exposure']}")


def test_rationale_records_what_the_software_did():
    """v5 §8.3：自动施加的事必须写清楚采用了什么，
    否则用户在界面上看到"曝光 -1.4"却不知道是谁加的。"""
    cs = _cset()
    apply_print_exposure(cs, _flat(0.6))
    for c in cs:
        assert "自动打印曝光" in (c.rationale or ""), f"{c.name} 没记录采用说明"
