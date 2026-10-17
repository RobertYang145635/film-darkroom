"""回归：技术正片高端剪裁的**归因**（输入端已死 vs 软件新引入）**与它决定的放行**。

背景（2026-09-09 实测）：全语料 21/169 帧因技术正片 B 通道高端剪裁 >1%
被 FAIL 挡住导出。但翻拍端的输入剪裁本身就有好几个 percent，
FAIL 文案里说不清"这 4% 是谁造成的"，用户只能在"软件把高光推爆了"和
"相机本来就没拍到"之间猜。归因口径是确定的物理对应：
**负片越密（t_in 越小）→ 正片越亮**，所以正片高端 ↔ 输入低端。

**㊴（2026-10-16）这条归因拿到了放行权，是用户定的**：
`blue 0.70→0.85` 之后全语料从 0 帧被挡变成 10 帧被挡，10 帧全部
`technical_preclip.high[B] == input_clip.low[B]`（软件新引入 0.00%）。
把标红的像素图给他看（`outputs/ref_look_probe/dead_input_1298_G200_G200-7.png`，
红点全在水面最亮的反光上），他说"**这种就是完全过曝的很正常吧**"
⇒ 高端的 ">1% 挡导出"只看**软件新引入**的那一份，纯继承的降级为警告。
⚠ 阈值一个没动（`CLIP_FAIL_FRAC` 仍是 1%），改的是**判据的主语**（总量 → 引入量）；
低端没有归因口径，仍按总量判；**归因说不清就不降级**（见本文件最后一条）。

锁五件事：
1. 剪裁**全部继承自输入**时，文案必须说"采集端就已丢失"，`high_introduced` 必须是 0，
   并且**不挡导出**（B-6 决定）；但 `clip_exceeded` 仍然为真——事实不许被抹掉；
2. 剪裁**由软件引入**时，文案必须点名"软件新引入"，并且**照旧挡**；
3. 1% 这条线**两侧都还有效**：引入量 0.03 挡、0.0005 放，阈值没被挪过；
4. 低端**不假装能归因**：输入端的对应量是 above_base，不在本口径内，
   必须返回 None + 原因，而不是顺手报个 0 冒充算过；
5. 没有输入端证据、或归因与统计量口径对不上时，判定**退回按总量**（不许借归因放行）。
"""

from __future__ import annotations

import numpy as np
import pytest

from negative_lab.diagnostics.quality_gate import (CLIP_FAIL_FRAC,
                                                   QualityGate, _clip_attribution)

N = 100          # 100x100 = 10000 像素，3% 就是 300 个


def _blown(channel: int, frac: float):
    """正片：指定通道有 frac 比例的像素顶到 1.0（高端剪裁）。"""
    pos = np.full((N, N, 3), 0.5, np.float32)
    k = int(round(frac * N * N))
    flat = pos.reshape(-1, 3)
    flat[:k, channel] = 1.0
    return pos


def _dead_at(pos: np.ndarray, channel: int):
    """输入端：与正片高端**同一批**像素已经死在 0（负片阴影=正片高光）。"""
    t = np.full((N, N, 3), 0.5, np.float32)
    n = int((pos.reshape(-1, 3)[:, channel] >= 1.0).sum())
    tf = t.reshape(-1, 3)
    tf[:n, channel] = 0.0
    return t


def _pre(high_b: float):
    return {"low": [0.0, 0.0, 0.0], "high": [0.0, 0.0, high_b], "n": N * N}


def test_fully_inherited_clip_is_an_acquisition_problem_and_no_longer_blocks():
    pos = _blown(2, 0.03)
    t = _dead_at(pos, 2)
    g = QualityGate.evaluate(t_in=t, positive_linear=pos, technical_preclip=_pre(0.03))
    #: **B-6（2026-10-16 用户决定）**：全部继承自输入的高端剪裁不再挡导出。
    #: 这条锁原先钉的是"照旧 fail、照旧挡"（归因只补信息）——那是**当时的**决定，
    #: ㊴ 把结构账摊开之后他改了口径 ⇒ **换主语，不删锁**：
    #: 现在它钉的是"降级必须带着证据降级"（文案、比例、`clip_exceeded` 一个都不能少）。
    assert g.status == "warn" and not g.blocks_export, (
        f"纯继承的剪裁仍被挡（status={g.status}）：B-6 的降级没生效，"
        "或者归因口径又对不上了")
    assert g.clip_exceeded, "降级不等于抹掉事实：超限这个旗子必须还立着"
    txt = " ".join(g.warnings)
    assert "采集端就已丢失" in txt, "整批剪裁都来自输入端，文案却不含归因"
    assert "重扫" in txt and "不挡导出" in txt, (
        f"降级必须说得出为什么放（{txt}）——不然用户以为没问题")
    attr = g.clip_attribution["B"]
    assert attr["high_inherited"] == pytest.approx(0.03, abs=1e-4)
    assert attr["high_introduced"] == pytest.approx(0.0, abs=1e-6), (
        "输入端同批像素已死，就不许把责任算到软件头上")
    # 输入端剪裁本身仍然是 warning（v5 §8.2：采集事实只警告）
    assert any("输入剪裁" in w for w in g.warnings)


def test_software_introduced_clip_is_named_as_such():
    pos = _blown(2, 0.03)
    t = np.full((N, N, 3), 0.5, np.float32)      # 输入端完好
    g = QualityGate.evaluate(t_in=t, positive_linear=pos, technical_preclip=_pre(0.03))
    assert g.status == "fail"
    txt = " ".join(g.failures)
    assert "软件新引入" in txt, "输入端完好却被剪裁，必须点名是软件造成的"
    assert "本软件能改" in txt
    attr = g.clip_attribution["B"]
    assert attr["high_introduced"] == pytest.approx(0.03, abs=1e-4)
    assert attr["high_inherited"] == pytest.approx(0.0, abs=1e-6)


def test_attribution_relaxes_only_the_inherited_half_never_the_introduced():
    """**B-6 的降级只作用于"纯继承"**：软件自己推爆的高光一律 FAIL，阈值一个数没动。

    ⚠ 这条锁原先叫 `test_attribution_never_relaxes_the_gate`，钉的是"归因不是放宽
    门禁的通道"（两种情形都照旧挡）。㊴ 把 10 帧被挡的账摊开、他看图之后说
    "这种就是完全过曝的很正常吧" ⇒ 决定改了，**判据的主语从"总量"换成"引入量"**；
    阈值 `CLIP_FAIL_FRAC` 仍是 1%，被换掉的只是"1% 里算谁的那一份"。
    所以这条锁继续存在，改钉"不许借归因把软件引入的那一半也放掉"。
    """
    pos = _blown(2, 0.03)
    clean_t = np.full((N, N, 3), 0.5, np.float32)
    dead_t = _dead_at(pos, 2)
    g_intro = QualityGate.evaluate(t_in=clean_t, positive_linear=pos,
                                   technical_preclip=_pre(0.03))
    assert g_intro.status == "fail" and g_intro.blocks_export, (
        "输入端完好、3% 全是软件推出去的 ⇒ 必须照旧挡")
    g_inh = QualityGate.evaluate(t_in=dead_t, positive_linear=pos,
                                 technical_preclip=_pre(0.03))
    assert g_inh.blocks_export is False and g_inh.status == "warn", (
        "同一条 3% 全部继承自输入端 ⇒ B-6 之后应当放行并警告")
    #: 混合：一半继承一半引入，**引入量 1.5% 自己就过了线** ⇒ 仍然挡
    t_mix = np.full((N, N, 3), 0.5, np.float32)
    t_mix.reshape(-1, 3)[:int(round(0.015 * N * N)), 2] = 0.0
    g_mix = QualityGate.evaluate(t_in=t_mix, positive_linear=_blown(2, 0.03),
                                 technical_preclip=_pre(0.03))
    assert g_mix.blocks_export, (
        f"混合帧被放过了（attr={g_mix.clip_attribution['B']}）："
        "降级只能吃掉继承的那一份，不许吃掉引入的")
    assert g_mix.clip_attribution["B"]["high_introduced"] == pytest.approx(0.015, abs=1e-4)
    #: 没过线的两侧都放（阈值没挪的证据）
    for t in (clean_t, dead_t):
        assert QualityGate.evaluate(
            t_in=t, positive_linear=_blown(2, 0.0005),
            technical_preclip=_pre(0.0005)).blocks_export is False
    assert CLIP_FAIL_FRAC == 0.01


def test_low_side_refuses_to_fake_an_attribution():
    pos = _blown(0, 0.03)
    t = np.full((N, N, 3), 0.5, np.float32)
    g = QualityGate.evaluate(t_in=t, positive_linear=pos, technical_preclip=_pre(0.0))
    attr = g.clip_attribution["R"]
    assert attr["low"] is None and "above_base" in attr["low_reason"], (
        "低端剪裁的输入端对应量是 above_base，报 0 会被读成"
        '"低端没有继承问题"，那是假消息')


def test_missing_or_mismatched_input_yields_no_attribution():
    pos = _blown(2, 0.03)
    assert _clip_attribution(None, pos, None) == {}
    assert _clip_attribution(np.full((4, 4, 3), 0.5, np.float32), pos, None) == {}
    g = QualityGate.evaluate(t_in=None, positive_linear=pos, technical_preclip=_pre(0.03))
    assert g.clip_attribution == {}
    assert g.status == "fail"          # 没有输入端证据时照旧按技术门禁判
