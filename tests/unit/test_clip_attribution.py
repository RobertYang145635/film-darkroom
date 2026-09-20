"""回归锁：技术正片高端剪裁的**归因**（输入端已死 vs 软件新引入）。

背景（2026-09-09 实测）：全语料 21/169 帧因技术正片 B 通道高端剪裁 >1%
被 FAIL 挡住导出。但翻拍端的输入剪裁本身就有好几个 percent，
FAIL 文案里说不清"这 4% 是谁造成的"，用户只能在"软件把高光推爆了"和
"相机本来就没拍到"之间猜。归因口径是确定的物理对应：
**负片越密（t_in 越小）→ 正片越亮**，所以正片高端 ↔ 输入低端。

锁四件事：
1. 剪裁**全部继承自输入**时，FAIL 文案必须说"采集端就已丢失"，
   且 `high_introduced` 必须是 0；
2. 剪裁**由软件引入**时，文案必须点名"软件新引入"；
3. **判定本身一个字都不许多改**：两种情形都照旧 fail、照旧挡导出
   （归因只补信息，不是放宽门禁的通道）；
4. 低端**不假装能归因**：输入端的对应量是 above_base，不在本口径内，
   必须返回 None + 原因，而不是顺手报个 0 冒充算过。
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


def test_fully_inherited_clip_says_it_is_an_acquisition_problem():
    pos = _blown(2, 0.03)
    t = _dead_at(pos, 2)
    g = QualityGate.evaluate(t_in=t, positive_linear=pos, technical_preclip=_pre(0.03))
    assert g.status == "fail" and g.blocks_export          # 判定不变
    txt = " ".join(g.failures)
    assert "采集端就已丢失" in txt, "整批剪裁都来自输入端，文案却不含归因"
    assert "重扫" in txt
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


def test_attribution_never_relaxes_the_gate():
    """归因不是后门：无论继承还是引入，>1% 一律 FAIL，<1% 一律不 FAIL。"""
    for t in (np.full((N, N, 3), 0.5, np.float32),
              _dead_at(_blown(2, 0.03), 2)):
        assert QualityGate.evaluate(
            t_in=t, positive_linear=_blown(2, 0.03),
            technical_preclip=_pre(0.03)).status == "fail"
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
