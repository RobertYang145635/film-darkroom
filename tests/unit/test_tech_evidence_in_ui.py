"""回归锁：归因必须真的**贴到右栏**，不能只停在数据结构里。

上一轮给门禁加了 `clip_attribution`、给片基估计加了 `best_rejected_region`，
但"用户看得见"要靠 UI 把它贴出来；只加字段不加出口的修复，
在用户那儿等于没修（本仓库已有先例：死代码审计里"只有 emit 没有 connect"）。
这里锁两件事：
1. `TechSection` 有 `evidence` 出口，且没证据时**不许**凭空多一行"证据："；
2. `MainWindow._update_tech_info` 真的把两条归因传下去
   （用 duck-typed stub 直接调未绑定方法，不拉起整个窗口）。
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("PySide6")

from negative_lab.diagnostics.explain import (  # noqa: E402
    base_fallback_note, clip_note)
from negative_lab.diagnostics.quality_gate import QualityGate  # noqa: E402
from negative_lab.ui import main_window as mw  # noqa: E402

_REJECTED = {
    "roi": (0, 0, 10, 10), "confidence": 0.298, "contamination": 0.2496,
    "kept_frac": 0.7504, "rel_spread": 0.0761, "mad": [0.0179, 0.0482, 0.0761],
    "scores": {"size": 1.0, "purity": 0.917, "spread": 0.298},
    "binding_score": "spread", "median_transmittance": [0.244, 0.093, 0.026],
}


def _gate_with_inherited_clip() -> QualityGate:
    pos = np.full((40, 40, 3), 0.5, np.float32)
    t = np.full((40, 40, 3), 0.5, np.float32)
    pos.reshape(-1, 3)[:480, 2] = 1.0
    t.reshape(-1, 3)[:480, 2] = 0.0
    return QualityGate.evaluate(
        t_in=t, positive_linear=pos,
        technical_preclip={"low": [0, 0, 0], "high": [0, 0, 0.30], "n": 1600})


def test_tech_section_renders_evidence_and_stays_quiet_without_it():
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])
    from negative_lab.ui.right_panel import TechSection

    sec = TechSection()
    evidence = "\n".join((base_fallback_note({"best_rejected_region": _REJECTED,
                                              "regions": [{}]}, "generic_prior"),
                          clip_note(_gate_with_inherited_clip())))
    sec.set_info(base_source="generic_prior", gate="fail", review=[], warnings=[],
                 evidence=evidence)
    txt = sec.lbl_body.text()
    assert txt.startswith("片基来源：generic_prior")
    assert "证据" in txt and "颗粒/背光散布" in txt and "采集端就已丢失" in txt

    sec.set_info(base_source="frame", gate="ok", review=[], warnings=[])
    assert "证据" not in sec.lbl_body.text(), "没有证据也不许摆一行空标题糊弄人"


def test_update_tech_info_actually_passes_the_attribution():
    """接线锁：`_update_tech_info` 必须把两条归因传进 set_tech_info。"""
    captured = {}

    class _Right:
        def set_tech_info(self, **kw):
            captured.update(kw)

    stub = SimpleNamespace(
        current=object(),
        pr=SimpleNamespace(
            base_source="generic_prior", frame_base={"best_rejected_region": _REJECTED,
                                                     "regions": [{}]},
            gate=_gate_with_inherited_clip(),
            report=SimpleNamespace(warnings=[])),
        right=_Right(),
        _consistency_text=lambda: "")
    mw.MainWindow._update_tech_info(stub)          # 未绑定方法 + duck-typed self

    assert captured, "根本没调用 set_tech_info，右栏永远看不到归因"
    assert "颗粒/背光散布" in captured["evidence"]
    assert "采集端就已丢失" in captured["evidence"]
    assert captured["base_source"] == "generic_prior"
