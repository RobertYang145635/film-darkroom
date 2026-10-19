r"""**平铺单帧扫描的取景规则**：没有片基证据的边一律不裁。

缺陷（2026-10-20 看图抓到，`docs/blockers.md` B-16 追加④）：ECN-2 是单帧平铺扫描
（检出画框占整幅 0.939~1.001），而 `detect_image_area` 的条带结构模型
`[帧边暗隙][片基亮带][暗印边带][画面]` 在这类扫描上会在**画面内部**找到一条亮带，
把带外全裁掉——5207-6 的 `image_area` 只剩整幅 **35.8%**，用户样片里的 LAWSON 招牌
整个被切在交付框外。与他的正片做尺度×位移互相关（`tmp_ecn2_rebate_frame.py`，8 帧两卷）：
不裁 s=1.05 |r|=0.90~0.98 八帧全中；现 `image_area` s=0.60~1.05 |r|=0.63~0.98。

规则：`frame` 占整幅 ≥ `FRAME_FLUSH_FRAC` ⇒ 只按 `detect_rebate_strips` 的**物理证据**
逐边内缩；条带扫描（G200 实测 0.760~0.764）走原路径不变。
阈值 0.85 两侧各留 ≥0.09 余量（`tmp_ecn2_frame_gate.py`）。

本文件三条都是**两臂**的：同一份合成输入，只改"帧占整幅比例"这一个变量。
"""
from __future__ import annotations

import numpy as np
import pytest

from negative_lab.calibration import border
from negative_lab.pipeline import service
from negative_lab.pipeline.geometry import OrientTransform
from negative_lab.library.models import GeometryRegions

H, W = 1000, 700


def _fake_det(frame, image_area, conf=0.9):
    return {"frame": frame, "image_area": image_area, "base": None,
            "confidence": conf, "warnings": []}


def _run(monkeypatch, frame, image_area, rebate):
    """走真实 `_auto_fill_rois`，只替换两个外部检测器。"""
    monkeypatch.setattr(service, "detect_regions",
                        lambda a, max_dim=800: _fake_det(frame, image_area))
    monkeypatch.setattr(border, "detect_rebate_strips", lambda a, **k: rebate)
    tr = OrientTransform((H, W), 0, False)
    geo, prov = GeometryRegions(), {"flags": []}
    out = service._auto_fill_rois(np.zeros((H, W, 3), np.float32), geo, tr, prov,
                                  keep_rebate=False)
    return out, prov


def test_flush_scan_cuts_only_where_base_evidence_exists(monkeypatch):
    """平铺 + 只有"上"边有片基 ⇒ 只内缩上边，其余三条边保持帧边。"""
    rebate = ([(0, 84, 12, 595)], [("上", 1.6, 0.42)])
    geo, prov = _run(monkeypatch, frame=(0, 0, H, W),
                     image_area=(120, 40, 940, 660), rebate=rebate)
    assert prov["image_area"] == "auto-rebate-bounded"
    y0, x0, y1, x1 = geo.image_area_roi
    assert (y0, x0, y1, x1) == (12, 0, H, W)          # 只有上边被证据裁掉 12px
    assert prov["image_area_inset"]["上"] == pytest.approx(0.012, abs=1e-3)
    assert prov["image_area_inset"]["左"] == 0.0


def test_strip_scan_keeps_the_structural_image_area(monkeypatch):
    """负控制：同样的"内部亮带"结果，帧只占 0.60 ⇒ 仍走条带模型，一个像素都不许回补。"""
    geo, prov = _run(monkeypatch, frame=(0, 0, int(H * 0.8), W),
                     image_area=(120, 40, 700, 660), rebate=([], []))
    assert prov["image_area"] == "auto"
    assert geo.image_area_roi == (120, 40, 700, 660)
    assert "image_area_inset" not in prov


def test_no_evidence_on_flush_scan_delivers_whole_frame(monkeypatch):
    """平铺 + 四条边都没有片基证据 ⇒ 交付整幅（这正是 5207-6 那一类）。"""
    geo, prov = _run(monkeypatch, frame=(0, 0, H, W),
                     image_area=(600, 7, 940, 680), rebate=([], []))
    assert prov["image_area"] == "auto-rebate-bounded"
    assert geo.image_area_roi == (0, 0, H, W)
    assert geo.output_crop_roi == (0, 0, H, W)        # 输出裁切跟着整幅走
    assert set(prov["image_area_inset"].values()) == {0.0}


def test_flush_gate_separates_the_two_corpora_on_real_numbers():
    """阈值的余量：ECN-2 实测 0.939~1.001，G200 实测 0.760~0.764（`tmp_ecn2_frame_gate.py`）。

    写成断言而不是注释，是因为这两个量都会随检测器改动而漂移；一旦两侧余量被吃掉，
    这条锁会先响，而不是等到某个片种的取景悄悄坏掉。
    """
    ecn2 = (0.939, 0.941, 1.001)
    g200 = (0.760, 0.762, 0.764)
    assert min(ecn2) > border.FRAME_FLUSH_FRAC
    assert max(g200) < border.FRAME_FLUSH_FRAC
    assert min(ecn2) - border.FRAME_FLUSH_FRAC >= 0.08
    assert border.FRAME_FLUSH_FRAC - max(g200) >= 0.08
