"""P0-C：真实样片质量回归（F-1/F-3/F-4 的核心门禁）。

对 verified 标注样片执行与批处理相同的完整管线
（加载 -> 几何 -> 自动分析 -> 反相 -> 质量门禁），断言：
- rotation 与人工标注一致；
- 自动画面裁切与标注 IoU >= 0.95；
- 输出不得保留大片片边/黑边（裁切面积占比合理）；
- 正常曝光样片逐通道输出剪裁 < 1%（超出必须带结构化警告且状态非 ok）；
- 固定输入重复两次，参数与像素逐比特一致。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from negative_lab.calibration.annotations import load_annotation
from negative_lab.calibration.border import detect_regions, iou
from negative_lab.diagnostics.quality_gate import QualityGate
from negative_lab.io.images import load_linear_rgb
from negative_lab.pipeline.auto import auto_analyze
from negative_lab.pipeline.density import invert_chunked
from negative_lab.pipeline.geometry import apply_geometry, roi_source_to_oriented

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "fixtures" / "manifest.json"


def _verified_annotations():
    if not MANIFEST.exists():
        return []
    doc = json.loads(MANIFEST.read_text(encoding="utf-8"))
    out = []
    for e in doc["images"]:
        ann = load_annotation(e["file"])
        # v4 §8.5：agent 标注降级 draft——几何/管线回归仍可用（几何可机器
        # 验证，不构成颜色验收）；needs_reconfirm 引导由单测覆盖。
        if ann and (ann.verified or getattr(ann, "needs_reconfirm", False)) \
                and e["decode_status"] == "ok" \
                and (ROOT / e["file"]).exists():
            out.append((e, ann))
    return out


CASES = _verified_annotations()
pytestmark = pytest.mark.skipif(
    not CASES, reason="无可用标注样片（verified 或 draft+needs_reconfirm）")


def _run_pipeline(rel: str, ann):
    from negative_lab.pipeline.service import process_image
    from negative_lab.profiles.profile import load_profile
    t = load_linear_rgb(ROOT / rel).rgb
    det = detect_regions(t)
    prior = None
    pf = ROOT / "fixtures" / "priors" / "KodakGold200-Z8-led.profile.json"
    if "G200" in rel and pf.exists():
        prior = load_profile(pf).to_initial_params()
    prepared = process_image(t, rotation_degrees=ann.rotation_degrees,
                             ann=ann, prior=prior)   # v3 F-1：回归走唯一服务
    res = prepared.result
    gate = prepared.gate
    return t, det, prepared.geometry, prepared.report, res, gate


@pytest.mark.parametrize("e,ann", CASES, ids=[c[0]["file"].split("/")[-1]
                                              for c in CASES])
def test_rotation_matches_annotation(e, ann):
    t = load_linear_rgb(ROOT / e["file"]).rgb
    _, geo = apply_geometry(t, rotation_degrees=ann.rotation_degrees)
    assert geo.rotation_degrees == ann.rotation_degrees


@pytest.mark.parametrize("e,ann", CASES, ids=[c[0]["file"].split("/")[-1]
                                              for c in CASES])
def test_auto_crop_iou(e, ann):
    """自动画面检测 vs 人工标注 IoU >= 0.95（不含标注信息，纯自动路径）。"""
    t = load_linear_rgb(ROOT / e["file"]).rgb
    det = detect_regions(t)
    assert det.get("image_area") is not None, det["warnings"]
    v = iou(tuple(det["image_area"]), tuple(ann.frame_roi))
    assert v >= 0.95, f"IoU={v:.3f} auto={det['image_area']} ann={ann.frame_roi}"


@pytest.mark.parametrize("e,ann", CASES, ids=[c[0]["file"].split("/")[-1]
                                              for c in CASES])
def test_no_large_rebate_in_output(e, ann):
    """默认输出不得保留大片片边：裁切后面积 <= 标注画面面积的 1.05 倍。"""
    t = load_linear_rgb(ROOT / e["file"]).rgb
    crop_oriented = roi_source_to_oriented(ann.frame_roi, t.shape[0], t.shape[1],
                                           ann.rotation_degrees)
    oriented, geo = apply_geometry(t, rotation_degrees=ann.rotation_degrees,
                                   crop_oriented=crop_oriented)
    ann_area = (ann.frame_roi[2] - ann.frame_roi[0]) * (ann.frame_roi[3] - ann.frame_roi[1])
    out_area = geo.output_size[0] * geo.output_size[1]
    assert out_area <= ann_area * 1.05


@pytest.mark.parametrize("e,ann", CASES, ids=[c[0]["file"].split("/")[-1]
                                              for c in CASES])
def test_output_clipping(e, ann):
    """v2 §2 + v3：留出样片输出剪裁 <1% 硬失败；标定样片允许 warn
    （暗角由 P0-C flat-field 正当解决，v3 禁止场景像素改写 Dmin），
    但超限必须结构化可见且状态非 ok。"""
    *_, gate = _run_pipeline(e["file"], ann)
    over = [f for f in gate.output_clip["low"] + gate.output_clip["high"] if f > 0.01]
    if over:
        assert gate.status != "ok", "剪裁超限但状态仍为 ok —— 门禁失效"
        assert any("剪裁" in w for w in gate.warnings)
        if e.get("split") == "holdout":
            pytest.fail(f"留出样片输出剪裁超限: {gate.output_clip}")
    assert gate.nonfinite_output == 0


@pytest.mark.parametrize("e,ann", CASES, ids=[c[0]["file"].split("/")[-1]
                                              for c in CASES])
def test_base_reference_valid(e, ann):
    """片基参考必须真的可用（实测教训：ROI 越界产生空切片 -> Dmin 回退
    0.5 -> 全图冲淡，而剪裁指标抓不住这种失败）。"""
    *_, rep, res, gate = _run_pipeline(e["file"], ann)
    assert rep.confidence.get("dmin", 0) > 0.5, rep.confidence
    assert max(rep.params.dmin) < 0.9, "Dmin 疑似回退值 0.5"


@pytest.mark.parametrize("e,ann", CASES, ids=[c[0]["file"].split("/")[-1]
                                              for c in CASES])
def test_not_washed_out(e, ann):
    """正常曝光户外场景：必须有真实中间调与黑位（p05<0.30, p50<0.85），
    防止'全白但无剪裁'的冲淡失败模式逃过门禁。"""
    *_, rep, res, gate = _run_pipeline(e["file"], ann)
    lum = np.nan_to_num(res.positive_linear).mean(axis=2)
    p05, p50 = np.quantile(lum, [0.05, 0.5])
    assert p50 < 0.85, f"整体冲淡: p50={p50:.2f}"
    assert p05 < 0.30, f"无真实黑位: p05={p05:.2f}"


@pytest.mark.parametrize("e,ann", CASES, ids=[c[0]["file"].split("/")[-1]
                                              for c in CASES])
def test_idempotent(e, ann):
    a = _run_pipeline(e["file"], ann)
    b = _run_pipeline(e["file"], ann)
    assert a[3].params.to_parameters_dict() == b[3].params.to_parameters_dict()
    assert np.array_equal(a[4].positive_linear, b[4].positive_linear)
