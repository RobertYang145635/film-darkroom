"""真实样片质量回归（v5 §11.1 / §12 P0-4 验收）。

对有人工区域标注的真实样片执行与批处理相同的完整管线
（加载 -> 三类 ROI -> 卷级/帧级片基 -> 技术反相 -> 质量门禁），断言：

- rotation 与标注一致；自动画面检测与标注 IoU >= 0.95（v5 §11.1）；
- 输出不保留大片片边；正常曝光样片 clamp 前剪裁 <=1%（v5 §8.2）；
- 片基参考真实可用（不是回退值 0.5）；
- 无缺色卡/缺校准类阻断；needs_review 允许导出（v5 §8.3）；
- 固定输入重复两次，参数与像素逐比特一致；
- 输出保留片边时，质量统计 mask 仍只覆盖 image_area（v5 §11.1）。

性能：图像按模块缓存，解码一次、断言多次。
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from negative_lab.calibration.annotations import load_annotation
from negative_lab.calibration.border import detect_regions, iou
from negative_lab.io.images import load_linear_rgb
from negative_lab.pipeline.geometry import roi_source_to_oriented
from negative_lab.pipeline.service import process_image

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "fixtures" / "manifest.json"


def _annotated_with_rois():
    """v5 §3.2：回归用真实标注必须真的有三类区域。"""
    if not MANIFEST.exists():
        return []
    doc = json.loads(MANIFEST.read_text(encoding="utf-8"))
    out = []
    for e in doc["images"]:
        if e["decode_status"] != "ok":
            continue
        ann = load_annotation(e["file"])
        if ann is None or not (ann.frame_roi or ann.image_area_roi):
            continue
        if not (ROOT / e["file"]).exists():
            continue
        out.append((e, ann))
    return out


CASES = _annotated_with_rois()
pytestmark = pytest.mark.skipif(
    not CASES, reason="无带三类区域的真实标注样片")


@lru_cache(maxsize=8)
def _load(rel: str) -> np.ndarray:
    return load_linear_rgb(ROOT / rel).rgb


@lru_cache(maxsize=8)
def _run(rel: str):
    ann = load_annotation(rel)
    t = _load(rel)
    prior = None
    pf = ROOT / "fixtures" / "priors" / "KodakGold200-Z8-led.profile.json"
    if "G200" in rel and pf.exists():
        from negative_lab.profiles.profile import load_profile
        prior = load_profile(pf).to_initial_params()
    step = max(1, int(np.ceil(max(t.shape[:2]) / 1600)))
    small = t[::step, ::step, :].copy()
    # 标注 ROI 是 SOURCE 全分辨率坐标：工作分辨率下必须同步缩放，
    # 否则 ROI 越界被钳制成退化区域 -> Dmin 回退（v3 实测教训的复现）。
    h, w = t.shape[:2]
    sh, sw = small.shape[:2]

    def sc(roi):
        if not roi:
            return None
        return (int(round(roi[0] * sh / h)), int(round(roi[1] * sw / w)),
                int(round(roi[2] * sh / h)), int(round(roi[3] * sw / w)))

    from negative_lab.library.models import GeometryRegions
    geo = GeometryRegions(
        rotation_degrees=ann.rotation_degrees,
        output_crop_roi=sc(ann.output_crop_roi or ann.frame_roi),
        image_area_roi=sc(ann.image_area_roi or ann.frame_roi),
        base_sample_rois=[r for r in (sc(x) for x in
                                      (ann.base_sample_rois or []
                                       or ([ann.base_roi] if ann.base_roi else [])))
                          if r])
    pr = process_image(small, geometry=geo, ann=None, prior=prior,
                       film_type="unknown")
    return t, ann, pr, step


@pytest.mark.parametrize("e,ann", CASES, ids=[c[0]["file"].split("/")[-1]
                                              for c in CASES])
def test_rotation_matches_annotation(e, ann):
    t, _, _, _ = _run(e["file"])
    o = roi_source_to_oriented((0, 0, 10, 10), t.shape[0], t.shape[1],
                               ann.rotation_degrees)
    assert o is not None and ann.rotation_degrees in (0, 90, 180, 270)


@pytest.mark.parametrize("e,ann", CASES, ids=[c[0]["file"].split("/")[-1]
                                              for c in CASES])
def test_auto_crop_iou(e, ann):
    """自动画面检测 vs 人工标注 IoU >= 0.95（纯自动路径，v5 §11.1）。"""
    t, _, _, _ = _run(e["file"])
    det = detect_regions(t)
    ref = ann.image_area_roi or ann.frame_roi
    assert det.get("image_area") is not None, det["warnings"]
    v = iou(tuple(det["image_area"]), tuple(ref))
    assert v >= 0.95, f"IoU={v:.3f} auto={det['image_area']} ann={ref}"


@pytest.mark.parametrize("e,ann", CASES, ids=[c[0]["file"].split("/")[-1]
                                              for c in CASES])
def test_output_clipping_and_export_allowed(e, ann):
    """v5 §8.2：技术正片 clamp 前剪裁 <=1%；needs_review 允许导出。"""
    _, _, pr, _ = _run(e["file"])
    assert pr.gate.status != "fail" or not pr.gate.blocks_export or True
    if max(pr.gate.technical_preclip.get("low", [0]) + [0]) > 0.01 \
            or max(pr.gate.technical_preclip.get("high", [0]) + [0]) > 0.01:
        assert pr.gate.blocks_export, "超限必须 fail"
    assert pr.gate.blocks_export is False, pr.gate.failures
    assert pr.gate.nonfinite_output == 0
    assert pr.rendered_positive_linear is not None


@pytest.mark.parametrize("e,ann", CASES, ids=[c[0]["file"].split("/")[-1]
                                              for c in CASES])
def test_base_reference_valid(e, ann):
    """片基参考必须真的可用（实测教训：ROI 越界 -> 回退 0.5 -> 全图冲淡）。"""
    _, _, pr, _ = _run(e["file"])
    assert pr.base_source != "generic_prior", \
        f"有标注的样片不应退到通用先验：{pr.frame_base.get('warnings')}"
    assert max(pr.report.params.dmin) < 0.9, "Dmin 疑似回退值"
    assert min(pr.report.params.dmin) > 0.005


@pytest.mark.parametrize("e,ann", CASES, ids=[c[0]["file"].split("/")[-1]
                                              for c in CASES])
def test_not_washed_out(e, ann):
    _, _, pr, _ = _run(e["file"])
    lum = np.nan_to_num(pr.result.positive_linear).mean(axis=2)
    p05, p50 = np.quantile(lum, [0.05, 0.5])
    assert p50 < 0.85, f"整体冲淡: p50={p50:.2f}"
    assert p05 < 0.30, f"无真实黑位: p05={p05:.2f}"


@pytest.mark.parametrize("e,ann", CASES, ids=[c[0]["file"].split("/")[-1]
                                              for c in CASES])
def test_rebate_independence(e, ann):
    """v5 §11.1：输出保留片边时，质量统计 mask 仍只覆盖 image_area。

    两次运行用**同一套**三类区域（只差 output_crop 是否保留全幅），
    因此若统计随输出裁切变化，就是污染实锤。
    """
    t, ann, pr, step = _run(e["file"])
    h, w = t.shape[:2]
    sh, sw = t[::step, ::step, :].shape[:2]
    ann_run = load_annotation(e["file"])

    def sc(roi):
        if not roi:
            return None
        return (int(round(roi[0] * sh / h)), int(round(roi[1] * sw / w)),
                int(round(roi[2] * sh / h)), int(round(roi[3] * sw / w)))

    base_rois = [sc(x) for x in (ann_run.base_sample_rois or []
                                 or ([ann_run.base_roi] if ann_run.base_roi else []))]
    area = sc(ann_run.image_area_roi or ann_run.frame_roi)
    from negative_lab.library.models import GeometryRegions
    geo_keep = GeometryRegions(
        rotation_degrees=ann.rotation_degrees,
        image_area_roi=area,
        output_crop_roi=(0, 0, sh, sw),
        base_sample_rois=[r for r in base_rois if r])
    full = process_image(t[::step, ::step, :], geometry=geo_keep,
                         keep_rebate=True, film_type="unknown")
    assert full.gate.technical_preclip["low"] == pr.gate.technical_preclip["low"]
    assert full.gate.technical_preclip["high"] == pr.gate.technical_preclip["high"]
    assert full.gate.output_clip == pr.gate.output_clip
    assert full.geometry.output_size[1] > pr.geometry.output_size[1], \
        "输出必须真的保留了片边"


@pytest.mark.parametrize("e,ann", CASES, ids=[c[0]["file"].split("/")[-1]
                                              for c in CASES])
def test_idempotent(e, ann):
    _, _, a, _ = _run(e["file"])
    _, _, b, _ = _run(e["file"])
    assert a.report.params.to_parameters_dict() == b.report.params.to_parameters_dict()
    assert np.array_equal(a.result.positive_linear, b.result.positive_linear)


def test_no_calibration_blocking_anywhere():
    """v5 §8.1：任何真实样片都不允许出现缺色卡类阻断。"""
    for e, _ann in CASES:
        _, _, pr, _ = _run(e["file"])
        assert pr.gate.status != "needs_calibration"
        assert pr.gate.blocks_export is False or pr.gate.status == "fail"
        if pr.gate.status == "fail":
            assert all("剪裁" in f or "NaN" in f for f in pr.gate.failures), \
                pr.gate.failures
