"""语料锁：画面内片基检测在**真帧**上的两头——坏帧被救，用户画的框不许被推翻。

对应实测（`tmp_orange_base_estimator.py` / `tmp_latd_roll_aim.py`，2026-10-11）：
- `1298_G200::G200-7`：片边取样被自己判污染（contamination 0.25）而整卷又是
  `roll_partial`，于是退回通用先验；先验的红通道电平比真片基暗 0.34~0.68 decade，
  后果是**该帧 55% 的像素透射率高于 assumed base**（负片物理上不可能），
  画面平均对数密度 APD(R−G) = **−1.034**，而同卷其余十帧全在 +0.147~+0.342。
  用户点名的"水是一片死青"就是这个数。
- `1299_G200::G200-7`：用户亲手画了片基框（conf 1.00）。**他的框是绝对真值**，
  新机制只允许在"没有框、也没有可用片边"时介入。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from negative_lab.calibration.annotations import load_annotation  # noqa: E402
from negative_lab.calibration.base import base_from_orange_signature  # noqa: E402
from negative_lab.io.images import load_linear_rgb  # noqa: E402
from negative_lab.library.fingerprint import capture_fingerprint  # noqa: E402
from negative_lab.library.models import FrameRecord, GeometryRegions  # noqa: E402
from negative_lab.pipeline.service import process_roll  # noqa: E402
from build_visual_contact_sheet import _downscale_geometry  # noqa: E402

MAN = json.loads((ROOT / "fixtures/acceptance/manifest.json").read_text(encoding="utf-8"))


def _record(fr: dict, max_dim: int = 1200):
    loaded = load_linear_rgb(ROOT / fr["file"])
    arr = loaded.rgb
    step = max(1, int(np.ceil(max(arr.shape[:2]) / float(max_dim))))
    img = arr[::step, ::step, :].copy()
    ann = load_annotation(fr["file"])
    if ann is not None and (ann.image_area_roi or ann.base_sample_rois):
        geo = GeometryRegions(
            rotation_degrees=ann.rotation_degrees,
            output_crop_roi=ann.output_crop_roi or ann.image_area_roi,
            image_area_roi=ann.image_area_roi or ann.output_crop_roi,
            base_sample_rois=list(ann.base_sample_rois or []))
        # **必须和生产/探针同一条缩放规则**：ROI 是源图坐标，直接拿去切缩小图
        # 会整块落到画面外，于是"用户手框"在测试里悄悄消失，测出来的就不是产品。
        geo = _downscale_geometry(geo, arr.shape[:2], img.shape[:2])
        area = geo.image_area_roi or (0, 0, img.shape[0], img.shape[1])
    else:
        geo = GeometryRegions(rotation_degrees=0)
        area = (0, 0, img.shape[0], img.shape[1])
    rec = FrameRecord(source_path=fr["file"], source_sha256=fr["source_sha256"],
                      input_facts=dict(loaded.meta),
                      capture_fingerprint=capture_fingerprint(
                          loaded.meta, source_path=fr["file"]).value,
                      geometry=geo, orientation_decided=ann is not None)
    return rec, img, area, geo


def _frame(fr_id: str) -> dict:
    hits = [f for f in MAN["frames"] if f["id"] == fr_id and f.get("required")]
    if not hits:
        pytest.skip(f"manifest 里没有 {fr_id}")
    if not (ROOT / hits[0]["file"]).exists():
        pytest.skip(f"源片不在仓库内：{hits[0]['file']}（外观真值语料不入库）")
    return hits[0]


def test_没有片边的坏帧不再吃通用先验():
    """**正锁**：`1298::G200-7` 必须从 `generic_prior` 变成画面内实测。"""
    fr = _frame("1298_G200::G200-7")
    rec, img, area, _geo = _record(fr)
    _, results = process_roll([(rec, img)])
    pr = results[0]
    assert pr.base_source == "content_orange", (
        f"这帧应当由画面内橙色亮区实测片基，实际 {pr.base_source}")
    dmin = np.asarray(pr.report.params.dmin, np.float64)
    # 先验给的是 R≈0.216；真片基必须更亮（否则 55% 像素比片基还亮这个矛盾不会消失）
    assert dmin[0] > 0.26, f"红通道片基应被抬到 0.26 以上，实得 {dmin.tolist()}"


def test_抬片基后高于片基的像素比例必须塌下去():
    """**对症量**：修复的直接后果是"透射率高于片基"这一物理矛盾消失。

    取样必须用 `pr.valid_mask`（生产实际看到的那块画面区）。用整幅源图会被
    黑衬底稀释，量出来的数就不是产品读到的数——这类"探针窗口 ≠ 生产窗口"
    的坑本轮已经踩过两次。
    """
    fr = _frame("1298_G200::G200-7")
    rec, img, area, _geo = _record(fr)
    _, results = process_roll([(rec, img)])
    pr = results[0]
    inner = pr.t[..., :3].reshape(-1, 3).astype(np.float64)
    m = pr.valid_mask
    if m is not None and m.shape == pr.t.shape[:2]:
        inner = pr.t[..., :3][m].reshape(-1, 3).astype(np.float64)
    prior = np.array([0.21637, 0.09737, 0.02596])       # 本轮实测的通用先验取值
    adopted = np.asarray(pr.report.params.dmin, np.float64)
    before = float((inner[:, 0] > prior[0]).mean())
    after = float((inner[:, 0] > adopted[0]).mean())
    assert before > 0.30, f"这条锁的前提是先验下红通道大面积越界，实得 {before:.3f}"
    assert after < 0.10, f"实测片基后红通道越界比例应塌到 10% 以下，实得 {after:.3f}"


def test_用户亲手画的片基框永远优先():
    """**纪律锁**：新机制不许推翻真值。这帧有手框，来源必须仍是实测取样。"""
    fr = _frame("1299_G200::G200-7")
    ann = load_annotation(fr["file"])
    assert ann is not None and ann.verified and ann.base_sample_rois, "前提：这帧有手框"
    rec, img, area, geo = _record(fr)
    assert geo.base_sample_rois, "前提：手框缩放后仍在图内"
    _, results = process_roll([(rec, img)])
    pr = results[0]
    assert pr.base_source in ("frame", "frame+roll"), (
        f"有用户手框时不许改用画面内估计：{pr.base_source}")
    assert pr.base_confidence > 0.9


def test_同一张真帧上手框与画面内估计的差在留出误差内():
    """把"0.02~0.03 decade"这条**留出证据**钉成锁。

    用 `1299` 里手框的帧：把框**摘掉**再跑画面内检测，估计值必须落在
    框所量到的片基附近。容差取 R/G 0.08 decade（留出实测中位 0.021/0.029，
    最差一帧 0.16 ⇒ 这里取"中位的三倍"而不是"最差值"，因为它量的是
    这条机制是否还成立，不是某一次运气）。
    """
    fr = _frame("1299_G200::G200-7")
    rec, img, area, geo = _record(fr)
    y0, x0, y1, x1 = geo.base_sample_rois[0]
    assert (y1 - y0) * (x1 - x0) >= 64, f"缩放后的框太小，量不出真值：{geo.base_sample_rois[0]}"
    truth = np.median(img[y0:y1, x0:x1, :3].reshape(-1, 3).astype(np.float64), axis=0)
    est = base_from_orange_signature(img, image_area_roi=area)
    assert est["dmin"] is not None, est["warnings"]
    err = np.log10(np.asarray(est["dmin"]) / truth)
    assert abs(err[0]) < 0.08 and abs(err[1]) < 0.08, (
        f"R/G 片基估计偏差应 <0.08 decade（留出中位 0.02~0.03），实得 {np.round(err, 3)}")
    assert abs(err[2]) < 0.25, f"B 通道允许更松（SNR 最低），实得 {np.round(err, 3)}"
