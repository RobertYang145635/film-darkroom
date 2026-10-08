"""回归：**用户亲手框的片基必须决定这一帧的去色罩**，在真实底片上量。

`1299_G200` 整卷翻拍时构图中常没有可用的未曝光片边，自动检测在 11 帧里
有 9 帧直接失败（`base_source=generic_prior`，置信度 0.00），另外 2 帧"成功"
但读出的密度比实际暗 0.4 decade。用户为此在 GUI 里把每一帧的片基框都画了
（`fixtures/annotations/1299_G200__*.json`，`verified_by=user`，2026-10-09）。

**当场实测**（900px 探针分辨率，`tmp_base_truth.py` / `tmp_user_base_diag.py`）：

| 帧 | 用户框实测 dmin | 自动 dmin | Δdecade(R,G,B) |
| --- | --- | --- | --- |
| G200-2 | 0.338 / 0.129 / 0.038 | 0.123 / 0.055 / 0.015 | +0.440 / +0.368 / +0.408 |
| G200-6 | 0.360 / 0.138 / 0.040 | 0.151 / 0.068 / 0.018 | +0.377 / +0.306 / +0.344 |
| G200-7 | 0.371 / 0.142 / 0.042 | 0.149 / 0.067 / 0.018 | +0.397 / +0.326 / +0.367 |

三帧用户值互相只差 0.04 decade（同一卷同一条片边，物理上就该一致），
而自动值/回退值比它暗 **0.4 decade**——片基读暗了，橙罩就减不够，
整帧影调与颜色一起歪。这个量级不是"精度差一点"，是"这一帧的颜色没有依据"。

因此本文件锁两件事，负控制是同一条帧**去掉**用户的框：
1. 给了框 ⇒ `base_source=frame`、置信度满值、dmin 落在他框里的实测值上；
2. 去掉框 ⇒ 退到 `generic_prior` 且红通道密度差超过 0.25 decade
   （也就是说：如果有人再把用户框丢掉——像探针里发生过的那样——这条锁会红，
   而不是让"标定没生效"再被读成"这卷拍基不可测"）。
"""

from __future__ import annotations

import importlib.util
import json
import pathlib

import numpy as np
import pytest

from negative_lab.calibration.annotations import load_annotation
from negative_lab.io.images import load_linear_rgb
from negative_lab.library.models import GeometryRegions
from negative_lab.pipeline.service import process_image

ROOT = pathlib.Path(__file__).resolve().parents[2]
_kit_spec = importlib.util.spec_from_file_location(
    "bvcs_user_base_lock", ROOT / "scripts" / "build_visual_contact_sheet.py")
kit = importlib.util.module_from_spec(_kit_spec)
_kit_spec.loader.exec_module(kit)

FRAME_ID = "1299_G200::G200-2"
# tmp_user_base_diag.py 当场打印的用户框实测值（900px）
USER_DMIN = np.array([0.338, 0.129, 0.0377])
MAX_DIM = 900


def _manifest_frame(fid: str) -> dict:
    man = json.loads((ROOT / "fixtures" / "acceptance" / "manifest.json")
                     .read_text(encoding="utf-8"))
    return next(f for f in man["frames"] if f["id"] == fid)


@pytest.fixture(scope="module")
def prepared():
    fr = _manifest_frame(FRAME_ID)
    path = ROOT / fr["file"]
    if not path.exists():
        pytest.skip(f"语料缺失：{fr['file']}")
    loaded = load_linear_rgb(path)
    arr = loaded.rgb
    step = max(1, int(np.ceil(max(arr.shape[:2]) / float(MAX_DIM))))
    return fr, arr, arr[::step, ::step, :].copy()


def _run(prepared, base_rois):
    fr, arr, img = prepared
    ann = load_annotation(fr["file"])
    assert ann is not None and ann.base_sample_rois, "标注里的用户片基框不见了"
    geo = GeometryRegions(rotation_degrees=ann.rotation_degrees,
                          output_crop_roi=ann.output_crop_roi or ann.image_area_roi,
                          image_area_roi=ann.image_area_roi or ann.output_crop_roi,
                          base_sample_rois=list(base_rois))
    geo = kit._downscale_geometry(geo, arr.shape[:2], img.shape[:2])
    return process_image(img, geometry=geo, ann=None, make_candidates=False,
                         orientation_decided=True, frame_id=fr["file"])


def test_user_base_box_drives_the_frame(prepared):
    pr = _run(prepared, load_annotation(_manifest_frame(FRAME_ID)["file"]).base_sample_rois)
    assert pr.base_source == "frame", (
        f"用户框没被采纳（base_source={pr.base_source}）——片基又退回了先验")
    assert pr.base_confidence >= 0.9
    dmin = np.array(pr.report.params.dmin, np.float64)
    # 密度域容差 0.02 decade（分辨率/取整带来的量级），不是相对容差
    assert float(np.max(np.abs(np.log10(dmin) - np.log10(USER_DMIN)))) < 0.02, (
        f"dmin 与他框里的实测值不符：{dmin} vs {USER_DMIN}")


def test_negative_control_removing_the_box_moves_base_04_decade(prepared):
    """负控制：去掉用户的框，这一帧的片基就换一个数量级。

    这条同时说明上面的通过**不是**因为"自动检测本来就能给出同一个值"。
    """
    pr = _run(prepared, [])
    dmin = np.array(pr.report.params.dmin, np.float64)
    gap = float(abs(np.log10(dmin[0]) - np.log10(USER_DMIN[0])))
    assert pr.base_source != "frame" or gap > 0.25, (
        f"去掉用户框后 base_source={pr.base_source}、Δdecade(R)={gap:.3f}："
        "自动检测居然也能给出同一个片基，上面的正锁就没有区分力")
    assert gap > 0.25, f"Δdecade(R)={gap:.3f} 过小，钉不住'标定丢失'这种回归"
