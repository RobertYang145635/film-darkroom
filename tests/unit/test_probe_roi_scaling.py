"""锁：探针/套具把源图缩小时，**不得丢掉用户框的窄片基带**。

**由来（2026-10-09 实测）**：`scripts/build_visual_contact_sheet.py::
_downscale_geometry()` 对三类 ROI 统一要求"缩放后每边 ≥16px"。未曝光片边是
一条窄带——`1299_G200::G200-2` 源图（8280×5520）里是 744×126，降到 900px
探针分辨率变成 74×**12**，12<16 ⇒ 整条被静默丢掉。后果不是"探针少个框"，
而是这一帧在管线里退回 `generic_prior`（置信度 0.00），于是当天量出来的
"用户标定了但没生效 / 这一卷片基不可测"是**探针的错**。当场对照：
修前 `base_sample_rois=[]`、`base_source=generic_prior/0.00`；
修后 `dmin=[0.338,0.129,0.038]`、`base_source=frame/1.00`。

规则：**片基取样区与构图区的退化标准不同**。片基区沿用生产同一条门槛
（`service.py` 里 `clamp_roi(..., min_side=4)`），构图/画面区维持 16px。
本文件同时锁住这条不对称，防止有人把它"统一"回去。
"""

from __future__ import annotations

import importlib.util
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "build_visual_contact_sheet.py"


def _load():
    spec = importlib.util.spec_from_file_location("bvcs_roi_scale", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mod = _load()
GeometryRegions = mod.GeometryRegions           # noqa: N806

# 实测的源/探针尺寸（8280×5520 → 步长 10）
SRC = (8280, 5520)
DST = (828, 552)
# 用户为 1299_G200::G200-2 亲手框的片基带（source 空间 (y0,x0,y1,x1)）
THIN_BASE = (2286, 5208, 3030, 5334)


def _geo(base_rois=(), crop=None, area=None):
    return GeometryRegions(rotation_degrees=0, output_crop_roi=crop,
                           image_area_roi=area, base_sample_rois=list(base_rois))


def test_thin_user_base_strip_survives_probe_downscale():
    """744×126 的片基带降到 74×12 仍然要留下——它是用户给的绝对真值。"""
    out = mod._downscale_geometry(_geo(base_rois=[THIN_BASE]), SRC, DST)
    assert len(out.base_sample_rois) == 1, (
        f"窄片基带被丢掉：{out.base_sample_rois}")
    y0, x0, y1, x1 = out.base_sample_rois[0]
    assert (y1 - y0, x1 - x0) == (74, 12)
    assert x0 > 0 and y0 > 0                     # 确实是等比缩放而非取整到 0


def test_base_threshold_is_the_production_four_pixels():
    """与生产 `clamp_roi(min_side=4)` 同一条线：4px 留、3px 去。"""
    keep = _geo(base_rois=[(0, 0, 200, 40)])     # 缩放后 20×4
    drop = _geo(base_rois=[(0, 0, 200, 30)])     # 缩放后 20×3
    assert len(mod._downscale_geometry(keep, SRC, DST).base_sample_rois) == 1
    assert mod._downscale_geometry(drop, SRC, DST).base_sample_rois == []


def test_degenerate_base_roi_is_still_dropped():
    """负控制：源图就退化的区（8×8）不能因为放宽而混进来。"""
    assert mod._downscale_geometry(_geo(base_rois=[(0, 0, 8, 8)]),
                                   SRC, DST).base_sample_rois == []


def test_crop_and_image_area_keep_the_strict_16px_rule():
    """放宽只针对片基区：构图区缩到 12px 仍应视为无意义。"""
    tiny = (0, 0, 80, 120)                       # 缩放后 8×12
    out = mod._downscale_geometry(_geo(crop=tiny, area=tiny), SRC, DST)
    assert out.output_crop_roi is None
    assert out.image_area_roi is None
    big = (1000, 1000, 7000, 5000)
    out2 = mod._downscale_geometry(_geo(crop=big, area=big), SRC, DST)
    assert out2.output_crop_roi and out2.image_area_roi
