"""共享工作流（整改 P0-C）：旋转 -> 检测（全图，含片基）-> 裁切 -> 自动分析。

关键顺序约束（真实样片教训）：片基 ROI 必须在裁切前的图上检测/提取——
先裁画面再找片基会退回外框启发式，把场景暗部当片基（Dmin 低估 ->
大量假性低端剪裁）。GUI/批处理/回归共用本函数，杜绝各入口漂移。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..calibration.annotations import Annotation
from ..calibration.border import detect_regions
from .auto import AutoReport, auto_analyze
from .geometry import GeometryRecord, apply_geometry, roi_source_to_oriented
from .params import InvertParams


@dataclass
class Prepared:
    t: np.ndarray                 # 裁切后的工作图（oriented 空间）
    geometry: GeometryRecord
    report: AutoReport
    used_annotation: bool
    base_arr: np.ndarray | None = None    # 裁切前图上提取的片基数组


def prepare_and_analyze(t_source: np.ndarray, *, rotation_degrees: int = 0,
                        ann: Annotation | None = None,
                        keep_rebate: bool = False,
                        prior: InvertParams | None = None) -> Prepared:
    # 1) 只旋转（保留片基与片边用于检测）
    oriented, _ = apply_geometry(t_source, rotation_degrees=rotation_degrees)
    # 2) 全图检测
    det = detect_regions(oriented)
    # 3) 画面裁切窗口：verified 标注优先，其次自动 image_area
    crop = None
    used_ann = False
    if not keep_rebate:
        if ann is not None and ann.verified and ann.frame_roi:
            crop = roi_source_to_oriented(tuple(ann.frame_roi),
                                          t_source.shape[0], t_source.shape[1],
                                          rotation_degrees)
            used_ann = True
        elif det.get("image_area") is not None:
            crop = tuple(det["image_area"])   # detect_regions 输出即 oriented 坐标
    t_work, geo = apply_geometry(oriented, rotation_degrees=0,
                                 crop_oriented=crop)
    # 几何记录必须反映真实总旋转（sidecar 可复现性）
    geo.rotation_degrees = rotation_degrees
    geo.source_size = (t_source.shape[0], t_source.shape[1])
    # 4) 片基 ROI：verified 标注优先；否则自动检测（都在裁切前的 oriented 图上）
    base_arr = None
    if ann is not None and ann.verified and ann.base_roi:
        by0, bx0, by1, bx1 = roi_source_to_oriented(
            tuple(ann.base_roi), t_source.shape[0], t_source.shape[1],
            rotation_degrees)
        base_arr = oriented[by0:by1, bx0:bx1, :]
    if (base_arr is None or base_arr.shape[0] * base_arr.shape[1] < 64) \
            and det.get("base") is not None:
        by0, bx0, by1, bx1 = det["base"]
        base_arr = oriented[by0:by1, bx0:bx1, :]
    if base_arr is not None and base_arr.shape[0] * base_arr.shape[1] < 64:
        base_arr = None   # 宁缺毋滥：过小区域会触发回退值伪装检测结果
    rep = auto_analyze(t_work, base_roi=base_arr, prior=prior)
    return Prepared(t=t_work, geometry=geo, report=rep, used_annotation=used_ann,
                    base_arr=base_arr)
