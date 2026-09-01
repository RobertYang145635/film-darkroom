"""统一处理服务（整改 v2 P0-D）：GUI、CLI、批处理唯一允许的管线入口。

固定顺序（v2 §2.1）：
  load/linearize(调用方) -> rotate_cw(source) -> 全图检测 frame/base/image_area
  -> 裁切前 oriented 坐标取片基 ROI -> crop(oriented) -> 透射率域除尘(可回退)
  -> auto_analyze(prior/中性点) -> invert_chunked -> QualityGate
入口禁止自行实现旋转/裁切/自动分析/质量判断。

sidecar v2（P0-F）：所有 ROI 以 SOURCE 坐标声明；crop_oriented 单独声明
ORIENTED 坐标；coordinate_space: "preview" 不再出现。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .. import __version__
from ..calibration.annotations import Annotation
from ..calibration.border import detect_regions
from ..diagnostics.quality_gate import QualityGate
from .auto import AutoReport, auto_analyze
from .density import InvertResult, invert_chunked
from .geometry import (GeometryRecord, apply_geometry, oriented_to_source,
                       rotate_cw, roi_source_to_oriented)
from .params import InvertParams


@dataclass
class ProcessResult:
    t: np.ndarray                       # 工作图（oriented+crop+dust 后）
    geometry: GeometryRecord
    report: AutoReport
    result: InvertResult
    gate: QualityGate
    meta: dict = field(default_factory=dict)

    def sidecar(self, *, source_sha256: str, source_path: str = "",
                film_type: str = "unknown", profile_id: str = "none",
                input_meta: dict | None = None) -> dict:
        """schema v2 旁车载荷（所有入口共用这一个 builder）。"""
        im = input_meta or self.meta or {}
        return {
            "schema_version": 2,
            "source_sha256": source_sha256,
            "source": source_path,
            "source_size": list(self.geometry.source_size),
            "rotation_degrees": self.geometry.rotation_degrees,
            "geometry": {
                "crop_oriented": (list(self.geometry.crop_oriented)
                                  if self.geometry.crop_oriented else None),
                "output_size": list(self.geometry.output_size),
                "keep_rebate": self.geometry.keep_rebate,
                "roi_source_space": self.geometry.roi_source_space,
            },
            "film_type": film_type,
            "device_profile_id": profile_id,
            "input": {
                "encoding": im.get("input_encoding", "unknown"),
                "encoding_detail": im.get("input_encoding_detail", "unknown"),
                "profile_sha256": im.get("input_icc_sha256", "none"),
                "raw_decode": im.get("raw_decode_params", {}),
                "white_balance_policy": im.get("white_balance_policy", "unknown"),
            },
            "working_space": im.get("working_space", "unknown"),
            "flatfield": (self.meta or {}).get("flatfield", {"coverage": "off"}),
            "output": {"encoding": "srgb",
                       "icc_sha256": im.get("output_icc_sha256", "")},
            "parameters": self.report.params.to_parameters_dict(),
            "confidence": self.report.confidence,
            "quality": self.gate.to_dict(),
            "warnings": sorted(set(self.report.warnings + self.result.warnings
                                   + self.gate.warnings + self.gate.failures)),
            "software_version": __version__,
        }


def process_image(t_source: np.ndarray, *,
                  rotation_degrees: int = 0,
                  ann: Annotation | None = None,
                  frame_roi_source: tuple | None = None,
                  base_roi_source: tuple | None = None,
                  frame_roi_oriented: tuple | None = None,
                  base_roi_oriented: tuple | None = None,
                  keep_rebate: bool = False,
                  prior: InvertParams | None = None,
                  neutral_bright_points: list | None = None,
                  neutral_dark_points: list | None = None,
                  dust_mask: np.ndarray | None = None,
                  dust_auto: bool = False,
                  force_params: InvertParams | None = None,
                  blank: np.ndarray | None = None,
                  flat_mode: str = "strip",
                  calibration=None,          # RollCalibration（v3 P0-E）
                  points_space: str = "oriented") -> ProcessResult:
    """执行固定顺序管线。points/dust_mask 默认在 ORIENTED（旋转后、裁切前）坐标。

    base_roi_source / frame_roi_source：SOURCE 坐标的用户覆盖（优先于标注，
    标注 verified 时优先于自动检测）。
    """
    if rotation_degrees not in (0, 90, 180, 270):
        raise ValueError(f"rotation_degrees 必须 0|90|180|270，得到 {rotation_degrees}")
    sh, sw = t_source.shape[:2]

    # 1) 顺时针旋转（geometry.rotate_cw 是唯一实现）
    oriented = rotate_cw(t_source, rotation_degrees)

    # 1b) flat-field（v3 P0-C：暗角/导光的正当解法，先于一切统计）
    flat_info: dict = {"coverage": "off"}
    if flat_mode != "off":
        from ..calibration.flatfield import apply_flat, fit_base_surface
        if blank is not None:
            fit = fit_base_surface(oriented, blank=rotate_cw(blank, rotation_degrees))
        else:
            _det0 = detect_regions(oriented)
            fit = (fit_base_surface(oriented, base_roi=_det0.get("base"))
                   if _det0.get("base") else
                   {"field": np.ones_like(oriented), "coverage": "none",
                    "warnings": ["无片基条带也无 blank：flat-field 跳过"]})
        oriented = apply_flat(oriented, fit)
        flat_info = {"coverage": fit["coverage"], "warnings": fit["warnings"]}

    # 2) 全图检测（裁切前，保证片基参考可用）
    det = detect_regions(oriented)
    warnings: list[str] = list(det.get("warnings", [])) + list(flat_info.get("warnings", []))

    # 3) 片基 ROI：用户 > verified 标注 > 自动
    if base_roi_oriented is not None:
        base_roi_source = oriented_to_source(tuple(base_roi_oriented), sh, sw,
                                             rotation_degrees)
    base_arr = None
    base_src_used = None
    if base_roi_source is not None:
        b = roi_source_to_oriented(tuple(base_roi_source), sh, sw, rotation_degrees)
        base_arr = oriented[b[0]:b[2], b[1]:b[3], :]
        base_src_used = list(base_roi_source)
    elif ann is not None and ann.verified and ann.base_roi:
        b = roi_source_to_oriented(tuple(ann.base_roi), sh, sw, rotation_degrees)
        base_arr = oriented[b[0]:b[2], b[1]:b[3], :]
        base_src_used = list(ann.base_roi)
    elif det.get("base") is not None:
        y0, x0, y1, x1 = det["base"]
        base_arr = oriented[y0:y1, x0:x1, :]

    # 4) 画面裁切：用户 > verified 标注 > 自动 image_area；keep_rebate 不裁
    crop = None
    crop_src_used = None
    if not keep_rebate:
        if frame_roi_oriented is not None:
            frame_roi_source = oriented_to_source(tuple(frame_roi_oriented),
                                                  sh, sw, rotation_degrees)
        if frame_roi_source is not None:
            crop = roi_source_to_oriented(tuple(frame_roi_source), sh, sw,
                                          rotation_degrees)
            crop_src_used = list(frame_roi_source)
        elif ann is not None and ann.verified and ann.frame_roi:
            crop = roi_source_to_oriented(tuple(ann.frame_roi), sh, sw,
                                          rotation_degrees)
            crop_src_used = list(ann.frame_roi)
        elif det.get("image_area") is not None:
            crop = tuple(det["image_area"])
        if crop is None:
            warnings.append("画面区域检测失败：未裁切（保留片边），请手动指定 frame ROI")

    # 5) 应用裁切（apply_geometry 记录最终几何事实）
    t_work, geo = apply_geometry(oriented, rotation_degrees=0, crop_oriented=crop)
    geo.rotation_degrees = rotation_degrees
    geo.source_size = (sh, sw)
    geo.roi_source_space = {
        "film_base": base_src_used if base_src_used else
                     (list(det["base"]) if det.get("base") else None),
        "frame": crop_src_used if crop_src_used else
                 (list(det["image_area"]) if det.get("image_area") else None),
        "coordinate_space": "source",
    }
    cy0 = crop[0] if crop else 0
    cx0 = crop[1] if crop else 0

    # 6) 除尘（透射率域，可回退；mask 从 oriented 平移到 crop 空间）
    if dust_auto:
        # §7：自动候选保守限定在片基/边缘均匀条带（内容区误检率高，手动）
        from ..restoration.dust import detect_dust as _dd
        region = None
        if det.get("base") is not None:
            by0, bx0, by1, bx1 = det["base"]
            region = np.zeros(oriented.shape[:2], bool)
            region[by0:by1, bx0:bx1] = True
        d = _dd(oriented, region=region)
        if d["mask"].any():
            dust_mask = d["mask"] if dust_mask is None else (dust_mask | d["mask"])
        warnings.extend(d["warnings"])
    if dust_mask is not None and dust_mask.any():
        from ..restoration.dust import safe_inpaint
        m = dust_mask
        if crop:
            h0, w0 = cy0, cx0
            h1 = min(m.shape[0], cy0 + t_work.shape[0])
            w1 = min(m.shape[1], cx0 + t_work.shape[1])
            sub = np.zeros(t_work.shape[:2], bool)
            sub[:h1 - h0, :w1 - w0] = m[h0:h1, w0:w1]
            m = sub
        t_work, err = safe_inpaint(t_work, m)
        if err:
            warnings.append(err)

    # 7) 中性点（oriented -> crop 坐标）
    nb = neutral_bright_points
    nd = neutral_dark_points
    if crop:
        def _shift(pts):
            return None if not pts else [(int(y - cy0), int(x - cx0))
                                         for (y, x) in pts
                                         if cy0 <= y < cy0 + t_work.shape[0]
                                         and cx0 <= x < cx0 + t_work.shape[1]]
        if points_space == "oriented":
            nb, nd = _shift(nb), _shift(nd)

    # 8) 自动分析（片基参考来自裁切前数组）+ 用户显式覆盖（§4.2.2）
    rep = auto_analyze(t_work, base_roi=base_arr, prior=prior,
                       neutral_bright_points=nb, neutral_dark_points=nd)
    # v3 P0-E 层级：verified 校准（user-points/chart）覆盖逐帧猜测，
    # 但 Dmin 保持本帧片基测量（物理量按帧）；force_params 最终裁决。
    if calibration is not None and calibration.usable_for_color:
        if calibration.dmax:
            rep.params.dmax = float(calibration.dmax)
        if calibration.red_ratio is not None:
            rep.params.red_ratio = float(calibration.red_ratio)
        if calibration.blue_ratio is not None:
            rep.params.blue_ratio = float(calibration.blue_ratio)
        if calibration.wb_high is not None:
            rep.params.wb_high = tuple(calibration.wb_high)
        if calibration.density_matrix is not None:
            rep.params.density_matrix = tuple(calibration.density_matrix)
        rep.confidence["calibration_source"] = calibration.source
        rep.warnings.append(
            f"已应用卷级校准（{calibration.source}，roll={calibration.roll}）")
    if force_params is not None:
        rep.params = force_params
    res = invert_chunked(t_work, rep.params)

    # 10) 质量门禁（v3 P0-G：缺有效标定的彩色负片自动结果 = needs_calibration）
    calib_level = None
    if calibration is not None and calibration.usable_for_color:
        calib_level = calibration.source
    elif (nb or nd) and (nb and nd):
        calib_level = "user-neutral-points"
    elif ann is not None and ann.verified and ann.neutral_rois:
        calib_level = "verified-roi"
    elif prior is not None and getattr(prior, "_calibrated", False):
        calib_level = "roll-profile"
    calibration = {"level": calib_level,
                   "confidence": min(rep.confidence.get("dmin", 0),
                                     rep.confidence.get("dmax", 0))}
    gate = QualityGate.evaluate(t_in=t_work, positive_linear=res.positive_linear,
                                input_nonfinite=res.diagnostics.get("input_nonfinite", 0),
                                extra_warnings=warnings + rep.warnings + res.warnings,
                                calibration=calibration)
    rep.warnings = sorted(set(warnings + rep.warnings))
    return ProcessResult(t=t_work, geometry=geo, report=rep, result=res,
                         gate=gate, meta={"flatfield": {
                             "coverage": flat_info["coverage"],
                             "warnings": flat_info.get("warnings", [])}})
