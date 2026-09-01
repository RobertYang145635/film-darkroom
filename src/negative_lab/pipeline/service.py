"""统一处理服务（v4 §2 唯一目标架构）：GUI、CLI、批处理唯一管线入口。

固定顺序：
  load/linearize(调用方) -> 线性 sRGB D65 工作空间 -> 几何/三 ROI/flat/dust
  -> 卷级校准选择与校验 -> 技术反相（选定 backend）-> 技术质量门禁
  -> RenderIntent（可选风格层）-> 输出诊断 -> 事务导出+sidecar v3

三 ROI 语义（v4 §2，不得混用）：
  output_crop_roi  最终构图（可有意保留齿孔/片边）
  image_area_roi   算法与质量统计区域（必须排除片边/齿孔/非画面）
  base_roi         片基/色罩估计取样
"""

from __future__ import annotations

from dataclasses import dataclass, field

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
from .render import RenderIntent, apply_render


@dataclass
class ProcessResult:
    t: np.ndarray                       # 输出裁切区工作图（反相输入）
    geometry: GeometryRecord
    report: AutoReport
    result: InvertResult                # 技术反相结果（positive_linear=技术正片）
    gate: QualityGate                   # 技术质量门禁（v4 §6.2：只读技术正片）
    rendered_positive_linear: np.ndarray | None = None
    rendered_diagnostics: dict = field(default_factory=dict)
    render_intent: RenderIntent = field(default_factory=RenderIntent)
    backend_name: str = "power-reciprocal"
    calibration_info: dict = field(default_factory=dict)
    valid_mask: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    # 兼容别名
    @property
    def technical_positive_linear(self) -> np.ndarray:
        return self.result.positive_linear

    @property
    def preview_srgb(self) -> np.ndarray:
        if self.rendered_positive_linear is not None:
            from .density import encode_srgb
            return encode_srgb(self.rendered_positive_linear)
        return self.result.preview_srgb

    def sidecar(self, *, source_sha256: str, source_path: str = "",
                film_type: str = "unknown", profile_id: str = "none",
                input_meta: dict | None = None) -> dict:
        """schema v3 旁车（v4 §2/§6.7：技术参数与风格参数分列、可追溯）。"""
        im = input_meta or self.meta or {}
        return {
            "schema_version": 3,
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
            "backend": self.backend_name,
            "calibration": self.calibration_info,
            "technical_parameters": self.report.params.to_parameters_dict(),
            "render_intent": self.render_intent.to_dict(),
            "confidence": self.report.confidence,
            "quality": self.gate.to_dict(),
            "rendered_diagnostics": self.rendered_diagnostics,
            "warnings": sorted(set(self.report.warnings + self.result.warnings
                                   + self.gate.warnings + self.gate.failures)),
            "software_version": __version__,
        }


def _roi_source(ann: Annotation | None, new_attr: str, legacy_attr: str):
    """标注 ROI 取值：新三字段优先，旧 frame_roi 兼容。"""
    if ann is None:
        return None
    v = getattr(ann, new_attr, None)
    if v:
        return tuple(v)
    return getattr(ann, legacy_attr, None)


def process_image(t_source: np.ndarray, *,
                  rotation_degrees: int = 0,
                  ann: Annotation | None = None,
                  output_crop_source: tuple | None = None,
                  image_area_source: tuple | None = None,
                  base_roi_source: tuple | None = None,
                  output_crop_oriented: tuple | None = None,
                  image_area_oriented: tuple | None = None,
                  base_roi_oriented: tuple | None = None,
                  frame_roi_source: tuple | None = None,   # 旧参数名兼容
                  frame_roi_oriented: tuple | None = None,
                  keep_rebate: bool = False,
                  prior: InvertParams | None = None,
                  neutral_bright_points: list | None = None,
                  neutral_dark_points: list | None = None,
                  dust_mask: np.ndarray | None = None,
                  dust_auto: bool = False,
                  force_params: InvertParams | None = None,
                  blank: np.ndarray | None = None,
                  flat_mode: str = "strip",
                  calibration=None,          # RollCalibration | None（v4 §5.1 强类型）
                  render_intent: RenderIntent | None = None,
                  film_type: str = "unknown",
                  points_space: str = "oriented") -> ProcessResult:
    """执行 v4 §2 固定顺序管线。ROI 参数：SOURCE 或 ORIENTED 坐标。

    兼容：frame_roi_source/frame_roi_oriented 旧参数名映射到 output_crop
    （并兼作 image_area，直到标注迁移完成）。
    """
    if rotation_degrees not in (0, 90, 180, 270):
        raise ValueError(f"rotation_degrees 必须 0|90|180|270，得到 {rotation_degrees}")
    sh, sw = t_source.shape[:2]

    # 1) 顺时针旋转
    oriented = rotate_cw(t_source, rotation_degrees)

    # 2) 三 ROI 解析（用户 > 标注 > 自动）：SOURCE -> ORIENTED 统一
    def _to_o(src_xy, o_xy, legacy):
        v = o_xy or src_xy or legacy
        if v is None:
            return None
        if o_xy is not None:
            return tuple(o_xy)
        return roi_source_to_oriented(tuple(v), sh, sw, rotation_degrees)

    crop_o = _to_o(output_crop_source, output_crop_oriented, None)
    area_o = _to_o(image_area_source, image_area_oriented, None)
    base_o = _to_o(base_roi_source, base_roi_oriented, None)
    # 旧 frame_roi 参数名兼容：同时承担 output_crop 与 image_area（v4 §2 迁移期）
    if frame_roi_oriented is not None:
        crop_o = crop_o or tuple(frame_roi_oriented)
        area_o = area_o or tuple(frame_roi_oriented)
    elif frame_roi_source is not None:
        _fo = roi_source_to_oriented(tuple(frame_roi_source), sh, sw, rotation_degrees)
        crop_o = crop_o or _fo
        area_o = area_o or _fo

    ann_crop = _roi_source(ann, "output_crop_roi", "frame_roi")
    ann_area = _roi_source(ann, "image_area_roi", "frame_roi")
    ann_base = _roi_source(ann, "base_roi", "base_roi")
    use_ann = ann is not None and ann.verified

    if crop_o is None and use_ann and ann_crop:
        crop_o = roi_source_to_oriented(tuple(ann_crop), sh, sw, rotation_degrees)
    if area_o is None and use_ann and ann_area:
        area_o = roi_source_to_oriented(tuple(ann_area), sh, sw, rotation_degrees)
    if base_o is None and use_ann and ann_base:
        base_o = roi_source_to_oriented(tuple(ann_base), sh, sw, rotation_degrees)

    manual = bool(crop_o and area_o and base_o)

    # 3) flat-field（先于一切统计；手动时用用户基区）
    flat_info: dict = {"coverage": "off"}
    if flat_mode != "off":
        from ..calibration.flatfield import apply_flat, fit_base_surface
        if blank is not None:
            fit = fit_base_surface(oriented, blank=rotate_cw(blank, rotation_degrees))
        elif base_o is not None:
            fit = fit_base_surface(oriented, base_roi=tuple(base_o))
        else:
            _det0 = detect_regions(oriented)
            fit = (fit_base_surface(oriented, base_roi=_det0.get("base"))
                   if _det0.get("base") else
                   {"field": np.ones_like(oriented), "coverage": "none",
                    "warnings": ["无片基条带也无 blank：flat-field 跳过"]})
        oriented = apply_flat(oriented, fit)
        flat_info = {"coverage": fit["coverage"], "warnings": fit["warnings"]}

    # 4) 自动检测（仅在区域不全时兜底；手动齐全则完全跳过）
    det: dict = {"warnings": []}
    if not (crop_o and area_o and base_o):
        det = detect_regions(oriented)
        if base_o is None and det.get("base"):
            base_o = tuple(det["base"])
        if area_o is None and det.get("image_area"):
            area_o = tuple(det["image_area"])
        if crop_o is None and det.get("image_area"):
            crop_o = tuple(det["image_area"])
    warnings: list[str] = list(det.get("warnings", [])) + list(flat_info.get("warnings", []))
    if manual:
        warnings.append("手动区域已指定：跳过自动检测（用户为准）")

    # 5) output crop（构图）：keep_rebate=True 保留全帧（旧兼容，明确语义）
    crop = None if keep_rebate else (tuple(crop_o) if crop_o else None)
    if keep_rebate:
        warnings.append("keep_rebate：输出为全帧（旧兼容）；"
                        "请迁移到显式 output_crop_roi（v4 §2）")
    if crop is None and not keep_rebate:
        warnings.append("画面区域不可用：未裁切（保留片边），请手动框选 output crop")

    t_out, geo = apply_geometry(oriented, rotation_degrees=0, crop_oriented=crop)
    geo.rotation_degrees = rotation_degrees
    geo.source_size = (sh, sw)
    geo.roi_source_space = {
        "output_crop_roi": oriented_to_source(crop_o, sh, sw, rotation_degrees)
        if (crop_o and not keep_rebate) else
        ([0, 0, sw, sh] if keep_rebate else None),
        "image_area_roi": oriented_to_source(area_o, sh, sw, rotation_degrees)
        if area_o else None,
        "base_roi": oriented_to_source(base_o, sh, sw, rotation_degrees)
        if base_o else None,
        "coordinate_space": "source",
    }
    cy0 = crop[0] if crop else 0
    cx0 = crop[1] if crop else 0

    # 6) 除尘（oriented 域，先于裁切；手动基区也作为保守候选区）
    if dust_auto:
        from ..restoration.dust import detect_dust as _dd
        region = None
        if base_o is not None:
            by0, bx0, by1, bx1 = base_o
            region = np.zeros(oriented.shape[:2], bool)
            region[by0:by1, bx0:bx1] = True
        d = _dd(oriented, region=region)
        if d["mask"].any():
            dust_mask = d["mask"] if dust_mask is None else (dust_mask | d["mask"])
        warnings.extend(d["warnings"])
    dust_applied = False
    if dust_mask is not None and dust_mask.any():
        from ..restoration.dust import safe_inpaint
        oriented, err = safe_inpaint(oriented, dust_mask)
        dust_applied = err is None
        if err:
            warnings.append(err)

    # 7) 分析区域（image_area）内的自动分析；输出区域（output crop）反相
    t_analysis = oriented[area_o[0]:area_o[2], area_o[1]:area_o[3], :] \
        if area_o else t_out
    base_arr = oriented[base_o[0]:base_o[2], base_o[1]:base_o[3], :] \
        if base_o else None

    nb = neutral_bright_points
    nd = neutral_dark_points
    if points_space == "oriented":
        def _shift(pts, y0, x0, h, w):
            if not pts:
                return None
            return [(int(y - y0), int(x - x0)) for (y, x) in pts
                    if y0 <= y < y0 + h and x0 <= x < x0 + w] or None
        ay0, ax0 = (area_o[0], area_o[1]) if area_o else (0, 0)
        nb = _shift(nb, ay0, ax0, t_analysis.shape[0], t_analysis.shape[1])
        nd = _shift(nd, ay0, ax0, t_analysis.shape[0], t_analysis.shape[1])

    rep = auto_analyze(t_analysis, base_roi=base_arr, prior=prior,
                       neutral_bright_points=nb, neutral_dark_points=nd)
    # v4 §5：verified 校准整体采用（原子）；Dmin 保持本帧物理测量
    calibration_info: dict = {"source": "none", "roll": None}
    if calibration is not None:
        calibration_info = {"source": calibration.source,
                            "roll": calibration.roll,
                            "verified": bool(calibration.verified),
                            "film_type": calibration.film_type}
        if calibration.usable_for_color:
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
            if calibration.backend_name:
                rep.params.backend_name = calibration.backend_name
            rep.confidence["calibration_source"] = calibration.source
            rep.warnings.append(
                f"已应用卷级校准（{calibration.source}，roll={calibration.roll}）")
    # 帧级黑位：不再自动下压 offset（backend 的解析软趾已把硬剪裁变为
    # 渐近逼近；全局压黑牺牲中间调彩度——G200-1 实测 sat_p90 0.83->0.44）。
    # 场景专属黑位调整是用户 offset 手动职责。
    if force_params is not None:
        rep.params = force_params   # 用户显式参数最终裁决（§4.2.2/v4 §6.6）

    # 8) 技术反相（v4 §4.3：注册表唯一路径）
    res = invert_chunked(t_out, rep.params)

    # 9) valid_mask：image_area ∩ output crop（最终输出坐标；v4 §7.3）
    valid_mask = None
    if area_o:
        ay0 = max(area_o[0] - cy0, 0)
        ax0 = max(area_o[1] - cx0, 0)
        ay1 = min(area_o[2] - cy0, t_out.shape[0])
        ax1 = min(area_o[3] - cx0, t_out.shape[1])
        if ay1 > ay0 and ax1 > ax0:
            valid_mask = np.zeros(t_out.shape[:2], bool)
            valid_mask[ay0:ay1, ax0:ax1] = True
        if dust_mask is not None and dust_mask.any() and dust_applied:
            pass  # 除尘区已修复，仍计入有效像素（sidecar 记录 dust_applied）
    gate_film = "b&w" if film_type in ("bw", "b&w", "黑白", "monochrome") else film_type

    # 10) 技术质量门禁（只读技术正片；v4 §6.2）
    calib_level = calibration_info.get("source") if calibration_info else None
    if calib_level in ("none", None, "base-only"):
        calib_level = None
    if calib_level in ("none", None):
        if nb and nd:
            calib_level = "user-neutral-points"
        elif ann is not None and ann.verified and getattr(ann, "neutral_refs", None):
            calib_level = "verified-roi"
        elif prior is not None and getattr(prior, "_calibrated", False):
            calib_level = "roll-profile"
    gate = QualityGate.evaluate(
        t_in=t_out, positive_linear=res.positive_linear,
        input_nonfinite=res.diagnostics.get("input_nonfinite", 0),
        extra_warnings=warnings + rep.warnings + res.warnings,
        calibration={"level": calib_level,
                     "confidence": min(rep.confidence.get("dmin", 0),
                                       rep.confidence.get("dmax", 0))},
        valid_mask=valid_mask,
        film_type=film_type,
        regions_ok=bool(crop_o and area_o and base_o))
    rep.warnings = sorted(set(warnings + rep.warnings))

    # 11) RenderIntent（v4 §6：中性恒等；渲染后剪裁单独记录，NaN 阻断导出）
    intent = render_intent or RenderIntent()
    rendered = apply_render(res.positive_linear, intent)
    r_diag: dict = {"clip": {}, "nonfinite": int((~np.isfinite(rendered)).sum())}
    from .density import channel_clip_ratios
    r_diag["clip"] = channel_clip_ratios(rendered)
    if r_diag["nonfinite"]:
        gate.failures.append(
            f"渲染后含 {r_diag['nonfinite']} 个 NaN/Inf（禁止进入导出）")
        gate.status = "fail"

    return ProcessResult(
        t=t_out, geometry=geo, report=rep, result=res, gate=gate,
        rendered_positive_linear=rendered, rendered_diagnostics=r_diag,
        render_intent=intent, backend_name=rep.params.backend_name,
        calibration_info=calibration_info, valid_mask=valid_mask,
        meta={"flatfield": {"coverage": flat_info["coverage"],
                            "warnings": flat_info.get("warnings", [])},
              "dust_applied": dust_applied})
