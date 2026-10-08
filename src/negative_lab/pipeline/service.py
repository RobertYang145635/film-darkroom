"""统一处理服务（v5 §0.3.4 / §13）：GUI、CLI、批处理与测试的唯一管线入口。

固定顺序（v5 §6.3 + §7.2）：

```
load/linearize(调用方) -> 线性工作空间
  -> orient(旋转 -> 镜像 -> 细微旋转)
  -> 三类独立 ROI（用户 > 标注 > 自动检测）
  -> 片基场（多 ROI 低阶可观测拟合）
  -> 帧级片基统计 -> 卷级片基解析（本帧/卷级/通用先验）
  -> 正密度 -> 受约束 3x3 + 单调曲线 + 技术端点 -> TechnicalPositive
  -> 技术质量门禁（只读技术正片）
  -> 三个自动候选（preserve/neutral/warm，共享同一技术底座）
  -> RenderIntent（色温/色调/影调/色彩）-> 渲染正片 -> sidecar v4
```

三类 ROI 语义（§3.2，不得混用）：
`output_crop_roi` 最终构图 / `image_area_roi` 算法与统计 /
`base_sample_rois[]` 未曝光片基取样（可多个）。

v5 §8.1：本服务**不接受**任何"缺色卡即失败"的逻辑；缺色卡、未知胶片型号、
没有 verified 标定都只影响来源说明与置信度。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..calibration.annotations import Annotation
from ..calibration.border import detect_regions
from ..diagnostics.quality_gate import QualityGate
from ..library.fingerprint import CaptureFingerprint, capture_fingerprint
from ..library.models import (CandidateSet, FrameRecord, GeometryRegions,
                              RollAnalysis, RollGroup)
from .auto import AutoReport, auto_analyze
from .candidates import build_candidates
from .density import InvertResult, invert_chunked, preclip_ratios
from .geometry import (GeometryRecord, OrientTransform, apply_geometry,
                       clamp_roi)
from .model_assets import deterministic_asset
from .params import InvertParams
from .print_scale import (LOOK_CHROMA_GAIN, anchor_refusal_reason, centre_region,
                          density_anchor, density_anchor_linear, picture_density,
                          print_curve)
from .render import RenderIntent, apply_render
from .roll_analysis import analyze_roll, frame_base_stats, resolve_base


@dataclass
class ProcessResult:
    """一次帧级处理的完整结果与诊断。"""

    t: np.ndarray                       # 输出裁切区工作图（反相输入）
    geometry: GeometryRecord
    report: AutoReport
    result: InvertResult                # 技术反相（positive_linear=技术正片）
    gate: QualityGate
    candidates: CandidateSet | None = None
    chosen_candidate: str = "preserve"
    rendered_positive_linear: np.ndarray | None = None
    rendered_diagnostics: dict = field(default_factory=dict)
    render_intent: RenderIntent = field(default_factory=RenderIntent)
    backend_name: str = "power-reciprocal"
    valid_mask: np.ndarray | None = None
    base_source: str = "frame"
    base_confidence: float = 0.0
    frame_base: dict = field(default_factory=dict)
    roll_analysis: RollAnalysis | None = None
    capture_fingerprint: str = ""
    roi_provenance: dict = field(default_factory=dict)
    model_asset: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    @property
    def technical_positive_linear(self) -> np.ndarray:
        return self.result.positive_linear

    @property
    def preview_srgb(self) -> np.ndarray:
        from .density import encode_srgb
        if self.rendered_positive_linear is not None:
            return encode_srgb(self.rendered_positive_linear)
        return self.result.preview_srgb

    @property
    def status(self) -> str:
        """v5 §3.3 的用户可见状态（不含任何缺色卡阻断）。"""
        if self.gate.failures:
            return "processing_error"
        if self.gate.review_reasons:
            return "needs_review"
        if "needs_crop_review" in self.roi_provenance.get("flags", []):
            return "needs_crop_review"
        return "converted"

    def sidecar(self, **kw) -> dict:
        from ..io.sidecar import build_sidecar_v4
        return build_sidecar_v4(self, **kw)


# ------------------------------------------------------------------ ROI 解析
def _ann_rois(ann: Annotation | None) -> dict:
    """标注 -> {output_crop, image_area, base_rois}（SOURCE 空间）。"""
    if ann is None:
        return {}
    base = [tuple(r) for r in (getattr(ann, "base_sample_rois", None) or [])]
    if not base and getattr(ann, "base_roi", None):
        base = [tuple(ann.base_roi)]
    return {"output_crop": tuple(ann.output_crop_roi) if ann.output_crop_roi else None,
            "image_area": tuple(ann.image_area_roi) if ann.image_area_roi else None,
            "base_rois": base or None}


def _resolve_rois(oriented_shape, *, geometry: GeometryRegions | None,
                  tr: OrientTransform, ann: Annotation | None,
                  output_crop_oriented, image_area_oriented, base_oriented,
                  keep_rebate: bool) -> tuple[GeometryRegions, dict]:
    """三类 ROI：用户显式 > 标注 > 自动检测（v5 §5.3）。"""
    oh, ow = oriented_shape
    prov: dict = {"flags": [], "auto": False}
    src = geometry or GeometryRegions()

    def from_oriented(o):
        return tr.bbox_oriented_to_source(tuple(o)) if o else None

    crop_s = src.output_crop_roi
    area_s = src.image_area_roi
    base_s = list(src.base_sample_rois)
    if crop_s is None and output_crop_oriented:
        crop_s = from_oriented(output_crop_oriented)
        prov["output_crop"] = "user"
    if area_s is None and image_area_oriented:
        area_s = from_oriented(image_area_oriented)
        prov["image_area"] = "user"
    if not base_s and base_oriented:
        base_s = [from_oriented(base_oriented)]
        prov["base"] = "user"
    a = _ann_rois(ann)
    if crop_s is None and a.get("output_crop"):
        crop_s = a["output_crop"]
        prov["output_crop"] = "annotation" + (
            "-verified" if ann.verified else "-draft")
    if area_s is None and a.get("image_area"):
        area_s = a["image_area"]
        prov["image_area"] = "annotation" + ("-verified" if ann.verified
                                             else "-draft")
    if not base_s and a.get("base_rois"):
        base_s = list(a["base_rois"])
        prov["base"] = "annotation" + ("-verified" if ann.verified else "-draft")

    # 缺失项由 _auto_fill_rois 用像素检测补齐（v5 §5.3）
    if crop_s is None or area_s is None or not base_s:
        prov["flags"].append("needs_detection")
    geo = GeometryRegions(rotation_degrees=tr.rotation_degrees,
                          fine_rotation_degrees=tr.fine_rotation_degrees,
                          mirror_horizontal=tr.mirror_horizontal,
                          output_crop_roi=crop_s, image_area_roi=area_s,
                          base_sample_rois=base_s)
    return geo, prov


def _auto_fill_rois(oriented: np.ndarray, geo: GeometryRegions,
                    tr: OrientTransform, prov: dict,
                    *, keep_rebate: bool,
                    crop_preference: str = "image_only") -> GeometryRegions:
    """自动检测缺失的 ROI，并给出每个 ROI 的独立置信度（v5 §5.3）。"""
    if geo.output_crop_roi and geo.image_area_roi and geo.base_sample_rois:
        prov["auto_confidence"] = None
        prov["flags"].append("manual_regions")
        prov["note"] = "三类区域均由用户/标注给出：跳过自动检测（用户为准）"
        return geo
    det = detect_regions(oriented)
    conf = float(det.get("confidence") or 0.0)
    prov["auto_confidence"] = round(conf, 3)
    prov["auto_warnings"] = list(det.get("warnings", []))
    src_h, src_w = tr.source_size

    def to_src(o_roi):
        return tr.bbox_oriented_to_source(tuple(o_roi))

    if geo.image_area_roi is None and det.get("image_area"):
        geo.image_area_roi = to_src(det["image_area"])
        prov["image_area"] = "auto"
    if geo.base_sample_rois == [] and det.get("base"):
        from ..calibration.border import refine_base_roi
        raw = tuple(det["base"])
        refined = refine_base_roi(oriented, raw)
        geo.base_sample_rois = [to_src(refined or raw)]
        prov["base"] = "auto-refined" if refined else "auto"
        if refined:
            prov["base_refined_from"] = to_src(raw)
    if geo.output_crop_roi is None:
        if keep_rebate:
            geo.output_crop_roi = (0, 0, src_h, src_w)
            prov["output_crop"] = "keep_rebate"
        elif det.get("image_area") and geo.image_area_roi:
            geo.output_crop_roi = geo.image_area_roi
            prov["output_crop"] = "auto-image-area"
        elif geo.image_area_roi:
            geo.output_crop_roi = geo.image_area_roi
            prov["output_crop"] = "fallback-image-area"
        elif det.get("frame"):
            geo.output_crop_roi = to_src(det["frame"])
            prov["output_crop"] = "auto-frame"
    if conf < 0.5 or det.get("image_area") is None:
        prov["flags"].append("needs_crop_review")
    return geo


# ------------------------------------------------------------------ 单帧处理
def process_image(t_source: np.ndarray, *,
                  geometry: GeometryRegions | None = None,
                  rotation_degrees: int = 0,
                  mirror_horizontal: bool = False,
                  fine_rotation_degrees: float = 0.0,
                  ann: Annotation | None = None,
                  output_crop_source: tuple | None = None,
                  image_area_source: tuple | None = None,
                  base_sample_source: list | None = None,
                  base_roi_source: tuple | None = None,      # 旧单值兼容
                  output_crop_oriented: tuple | None = None,
                  image_area_oriented: tuple | None = None,
                  base_roi_oriented: tuple | None = None,
                  frame_roi_source: tuple | None = None,     # 已废弃别名
                  frame_roi_oriented: tuple | None = None,
                  keep_rebate: bool = False,
                  orientation_decided: bool | None = None,
                  prior: InvertParams | None = None,
                  force_params: InvertParams | None = None,
                  roll_analysis: RollAnalysis | None = None,
                  neutral_bright_points: list | None = None,
                  neutral_dark_points: list | None = None,
                  blank: np.ndarray | None = None,
                  flat_mode: str = "strip",
                  render_intent: RenderIntent | None = None,
                  candidate: str | None = None,
                  make_candidates: bool = True,
                  frame_id: str | None = None,
                  capture_fingerprint: CaptureFingerprint | str | None = None,
                  model_asset=None,
                  dust_mask: np.ndarray | None = None,
                  dust_auto: bool = False,
                  points_space: str = "oriented") -> ProcessResult:
    """执行 v5 固定顺序管线。ROI 参数可给 SOURCE 或 ORIENTED 坐标。

    2026-09-08 死参数清理（交接 §4-2，AST 全量核实）：`film_type` 与
    `calibration` 两形参在函数体内从未被读取（34 形参中仅有的两个死参），
    全仓 43 处调用点、11 处传参全部为无效果的仪式性实参，已随形参一并移除。
    「未知胶片型号不阻断导出」（v5 §6.3）本就是门禁的既有行为，
    film_type 宣称若将来需要，应走 sidecar 字段而非管线形参。
    """
    if rotation_degrees not in (0, 90, 180, 270):
        raise ValueError(f"rotation_degrees 必须 0|90|180|270，得到 {rotation_degrees}")
    t_source = np.asarray(t_source)
    sh, sw = t_source.shape[:2]

    # ---- 1) 兼容旧参数名：frame_roi -> output_crop + image_area
    legacy_crop = frame_roi_source or output_crop_source
    legacy_area = frame_roi_source or image_area_source
    legacy_base = list(base_sample_source or [])
    if not legacy_base and base_roi_source:
        legacy_base = [tuple(base_roi_source)]
    geo_in = geometry or GeometryRegions(
        rotation_degrees=rotation_degrees,
        fine_rotation_degrees=fine_rotation_degrees,
        mirror_horizontal=mirror_horizontal,
        output_crop_roi=legacy_crop, image_area_roi=legacy_area,
        base_sample_rois=legacy_base)
    geo_in = GeometryRegions.from_dict(geo_in.to_dict())      # 深拷贝
    tr = OrientTransform((sh, sw), geo_in.rotation_degrees or rotation_degrees,
                         geo_in.mirror_horizontal, geo_in.fine_rotation_degrees)

    # ---- 2) orient
    oriented = tr.apply(t_source)
    oh, ow = oriented.shape[:2]

    # ---- 3) 三类 ROI
    geo, prov = _resolve_rois(
        (oh, ow), geometry=geo_in, tr=tr, ann=ann,
        output_crop_oriented=output_crop_oriented,
        image_area_oriented=image_area_oriented,
        base_oriented=base_roi_oriented, keep_rebate=keep_rebate)
    if frame_roi_oriented is not None and prov.get("output_crop") is None:
        fr = tr.bbox_oriented_to_source(tuple(frame_roi_oriented))
        geo.output_crop_roi = geo.output_crop_roi or fr
        geo.image_area_roi = geo.image_area_roi or fr
        prov["output_crop"] = prov["output_crop"] or "user-oriented"
    geo = _auto_fill_rois(oriented, geo, tr, prov, keep_rebate=keep_rebate)

    # ---- 3b) 方向**是否有人决定过**（B-12）。`rotation_degrees=0` 有两种含义：
    # "看过、确认就是 0°"和"从没决定过的默认值"，像素完全一样，只有这个字段能分开。
    # 与用户手工参考对照实测：决定过的帧内容相关领先 0.35~0.46，
    # 默认帧肉眼可见地差 90°/180°。软件不猜方向，但必须说出"没决定过"。
    prov["orientation"] = ("unstated" if orientation_decided is None
                           else "decided" if orientation_decided else "unconfirmed")
    orientation_reason = ""
    if orientation_decided is False:
        extra = []
        if tr.mirror_horizontal:
            extra.append("水平镜像")
        if abs(float(tr.fine_rotation_degrees)) > 1e-9:
            extra.append(f"微调 {float(tr.fine_rotation_degrees):+.1f}°")
        tail = ("、" + "、".join(extra)) if extra else ""
        orientation_reason = (
            f"画面方向未经确认（当前按 {int(tr.rotation_degrees)}°{tail} 输出）："
            "请在裁切页确认方向，未确认前这一帧可能是横放或倒置的")

    # ORIENTED 空间 ROI（钳制到画布）
    crop_o = tr.bbox_source_to_oriented(geo.output_crop_roi) if geo.output_crop_roi else None
    area_o = tr.bbox_source_to_oriented(geo.image_area_roi) if geo.image_area_roi else None
    base_o = [tr.bbox_source_to_oriented(r) for r in geo.base_sample_rois]
    crop_o = clamp_roi(crop_o, (oh, ow)) if crop_o else None
    area_o = clamp_roi(area_o, (oh, ow)) if area_o else None
    base_o = [r for r in (clamp_roi(x, (oh, ow), min_side=4) for x in base_o) if r]

    warnings: list[str] = []
    if "manual_regions" in prov.get("flags", []):
        warnings.append("三类区域已按用户/标注确认，自动检测已跳过")
    for w in prov.get("auto_warnings", []):
        warnings.append(w)

    # ---- 4) 片基场（只在实际可观测方向拟合）
    flat_info: dict = {"coverage": "off"}
    if flat_mode != "off":
        from ..calibration.flatfield import apply_flat, fit_base_surface
        if blank is not None:
            fit = fit_base_surface(oriented, blank=tr.apply(blank))
        elif base_o:
            fit = fit_base_surface(oriented, base_rois=base_o)
        else:
            fit = {"field": np.ones_like(oriented), "coverage": "none",
                   "warnings": ["没有可用片基取样区：flat-field 跳过"],
                   "observed": {"spans": [0.0, 0.0], "n_regions": 0}}
        oriented = apply_flat(oriented, fit)
        flat_info = {"coverage": fit["coverage"], "warnings": fit["warnings"],
                     "observed": fit.get("observed", {})}
        warnings.extend(fit["warnings"])

    # ---- 5) 帧级片基统计 + 卷级解析（v5 §6.2）
    fbase = frame_base_stats(oriented, base_o, image_area_roi=area_o)
    warnings.extend(fbase["warnings"])
    resolved = resolve_base(fbase, roll_analysis, t_oriented=oriented,
                            image_area_roi=area_o, frame_id=frame_id)
    warnings.extend(resolved["warnings"])
    base_source = resolved["base_source"]
    base_conf = resolved["confidence"]

    # ---- 6) 输出裁切
    crop = None if keep_rebate else crop_o
    if crop is None and not keep_rebate:
        warnings.append("输出裁切区不可用：本次输出保留整幅（请在裁切页确认）")
    t_out, geo_rec = apply_geometry(
        oriented, rotation_degrees=0, crop_oriented=crop,
        roi_source_space=None)
    geo_rec.rotation_degrees = tr.rotation_degrees
    geo_rec.mirror_horizontal = tr.mirror_horizontal
    geo_rec.fine_rotation_degrees = tr.fine_rotation_degrees
    geo_rec.source_size = (sh, sw)
    geo_rec.keep_rebate = keep_rebate
    geo_rec.roi_source_space = {
        "output_crop_roi": list(geo.output_crop_roi) if geo.output_crop_roi else None,
        "image_area_roi": list(geo.image_area_roi) if geo.image_area_roi else None,
        "base_sample_rois": [list(r) for r in geo.base_sample_rois],
        "rotation_degrees": tr.rotation_degrees,
        "mirror_horizontal": tr.mirror_horizontal,
        "fine_rotation_degrees": tr.fine_rotation_degrees,
        "coordinate_space": "source",
        "provenance": {k: v for k, v in prov.items()
                       if k in ("output_crop", "image_area", "base",
                                "auto_confidence")},
    }
    cy0 = crop[0] if crop else 0
    cx0 = crop[1] if crop else 0

    # ---- 7) 有效画面 mask（image_area ∩ output crop，输出坐标）
    valid_mask = None
    if area_o:
        ay0, ax0 = max(area_o[0] - cy0, 0), max(area_o[1] - cx0, 0)
        ay1 = min(area_o[2] - cy0, t_out.shape[0])
        ax1 = min(area_o[3] - cx0, t_out.shape[1])
        if ay1 > ay0 and ax1 > ax0:
            valid_mask = np.zeros(t_out.shape[:2], bool)
            valid_mask[ay0:ay1, ax0:ax1] = True

    # ---- 8) 技术端点分析（在有效画面内）+ 技术反相
    t_analysis = oriented[area_o[0]:area_o[2], area_o[1]:area_o[3], :] \
        if area_o else t_out
    nb, nd = neutral_bright_points, neutral_dark_points
    if points_space == "oriented" and area_o:
        ay0, ax0 = area_o[0], area_o[1]

        def _shift(pts):
            if not pts:
                return None
            out = [(int(y - ay0), int(x - ax0)) for (y, x) in pts
                   if ay0 <= y < area_o[2] and ax0 <= x < area_o[3]]
            return out or None
        nb, nd = _shift(nb), _shift(nd)

    rep = auto_analyze(t_analysis, base_roi=None, film_roi=t_analysis,
                       neutral_bright_points=nb, neutral_dark_points=nd,
                       prior=prior, resolved_dmin=resolved["dmin"],
                       base_source=base_source)
    # A density matrix trained only by channel decorrelation is not a colour
    # calibration.  It can change correlation while making neutral objects
    # purple/blue, so it must never be silently enabled in the ordinary path.
    # An explicitly supplied asset remains available for controlled experiments
    # and future colour-validated models.
    asset = (model_asset if model_asset is not None else
             deterministic_asset(reason="默认关闭：通用矩阵尚未通过颜色真值验证"))
    asset_info = asset.apply_to(rep.params)
    if force_params is not None:
        rep.params = force_params            # 用户显式技术参数最终裁决
    res = invert_chunked(t_out, rep.params, mask=valid_mask)
    # 自动打印尺度（B-11）**不在此处施加于技术正片**：
    # `test_rebate_independence` 比的是技术层门禁统计，而"印相"是美学层动作；
    # 接在技术层既违反分层，也会让 2.29% 的高光削顶把整帧挡掉导出。
    # 正确落点是 `RenderIntent`（随候选存储、预览与导出同源），
    # 需要先定 sidecar schema 的取舍 —— 见 docs/blockers.md B-11 追加⑥。
    print_scale_info = None
    preclip = preclip_ratios(res.diagnostics)

    # ---- 9) 三候选（共享同一技术底座）
    fp_value = ""
    if capture_fingerprint is not None:
        fp_value = (capture_fingerprint.value
                    if isinstance(capture_fingerprint, CaptureFingerprint)
                    else str(capture_fingerprint))
    cset: CandidateSet | None = None
    chosen = candidate or "preserve"
    if make_candidates:
        cset = build_candidates(_CandidateInput(
            result=res, geometry=geo_rec, valid_mask=valid_mask, t=t_out,
            roll_analysis=roll_analysis, base_source=base_source))
        if candidate is None:
            chosen = cset.default_name
        cand = cset.get(chosen)
        intent = RenderIntent.from_dict(cand.intent)
        if render_intent is not None:
            # 用户微调叠加在候选起点之上（v5 §7.3：同步只复制选择与 intent）
            intent = render_intent.copy()
    else:
        intent = (render_intent or RenderIntent()).copy()
    if render_intent is not None:
        intent = render_intent.copy()

    # ---- 9b) 自动打印尺度（B-11）：把这一帧的分位（取在 `picture_density` 域，
    # 即重映射自己会读到的那个域）作为"印相"决定，
    # **写进每个候选的 intent**（可见、进 sidecar、可复算），由 `apply_render`
    # 施加。技术正片与门禁口径一字不动（v5 §8.2），
    # "恒等 intent == 技术正片"的 v4 契约也仍然成立（无锚点即恒等）。
    # 用户显式 `force_params` / `render_intent` 是最终裁决，此时不施加。
    print_scale_info = None
    print_refused = ""
    if force_params is None and render_intent is None:
        # 取样区 = 画面区包围盒的中心（边角照度 falloff / 灰尘不该决定黑点），
        # 施加区仍是整个画面区（片边不动）。
        region = centre_region(valid_mask,
                              shape=res.positive_linear.shape[:2])
        samples = picture_density(res.positive_linear, rep.params.dmax, region)
        anchor = density_anchor(samples)
        linear_fallback = ""
        if anchor is None:
            print_refused = anchor_refusal_reason(samples)
        else:
            try:
                print_curve(anchor)
            except ValueError as exc:
                #: 三点曲线解不出（趾部挤成一团）不等于"这帧不该印"。
                #: 退成两点直线仍能压掉雾；实测 `1298_G200::G200-10`
                #: 不退时该帧 `p05=0.292`（用户参考 0.036），是标定后最差的一帧。
                two = density_anchor_linear(samples)
                if two is None:
                    print_refused = str(exc)
                else:
                    linear_fallback = str(exc)
                    anchor = two
        if anchor is not None:
            a = [round(float(v), 5) for v in anchor]
            #: 片基没实测的帧**不放大色彩**：`LOOK_CHROMA_GAIN` 是对着
            #: "已标定"的帧拟合出来的口味系数，乘到一片不知道错在哪的颜色上
            #: 只会把错误放大（实测 1299 两帧饱和 p95 0.537→0.910、0.341→0.769）。
            gain_off = (1.0 if base_source == "generic_prior" else 0.0)
            for cand in (cset if cset is not None else []):
                merged = dict(cand.intent or {})
                merged["print_anchor"] = list(a)
                merged["print_dmax"] = float(rep.params.dmax)
                merged["print_chroma_gain"] = gain_off
                cand.intent = merged
            intent = RenderIntent.from_dict(
                {**intent.to_dict(), "print_anchor": a,
                 "print_dmax": float(rep.params.dmax),
                 "print_chroma_gain": gain_off})
            print_scale_info = (
                {"d_lo": a[0], "d_hi": a[1], "points": 2}
                if len(a) == 2 else
                {"d_lo": a[0], "d_mid": a[1], "d_hi": a[2], "points": 3})
            print_scale_info.update({
                "dmax_old": float(rep.params.dmax),
                "chroma_gain": (gain_off or LOOK_CHROMA_GAIN),
                "chroma_gain_reason": (
                    "片基未实测（generic_prior），色彩不做放大"
                    if gain_off else "片基已实测")})
            if linear_fallback:
                print_scale_info["two_point_fallback"] = linear_fallback
        else:
            print_scale_info = {"refused": print_refused}

    # ---- 10) 渲染
    rendered = apply_render(res.positive_linear, intent)

    # ---- 11) 技术质量门禁（只读技术正片；v5 §8.2）
    scene_flags = (cset.diagnostics.get("scene", {}) if cset else {})
    cand_conf = float(cset.get(chosen).confidence) if cset else 1.0
    crop_conf = float(prov.get("auto_confidence") or 1.0)
    if isinstance(capture_fingerprint, CaptureFingerprint):
        fp_conf = capture_fingerprint.completeness
    else:
        fp_conf = 1.0 if fp_value else 0.0
    gate = QualityGate.evaluate(
        t_in=t_out, positive_linear=res.positive_linear,
        technical_preclip=preclip,
        rendered_positive=rendered,
        input_nonfinite=res.diagnostics.get("input_nonfinite", 0),
        extra_warnings=warnings + rep.warnings + res.warnings,
        valid_mask=valid_mask,
        base_source=base_source, base_confidence=base_conf,
        candidate_confidence=cand_conf, crop_confidence=crop_conf,
        fingerprint_confidence=fp_conf,
        scene_flags={"night": bool(scene_flags.get("dark_frac", 0) > 0.5),
                     "monochrome_light":
                         bool(scene_flags.get("hue_concentration", 0) > 0.7),
                     "high_saturation":
                         bool(scene_flags.get("sat_frac", 0) > 0.6)},
        extra_review_reasons=[r for r in (
            orientation_reason,
            (f"自动打印尺度未施加（{print_refused}）：本帧按未印相的技术正片输出，"
             "需要人工定 Grade") if print_refused else "") if r])
    gate.base_diagnostics = {
        "base_source": base_source, "confidence": base_conf,
        "provenance": resolved.get("provenance", {}),
        "regions_used": int(fbase.get("n_regions_used") or 0),
        "regions_rejected": list(fbase.get("rejected") or []),
        "contamination": fbase.get("contamination"),
        "spatial_trend": fbase.get("spatial_trend"),
        "black_level": fbase.get("black_level"),
        "flatfield": flat_info,
    }
    rep.warnings = sorted(set(warnings + rep.warnings))

    return ProcessResult(
        t=t_out, geometry=geo_rec, report=rep, result=res, gate=gate,
        candidates=cset, chosen_candidate=chosen,
        rendered_positive_linear=rendered,
        rendered_diagnostics={"preclip": preclip,
                              "nonfinite": int((~np.isfinite(rendered)).sum())},
        render_intent=intent, backend_name=rep.params.backend_name,
        valid_mask=valid_mask, base_source=base_source,
        base_confidence=base_conf, frame_base=fbase,
        roll_analysis=roll_analysis, capture_fingerprint=fp_value,
        roi_provenance=prov,
        model_asset=asset_info,
        meta={"flatfield": flat_info,
              "print_scale": print_scale_info or {},
              "input_facts": {},
              "dust_deferred": bool(dust_mask is not None or dust_auto)})


class _CandidateInput:
    """build_candidates 的最小输入视图（避免复制大数组）。"""

    def __init__(self, *, result, geometry, valid_mask, t, roll_analysis,
                 base_source):
        self.result = result
        self.geometry = geometry
        self.valid_mask = valid_mask
        self.t = t
        self.roll_analysis = roll_analysis
        self.base_source = base_source


# ------------------------------------------------------------------ 候选出口
def candidate_positive(pr: ProcessResult, name: str,
                       render_intent: RenderIntent | None = None) -> np.ndarray:
    """按候选名渲染正片——与导出走同一 service 出口（v5 §12 P0-5.5）。"""
    if pr.candidates is None:
        raise ValueError("该 ProcessResult 未生成候选集（make_candidates=False）")
    cand = pr.candidates.get(name)
    intent = render_intent if render_intent is not None \
        else RenderIntent.from_dict(cand.intent)
    return apply_render(pr.result.positive_linear, intent)


# ------------------------------------------------------------------ 整卷处理
def process_roll(frames: list[tuple[FrameRecord, np.ndarray]], *,
                 roll: RollGroup | None = None,
                 prior: InvertParams | None = None,
                 render_intent: RenderIntent | None = None,
                 candidate: str | None = None,
                 flat_mode: str = "strip",
                 make_candidates: bool = True,
                 capture_fingerprints: dict | None = None,
                 model_asset=None,
                 on_frame=None) -> tuple[RollAnalysis, list[ProcessResult]]:
    """整卷两遍处理（v5 §6.2 / §1.3）。

    第一遍：逐帧走同一 `process_image`（自动检测三类 ROI + 帧级片基），
    不生成候选；第二遍：汇总卷级片基模型后重跑，让本帧可靠片基与卷级
    估计按 §6.2.3 加权结合、无片基帧自动继承卷级模型。

    两遍成本约 2×（解码只做一次，由调用方缓存图像）；这是"先有卷统计
    再有帧结果"的算法必然顺序，v5 §6.2.2-3 明确要求。

    `model_asset`：密度域解混资产（v5 §6.3）。此前只有单帧入口能接收，
    而生产主路径是整卷 ⇒ 资产在整卷路径上**根本不可达**，`apply_to` 与
    `fixtures/models/` 里已训练的矩阵全是死路。两遍都必须转发，否则第一遍
    与第二遍用不同的反相 backend，卷级片基与帧结果对不上。
    """
    first: list[ProcessResult] = []
    fps: list[str] = []
    for rec, img in frames:
        fp = (capture_fingerprints or {}).get(rec.source_path) \
            or capture_fingerprint_from_record(rec, img)
        fps.append(fp.value if fp else "")
        pr = process_image(img, geometry=rec.geometry, ann=None, prior=prior,
                           roll_analysis=None, render_intent=None,
                           make_candidates=False,
                           flat_mode=flat_mode,
                           orientation_decided=rec.orientation_decided,
                           model_asset=model_asset,
                           capture_fingerprint=fp)
        pr.meta["input_facts"] = dict(rec.input_facts or {})
        first.append(pr)
    main_fp = ""
    vals = [f for f in fps if f]
    if vals:
        main_fp = max(set(vals), key=vals.count)
    stats = [{
        "frame_id": rec.source_path,
        "capture_fingerprint": fp,
        "base_dmin": pr.frame_base.get("dmin"),
        "base_confidence": pr.frame_base.get("confidence", 0.0),
        "contamination": pr.frame_base.get("contamination"),
        "black_level": pr.frame_base.get("black_level"),
        "spatial_trend": pr.frame_base.get("spatial_trend"),
        # 通用先验回退帧的共同曝光尺度（§6.2.5：全卷共用）
        "generic_scale": (sum(pr.report.params.dmin)
                          if pr.base_source == "generic_prior" else None),
    } for (rec, _img), pr, fp in zip(frames, first, fps)]
    ra = analyze_roll(stats, fingerprint=main_fp)
    if roll is not None:
        roll.roll_analysis = ra
        roll.capture_fingerprint = main_fp
    results: list[ProcessResult] = []
    for (rec, img), pr0, fp in zip(frames, first, fps):
        same_fp = (not main_fp or fp == main_fp)
        # 同指纹的所有帧都拿到卷分析：参与卷模型的帧用卷级 dmin，
        # 未参与的帧只继承本卷共同曝光尺度（§6.2.5）
        shared = ra if same_fp and (ra.shared_with_frames or ra.generic_scale) else None
        rec.geometry = GeometryRegions.from_dict(pr0.geometry.roi_source_space)
        pr = process_image(img, geometry=rec.geometry, ann=None, prior=prior,
                           roll_analysis=shared,
                           render_intent=render_intent, candidate=candidate,
                           make_candidates=make_candidates,
                           flat_mode=flat_mode,
                           orientation_decided=rec.orientation_decided,
                           frame_id=rec.source_path,
                           model_asset=model_asset,
                           capture_fingerprint=fp)
        pr.meta["input_facts"] = dict(rec.input_facts or {})
        rec.frame_analysis = {
            "base_dmin": pr.frame_base.get("dmin"),
            "base_confidence": pr.frame_base.get("confidence"),
            "contamination": pr.frame_base.get("contamination"),
            "black_level": pr.frame_base.get("black_level"),
            "spatial_trend": pr.frame_base.get("spatial_trend"),
            "regions": [list(r) for r in rec.geometry.base_sample_rois]}
        rec.chosen_candidate = pr.chosen_candidate
        results.append(pr)
        if on_frame:
            on_frame(rec, pr)
    return ra, results


def capture_fingerprint_from_record(rec: FrameRecord, img: np.ndarray):
    """没有加载 meta 时从输入事实推导指纹（可能不完整，诚实降级）。"""
    facts = dict(rec.input_facts or {})
    if not facts:
        return None
    return capture_fingerprint(facts, source_path=rec.source_path)
