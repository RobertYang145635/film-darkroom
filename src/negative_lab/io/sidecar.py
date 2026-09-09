"""Sidecar schema v4（v5 §3.4）与 v3 只读兼容迁移（§12 P0-1.2）。

v4 必须记录（缺一不可，全部来自实际执行事实，不写猜测值）：

- 源文件路径、SHA-256、尺寸、输入类型；
- 实际 RAW/TIFF/DNG 解码参数；
- ICC / TRC / 工作空间事实；
- 自动采集指纹及其参与字段；
- 所属卷与卷分析版本；
- 三类 ROI 与全部几何变换（SOURCE 坐标 + 实际输出变换）；
- 片基估计来源（本帧 / 本卷其他帧 / 通用先验回退）；
- 技术反相模型、参数与模型资产哈希；
- 三个候选的参数、置信度与默认选择原因；
- 用户选定候选与微调参数；
- 技术正片与渲染正片的独立诊断；
- 软件版本、Git commit、时间与警告。
"""

from __future__ import annotations

from .. import storage

import subprocess
import time
from pathlib import Path

SCHEMA_VERSION = 4
LEGACY_READABLE = (1, 2, 3)

_ROOT = storage.repo_path()
_commit_cache: str | None = None


def git_commit() -> str:
    global _commit_cache
    if _commit_cache is not None:
        return _commit_cache
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_ROOT,
                             capture_output=True, text=True, timeout=5)
        _commit_cache = out.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001 —— 无 git 环境不得阻断导出
        _commit_cache = "unknown"
    return _commit_cache


def build_sidecar_v4(pr, *, source_sha256: str, source_path: str = "",
                     film_type: str = "unknown", roll_id: str | None = None,
                     input_meta: dict | None = None,
                     fingerprint_fields: dict | None = None,
                     batch_assumption: dict | None = None,
                     model_asset: dict | None = None) -> dict:
    """从 ProcessResult 生成 v4 sidecar（唯一 builder）。"""
    from .. import __version__
    im = dict(input_meta or {})
    im.update(pr.meta.get("input_facts") or {})
    geo = pr.geometry
    cset = pr.candidates
    chosen = cset.get(pr.chosen_candidate) if cset else None
    base_diag = pr.gate.base_diagnostics or {}
    return {
        "schema_version": SCHEMA_VERSION,
        "software_version": __version__,
        "git_commit": git_commit(),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        # ---- 源文件事实
        "source": source_path,
        "source_sha256": source_sha256,
        "source_size": list(geo.source_size),
        "input_type": im.get("kind") or im.get("input_type") or "unknown",
        "bit_depth": im.get("bit_depth") or im.get("bits"),
        # ---- 解码事实
        "decode": {
            "raw_decode_params": im.get("raw_decode_params", {}),
            "decoder": im.get("decoder", "n/a"),
            "white_balance_policy": im.get("white_balance_policy", "unknown"),
            "applied_wb_daylight": im.get("applied_wb_daylight"),
            "camera_wb_as_shot": im.get("camera_wb_as_shot"),
            "black_level": im.get("black_level_per_channel"),
            "white_level": im.get("white_level"),
            "dng_kind": im.get("dng_kind"),
            "normalized_by": im.get("normalized_by"),
        },
        # ---- 色彩事实
        "color": {
            "input_encoding": im.get("input_encoding", "unknown"),
            "input_encoding_detail": im.get("input_encoding_detail", "unknown"),
            "input_icc_sha256": im.get("input_icc_sha256", "none"),
            "input_trc_sha256": im.get("input_trc_sha256", "none"),
            "working_space": im.get("working_space", "unknown"),
            "output_encoding": "srgb",
            "output_icc_sha256": im.get("output_icc_sha256", ""),
        },
        # ---- 采集指纹
        "capture_fingerprint": {
            "value": pr.capture_fingerprint or "",
            "fields": fingerprint_fields or {},
            "editable": False,
            "batch_encoding_assumption": batch_assumption or {},
        },
        # ---- 卷
        "roll": {
            "id": roll_id,
            "analysis_version": getattr(pr.roll_analysis, "analysis_version", None),
            "analysis_confidence": getattr(pr.roll_analysis, "confidence", None),
            "analysis_source_frames": list(
                getattr(pr.roll_analysis, "source_frames", []) or []),
            "analysis_warnings": list(
                getattr(pr.roll_analysis, "warnings", []) or []),
            "split_suggested": bool(
                getattr(pr.roll_analysis, "split_suggested", False)),
        },
        # ---- 三类 ROI 与全部几何变换
        "geometry": geo.to_dict(),
        "output_transform": {
            "rotation_degrees": geo.rotation_degrees,
            "mirror_horizontal": bool(geo.mirror_horizontal),
            "fine_rotation_degrees": float(geo.fine_rotation_degrees),
            "crop_oriented": list(geo.crop_oriented) if geo.crop_oriented else None,
            "output_size": list(geo.output_size),
            "keep_rebate": bool(geo.keep_rebate),
        },
        # ---- 片基来源
        "base": {
            "dmin": [float(v) for v in pr.report.params.dmin],
            "base_source": pr.base_source,
            "confidence": float(pr.base_confidence),
            "provenance": base_diag.get("provenance", {}),
            "regions_used": base_diag.get("regions_used"),
            "regions_rejected": base_diag.get("regions_rejected", []),
            "contamination": base_diag.get("contamination"),
            "spatial_trend": base_diag.get("spatial_trend"),
            "black_level": base_diag.get("black_level"),
            "flatfield": base_diag.get("flatfield", {"coverage": "off"}),
            "generic_prior_asset": None if pr.base_source != "generic_prior"
            else _generic_asset(),
        },
        # ---- 技术模型
        "film_type": film_type,
        "technical": {
            "backend": pr.backend_name,
            "parameters": pr.report.params.to_parameters_dict(),
            "model_asset": model_asset or pr.model_asset or {},
            "confidence": pr.report.confidence,
            "stages": pr.report.stages,
        },
        # ---- 三候选
        "candidates": cset.to_dict() if cset else None,
        "chosen_candidate": pr.chosen_candidate,
        "default_candidate": cset.default_name if cset else None,
        "default_reason": chosen.rationale if chosen else "",
        # ---- 微调
        "render_intent": pr.render_intent.to_dict(),
        # ---- 独立诊断
        "diagnostics": {
            "technical": {**pr.result.diagnostics,
                          "gate": pr.gate.to_dict()},
            "rendered": pr.rendered_diagnostics,
        },
        "warnings": sorted(set(pr.report.warnings + pr.result.warnings
                               + pr.gate.warnings + pr.gate.failures
                               + pr.gate.review_reasons)),
        "review_reasons": list(pr.gate.review_reasons),
        "status": pr.status,
    }


def _generic_asset():
    from ..pipeline.roll_analysis import GENERIC_PRIOR
    return dict(GENERIC_PRIOR)


def validate_sidecar(doc: dict) -> list[str]:
    """结构校验（ASCII 错误信息）。返回空列表表示合法。"""
    errs: list[str] = []
    if not isinstance(doc, dict):
        return ["sidecar must be a JSON object"]
    v = doc.get("schema_version")
    if v not in (*LEGACY_READABLE, SCHEMA_VERSION):
        errs.append(f"unsupported schema_version {v!r}")
    if v == SCHEMA_VERSION:
        for key in ("source_sha256", "geometry", "base", "technical",
                    "candidates", "render_intent", "diagnostics",
                    "capture_fingerprint", "roll", "color", "decode"):
            if key not in doc:
                errs.append(f"missing required key {key!r}")
        sha = doc.get("source_sha256", "")
        if len(sha) != 64:
            errs.append(f"source_sha256 must be 64 hex chars, got {sha!r}")
        geo = doc.get("geometry") or {}
        rss = geo.get("roi_source_space") or {}
        if rss.get("coordinate_space") != "source":
            errs.append("geometry.roi_source_space.coordinate_space must be 'source'")
        for key in ("output_crop_roi", "image_area_roi", "base_sample_rois"):
            if key not in rss:
                errs.append(f"geometry.roi_source_space missing {key!r}")
        if rss.get("base_sample_rois") is not None \
                and not isinstance(rss.get("base_sample_rois"), list):
            errs.append("base_sample_rois must be a list")
        if doc.get("base", {}).get("base_source") not in (
                "frame", "roll", "roll_partial", "frame+roll", "generic_prior",
                "none"):
            errs.append(f"unknown base.base_source "
                        f"{doc.get('base', {}).get('base_source')!r}")
    return errs


def migrate_v3_to_v4(doc: dict) -> dict:
    """v3（及更早）sidecar 的**只读**兼容迁移。

    原则：能确定的字段照搬；无法确定的新字段填 None 并写入 `migration`
    说明，绝不伪造（例如不猜测采集指纹）。
    """
    src = dict(doc or {})
    v = int(src.get("schema_version") or 0)
    if v == SCHEMA_VERSION:
        return src
    if v not in LEGACY_READABLE:
        raise ValueError(f"无法迁移 schema_version={v!r} 的 sidecar")
    geo = src.get("geometry") or {}
    rss = dict(geo.get("roi_source_space") or {})
    crop = rss.get("output_crop_roi") or rss.get("frame_roi")
    area = rss.get("image_area_roi") or crop
    base = rss.get("base_roi") or rss.get("base")
    tech = src.get("technical_parameters") or src.get("parameters") or {}
    from ..pipeline.render import RenderIntent
    ri = src.get("render_intent")
    if ri:
        intent = RenderIntent.from_dict(ri).to_dict()
    else:
        intent = RenderIntent.from_legacy_parameters(tech).to_dict()
    out = {
        "schema_version": SCHEMA_VERSION,
        "software_version": src.get("software_version", "unknown"),
        "git_commit": src.get("git_commit", "unknown"),
        "created_utc": src.get("created_utc") or src.get("finished_utc", ""),
        "source": src.get("source", ""),
        "source_sha256": src.get("source_sha256", ""),
        "source_size": src.get("source_size") or geo.get("source_size") or [None, None],
        "input_type": src.get("input_type", "unknown"),
        "bit_depth": src.get("bit_depth"),
        "decode": {"raw_decode_params": (src.get("input") or {}).get("raw_decode_params", {}),
                   "decoder": "unknown (migrated)",
                   "white_balance_policy": (src.get("input") or {}).get(
                       "white_balance_policy", "unknown"),
                   "applied_wb_daylight": None, "camera_wb_as_shot": None,
                   "black_level": None, "white_level": None,
                   "dng_kind": None, "normalized_by": None},
        "color": {"input_encoding": (src.get("input") or {}).get("encoding", "unknown"),
                  "input_encoding_detail": (src.get("input") or {}).get(
                      "encoding_detail", "unknown"),
                  "input_icc_sha256": (src.get("input") or {}).get(
                      "profile_sha256", "none"),
                  "input_trc_sha256": "none",
                  "working_space": src.get("working_space", "unknown"),
                  "output_encoding": "srgb",
                  "output_icc_sha256": (src.get("output") or {}).get(
                      "icc_sha256", "")},
        "capture_fingerprint": {"value": "", "fields": {}, "editable": False,
                                "batch_encoding_assumption": {}},
        "roll": {"id": None, "analysis_version": None,
                 "analysis_confidence": None, "analysis_source_frames": [],
                 "analysis_warnings": [], "split_suggested": False},
        "geometry": {**geo, "roi_source_space": {
            **rss,
            "output_crop_roi": list(crop) if crop else None,
            "image_area_roi": list(area) if area else None,
            "base_sample_rois": [list(base)] if base else [],
            "coordinate_space": rss.get("coordinate_space", "source"),
            "provenance": {"migrated_from": v}}},
        "output_transform": {
            "rotation_degrees": src.get("rotation_degrees", 0),
            "mirror_horizontal": False, "fine_rotation_degrees": 0.0,
            "crop_oriented": geo.get("crop_oriented"),
            "output_size": geo.get("output_size") or [None, None],
            "keep_rebate": bool(geo.get("keep_rebate", False))},
        "base": {"dmin": [float(x) for x in tech.get("dmin", [0.5, 0.5, 0.5])],
                 "base_source": "frame", "confidence": None, "provenance": {},
                 "regions_used": None, "regions_rejected": [],
                 "contamination": None, "spatial_trend": None,
                 "black_level": None,
                 "flatfield": src.get("flatfield", {"coverage": "off"}),
                 "generic_prior_asset": None},
        "film_type": src.get("film_type", "unknown"),
        "technical": {"backend": src.get("backend", "power-reciprocal"),
                      "parameters": tech, "model_asset": {},
                      "confidence": src.get("confidence", {}), "stages": []},
        "candidates": None,
        "chosen_candidate": "preserve", "default_candidate": "preserve",
        "default_reason": "migrated：旧 sidecar 没有候选集",
        "render_intent": intent,
        "diagnostics": {"technical": {"gate": src.get("quality") or {}},
                        "rendered": src.get("rendered_diagnostics") or {}},
        "warnings": list(src.get("warnings") or []),
        "review_reasons": [],
        "status": src.get("status", "converted"),
        "migration": {"from_schema_version": v,
                      "notes": ["采集指纹、卷分析、候选集与片基来源在旧版本中"
                                "不存在，迁移后为 None/默认，不伪造事实"]},
    }
    return out


def load_sidecar(path: str | Path) -> dict:
    """读取 sidecar；旧版本自动迁移为 v4（只读兼容）。"""
    import json

    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    return migrate_v3_to_v4(doc) if doc.get("schema_version") != SCHEMA_VERSION \
        else doc
