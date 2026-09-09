"""整卷批量导出（v5 §12 P0-7）。

- 逐帧走**同一个** `pipeline.service.process_image`（GUI/批处理/测试一致）；
- 每帧事务导出 TIFF16 + JPEG8 + sidecar v4（失败回滚，不留半成品）；
- 唯一命名、可取消、批次报告 `roll-report.json`；
- 原始文件永远只读：导出前校验目标不等于任何源文件。
"""

from __future__ import annotations

from .. import storage

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


from .. import __version__
from ..io.export import ExportError, export_triplet
from ..io.images import ReadError, load_linear_rgb
from ..library.fingerprint import capture_fingerprint
from ..library.models import FrameRecord, FrameStatus
from ..pipeline.render import RenderIntent
from .service import process_image


# ---------------------------------------------------------- 导出输入快照
# 审查 §4-1（2026-09-08，最高危未修项）：此前后台导出线程逐帧读
# `rec.geometry / keep_rebate / render_intent_override / frame_analysis`
# 等**活对象**，导出期间 UI 的任何编辑都会漏进成品（错误产物安静出厂）；
# worker 还直接写 `rec.status / rec.capture_fingerprint`（跨线程改共享状态）。
# 现在调用方在启动导出前用 `snapshot_frame()` 把逐帧输入钉在原地，
# 导出全程只读快照；状态回写由调用方在**主线程**按 items 调
# `apply_status_writeback()` 完成。同步调用方（CLI/测试）不传 presnapshot
# 时行为不变：入口快照、退出即回写，等价于旧实现。
@dataclass
class _FrameSnapshot:
    """一帧导出输入在启动时刻的冻结副本（只读语义）。"""

    source_path: str
    source_sha256: str | None = None
    input_facts: dict | None = None
    geometry: Any = None
    keep_rebate: bool = False
    render_intent_override: dict | None = None
    chosen_candidate: str | None = None
    frame_analysis: dict | None = None
    capture_fingerprint: str | None = None
    status: str | None = None
    roll_id: str | None = None


def snapshot_frame(rec: FrameRecord) -> _FrameSnapshot:
    """把一帧的导出输入钉在原地。

    geometry 深拷贝（UI 会原地改 fine_rotation/mirror_horizontal，
    浅引用挡不住）；其余字段浅拷贝即可：dict 会被 UI 整体替换而不是原地改。
    """
    from ..library.models import GeometryRegions
    geo = rec.geometry
    if geo is not None:
        geo = GeometryRegions.from_dict(geo.to_dict())
    return _FrameSnapshot(
        source_path=rec.source_path,
        source_sha256=rec.source_sha256,
        input_facts=dict(rec.input_facts or {}),
        geometry=geo,
        keep_rebate=bool(rec.keep_rebate),
        render_intent_override=(dict(rec.render_intent_override)
                                if rec.render_intent_override else None),
        chosen_candidate=rec.chosen_candidate,
        frame_analysis=(dict(rec.frame_analysis)
                        if rec.frame_analysis else None),
        capture_fingerprint=rec.capture_fingerprint,
        status=rec.status,
        roll_id=rec.roll_id)


def apply_status_writeback(session, items: list[dict]) -> None:
    """把导出结果按 items 回写到会话 FrameRecord（**必须主线程调用**）。

    items 由 export_roll_frames 返回（cancelled 之外每帧恰一项）。
    cancelled 的帧没有产出，状态保持原样——不伪造任何结论。
    """
    if session is None:
        return
    for entry in items:
        rec = session.frame(str(entry.get("file") or ""))
        if rec is None:
            continue
        st = entry.get("status")
        if st == "ok":
            rec.status = FrameStatus.EXPORTED.value
        elif st == "needs_review":
            rec.status = FrameStatus.NEEDS_REVIEW.value
        elif st == "failed":
            err = str(entry.get("error") or "")
            rec.status = (FrameStatus.DECODE_ERROR.value
                          if err.startswith("decode:")
                          else FrameStatus.PROCESSING_ERROR.value)
        cf = entry.get("capture_fingerprint")
        if cf:
            rec.capture_fingerprint = cf


def freeze_export_inputs(session, roll_id: str | None, frames,
                         default_intent: RenderIntent | None = None) -> dict:
    """一次性冻结整卷导出的全部输入（**必须主线程调用**，§4-1）。

    返回 {"records","intents","tiff_encoding","roll_stats","batch_assumption"}。
    worker 线程只消费这份冻结结果（配合 export_roll_frames(presnapshot=True,
    session=None)），从结构上保证导出全程零读零写活对象。
    """
    records = [snapshot_frame(f) for f in frames]
    base = (default_intent.to_dict() if default_intent is not None else {})
    intents = {f.source_path: RenderIntent.from_dict(
        f.render_intent_override or base) for f in frames}
    tiff_encoding = None
    batch_assumption = None
    roll_stats = None
    if session is not None:
        batch_assumption = session.batch_encoding_assumption
        tiff_encoding = (batch_assumption or {}).get("encoding")
        if roll_id:
            roll_stats = _collect_roll_stats(session, roll_id)
    return {"records": records, "intents": intents,
            "tiff_encoding": tiff_encoding, "roll_stats": roll_stats,
            "batch_assumption": batch_assumption}


def export_roll_frames(frames: list[FrameRecord], outdir: Path, *,
                       session=None, roll_id: str | None = None,
                       intent_resolver=None, tiff_assumption: str | None = None,
                       jpeg_quality: int = 95, export_tiff: bool = True,
                       export_jpeg: bool = True, cancel: threading.Event | None = None,
                       progress=None, presnapshot: bool = False,
                       frozen_roll_stats: list | None = None,
                       frozen_batch_assumption: dict | None = None) -> dict:
    """导出一卷。返回 {"ok","needs_review","failed","cancelled","items","report_path"}。

    四类计数之和恒等于 len(frames)：取消的帧也必须出现在报告里，
    否则"整卷导出完成"会在静默丢帧的情况下显示为成功（v6.2 §6.6）。
    items 一并返回，使导出面板能够跳到失败帧。

    presnapshot=True（后台线程调用方，如 GUI）：入口先把全部输入快照冻结，
    全程不写回任何 rec 字段——状态回写由调用方在主线程按 items 调
    apply_status_writeback()。配合 session=None + freeze_export_inputs()
    可从结构上保证 worker 零读零写活对象。默认 False（同步调用方，
    CLI/测试）：行为与旧实现一致（读活对象、状态即写即生效）。
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    records = [snapshot_frame(r) for r in frames] if presnapshot else list(frames)
    if presnapshot:
        # 快照模式：卷统计与批次编码假设用调用方冻结的值；
        # 未冻结则兜底收集（仅同步快照调用会走到）。
        batch_assumption = frozen_batch_assumption
        roll_stats = frozen_roll_stats
        if roll_stats is None and session is not None and roll_id:
            roll_stats = _collect_roll_stats(session, roll_id)
    else:
        batch_assumption = (session.batch_encoding_assumption
                            if session is not None else None)
        roll_stats = None
    items: list[dict] = []
    counts = {"ok": 0, "needs_review": 0, "failed": 0, "cancelled": 0}
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    for rec in records:
        if cancel is not None and cancel.is_set():
            items.append({"file": rec.source_path, "status": "cancelled"})
            counts["cancelled"] += 1
            continue
        if progress:
            progress(f"导出 {Path(rec.source_path).name} …")
        try:
            entry = _export_one(rec, outdir, session=session, roll_id=roll_id,
                                intent_resolver=intent_resolver,
                                tiff_assumption=tiff_assumption,
                                jpeg_quality=jpeg_quality,
                                export_tiff=export_tiff, export_jpeg=export_jpeg,
                                roll_stats=roll_stats,
                                batch_assumption=batch_assumption)
        except ReadError as e:
            if not presnapshot:
                rec.status = FrameStatus.DECODE_ERROR.value
            entry = {"file": rec.source_path, "status": "failed",
                     "error": f"decode: {e}"}
        except ExportError as e:
            if not presnapshot:
                rec.status = FrameStatus.PROCESSING_ERROR.value
            entry = {"file": rec.source_path, "status": "failed",
                     "error": f"export: {e}"}
        except Exception as e:  # noqa: BLE001 —— 单帧失败不影响整卷
            if not presnapshot:
                rec.status = FrameStatus.PROCESSING_ERROR.value
            entry = {"file": rec.source_path, "status": "failed",
                     "error": f"{type(e).__name__}: {e}"}
        if not presnapshot:
            # 同步调用方（CLI/测试）：保持旧语义，状态随导出即写。
            if entry["status"] == "ok":
                rec.status = FrameStatus.EXPORTED.value
            elif entry["status"] == "needs_review":
                rec.status = FrameStatus.NEEDS_REVIEW.value
        key = "ok" if entry["status"] == "ok" else (
            "needs_review" if entry["status"] == "needs_review" else "failed")
        counts[key] = counts.get(key, 0) + 1
        items.append(entry)
        if progress:
            progress(f"{Path(rec.source_path).name}: {entry['status']}")
    report = {
        # ---- 报告 schema 统一（2026-09-08，交接 §4-3 后半）----
        # 与 batch.report() 共用核心键集（schema_version / software_version /
        # started_utc / finished_utc / outdir / kind / summary /
        # summary_complete / items 至少含 file、status、失败含 error）。
        # kind 区分批次与整卷导出；本侧保留 v1 的 roll 字段，
        # schema_version 从 1 提升到 2 以对齐键集口径。
        "schema_version": 2,
        "software_version": __version__,
        "started_utc": started,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "outdir": outdir.as_posix(),
        "kind": "roll",
        "roll": roll_id,
        "summary": dict(counts),
        "summary_complete": sum(counts.values()) == len(frames),
        "items": items,
    }
    rp = outdir / "roll-report.json"
    from ..io.atomic import atomic_write_bytes    # 原子写（workers.py 契约）
    atomic_write_bytes(rp, json.dumps(report, ensure_ascii=False, indent=2)
                       .encode("utf-8"), overwrite=True)
    result = dict(counts)
    result["items"] = items
    # 快照模式：capture_fingerprint 只进 items（entry["capture_fingerprint"]），
    # 由主线程 apply_status_writeback 统一回写；同步模式保持旧语义即写即生效。
    if not presnapshot:
        for entry in items:
            cf = entry.get("capture_fingerprint")
            if cf:
                rec_for = next((r for r in records
                                if r.source_path == entry.get("file")), None)
                if rec_for is not None:
                    rec_for.capture_fingerprint = cf
    # 审查修复：summary_complete 此前只写进落盘报告，返回体里没有
    # -> UI 想如实提示「计数不完整」也读不到。补进返回值。
    result["summary_complete"] = sum(counts.values()) == len(frames)
    result["report_path"] = rp.as_posix()
    return result


def _collect_roll_stats(session, roll_id: str) -> list[dict]:
    """快照模式下的卷级统计输入：从会话全卷帧收集（入口时刻冻结）。"""
    stats = []
    for other in _roll_frames(session, roll_id):
        st = other.frame_analysis or {}
        if st.get("base_dmin"):
            stats.append({
                "frame_id": other.source_path,
                "capture_fingerprint": other.capture_fingerprint,
                "base_dmin": st["base_dmin"],
                "base_confidence": st.get("base_confidence", 0.0),
                "contamination": st.get("contamination"),
                "black_level": st.get("black_level"),
                "spatial_trend": st.get("spatial_trend")})
    return stats


def _export_geometry(rec):
    """导出前必须尊重**已落盘的用户标定**（冷读审查 L5-②，2026-09-08）。

    本会话没打开过的帧，rec.geometry 还是导入期的空值，此前直接交给
    process_image 走自动检测 —— 于是用户上个会话签过字的几何被**静默忽略**，
    与 user-quickstart §5「确认标注会影响导出」的承诺相反。这里在几何不完整时
    回读确认标注：确认优先于草稿，只有签字基线才是可交付依据。
    """
    geo = rec.geometry
    if geo is not None and geo.is_complete():
        return geo
    try:
        from ..calibration.annotations import load_annotation
        from ..calibration.state import is_confirmed
        ann = load_annotation(rec.source_path)
    except Exception:                        # 损坏不得阻断整卷导出
        return geo
    if ann is None or not is_confirmed(ann):
        return geo
    if (getattr(ann, "source_sha256", "") and rec.source_sha256
            and ann.source_sha256 != rec.source_sha256):
        return geo                           # 源图已换，标定不再适用
    from ..library.models import GeometryRegions
    return GeometryRegions(
        rotation_degrees=ann.rotation_degrees,
        fine_rotation_degrees=float(
            getattr(ann, "fine_rotation_degrees", 0.0) or 0.0),
        mirror_horizontal=bool(getattr(ann, "mirror_horizontal", False)),
        output_crop_roi=ann.output_crop_roi or ann.image_area_roi,
        image_area_roi=ann.image_area_roi,
        base_sample_rois=list(ann.base_sample_rois or []))


def _export_one(rec, outdir: Path, *, session, roll_id,
                intent_resolver, tiff_assumption, jpeg_quality,
                export_tiff, export_jpeg, roll_stats=None,
                batch_assumption=None) -> dict:
    """导出一帧。rec 可以是 FrameRecord 或 snapshot_frame() 的冻结快照；
    本函数**不写** rec 的任何字段（§4-1：worker 线程不碰共享状态），
    capture_fingerprint 进 entry，status 由调用方按 entry 回写。
    batch_assumption/roll_stats 已在入口冻结；快照模式下本函数对
    session 的唯一用途是同步路径的卷统计回退。"""
    from ..color.management import icc_sha256, srgb_icc_bytes
    from ..io.sidecar import build_sidecar_v4
    from .roll_analysis import analyze_roll

    path = _abs(rec.source_path, rec)
    loaded = load_linear_rgb(path, tiff_encoding=tiff_assumption)
    fp = capture_fingerprint(loaded.meta, source_path=rec.source_path,
                             batch_assumption=batch_assumption)
    roll_analysis = None
    if roll_stats is None:
        # 同步路径（CLI/测试）：与旧实现一致，从会话逐帧取统计。
        stats = []
        if session is not None and roll_id:
            for other in _roll_frames(session, roll_id):
                st = other.frame_analysis or {}
                if st.get("base_dmin"):
                    stats.append({
                        "frame_id": other.source_path,
                        "capture_fingerprint": other.capture_fingerprint,
                        "base_dmin": st["base_dmin"],
                        "base_confidence": st.get("base_confidence", 0.0),
                        "contamination": st.get("contamination"),
                        "black_level": st.get("black_level"),
                        "spatial_trend": st.get("spatial_trend")})
    else:
        # 快照路径（GUI 后台线程）：统计在入口已冻结。
        stats = roll_stats
    if stats:
        roll_analysis = analyze_roll(stats, fingerprint=fp.value)
    intent = intent_resolver(rec) if intent_resolver else RenderIntent()
    candidate = rec.chosen_candidate or None
    # v6.1 §4：导出链路必须传当前帧的用户选择；不得在导出层自己裁切数组
    pr = process_image(loaded.rgb, geometry=_export_geometry(rec),
                       roll_analysis=roll_analysis, render_intent=intent,
                       candidate=candidate, capture_fingerprint=fp,
                       keep_rebate=bool(getattr(rec, "keep_rebate", False)))
    if pr.gate.blocks_export:
        raise ExportError("质量门禁阻断：" + "; ".join(pr.gate.failures))
    meta = dict(loaded.meta)
    meta["output_icc_sha256"] = icc_sha256(srgb_icc_bytes())
    payload = build_sidecar_v4(
        pr, source_sha256=meta.get("sha256", ""), source_path=path.as_posix(),
        film_type="unknown", roll_id=roll_id, input_meta=meta,
        fingerprint_fields=fp.fields, batch_assumption=batch_assumption)
    written = export_triplet(outdir, path.stem, pr.rendered_positive_linear,
                             payload, sources=[path], jpeg_quality=jpeg_quality,
                             export_tiff=export_tiff, export_jpeg=export_jpeg)
    return {"file": rec.source_path,
            "status": "needs_review" if pr.gate.review_reasons else "ok",
            "outputs": [p.as_posix() for p in written],
            "gate": pr.gate.status,
            "base_source": pr.base_source,
            "candidate": pr.chosen_candidate,
            "review_reasons": list(pr.gate.review_reasons),
            "clip": pr.gate.technical_preclip,
            "capture_fingerprint": fp.value}


def _roll_frames(session, roll_id: str) -> list[FrameRecord]:
    if session is None:
        return []
    r = session.roll(roll_id)
    if r is None:
        return []
    return [f for f in (session.frame(p) for p in r.frame_ids) if f is not None]


def _abs(path: str, rec: FrameRecord | None = None) -> Path:
    if rec is not None:
        abs_path = (rec.input_facts or {}).get("abs_path")
        if abs_path and Path(abs_path).exists():
            return Path(abs_path)
    p = Path(path)
    if p.is_absolute():
        return p
    return storage.repo_path(*Path(path).parts)
