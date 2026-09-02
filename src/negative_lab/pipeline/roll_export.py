"""整卷批量导出（v5 §12 P0-7）。

- 逐帧走**同一个** `pipeline.service.process_image`（GUI/批处理/测试一致）；
- 每帧事务导出 TIFF16 + JPEG8 + sidecar v4（失败回滚，不留半成品）；
- 唯一命名、可取消、批次报告 `roll-report.json`；
- 原始文件永远只读：导出前校验目标不等于任何源文件。
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np

from .. import __version__
from ..io.export import ExportError, export_triplet
from ..io.images import ReadError, load_linear_rgb
from ..library.fingerprint import capture_fingerprint
from ..library.models import FrameRecord, FrameStatus
from ..pipeline.render import RenderIntent
from .service import process_image


class Cancelled(Exception):
    pass


def export_roll_frames(frames: list[FrameRecord], outdir: Path, *,
                       session=None, roll_id: str | None = None,
                       intent_resolver=None, tiff_assumption: str | None = None,
                       jpeg_quality: int = 95, export_tiff: bool = True,
                       export_jpeg: bool = True, cancel: threading.Event | None = None,
                       progress=None) -> dict:
    """导出一卷。返回 {"ok","needs_review","failed","items","report_path"}。"""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    items: list[dict] = []
    counts = {"ok": 0, "needs_review": 0, "failed": 0}
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    for rec in frames:
        if cancel is not None and cancel.is_set():
            items.append({"file": rec.source_path, "status": "cancelled"})
            continue
        if progress:
            progress(f"导出 {Path(rec.source_path).name} …")
        try:
            entry = _export_one(rec, outdir, session=session, roll_id=roll_id,
                                intent_resolver=intent_resolver,
                                tiff_assumption=tiff_assumption,
                                jpeg_quality=jpeg_quality,
                                export_tiff=export_tiff, export_jpeg=export_jpeg)
        except ReadError as e:
            rec.status = FrameStatus.DECODE_ERROR.value
            entry = {"file": rec.source_path, "status": "failed",
                     "error": f"decode: {e}"}
        except ExportError as e:
            rec.status = FrameStatus.PROCESSING_ERROR.value
            entry = {"file": rec.source_path, "status": "failed",
                     "error": f"export: {e}"}
        except Exception as e:  # noqa: BLE001 —— 单帧失败不影响整卷
            rec.status = FrameStatus.PROCESSING_ERROR.value
            entry = {"file": rec.source_path, "status": "failed",
                     "error": f"{type(e).__name__}: {e}"}
        key = "ok" if entry["status"] == "ok" else (
            "needs_review" if entry["status"] == "needs_review" else "failed")
        counts[key] = counts.get(key, 0) + 1
        items.append(entry)
        if progress:
            progress(f"{Path(rec.source_path).name}: {entry['status']}")
    report = {
        "schema_version": 1,
        "software_version": __version__,
        "started_utc": started,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "outdir": outdir.as_posix(),
        "roll": roll_id,
        "summary": counts,
        "items": items,
    }
    rp = outdir / "roll-report.json"
    rp.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                  encoding="utf-8")
    counts["report_path"] = rp.as_posix()
    return counts


def _export_one(rec: FrameRecord, outdir: Path, *, session, roll_id,
                intent_resolver, tiff_assumption, jpeg_quality,
                export_tiff, export_jpeg) -> dict:
    from ..color.management import icc_sha256, srgb_icc_bytes
    from ..io.sidecar import build_sidecar_v4
    from .roll_analysis import analyze_roll

    path = _abs(rec.source_path, rec)
    loaded = load_linear_rgb(path, tiff_encoding=tiff_assumption)
    fp = capture_fingerprint(loaded.meta, source_path=rec.source_path,
                             batch_assumption=(session.batch_encoding_assumption
                                               if session else None))
    rec.capture_fingerprint = fp.value
    roll_analysis = None
    if session is not None and roll_id:
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
        if stats:
            roll_analysis = analyze_roll(stats, fingerprint=fp.value)
    intent = intent_resolver(rec) if intent_resolver else RenderIntent()
    candidate = rec.chosen_candidate or None
    pr = process_image(loaded.rgb, geometry=rec.geometry,
                       roll_analysis=roll_analysis, render_intent=intent,
                       candidate=candidate, capture_fingerprint=fp,
                       film_type="unknown")
    if pr.gate.blocks_export:
        rec.status = FrameStatus.PROCESSING_ERROR.value
        raise ExportError("质量门禁阻断：" + "; ".join(pr.gate.failures))
    meta = dict(loaded.meta)
    meta["output_icc_sha256"] = icc_sha256(srgb_icc_bytes())
    payload = build_sidecar_v4(
        pr, source_sha256=meta.get("sha256", ""), source_path=path.as_posix(),
        film_type="unknown", roll_id=roll_id, input_meta=meta,
        fingerprint_fields=fp.fields,
        batch_assumption=(session.batch_encoding_assumption if session else None))
    written = export_triplet(outdir, path.stem, pr.rendered_positive_linear,
                             payload, sources=[path], jpeg_quality=jpeg_quality,
                             export_tiff=export_tiff, export_jpeg=export_jpeg)
    rec.status = (FrameStatus.NEEDS_REVIEW.value if pr.gate.review_reasons
                  else FrameStatus.EXPORTED.value)
    return {"file": rec.source_path,
            "status": "needs_review" if pr.gate.review_reasons else "ok",
            "outputs": [p.as_posix() for p in written],
            "gate": pr.gate.status,
            "base_source": pr.base_source,
            "candidate": pr.chosen_candidate,
            "review_reasons": list(pr.gate.review_reasons),
            "clip": pr.gate.technical_preclip}


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
    return Path(__file__).resolve().parents[3] / path
