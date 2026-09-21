"""批处理队列（v5 §2.1、§12 P0-7）。

- 每张图独立分析；配置文件只作初始先验，不强行覆盖单张结果；
- 单张失败不影响其余；失败项可单独重试；取消不留半成品；
- 队列状态：pending/running/ok/warn/needs_review/failed/cancelled；
- **没有色卡、没有 verified 校准、未知胶片型号都不再是状态**（v5 §8.1）；
- 结束生成 batch-report.json + 可读摘要。

所有帧都通过 `pipeline.service.process_image` 这一唯一入口（§0.3.4）。
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path


from .. import __version__
from ..io.images import load_linear_rgb
from ..library.fingerprint import capture_fingerprint
from ..library.models import GeometryRegions
from ..pipeline.render import RenderIntent
from .service import process_image          # 唯一管线入口（可被测试替换）

# ASCII 状态词（默认 GBK 控制台不得崩溃）
STATUS_ICON = {"pending": "PENDING", "running": "RUNNING", "ok": "OK",
               "warn": "WARN", "needs_review": "NEEDS-REVIEW",
               "failed": "FAILED", "blocked": "BLOCKED", "cancelled": "CANCELLED"}


@dataclass
class BatchItem:
    path: Path
    status: str = "pending"
    outputs: list[str] = field(default_factory=list)
    parameters: dict | None = None
    confidence: dict | None = None
    warnings: list[str] = field(default_factory=list)
    diagnostics: dict | None = None
    quality: dict | None = None          # QualityGate.to_dict()
    geometry: dict | None = None         # GeometryRecord.to_dict()
    base_source: str | None = None
    candidate: str | None = None
    error: str | None = None
    seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "file": self.path.as_posix(),
            "status": self.status,
            "outputs": self.outputs,
            "parameters": self.parameters,
            "confidence": self.confidence,
            "base_source": self.base_source,
            "candidate": self.candidate,
            "warnings": self.warnings,
            "diagnostics": self.diagnostics,
            "quality": self.quality,
            "geometry": self.geometry,
            "error": self.error,
            "seconds": round(self.seconds, 2),
        }


class BatchJob:
    def __init__(self, files: list[Path], outdir: Path,
                 *, prior=None,
                 jpeg_quality: int = 95,
                 export_jpeg: bool = True,
                 export_tiff: bool = True,
                 tiff_encoding: str | None = None,
                 keep_rebate: bool = False,
                 geometry: GeometryRegions | None = None,
                 render_intent: RenderIntent | None = None,
                 candidate: str | None = None,
                 roll_analysis=None,
                 batch_assumption: dict | None = None):
        # 2026-09-08 死参数清理：film_type 构造参数此前只转发给
        # process_image 的死形参（从未参与任何行为），随签名一并移除；
        # sidecar 的 film_type 字段仍按 v5 口径写（prior 命名或 unknown）。
        self.items = [BatchItem(path=Path(f)) for f in files]
        self.outdir = Path(outdir)
        self.prior = prior
        self.jpeg_quality = jpeg_quality
        self.export_jpeg = export_jpeg
        self.export_tiff = export_tiff
        # 无 ICC/Gamma 标签 TIFF 的批次级编码假设（v5 §4.3.3）
        self.tiff_encoding = tiff_encoding
        self.batch_assumption = batch_assumption
        self.keep_rebate = keep_rebate
        self.geometry = geometry
        self.render_intent = render_intent
        self.candidate = candidate
        self.roll_analysis = roll_analysis
        self._cancel = threading.Event()
        self.started_utc = ""

    # ---- 控制

    def cancel(self) -> None:
        self._cancel.set()

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def failed_items(self) -> list[BatchItem]:
        return [it for it in self.items if it.status == "failed"]

    # ---- 执行

    def run(self, on_update=None) -> dict:
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        for it in self.items:
            if self.cancelled:
                it.status = "cancelled"
                if on_update:
                    on_update(it)
                continue
            it.status = "running"
            if on_update:
                on_update(it)
            t0 = time.perf_counter()
            try:
                self._process_one(it)
            except Exception as e:  # noqa: BLE001 —— 单张失败不影响其余
                it.status = "failed"
                it.error = f"{type(e).__name__}: {e}"
            it.seconds = time.perf_counter() - t0
            if on_update:
                on_update(it)
        rep = self.report()
        try:
            self.write_report()
        except OSError:
            pass
        return rep

    def _process_one(self, it: BatchItem) -> None:
        src = it.path
        loaded = load_linear_rgb(src, tiff_encoding=self.tiff_encoding)
        from ..calibration.annotations import load_annotation
        from ..io.sidecar import build_sidecar_v4

        ann = None
        try:
            ann = load_annotation(src.resolve().relative_to(
                Path(__file__).resolve().parents[3]).as_posix())
        except (ValueError, OSError):
            pass
        fp = capture_fingerprint(loaded.meta, source_path=str(src),
                                 batch_assumption=self.batch_assumption)
        pr = process_image(
            loaded.rgb, geometry=self.geometry, ann=ann,
            keep_rebate=self.keep_rebate,
            prior=self.prior.to_initial_params() if self.prior else None,
            roll_analysis=self.roll_analysis,
            render_intent=self.render_intent, candidate=self.candidate,
            capture_fingerprint=fp)
        if self.cancelled:
            it.status = "cancelled"
            return
        # ---- fail 在写文件之前阻断（不产出任何输出）
        if pr.gate.blocks_export:
            from ..diagnostics.explain import clip_note
            # B-7：批量路径与整卷导出对齐——门禁阻断是**故意不给导**，
            # 不再冒充 failed。counts 按状态动态统计（见 report()），
            # 所以 blocked 会自动进 summary，守恒式仍然成立。
            it.status = "blocked"
            # 批量路径同样要说清责任：这帧是"采集端就已丢失"还是"软件能改"。
            # 注意：本路径的 blocked/failed **分桶仍未做**（见 blockers.md B-7 遗留），
            # 这里只补归因，不动状态语义，避免一次改两层。
            note = clip_note(pr.gate)
            it.error = ("; ".join(pr.gate.failures)
                        + (("；" + note) if note else ""))
            it.quality = pr.gate.to_dict()
            it.geometry = pr.geometry.to_dict()
            return
        it.parameters = pr.report.params.to_parameters_dict()
        it.confidence = pr.report.confidence
        it.diagnostics = pr.result.diagnostics
        it.geometry = pr.geometry.to_dict()
        it.quality = pr.gate.to_dict()
        it.warnings = pr.gate.warnings
        it.base_source = pr.base_source
        it.candidate = pr.chosen_candidate

        from ..color.management import icc_sha256, srgb_icc_bytes
        loaded.meta["output_icc_sha256"] = icc_sha256(srgb_icc_bytes())
        payload = build_sidecar_v4(
            pr, source_sha256=loaded.meta.get("sha256", ""),
            source_path=src.as_posix(),
            film_type=(self.prior.film_type if self.prior else "unknown"),
            input_meta=loaded.meta, fingerprint_fields=fp.fields,
            batch_assumption=self.batch_assumption)
        from ..io.export import ExportError, export_triplet
        try:
            written = export_triplet(self.outdir, src.stem,
                                     pr.rendered_positive_linear, payload,
                                     sources=[src], jpeg_quality=self.jpeg_quality,
                                     export_tiff=self.export_tiff,
                                     export_jpeg=self.export_jpeg)
        except ExportError as e:
            it.status = "failed"
            it.error = str(e)
            return
        it.outputs = [p.as_posix() for p in written]
        if pr.gate.review_reasons:
            it.status = "needs_review"      # 允许导出，但必须逐张复核
        elif pr.gate.warnings:
            it.status = "warn"
        else:
            it.status = "ok"

    # ---- 报告

    def report(self) -> dict:
        counts: dict[str, int] = {}
        for it in self.items:
            counts[it.status] = counts.get(it.status, 0) + 1
        return {
            "schema_version": 2,
            "software_version": __version__,
            "started_utc": self.started_utc,
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "outdir": self.outdir.as_posix(),
            # ---- 报告 schema 统一（2026-09-08，交接 §4-3 后半）----
            # batch.report(v2) 与 roll_export(v1) 键集漂移是本轮审查发现的
            # 重复实现问题。统一后两者共用核心键集：
            #   schema_version / software_version / started_utc / finished_utc /
            #   outdir / kind / summary / summary_complete / items（items 至少含
            #   file、status，失败含 error）。kind 区分批次与整卷导出；
            #   roll 侧另有 roll 字段、batch 侧另有 profile 字段。
            "kind": "batch",
            "profile": self.prior.name if self.prior else None,
            "summary": counts,
            "summary_complete": sum(counts.values()) == len(self.items),
            "items": [it.to_dict() for it in self.items],
        }

    def write_report(self) -> Path:
        out = self.outdir / "batch-report.json"
        from ..io.atomic import atomic_write_bytes    # 原子写（v6.2 审查批次 3）
        atomic_write_bytes(out, json.dumps(self.report(), ensure_ascii=False,
                                           indent=2).encode("utf-8"),
                           overwrite=True)
        doc = self.report()
        lines = [f"批处理报告  {out}",
                 f"总计 {len(self.items)} 张：",
                 "  " + "  ".join(f"{STATUS_ICON.get(k, k)}={v}"
                                  for k, v in sorted(doc["summary"].items()))]
        for it in self.items:
            if it.status == "failed":
                lines.append(f"  失败 {it.path.name}: {it.error}")
            elif it.status == "blocked":
                # 被门禁挡住的帧也必须出现在人读报告里，否则它在 txt 里凭空消失；
                # 措辞也要与"失败"分开（B-7：处置方式是重扫/重拍，不是重试导出）
                lines.append(f"  被门禁挡住 {it.path.name}: {it.error}")
        atomic_write_bytes(self.outdir / "batch-report.txt",
                           "\n".join(lines).encode("utf-8"), overwrite=True)
        return out


def collect_files(root: Path, exts=(".tif", ".tiff", ".nef", ".dng", ".cr2",
                                    ".arw", ".png")) -> list[Path]:
    root = Path(root)
    if root.is_file():
        return [root]
    return sorted(f for f in root.rglob("*")
                  if f.suffix.lower() in exts and not f.name.lower().endswith(".downloading"))
