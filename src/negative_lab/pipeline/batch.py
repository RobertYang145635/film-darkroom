"""批处理队列（规格 §2.2.1、§5.2、§8.1.6）。

- 每张图独立自动分析；配置文件只作初始先验，不强行覆盖单张结果。
- 单张失败不影响其余；失败项可单独重试；取消不留半成品
  （导出走原子写入，取消检查点在文件边界与导出前）。
- 队列状态：pending/running/ok/warn/failed/cancelled。
- 结束生成 batch-report.json + 可读摘要。
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .. import __version__
from ..io.atomic import ensure_not_source
from ..io.images import load_linear_rgb, save_jpeg8, save_tiff16
from ..pipeline.auto import auto_analyze
from ..pipeline.density import invert_chunked
from ..profiles.profile import FilmProfile

# ASCII 状态词（整改 I-7/P0-D：默认 GBK 控制台不得崩溃）
STATUS_ICON = {"pending": "PENDING", "running": "RUNNING", "ok": "OK",
               "warn": "WARN", "failed": "FAILED", "cancelled": "CANCELLED",
               "needs_calibration": "NEEDS-CAL"}


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
    error: str | None = None
    seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "file": self.path.as_posix(),
            "status": self.status,
            "outputs": self.outputs,
            "parameters": self.parameters,
            "confidence": self.confidence,
            "warnings": self.warnings,
            "diagnostics": self.diagnostics,
            "quality": self.quality,
            "geometry": self.geometry,
            "error": self.error,
            "seconds": round(self.seconds, 2),
        }


class BatchJob:
    def __init__(self, files: list[Path], outdir: Path,
                 prior: FilmProfile | None = None,
                 jpeg_quality: int = 95,
                 export_jpeg: bool = True,
                 export_tiff: bool = True,
                 dust_auto: bool = False,
                 tiff_encoding: str | None = None,
                 keep_rebate: bool = False):
        self.items = [BatchItem(path=Path(f)) for f in files]
        self.outdir = Path(outdir)
        self.prior = prior
        self.jpeg_quality = jpeg_quality
        self.export_jpeg = export_jpeg
        self.export_tiff = export_tiff
        self.dust_auto = dust_auto   # §7：自动检测是保守候选，默认关闭
        # 无 ICC/Gamma 标签 TIFF 的显式传递函数假设（None=拒绝，整改 P0-B）
        self.tiff_encoding = tiff_encoding
        # 整改 P0-C：默认裁切到画面区域（排除片边/齿孔/片基）；keep_rebate=True 保留
        self.keep_rebate = keep_rebate
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
        """顺序处理（单线程；GUI 侧放 QThread）。返回报告 dict。"""
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
        # 成功收尾必须写报告（整改 I-5/I-8：GUI 完成标签只能显示真实存在的报告）
        try:
            self.write_report()
        except OSError:
            pass
        return rep

    def _process_one(self, it: BatchItem) -> None:
        src = it.path
        loaded = load_linear_rgb(src, tiff_encoding=self.tiff_encoding)
        # ---- 统一服务（整改 v2 P0-D：与 GUI/CLI 同一管线，禁止入口自行实现）
        from ..calibration.annotations import load_annotation
        from .service import process_image
        ann = None
        try:
            ann = load_annotation(src.resolve().relative_to(
                Path(__file__).resolve().parents[3]).as_posix())
        except ValueError:
            pass
        pr = process_image(
            loaded.rgb,
            rotation_degrees=ann.rotation_degrees if ann else 0,
            ann=ann, keep_rebate=self.keep_rebate,
            prior=self.prior.to_initial_params() if self.prior else None,
            dust_auto=self.dust_auto)
        if self.cancelled:
            it.status = "cancelled"
            return
        # ---- fail 在写文件之前阻断（整改 I-3：不产出任何输出）
        if pr.gate.status == "fail":
            it.status = "failed"
            it.error = "; ".join(pr.gate.failures)
            it.quality = pr.gate.to_dict()
            it.geometry = pr.geometry.to_dict()
            return
        it.parameters = pr.report.params.to_parameters_dict()
        it.confidence = pr.report.confidence
        it.diagnostics = pr.result.diagnostics
        it.geometry = pr.geometry.to_dict()
        it.quality = pr.gate.to_dict()
        it.warnings = pr.gate.warnings

        from ..color.management import icc_sha256, srgb_icc_bytes
        loaded.meta["output_icc_sha256"] = icc_sha256(srgb_icc_bytes())
        payload = pr.sidecar(
            source_sha256=loaded.meta.get("sha256", ""),
            source_path=src.as_posix(),
            film_type=(self.prior.film_type if self.prior else "unknown"),
            profile_id=(self.prior.profile_id if self.prior else "none"),
            input_meta=loaded.meta)
        # 事务三件套（v3 P0-H：staging->校验->提交，失败零残留）
        from ..io.export import ExportError, export_triplet
        try:
            written = export_triplet(self.outdir, src.stem,
                                     pr.result.positive_linear, payload,
                                     sources=[src], jpeg_quality=self.jpeg_quality)
        except ExportError as e:
            it.status = "failed"
            it.error = str(e)
            return
        it.outputs = [p.as_posix() for p in written]
        low_conf = any(v < 0.5 for k, v in pr.report.confidence.items()
                       if k in ("dmin", "dmax"))
        if pr.gate.status == "needs_calibration":
            it.status = "needs_calibration"   # v3：可导出但明确非标定状态
        elif pr.gate.warnings or low_conf:
            it.status = "warn"
        else:
            it.status = "ok"

    @staticmethod
    def _unique(p: Path) -> Path:
        i = 1
        out = p
        while out.exists():
            out = p.with_name(f"{p.stem}-{i}{p.suffix}")
            i += 1
        return out

    # ---- 报告

    def report(self) -> dict:
        counts: dict[str, int] = {}
        for it in self.items:
            counts[it.status] = counts.get(it.status, 0) + 1
        return {
            "schema_version": 1,
            "software_version": __version__,
            "started_utc": self.started_utc,
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "outdir": self.outdir.as_posix(),
            "profile": self.prior.name if self.prior else None,
            "summary": counts,
            "items": [it.to_dict() for it in self.items],
        }

    def write_report(self) -> Path:
        out = self.outdir / "batch-report.json"
        out.write_text(json.dumps(self.report(), ensure_ascii=False, indent=2),
                       encoding="utf-8")
        lines = [f"批处理报告  {out}",
                 f"总计 {len(self.items)} 张：",
                 "  " + "  ".join(f"{STATUS_ICON.get(k,k)}{k}={v}"
                                  for k, v in sorted(
                                      self.report()['summary'].items()))]
        for it in self.items:
            if it.status == "failed":
                lines.append(f"  失败 {it.path.name}: {it.error}")
        (self.outdir / "batch-report.txt").write_text("\n".join(lines),
                                                      encoding="utf-8")
        return out


def collect_files(root: Path, exts=(".tif", ".tiff", ".nef", ".dng", ".cr2",
                                    ".arw", ".png")) -> list[Path]:
    root = Path(root)
    if root.is_file():
        return [root]
    return sorted(f for f in root.rglob("*")
                  if f.suffix.lower() in exts and not f.name.lower().endswith(".downloading"))
