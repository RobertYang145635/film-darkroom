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

STATUS_ICON = {"pending": "○", "running": "◐", "ok": "✓",
               "warn": "⚠", "failed": "✗", "cancelled": "⊘"}


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
        return self.report()

    def _process_one(self, it: BatchItem) -> None:
        src = it.path
        loaded = load_linear_rgb(src, tiff_encoding=self.tiff_encoding)
        t = loaded.rgb
        # ---- 共享工作流（整改 P0-C/I-1：与回归/GUI 同一实现；
        # 片基检测在裁切前的图上做，避免 Dmin 参考丢失）
        from ..calibration.annotations import load_annotation
        from .workflow import prepare_and_analyze
        ann = None
        try:
            ann = load_annotation(src.resolve().relative_to(
                Path(__file__).resolve().parents[3]).as_posix())
        except ValueError:
            pass
        prepared = prepare_and_analyze(
            t, rotation_degrees=ann.rotation_degrees if ann else 0,
            ann=ann, keep_rebate=self.keep_rebate,
            prior=self.prior.to_initial_params() if self.prior else None)
        t = prepared.t
        it.geometry = prepared.geometry.to_dict()
        if self.dust_auto:
            # §7：自动检测保守限定在片基/边缘均匀条带；画面内容区候选密度
            # 过高（真实样片实测 21%），内容尘点走手动蒙版，不自动冒险。
            from ..calibration.border import detect_regions
            from ..restoration.dust import detect_dust, safe_inpaint
            det_r = detect_regions(t)
            region = None
            if det_r.get("base") is not None:
                by0, bx0, by1, bx1 = det_r["base"]
                region = np.zeros(t.shape[:2], bool)
                region[by0:by1, bx0:bx1] = True
            det = detect_dust(t, region=region)
            t, err = safe_inpaint(t, det["mask"])
            if err:
                loaded.warnings.append(err)   # 修复失败不阻断，只警告
            elif det["mask"].any():
                loaded.warnings.append(
                    f"片基条带自动除尘：{det['n_spots']} 斑点像素/{det['n_scratches']} "
                    f"划痕像素已修复（画面内容区请用手动画笔蒙版）")
                loaded.warnings.extend(det["warnings"])
            else:
                loaded.warnings.extend(det["warnings"])
        # 除尘改变了像素：用裁切前提取的片基数组重跑分析（保持 Dmin 参考一致）
        rep = auto_analyze(t, base_roi=prepared.base_arr,
                           prior=self.prior.to_initial_params() if self.prior else None) \
            if self.dust_auto else prepared.report
        if self.cancelled:
            it.status = "cancelled"
            return
        res = invert_chunked(t, rep.params)
        warnings = sorted(set(loaded.warnings + rep.warnings + res.warnings))
        # ---- 质量门禁（整改 F-4：结构化状态，不靠中文字符串猜）
        from ..diagnostics.quality_gate import QualityGate
        gate = QualityGate.evaluate(
            t_in=t, positive_linear=res.positive_linear,
            input_nonfinite=res.diagnostics.get("input_nonfinite", 0),
            extra_warnings=warnings)
        it.parameters = rep.params.to_parameters_dict()
        it.confidence = rep.confidence
        it.diagnostics = res.diagnostics
        it.warnings = gate.warnings + gate.failures
        it.quality = gate.to_dict()
        low_conf = any(v < 0.5 for k, v in rep.confidence.items()
                       if k in ("dmin", "dmax"))

        stem = src.stem
        written: list[Path] = []
        from ..color.management import OutputEncoding
        if self.export_tiff:
            dst = self._unique(self.outdir / f"{stem}-positive.tif")
            ensure_not_source(dst, [src])
            save_tiff16(dst, res.positive_linear, OutputEncoding.SRGB, sources=[src])
            written.append(dst)
        if self.export_jpeg:
            dst = self._unique(self.outdir / f"{stem}-positive.jpg")
            ensure_not_source(dst, [src])
            save_jpeg8(dst, res.positive_linear, quality=self.jpeg_quality,
                       sources=[src])
            written.append(dst)
        if written:
            side = Path(str(written[0]) + ".negative.json")
            from ..color.management import icc_sha256, srgb_icc_bytes
            side.write_text(json.dumps({
                "schema_version": 1,
                "source": src.as_posix(),
                "source_sha256": loaded.meta.get("sha256", ""),
                "film_type": self.prior.film_type if self.prior else "unknown",
                "device_profile_id": self.prior.profile_id if self.prior else "none",
                "input_profile": loaded.meta.get("input_encoding", "unknown"),
                "input_icc_sha256": loaded.meta.get("input_icc_sha256", "none"),
                "working_space": loaded.meta.get("working_space", "unknown"),
                "output_encoding": "srgb",
                "output_icc_sha256": icc_sha256(srgb_icc_bytes()),
                "geometry": it.geometry,
                "quality": it.quality,
                "parameters": it.parameters,
                "warnings": it.warnings,
                "diagnostics": {**res.diagnostics, "confidence": rep.confidence},
                "software_version": __version__,
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            written.append(side)
        it.outputs = [p.as_posix() for p in written]
        # 状态来自结构化枚举：gate.failures -> failed(导出无效) / gate 警告或
        # 低置信度 -> warn / 否则 ok（整改 I-5：禁止中文字符串匹配）
        if gate.failures:
            it.status = "warn"     # 输出仍生成，但质量门禁失败必须可见
        elif gate.warnings or low_conf:
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
