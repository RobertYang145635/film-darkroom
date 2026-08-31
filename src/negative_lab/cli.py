"""Phase 1 命令行接口（无 GUI，规格 §10 Phase 1 交付）。

用法：
    python -m negative_lab.cli invert INPUT -o OUTDIR [--auto] [--base-roi x y w h]
                                     [--params params.json] [--jpeg-quality 95]

导出：16 位 TIFF + 8 位 JPEG（固定 8 位，明确提示）+ 旁车 <out>.negative.json。
原始文件只读，绝不覆盖（§9）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from . import __version__
from .io.images import ReadError, load_linear_rgb, save_jpeg8, save_tiff16
from .pipeline.auto import auto_analyze
from .pipeline.density import invert
from .pipeline.params import InvertParams


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def build_sidecar(src: Path, params: InvertParams, roi: dict, diagnostics: dict,
                  meta: dict, film_type: str = "unknown") -> dict:
    """规格 §6 旁车 schema_version=1。"""
    return {
        "schema_version": 1,
        "source_sha256": _sha256(src),
        "film_type": film_type,
        "device_profile_id": meta.get("device_profile_id", "none"),
        "input_profile": meta.get("input_profile", "linear-normalized"),
        "roi": roi,
        "parameters": params.to_parameters_dict(),
        "diagnostics": diagnostics,
        "software_version": __version__,
    }


def cmd_invert(args: argparse.Namespace) -> int:
    src = Path(args.input)
    outdir = Path(args.output)
    try:
        loaded = load_linear_rgb(src)
    except ReadError as e:
        print(f"[读取失败]\n{e}", file=sys.stderr)
        return 2

    t = loaded.rgb
    warnings = list(loaded.warnings)
    roi = {"frame": [0, 0, t.shape[1], t.shape[0]], "film_base": None}

    if args.params is not None:
        pd = json.loads(Path(args.params).read_text(encoding="utf-8"))
        params = InvertParams.from_parameters_dict(pd.get("parameters", pd))
        report_diagnostics = invert(t, params).diagnostics
        confidence = {}
    else:
        base_roi = None
        if args.base_roi is not None:
            x, y, w, h = args.base_roi
            base_roi = t[y:y + h, x:x + w, :]
            roi["film_base"] = [x, y, w, h]
        rep = auto_analyze(t, base_roi=base_roi)
        params = rep.params
        warnings.extend(rep.warnings)
        confidence = rep.confidence
        report_diagnostics = rep.diagnostics
        if args.base_roi is None:
            roi["film_base"] = ["auto-border-strip"]

    res = invert(t, params)
    warnings.extend(res.warnings)

    outdir.mkdir(parents=True, exist_ok=True)
    stem = args.prefix or src.stem
    tif_path = outdir / f"{stem}.tif"
    jpg_path = outdir / f"{stem}.jpg"
    side_path = outdir / f"{stem}.negative.json"

    save_tiff16(tif_path, res.preview_srgb, sources=[src])
    save_jpeg8(jpg_path, res.preview_srgb, quality=args.jpeg_quality, sources=[src])
    side = build_sidecar(src, params, roi, report_diagnostics, loaded.meta,
                        film_type=args.film_type)
    side_path.write_text(json.dumps(side, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"参数: dmin={[round(v,4) for v in params.dmin]} dmax={params.dmax:.3f} "
          f"offset={params.offset:.3f} k(R/G/B)="
          f"{params.reference_exponent*params.red_ratio:.3f}/"
          f"{params.reference_exponent:.3f}/"
          f"{params.reference_exponent*params.blue_ratio:.3f}")
    print(f"置信度: {confidence if confidence else '（手动参数）'}")
    for w in dict.fromkeys(warnings):  # 去重保序
        print(f"警告: {w}")
    print(f"导出: {tif_path} (16-bit TIFF)")
    print(f"导出: {jpg_path} (JPEG —— 注意：JPEG 固定为 8 位)")
    print(f"旁车: {side_path}")
    return 0


def cmd_batch(args: argparse.Namespace) -> int:
    from .pipeline.batch import BatchJob, collect_files
    from .profiles.profile import load_profile

    files = collect_files(Path(args.input))
    if not files:
        print(f"未找到可处理文件：{args.input}", file=sys.stderr)
        return 2
    prior = load_profile(args.profile) if args.profile else None
    job = BatchJob(files, Path(args.output), prior=prior,
                   jpeg_quality=args.jpeg_quality, dust_auto=args.dust_auto)

    def show(it):
        icon = {"ok": "✓", "warn": "⚠", "failed": "✗", "cancelled": "⊘"}.get(it.status, "…")
        extra = f"  {it.error}" if it.error else ""
        print(f"{icon} {it.path.name} [{it.status}]{extra}", flush=True)

    job.run(on_update=show)
    report_path = job.write_report()
    rep = job.report()
    print(f"报告：{report_path}")
    print(f"汇总：{rep['summary']}")
    print("提示：JPEG 固定为 8 位；TIFF 为 16 位；源文件只读未改动")
    return 1 if rep["summary"].get("failed", 0) else 0


def cmd_profile(args: argparse.Namespace) -> int:
    from .profiles.profile import (FilmProfile, list_profiles, load_profile,
                                   save_profile)
    if args.action == "list":
        for p in list_profiles():
            print(f"{p.name}  id={p.profile_id} v{p.version} film={p.film_type} "
                  f"device={p.device} light={p.light_source}")
        return 0
    if args.action == "show":
        p = load_profile(args.name)
        print(json.dumps(p.to_dict(), ensure_ascii=False, indent=2))
        return 0
    if args.action == "create":
        prof = FilmProfile(
            name=args.name, film_type=args.film_type or "unknown",
            device=args.device or "unknown", light_source=args.light or "unknown",
            base_prior=list(args.dmin) if args.dmin else None,
            exponent_prior={"reference_exponent": args.ref_exponent,
                            "red_ratio": args.red_ratio, "blue_ratio": args.blue_ratio}
            if None not in (args.ref_exponent, args.red_ratio, args.blue_ratio) else None,
            notes=args.notes or "")
        out = save_profile(prof)
        print(f"已保存：{out}")
        return 0
    print("action 需为 create/list/show", file=sys.stderr)
    return 2


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="negative_lab", description="负片去色罩 CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("invert", help="单张自动/参数反相导出")
    p.add_argument("input")
    p.add_argument("-o", "--output", required=True, help="输出目录（新文件，不覆盖源）")
    p.add_argument("--prefix", default=None)
    p.add_argument("--params", default=None, help="旁车/参数字段 JSON（跳过自动分析）")
    p.add_argument("--base-roi", type=int, nargs=4, default=None, metavar=("X", "Y", "W", "H"),
                   help="未曝光片基取样框")
    p.add_argument("--jpeg-quality", type=int, default=95)
    p.add_argument("--film-type", default="unknown",
                   choices=["unknown", "c41_color", "bw", "ecn2_color"])
    p.set_defaults(fn=cmd_invert)

    b = sub.add_parser("batch", help="批处理：文件夹逐张独立分析导出")
    b.add_argument("input", help="输入目录或文件")
    b.add_argument("-o", "--output", required=True, help="输出目录")
    b.add_argument("--profile", default=None, help="配置文件 .profile.json（仅作先验）")
    b.add_argument("--dust-auto", action="store_true",
                   help="启用保守自动除尘/划痕候选（§7，默认关闭）")
    b.add_argument("--jpeg-quality", type=int, default=95)
    b.set_defaults(fn=cmd_batch)

    q = sub.add_parser("profile", help="胶片/设备配置文件管理")
    q.add_argument("action", choices=["create", "list", "show"])
    q.add_argument("name", nargs="?", default=None)
    q.add_argument("--film-type", default=None)
    q.add_argument("--device", default=None)
    q.add_argument("--light", default=None)
    q.add_argument("--dmin", type=float, nargs=3, default=None)
    q.add_argument("--ref-exponent", type=float, default=None)
    q.add_argument("--red-ratio", type=float, default=None)
    q.add_argument("--blue-ratio", type=float, default=None)
    q.add_argument("--notes", default=None)
    q.set_defaults(fn=cmd_profile)
    args = ap.parse_args(argv)
    if args.cmd == "profile" and args.action in ("create", "show") and not args.name:
        ap.error("create/show 需要 name")
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
