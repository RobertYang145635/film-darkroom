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
from .color.management import OutputEncoding
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
    """规格 §6 旁车 schema_version=1（色彩字段按整改 P0-B 记录实测值，不硬编码）。"""
    return {
        "schema_version": 1,
        "source_sha256": _sha256(src),
        "film_type": film_type,
        "device_profile_id": meta.get("device_profile_id", "none"),
        "input_profile": meta.get("input_encoding", "unknown"),
        "input_profile_detail": meta.get("input_encoding_detail",
                                         meta.get("white_balance_policy", "unknown")),
        "input_icc_sha256": meta.get("input_icc_sha256", "none"),
        "working_space": meta.get("working_space", "camera-linear-normalized"),
        "output_encoding": "srgb",
        "output_icc_sha256": meta.get("output_icc_sha256", ""),
        "roi": roi,
        "parameters": params.to_parameters_dict(),
        "diagnostics": diagnostics,
        "software_version": __version__,
    }


def cmd_invert(args: argparse.Namespace) -> int:
    """单张反相（整改 v2 P0-D：与 GUI/批处理同一 process_image 管线）。"""
    from .pipeline.density import invert_chunked
    from .pipeline.service import process_image
    from .profiles.profile import load_profile

    src = Path(args.input)
    outdir = Path(args.output)
    try:
        loaded = load_linear_rgb(src, tiff_encoding=args.tiff_encoding)
    except ReadError as e:
        print(f"[读取失败]\n{e}", file=sys.stderr)
        return 2

    frame = tuple(args.frame_roi) if args.frame_roi else None
    base = tuple(args.base_roi) if args.base_roi else None
    # base-roi 兼容旧参数序 x y w h -> y0 x0 y1 x1
    base_yx = (base[1], base[0], base[1] + base[3], base[0] + base[2]) \
        if base else None
    prior = load_profile(args.profile).to_initial_params() if args.profile else None
    blank = None
    if args.blank_frame:
        blank = load_linear_rgb(Path(args.blank_frame),
                                tiff_encoding=args.tiff_encoding).rgb
    pr = process_image(loaded.rgb, rotation_degrees=args.rotation,
                       frame_roi_source=frame, base_roi_source=base_yx,
                       keep_rebate=args.keep_rebate, prior=prior,
                       dust_auto=args.dust_auto,
                       blank=blank, flat_mode=args.flat_mode)

    if args.params is not None:
        pd = json.loads(Path(args.params).read_text(encoding="utf-8"))
        pr.report.params = InvertParams.from_parameters_dict(
            pd.get("parameters", pd))
        pr.result = invert_chunked(pr.t, pr.report.params)
        from .diagnostics.quality_gate import QualityGate
        pr.gate = QualityGate.evaluate(
            t_in=pr.t, positive_linear=pr.result.positive_linear,
            input_nonfinite=pr.result.diagnostics.get("input_nonfinite", 0),
            extra_warnings=pr.report.warnings + pr.result.warnings)

    if pr.gate.status == "fail":
        print("FAILED: 质量门禁阻断导出：", file=sys.stderr)
        for f in pr.gate.failures:
            print(f"  - {f}", file=sys.stderr)
        return 3

    outdir.mkdir(parents=True, exist_ok=True)
    stem = args.prefix or src.stem
    tif_path = _unique(outdir / f"{stem}.tif")
    jpg_path = tif_path.with_suffix(".jpg")
    side_path = Path(str(tif_path) + ".negative.json")

    save_tiff16(tif_path, pr.result.positive_linear, OutputEncoding.SRGB,
                sources=[src])
    save_jpeg8(jpg_path, pr.result.positive_linear, quality=args.jpeg_quality,
               sources=[src])
    from .color.management import icc_sha256, srgb_icc_bytes
    loaded.meta["output_icc_sha256"] = icc_sha256(srgb_icc_bytes())
    payload = pr.sidecar(source_sha256=loaded.meta.get("sha256", ""),
                         source_path=src.as_posix(),
                         film_type=args.film_type,
                         profile_id=(args.profile or "none"),
                         input_meta=loaded.meta)
    side_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                         encoding="utf-8")

    p = pr.report.params
    print(f"OK params: dmin={[round(v,4) for v in p.dmin]} dmax={p.dmax:.3f} "
          f"offset={p.offset:.3f} rot={args.rotation} "
          f"crop={pr.geometry.crop_oriented}")
    print(f"confidence: {pr.report.confidence}  quality: {pr.gate.status}")
    for w in dict.fromkeys(pr.gate.warnings):
        print(f"WARN: {w}")
    print(f"export: {tif_path} (16-bit TIFF, sRGB ICC)")
    print(f"export: {jpg_path} (JPEG, fixed 8-bit)")
    print(f"sidecar: {side_path} (schema v2)")
    return 0


def _unique(p: Path) -> Path:
    i = 1
    out = p
    while out.exists():
        out = p.with_name(f"{p.stem}-{i}{p.suffix}")
        i += 1
    return out


def cmd_batch(args: argparse.Namespace) -> int:
    from .pipeline.batch import BatchJob, collect_files
    from .profiles.profile import load_profile

    files = collect_files(Path(args.input))
    if not files:
        print(f"未找到可处理文件：{args.input}", file=sys.stderr)
        return 2
    prior = load_profile(args.profile) if args.profile else None
    job = BatchJob(files, Path(args.output), prior=prior,
                   jpeg_quality=args.jpeg_quality, dust_auto=args.dust_auto,
                   tiff_encoding=args.tiff_encoding)

    def show(it):
        print(f"{it.status:9s} {it.path.name}"
              + (f"  {it.error}" if it.error else ""), flush=True)

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
                   help="未曝光片基取样框（SOURCE 坐标）")
    p.add_argument("--frame-roi", type=int, nargs=4, default=None,
                   metavar=("Y0", "X0", "Y1", "X1"),
                   help="画面裁切框（SOURCE 坐标 y0 x0 y1 x1）")
    p.add_argument("--rotation", type=int, default=0, choices=[0, 90, 180, 270],
                   help="顺时针旋转角度")
    p.add_argument("--keep-rebate", action="store_true",
                   help="保留片边不裁切（默认裁到画面区域）")
    p.add_argument("--profile", default=None, help="配置文件（先验）")
    p.add_argument("--dust-auto", action="store_true", help="片基条带保守自动除尘")
    p.add_argument("--flat-mode", choices=["off", "strip", "blank"], default="strip",
                   help="flat-field：off 关闭 / strip 片基条带一维曲面（默认） / "
                        "blank 需 --blank-frame 完整二维")
    p.add_argument("--blank-frame", default=None,
                   help="同装置同几何未放胶片曝光帧（blank light frame）路径")
    p.add_argument("--jpeg-quality", type=int, default=95)
    p.add_argument("--tiff-encoding", default=None,
                   choices=["linear", "srgb", "adobe"],
                   help="无 ICC/Gamma 标签 TIFF 的传递函数假设（否则拒绝处理）")
    p.add_argument("--film-type", default="unknown",
                   choices=["unknown", "c41_color", "bw", "ecn2_color"])
    p.set_defaults(fn=cmd_invert)

    b = sub.add_parser("batch", help="批处理：文件夹逐张独立分析导出")
    b.add_argument("input", help="输入目录或文件")
    b.add_argument("-o", "--output", required=True, help="输出目录")
    b.add_argument("--profile", default=None, help="配置文件 .profile.json（仅作先验）")
    b.add_argument("--dust-auto", action="store_true",
                   help="启用保守自动除尘/划痕候选（§7，默认关闭）")
    b.add_argument("--tiff-encoding", default=None,
                   choices=["linear", "srgb", "adobe"],
                   help="无 ICC/Gamma 标签 TIFF 的传递函数假设")
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
