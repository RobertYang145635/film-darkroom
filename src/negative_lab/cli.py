"""命令行接口（v5）。

子命令：
    import   扫描文件夹 -> 会话 JSON（只探针+哈希，不解码全图）
    invert   单帧：三类 ROI + 卷级分析 + 三候选 + 微调 -> 三件套
    batch    文件夹批量（同一 service）
    roll     按会话里的卷分组整卷导出
    profile  研发用配置文件管理（普通用户路径不需要）

原始文件永远只读；导出目录必须显式给出（v5 §0.3.2）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


from .io.images import ReadError, load_linear_rgb
from .library.fingerprint import capture_fingerprint
from .library.models import FrameRecord, GeometryRegions, ImportSession
from .pipeline.params import InvertParams
from .pipeline.render import RenderIntent

_TONE_FLAGS = ("temperature", "tint", "exposure", "contrast", "highlights",
               "shadows", "whites", "blacks", "vibrance", "saturation")


def _intent_from_args(args) -> RenderIntent:
    it = RenderIntent()
    for name in _TONE_FLAGS:
        v = getattr(args, name.replace("-", "_"), None)
        if v is not None:
            setattr(it, name, float(v))
    return it


def _geometry_from_args(args) -> GeometryRegions:
    base = []
    for spec in (args.base_sample or []):
        y0, x0, y1, x1 = (int(v) for v in spec)
        base.append((y0, x0, y1, x1))
    if args.base_roi:
        y0, x0, y1, x1 = (int(v) for v in args.base_roi)
        base.append((y0, x0, y1, x1))
    return GeometryRegions(
        rotation_degrees=args.rotation,
        fine_rotation_degrees=args.fine_rotation,
        mirror_horizontal=bool(args.mirror),
        output_crop_roi=tuple(args.output_crop) if args.output_crop else None,
        image_area_roi=tuple(args.image_area) if args.image_area else None,
        base_sample_rois=base)


def _load_roll_analysis(path: str | None):
    if not path:
        return None
    from .library.models import RollAnalysis
    return RollAnalysis.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def cmd_import(args) -> int:
    from .library.session import import_session

    sess = import_session(Path(args.input))
    out = Path(args.output)
    sess.save(out)
    print(f"OK imported {len(sess.files)} files -> {out}")
    bad = [f.source_path for f in sess.files
           if f.status == "decode_error"]
    if bad:
        print(f"NOTE {len(bad)} 个文件探针失败（详见会话 JSON），"
              "其余仍可处理；缺样片的输入类别见 docs/support-matrix.md")
    return 0


def cmd_invert(args) -> int:
    from .io.export import ExportError, export_triplet
    from .io.sidecar import build_sidecar_v4
    from .pipeline.service import process_image
    from .profiles.profile import load_profile

    src = Path(args.input)
    outdir = Path(args.output)
    assumption = None
    if args.tiff_encoding:
        assumption = {"encoding": args.tiff_encoding,
                      "id": f"cli-{args.tiff_encoding}",
                      "confirmed_by": "user"}
    try:
        loaded = load_linear_rgb(src, tiff_encoding=args.tiff_encoding)
    except ReadError as e:
        print(f"[读取失败]\n{e}", file=sys.stderr)
        return 2
    fp = capture_fingerprint(loaded.meta, source_path=str(src),
                             batch_assumption=assumption)
    prior = load_profile(args.profile).to_initial_params() if args.profile else None
    force_params = None
    if args.params is not None:
        pd = json.loads(Path(args.params).read_text(encoding="utf-8"))
        tech = (pd.get("technical", {}) or {}).get("parameters") \
            or pd.get("parameters") or pd.get("technical_parameters") or pd
        force_params = InvertParams.from_parameters_dict(tech)
    blank = None
    if args.blank_frame:
        blank = load_linear_rgb(Path(args.blank_frame),
                                tiff_encoding=args.tiff_encoding).rgb
    pr = process_image(
        loaded.rgb, geometry=_geometry_from_args(args),
        keep_rebate=args.keep_rebate, prior=prior,
        roll_analysis=_load_roll_analysis(args.roll_analysis),
        render_intent=_intent_from_args(args), candidate=args.candidate,
        blank=blank, flat_mode=args.flat_mode,
        force_params=force_params,   # v6.2 审查：此前解析完从未传入，--params 是死开关
        capture_fingerprint=fp)

    if pr.gate.blocks_export:
        print("FAILED: 质量门禁阻断导出：", file=sys.stderr)
        for f in pr.gate.failures:
            print(f"  - {f}", file=sys.stderr)
        return 3
    outdir.mkdir(parents=True, exist_ok=True)
    stem = args.prefix or src.stem
    from .color.management import icc_sha256, srgb_icc_bytes
    loaded.meta["output_icc_sha256"] = icc_sha256(srgb_icc_bytes())
    payload = build_sidecar_v4(
        pr, source_sha256=loaded.meta.get("sha256", ""),
        source_path=src.as_posix(), film_type=args.film_type,
        roll_id=args.roll_id, input_meta=loaded.meta,
        fingerprint_fields=fp.fields, batch_assumption=assumption)
    try:
        written = export_triplet(outdir, stem, pr.rendered_positive_linear,
                                 payload, sources=[src],
                                 jpeg_quality=args.jpeg_quality,
                                 export_tiff=not args.no_tiff,
                                 export_jpeg=not args.no_jpeg)
    except ExportError as e:
        print(f"FAILED: {e}", file=sys.stderr)
        return 4
    p = pr.report.params
    print(f"OK params: dmin={[round(v, 4) for v in p.dmin]} dmax={p.dmax:.3f} "
          f"offset={p.offset:.3f} rot={pr.geometry.rotation_degrees} "
          f"crop={pr.geometry.crop_oriented}")
    print(f"base_source={pr.base_source} candidate={pr.chosen_candidate} "
          f"quality={pr.gate.status} fingerprint={fp.short()}")
    if pr.candidates is not None:
        for c in pr.candidates:
            print(f"  candidate {c.name}: conf={c.confidence:.2f} "
                  f"temp={c.intent.get('temperature', 0):+.1f} "
                  f"tint={c.intent.get('tint', 0):+.1f} "
                  f"{'(回退 ' + c.fallback_to + ')' if c.fallback_to else ''}")
    for r in pr.gate.review_reasons:
        print(f"REVIEW: {r}")
    for w in dict.fromkeys(pr.gate.warnings):
        print(f"WARN: {w}")
    for f in written:
        print(f"export: {f}")
    print("提示：JPEG 固定 8 位；TIFF 16 位；sidecar schema v4；源文件只读")
    return 0


def cmd_batch(args) -> int:
    from .pipeline.batch import BatchJob, collect_files
    from .profiles.profile import load_profile

    files = collect_files(Path(args.input))
    if not files:
        print(f"未找到可处理文件：{args.input}", file=sys.stderr)
        return 2
    prior = load_profile(args.profile) if args.profile else None
    assumption = None
    if args.tiff_encoding:
        assumption = {"encoding": args.tiff_encoding,
                      "id": f"batch-{args.tiff_encoding}", "confirmed_by": "user"}
    job = BatchJob(files, Path(args.output), prior=prior,
                   jpeg_quality=args.jpeg_quality,
                   tiff_encoding=args.tiff_encoding,
                   batch_assumption=assumption,
                   geometry=_geometry_from_args(args) if args.rotation else None,
                   render_intent=_intent_from_args(args),
                   candidate=args.candidate,
                   roll_analysis=_load_roll_analysis(args.roll_analysis))

    def show(it):
        print(f"{it.status:12s} {it.path.name}"
              + (f"  {it.error}" if it.error else ""), flush=True)

    job.run(on_update=show)
    report_path = job.write_report()
    rep = job.report()
    print(f"报告：{report_path}")
    print(f"汇总：{rep['summary']}")
    return 1 if rep["summary"].get("failed", 0) else 0


def cmd_roll(args) -> int:
    from .pipeline.roll_export import export_exit_code, export_roll_frames

    sess = ImportSession.load(Path(args.session))
    roll = sess.roll(args.roll)
    if roll is None:
        named = [r for r in sess.roll_groups if r.display_name == args.roll]
        roll = named[0] if named else None
    if roll is None:
        print(f"会话中没有卷 {args.roll!r}", file=sys.stderr)
        return 2
    frames = [f for f in (sess.frame(p) for p in roll.frame_ids) if f]
    intent = _intent_from_args(args)

    def resolver(rec: FrameRecord) -> RenderIntent:
        if rec.render_intent_override:
            return RenderIntent.from_dict(rec.render_intent_override)
        return intent

    rep = export_roll_frames(frames, Path(args.output), session=sess,
                             roll_id=roll.id, intent_resolver=resolver,
                             tiff_assumption=args.tiff_encoding,
                             progress=lambda m: print(m, flush=True))
    print(f"OK roll={roll.id} ok={rep['ok']} needs_review="
          f"{rep['needs_review']} blocked={rep.get('blocked', 0)} failed={rep['failed']}")
    print(f"报告：{rep['report_path']}")
    # 只看 failed 会让"全卷被门禁挡住"退出 0（假绿），见 export_exit_code
    return export_exit_code(rep)


def cmd_profile(args) -> int:
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
            notes=args.notes or "")
        out = save_profile(prof)
        print(f"已保存（研发用先验，普通路径不需要）：{out}")
        return 0
    print("action 需为 create/list/show", file=sys.stderr)
    return 2


def _add_geometry_args(p):
    p.add_argument("--rotation", type=int, default=0, choices=[0, 90, 180, 270],
                   help="顺时针旋转角度")
    p.add_argument("--mirror", action="store_true", help="水平镜像")
    p.add_argument("--fine-rotation", dest="fine_rotation", type=float, default=0.0,
                   help="细微旋转/拉直（度，±10）")
    p.add_argument("--output-crop", type=int, nargs=4, default=None,
                   metavar=("Y0", "X0", "Y1", "X1"),
                   help="最终输出裁切（SOURCE 坐标）")
    p.add_argument("--image-area", dest="image_area", type=int, nargs=4,
                   default=None, metavar=("Y0", "X0", "Y1", "X1"),
                   help="有效画面分析区（SOURCE 坐标）")
    p.add_argument("--base-sample", dest="base_sample", type=int, nargs=4,
                   action="append", default=None, metavar=("Y0", "X0", "Y1", "X1"),
                   help="未曝光片基取样区（可重复给多个）")
    p.add_argument("--base-roi", dest="base_roi", type=int, nargs=4, default=None,
                   metavar=("Y0", "X0", "Y1", "X1"),
                   help="（旧参数名，等价于一次 --base-sample）")
    p.add_argument("--keep-rebate", action="store_true",
                   help="输出保留片边（颜色分析仍只用 --image-area）")


def _add_tone_args(p):
    for name in _TONE_FLAGS:
        p.add_argument(f"--{name.replace('_', '-')}", type=float, default=None,
                       help=f"微调 {name}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="negative_lab",
                                 description="负片裁切与去色罩 CLI（v5）")
    sub = ap.add_subparsers(dest="cmd", required=True)

    i = sub.add_parser("import", help="导入文件夹 -> 会话 JSON（只探针与哈希）")
    i.add_argument("input")
    i.add_argument("-o", "--output", required=True, help="会话 JSON 路径")
    i.set_defaults(fn=cmd_import)

    p = sub.add_parser("invert", help="单帧裁切+转换+导出")
    p.add_argument("input")
    p.add_argument("-o", "--output", required=True, help="输出目录（新文件，不覆盖源）")
    p.add_argument("--prefix", default=None)
    p.add_argument("--roll-id", dest="roll_id", default=None)
    p.add_argument("--params", default=None, help="技术参数 JSON（跳过自动分析）")
    p.add_argument("--candidate", default=None,
                   choices=["preserve", "neutral", "warm"],
                   help="自动颜色候选（默认由卷级统计选择）")
    p.add_argument("--roll-analysis", dest="roll_analysis", default=None,
                   help="卷级分析 JSON（RollAnalysis；普通路径无需测色图）")
    _add_geometry_args(p)
    _add_tone_args(p)
    p.add_argument("--profile", default=None, help="（研发）配置先验")
    p.add_argument("--flat-mode", choices=["off", "strip", "blank"], default="strip",
                   help="片基场：off 关闭 / strip 可观测方向拟合（默认）")
    p.add_argument("--blank-frame", default=None,
                   help="同装置同几何的 blank light frame 路径")
    p.add_argument("--jpeg-quality", type=int, default=95)
    p.add_argument("--no-tiff", action="store_true")
    p.add_argument("--no-jpeg", action="store_true")
    p.add_argument("--tiff-encoding", default=None,
                   choices=["linear", "srgb", "adobe"],
                   help="无法判定传递函数时的**批次级**编码假设")
    p.add_argument("--film-type", default="unknown",
                   choices=["unknown", "c41_color", "ecn2_color", "bw"],
                   help="可选标签；缺失不阻断（v5 §6.3）")
    p.set_defaults(fn=cmd_invert)

    b = sub.add_parser("batch", help="文件夹批量")
    b.add_argument("input", help="输入目录或文件")
    b.add_argument("-o", "--output", required=True, help="输出目录")
    b.add_argument("--profile", default=None)
    b.add_argument("--candidate", default=None,
                   choices=["preserve", "neutral", "warm"])
    b.add_argument("--roll-analysis", dest="roll_analysis", default=None)
    b.add_argument("--rotation", type=int, default=0, choices=[0, 90, 180, 270])
    _add_tone_args(b)
    b.add_argument("--tiff-encoding", default=None,
                   choices=["linear", "srgb", "adobe"])
    b.add_argument("--jpeg-quality", type=int, default=95)
    b.set_defaults(fn=cmd_batch)

    r = sub.add_parser("roll", help="按会话整卷导出")
    r.add_argument("--session", required=True, help="会话 JSON")
    r.add_argument("--roll", required=True, help="卷 id 或显示名")
    r.add_argument("-o", "--output", required=True)
    _add_tone_args(r)
    r.add_argument("--tiff-encoding", default=None,
                   choices=["linear", "srgb", "adobe"])
    r.set_defaults(fn=cmd_roll)

    q = sub.add_parser("profile", help="（研发）胶片/设备配置先验管理")
    q.add_argument("action", choices=["create", "list", "show"])
    q.add_argument("name", nargs="?", default=None)
    q.add_argument("--film-type", default=None)
    q.add_argument("--device", default=None)
    q.add_argument("--light", default=None)
    q.add_argument("--dmin", type=float, nargs=3, default=None)
    q.add_argument("--notes", default=None)
    q.set_defaults(fn=cmd_profile)

    args = ap.parse_args(argv)
    if args.cmd == "profile" and args.action in ("create", "show") and not args.name:
        ap.error("create/show 需要 name")
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
