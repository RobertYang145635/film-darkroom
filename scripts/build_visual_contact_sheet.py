"""逐卷视觉联系表（v5 §10.2 / §12 P0-8.1）。

每张并列展示**相同画面裁切**：

```
原始负片 | 我们的 preserve | 我们的 neutral | 我们的 warm
我们的当前默认 + 微调 | NLP 默认参考 | 用户满意参考（若有）
```

输出：
    outputs/quality/<run-id>/rolls/<roll>/contact-sheet.jpg
    outputs/quality/<run-id>/rolls/<roll>/decisions.json   （用户填写）
    outputs/quality/<run-id>/rolls/<roll>/metrics.json     （自动指标）

纪律：manifest 中所有 required=true 帧必须出现，不得只挑好看的（§10.2）；
最终视觉结论只有用户有权填写（§10.2 末段）。

运行：.venv\\Scripts\\python.exe scripts/build_visual_contact_sheet.py
       [--roll ROLL] [--max-dim 1200] [--run-id ID]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from PIL import Image, ImageDraw  # noqa: E402

from negative_lab.calibration.annotations import load_annotation  # noqa: E402
from negative_lab.io.images import ReadError, load_linear_rgb  # noqa: E402
from negative_lab.io.sidecar import git_commit  # noqa: E402
from negative_lab.library.fingerprint import capture_fingerprint  # noqa: E402
from negative_lab.library.models import GeometryRegions  # noqa: E402
from negative_lab.pipeline.service import (candidate_positive,  # noqa: E402
                                           process_roll)

ACCEPTANCE = ROOT / "fixtures" / "acceptance" / "manifest.json"
TILE_H = 300
LABELS = ["原始负片", "preserve", "neutral", "warm", "默认+微调",
          "NLP 默认参考", "用户满意参考"]


def _to_img(a01: np.ndarray) -> Image.Image:
    x = np.clip(a01, 0, 1)
    return Image.fromarray((x * 255 + 0.5).astype(np.uint8)).convert("RGB")


def _norm_negative(a: np.ndarray) -> np.ndarray:
    x = np.clip(a, 0, None)
    return x / max(float(x.max()), 1e-6)


def _fit(im: Image.Image, h: int = TILE_H) -> Image.Image:
    w = max(1, int(im.width * h / im.height))
    return im.resize((w, h))


def _row(tiles: list[tuple[str, Image.Image | None]], idx: int,
         caption: str) -> Image.Image:
    ims = []
    for label, im in tiles:
        if im is None:
            im = Image.new("RGB", (120, TILE_H), (40, 40, 40))
            label = f"{label}（缺）"
        ims.append((label, _fit(im)))
    W = sum(i.width for _, i in ims) + 8 * (len(ims) + 1)
    H = TILE_H + 40
    out = Image.new("RGB", (max(W, 900), H), (18, 18, 18))
    d = ImageDraw.Draw(out)
    d.text((4, 2), f"[{idx}] {caption}", fill=(255, 230, 140))
    x = 4
    for label, im in ims:
        d.text((x + 2, 20), label, fill=(180, 220, 255))
        out.paste(im, (x, 38))
        x += im.width + 8
    return out


def _stack(rows: list[Image.Image], out_path: Path) -> None:
    W = max(r.width for r in rows)
    H = sum(r.height for r in rows) + 6 * len(rows)
    sheet = Image.new("RGB", (W, H), (12, 12, 12))
    y = 0
    for r in rows:
        sheet.paste(r, (0, y))
        y += r.height + 6
    sheet.save(out_path, quality=84)


def _orientation_from(frame: dict) -> tuple[int, float, bool]:
    """从**标注**取方向（旋转/微调/镜像）。

    以前这里恒为 0，而 `fixtures/annotations/1298_G200__G200-1.json` 明写
    `rotation_degrees: 180`（备注"倒置"）——于是 48 帧判定条与 `metrics.json`
    里取画面上三分之一的 `top_mean/top_std` 分层，**一直是按倒像算的**。

    标注缺失 → 保持 0（与旧行为一致）。
    标注声明 `oriented` 空间 → **停下**：清单里的 ROI 是 source 空间，
    带着非零旋转去解释 oriented 空间的框会把裁切挪到别处；
    而旋转为 0 时两种空间重合，**这个错以前根本看不出来**。
    """
    try:
        ann = load_annotation(frame["file"])
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            f"{frame['id']}: 标注读不出来（{type(e).__name__}: "
            f"{str(e)[:120]}）——宁可停下，也不要静默按 0 度出图") from e
    if ann is None:
        return 0, 0.0, False
    if ann.coordinate_space != "source":
        raise RuntimeError(
            f"{frame['id']}: 标注 coordinate_space={ann.coordinate_space!r}，"
            "而清单里的 ROI 是 source 空间，套具无法安全施加旋转")
    return (int(ann.rotation_degrees), float(ann.fine_rotation_degrees),
            bool(ann.mirror_horizontal))


def _orientation_decided(frame: dict) -> bool:
    """这一帧的方向**有没有人决定过**（B-12）。

    标注里写 `0°` 与"根本没有标注、默认 0°"在像素上一模一样，
    只有这个字段能分开二者；实测 `1298_G200` 12 帧只有 4 帧有标注文件。
    读不出标注按"没决定过"算——它不是决定。
    """
    try:
        return load_annotation(frame["file"]) is not None
    except Exception:  # noqa: BLE001
        return False


def _geometry(frame: dict) -> GeometryRegions:
    r = frame["rois"]
    rot, fine, mirror = _orientation_from(frame)
    return GeometryRegions(
        rotation_degrees=rot,
        fine_rotation_degrees=fine,
        mirror_horizontal=mirror,
        output_crop_roi=tuple(r["output_crop_roi"]) if r["output_crop_roi"] else None,
        image_area_roi=tuple(r["image_area_roi"]) if r["image_area_roi"] else None,
        base_sample_rois=[tuple(x) for x in r["base_sample_rois"]])


def process_roll_sheet(roll: str, frames: list[dict], out_dir: Path,
                       *, max_dim: int) -> dict:
    """整卷走**生产服务** `process_roll`（§0.3.4：联系表不得有第二条管线）。"""
    from negative_lab.library.models import FrameRecord

    out_dir.mkdir(parents=True, exist_ok=True)
    metrics, metrics_rows, decisions = [], [], []
    records, images, ctxs = [], [], []
    for fr in frames:
        p = ROOT / fr["file"]
        try:
            loaded = load_linear_rgb(p)
        except ReadError as e:
            metrics.append({"id": fr["id"], "status": "decode_error",
                            "error": str(e).splitlines()[-1]})
            continue
        arr = loaded.rgb
        step = max(1, int(np.ceil(max(arr.shape[:2]) / float(max_dim))))
        small = arr[::step, ::step, :].copy()
        fp = capture_fingerprint(loaded.meta, source_path=fr["file"])
        geo = _downscale_geometry(_geometry(fr), arr.shape[:2], small.shape[:2])
        if not geo.is_complete():
            geo = GeometryRegions(rotation_degrees=geo.rotation_degrees)
        rec = FrameRecord(source_path=fr["file"], source_sha256=fr["source_sha256"],
                          input_facts=dict(loaded.meta),
                          capture_fingerprint=fp.value, geometry=geo,
                          orientation_decided=_orientation_decided(fr))
        records.append(rec)
        images.append(small)
        ctxs.append({"fp": fp, "step": step})
    ra, results = process_roll(list(zip(records, images)))
    for fr, rec, small, pr, ctx in zip(frames, records, images, results, ctxs):
        cs = pr.candidates
        renders = {c.name: candidate_positive(pr, c.name) for c in cs} \
            if cs else {}
        default_render = renders.get(cs.default_name) if cs else \
            pr.rendered_positive_linear
        neg = _to_img(_norm_negative(small))
        nlp = None
        if fr.get("nlp_reference") and (ROOT / fr["nlp_reference"]).exists():
            nlp = Image.open(str(ROOT / fr["nlp_reference"])).convert("RGB")
        metrics_rows.append(_row(
            [("原始负片", neg),
             ("preserve", _to_img(_srgb(renders["preserve"])) if "preserve" in renders else None),
             ("neutral", _to_img(_srgb(renders["neutral"])) if "neutral" in renders else None),
             ("warm", _to_img(_srgb(renders["warm"])) if "warm" in renders else None),
             (f"默认 {pr.chosen_candidate}", _to_img(_srgb(default_render))
              if default_render is not None else None),
             ("NLP 默认参考", nlp),
             ("用户满意参考", None)],
            len(metrics_rows) + 1,
            f"{Path(fr['file']).name}  片基={pr.base_source}  "
            f"门禁={pr.gate.status}"))
        metrics.append({
            "id": fr["id"], "file": fr["file"], "roll": roll,
            "status": "ok", "gate": pr.gate.status,
            "blocks_export": pr.gate.blocks_export,
            "base_source": pr.base_source, "base_confidence": pr.base_confidence,
            "default_candidate": pr.chosen_candidate,
            "candidate_confidence": {c.name: c.confidence for c in cs} if cs else {},
            "neutral_fallback": cs.neutral.fallback_to if cs else None,
            "correction_capped": {c.name: c.correction_capped for c in cs} if cs else {},
            "technical_preclip": pr.gate.technical_preclip,
            "input_clip": pr.gate.input_clip,
            "chroma_stats": pr.gate.chroma_stats,
            "review_reasons": pr.gate.review_reasons,
            "roi_provenance": pr.roi_provenance,
            "render_intent": pr.render_intent.to_dict(),
            "working_step": ctx["step"],
        })
        decisions.append({
            "id": fr["id"], "file": fr["file"],
            "parity_required": fr["parity_required"],
            "required": fr["required"],
            "accepted": None, "chosen_candidate": None,
            "relative_to_nlp": None, "issue_tags": [], "note": "",
        })
    sheet = out_dir / "contact-sheet.jpg"
    _stack(metrics_rows, sheet)
    (out_dir / "decisions.json").write_text(json.dumps({
        "roll": roll, "spec": "v5 §10.2",
        "signed_by": None, "signed_date": None,
        "note": "只有用户有权填写最终视觉结论；执行 agent 不得代签。",
        "frames": decisions}, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps({
        "roll": roll, "roll_analysis": ra.to_dict(), "frames": metrics,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"roll": roll, "sheet": sheet.as_posix(),
            "frames": len(metrics),
            "failed": sum(1 for m in metrics if m.get("status") != "ok"),
            "needs_review": sum(1 for m in metrics
                                if m.get("gate") == "needs_review"),
            "blocked": sum(1 for m in metrics if m.get("blocks_export")),
            "roll_base_source": ra.base_source,
            "roll_split_suggested": ra.split_suggested}


def _srgb(pos: np.ndarray) -> np.ndarray:
    from negative_lab.pipeline.density import encode_srgb
    return encode_srgb(pos)


def _downscale_geometry(geo: GeometryRegions, src_shape, dst_shape) -> GeometryRegions:
    sy = dst_shape[0] / src_shape[0]
    sx = dst_shape[1] / src_shape[1]

    def sc(roi, min_side=16):
        if not roi:
            return None
        out = (int(round(roi[0] * sy)), int(round(roi[1] * sx)),
               int(round(roi[2] * sy)), int(round(roi[3] * sx)))
        return out if (out[2] - out[0] >= min_side
                       and out[3] - out[1] >= min_side) else None

    # **片基框的门槛必须比构图区低，且与生产同源**：未曝光片边是一条窄带
    # （实测 `1299_G200::G200-2` 源图 744×126），降到 900px 探针分辨率就变成
    # 74×12，用统一的 16px 规则会**静默丢掉用户亲手框的片基**，于是这一帧退回
    # `generic_prior`——量到的"用户标定没生效"是探针的错，不是产品的错
    # （当场实测：修前 base_sample_rois=[]、base_source=generic_prior/0.00；
    # 修后取到 dmin=[0.338,0.129,0.038]、base_source=frame/1.00）。
    # 4px 与 `service.py` 里 `clamp_roi(..., min_side=4)` 同一条规则，不再自创第三个门槛。
    return GeometryRegions(
        rotation_degrees=geo.rotation_degrees,
        fine_rotation_degrees=geo.fine_rotation_degrees,
        mirror_horizontal=geo.mirror_horizontal,
        output_crop_roi=sc(geo.output_crop_roi),
        image_area_roi=sc(geo.image_area_roi),
        base_sample_rois=[r for r in (sc(x, min_side=4)
                                      for x in geo.base_sample_rois) if r])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roll", default=None, help="只处理指定卷")
    ap.add_argument("--max-dim", type=int, default=1200,
                    help="联系表工作分辨率（最长边）；全分辨率验收以 GUI 导出为准")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--limit", type=int, default=None, help="每卷最多处理帧数（调试）")
    args = ap.parse_args()

    if not ACCEPTANCE.exists():
        print("先运行 scripts/build_acceptance_manifest.py", file=sys.stderr)
        return 2
    doc = json.loads(ACCEPTANCE.read_text(encoding="utf-8"))
    run_id = args.run_id or time.strftime("%Y-%m-%d-%H%M%S")
    out_root = ROOT / "outputs" / "quality" / run_id
    (out_root / "rolls").mkdir(parents=True, exist_ok=True)
    by_roll: dict[str, list[dict]] = {}
    for f in doc["frames"]:
        if args.roll and f["roll"] != args.roll:
            continue
        by_roll.setdefault(f["roll"], []).append(f)
    summaries = []
    for roll in sorted(by_roll):
        frames = by_roll[roll]
        if args.limit:
            frames = frames[:args.limit]
        print(f"== {roll}: {len(frames)} 帧", flush=True)
        t0 = time.perf_counter()
        try:
            s = process_roll_sheet(roll, frames, out_root / "rolls" / roll,
                                   max_dim=args.max_dim)
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL {type(e).__name__}: {e}", file=sys.stderr)
            s = {"roll": roll, "error": f"{type(e).__name__}: {e}"}
        s["seconds"] = round(time.perf_counter() - t0, 1)
        summaries.append(s)
        print("  " + json.dumps(s, ensure_ascii=False), flush=True)
    (out_root / "contact-sheet-summary.json").write_text(json.dumps({
        "run_id": run_id, "git_commit": git_commit(),
        "max_dim": args.max_dim,
        "note": "联系表为预览分辨率（最长边 max_dim）；逐张视觉结论由用户填写。",
        "rolls": summaries}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"输出目录：{out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
