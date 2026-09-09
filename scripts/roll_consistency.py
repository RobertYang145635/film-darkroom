"""去色罩算法研究 R3：真实样卷一致性基线（无真值可算的数字）。

对给定卷跑完整默认管线（自动 ROI + 卷级片基 + 默认候选），输出：
- 每帧：渲染正片中位色偏 (a*,b*)、中位 L*、场景置信度、
  候选诊断（估计器/置信度）、使用的片基来源；
- 卷级：两两 Lab ΔE2000 矩阵、均值/p95、离群帧（色偏中心距卷中位 > 4）。

用途：给"同卷一致性"第一个数字基线；后续任何算法改动在此上复测。
运行：.venv\\Scripts\\python.exe scripts/roll_consistency.py <卷目录> [--max N]
输出：outputs/research/roll_consistency/<卷名>.json + 汇总 markdown
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402


def _jsonable(obj):
    """递归把 NaN/Inf 变成 null，保证产出是合法 JSON（RFC 8259）。"""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if np.isfinite(f) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    max_frames = 12
    if "--max" in sys.argv:
        i = sys.argv.index("--max")
        max_frames = int(sys.argv[i + 1])
    folder = Path(args[0]) if args else (
        ROOT / "20260709_SAMPLE-ID(1)" / "1298_G200" / "翻拍")
    # 相对路径统一按仓库根解析（此前依赖调用者的 cwd）
    if not folder.is_absolute() and not folder.exists():
        folder = ROOT / folder
    if not folder.exists():
        print(f"卷目录不存在：{folder}")
        return 1
    out_dir: Path | None = None
    if "--out" in sys.argv:
        i = sys.argv.index("--out")
        out_dir = Path(sys.argv[i + 1])
    roll_override = None
    if "--roll" in sys.argv:
        i = sys.argv.index("--roll")
        roll_override = sys.argv[i + 1]

    from negative_lab.io.images import load_linear_rgb
    from negative_lab.pipeline.candidates import (estimate_illuminants,
                                                  scene_confidence)
    from negative_lab.diagnostics.consistency import roll_consistency
    from negative_lab.pipeline.render import apply_render, RenderIntent
    from negative_lab.pipeline.service import process_image
    from negative_lab.library.session import iter_supported_files

    files = sorted(iter_supported_files(folder))[:max_frames]
    if not files:
        print("卷内没有支持的文件")
        return 1
    # 卷名：--roll 优先；否则取父目录（样片常放在 <卷>/翻拍/ 下）。
    # 曾产出过 roll="翻拍" 的重复文件（审查 §2.3），故显式可覆盖并记录 roll_dir。
    roll_name = roll_override or (folder.parent.name if folder.parent.name
                                  not in ("", ".") else folder.name)
    print(f"卷：{roll_name}（{folder}），{len(files)} 帧（max={max_frames}）")

    rendered: list[np.ndarray] = []
    frames_meta: list[dict] = []
    t0 = time.time()
    for i, f in enumerate(files):
        try:
            t0f = time.time()
            loaded = load_linear_rgb(f)
            pr = process_image(loaded.rgb)
            pos = pr.result.positive_linear
            # 渲染默认候选（preserve = 恒等；取 sidecar 口径的默认候选）
            chosen = pr.chosen_candidate or "preserve"
            intent = RenderIntent.from_dict(
                pr.candidates.get(chosen).intent) if pr.candidates else RenderIntent()
            img = np.asarray(apply_render(pos, intent))
            # 只统计输出裁切内的画面（构图不同会污染 ΔE）
            crop = pr.geometry.roi_source_space.get("output_crop_roi")
            if crop:
                y0, x0, y1, x1 = (int(v) for v in crop)
                h, w = img.shape[:2]
                img = img[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
            rendered.append(img)
            est = estimate_illuminants(pos, getattr(pr, "valid_mask", None),
                                       getattr(pr, "t", None))
            conf, stats = scene_confidence(pos, getattr(pr, "valid_mask", None),
                                           est)
            dmin_used = [round(float(v), 6)
                         for v in pr.report.params.dmin]
            frames_meta.append({
                "file": f.name,
                "base_source": pr.base_source,
                "base_dmin_used": dmin_used,
                # v6.2 审查 §3.6：逐帧**实际被施加的量**必须落盘。
                # 片基色度漂移量的是"胶片和冲洗是否一致"，而这几项量的是
                # "我们把同卷各帧处理得是否一致"——后者才是产品要背的指标。
                "applied": {
                    "dmax": round(float(getattr(pr.report.params, "dmax", 0.0)), 6),
                    "temperature": round(float(getattr(intent, "temperature", 0.0)), 3),
                    "tint": round(float(getattr(intent, "tint", 0.0)), 3),
                    "flat_mode": getattr(pr.report, "flat_mode", None),
                },
                "candidate_default": chosen,
                "scene_confidence": round(float(conf), 3),
                "neutral_frac": stats.get("neutral_frac"),
                "median_ab": None,           # 由 roll_consistency 填
                "shape": [int(img.shape[0]), int(img.shape[1])],
                "sec_per_frame": round(time.time() - t0f, 1),
            })
            print(f"  [{i + 1}/{len(files)}] {f.name} "
                  f"({frames_meta[-1]['sec_per_frame']}s, "
                  f"conf={conf:.2f}, base={pr.base_source})", flush=True)
        except Exception as e:  # noqa: BLE001 —— 单帧失败不拖垮整卷统计
            frames_meta.append({"file": f.name, "error": f"{type(e).__name__}: {e}"})
            print(f"  [{i + 1}/{len(files)}] {f.name} FAILED: {e}", flush=True)

    ok = [i for i, m in enumerate(frames_meta) if "error" not in m]
    rc = roll_consistency([rendered[i] for i in ok])
    for k, i in enumerate(ok):
        frames_meta[i]["median_ab"] = rc["median_ab"][k]

    # 场景无关锚点：片基 Dmin 色度散布（同卷同冲洗片基色度应恒定）。
    # log10(通道/亮度均值) 的逐帧极差 = 卷内片基色度漂移（decade）。
    # 口径：只统计实测片基帧（base_source != generic_prior）——通用先验
    # 的形状是资产常量，混进来会伪造"片基漂移"（CN400 实测：3 帧
    # generic_prior 把 max_abs_drift 从 0.03 抬到 0.18）。
    dmins = [np.asarray(m["base_dmin_used"], np.float64)
             for m in frames_meta
             if m.get("base_dmin_used")
             and m.get("base_source") != "generic_prior"]
    base_chroma = None
    if len(dmins) >= 2:
        C = np.stack([np.log10(np.maximum(d / max(d.mean(), 1e-9), 1e-9))
                      for d in dmins])
        base_chroma = {
            "per_frame": [[round(float(v), 4) for v in row] for row in C],
            "max_abs_drift": round(float(np.max(C.max(axis=0) - C.min(axis=0))), 4),
            "spread_std": [round(float(v), 4) for v in C.std(axis=0)],
            "n_with_base": len(dmins),
            "excluded_generic_prior": sum(
                1 for m in frames_meta
                if m.get("base_source") == "generic_prior"),
            "note": ("片基色度（log10 通道/均值，decade）是场景无关的处理一致性"
                     "锚点：全图 ΔE 被场景内容主导时，只有这里能回答"
                     "「同卷处理是否一致」。只统计实测片基帧"
                     "（generic_prior 形状是资产常量，混入会伪造漂移）"),
        }
    # 逐帧施加量的散布（与场景无关）：>0 就说明我们**有意**对同卷帧做了不同处理
    applied_rows = [m["applied"] for m in frames_meta if m.get("applied")]
    applied_consistency = None
    if len(applied_rows) >= 2:
        def _spread(key):
            vals = [float(r[key]) for r in applied_rows
                    if isinstance(r.get(key), (int, float))]
            if len(vals) < 2:
                return None
            return {"min": round(min(vals), 6), "max": round(max(vals), 6),
                    "spread": round(max(vals) - min(vals), 6)}
        ident = [r.get("dmax") for r in applied_rows]
        applied_consistency = {
            "n_frames": len(applied_rows),
            "dmax": _spread("dmax"),
            "temperature": _spread("temperature"),
            "tint": _spread("tint"),
            "identical_dmax_frames": len(ident) - len(set(ident)),
            "note": ("同卷各帧实际施加的增益/渲染意图散布。全为 0 = 整卷用同一套参数处理；"
                     "非 0 = 逐帧自适应的幅度，是「卷内一致性」最直接的可控项"
                     "（场景内容不参与）"),
        }
    named_outliers = [
        {"index": int(i),
         "file": (frames_meta[ok[i]]["file"] if i < len(ok) else None),
         "score": (round(float(rc["outlier_score"][i]), 3)
                   if i < len(rc.get("outlier_score") or []) else None)}
        for i in (rc.get("outliers") or [])]
    summary = {
        "roll": roll_name,
        "roll_dir": str(folder),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_frames": rc["n_frames"],
        "n_pairs": rc["n_pairs"],
        "de_mean": rc["de_mean"],
        "de_p50": rc["de_p50"],
        "de_p95": rc["de_p95"],
        "median_ab": rc["median_ab"],
        "outlier_score": rc["outlier_score"],
        "outliers": named_outliers,
        "de_matrix": rc["de_matrix"],
        "base_chroma": base_chroma,
        "applied_consistency": applied_consistency,
        "frames": frames_meta,
        "total_sec": round(time.time() - t0, 1),
        "method": ("全默认管线（自动 ROI + 卷级片基 + 默认候选渲染）；"
                   "输出裁切内全画面 Lab ΔE2000 两两统计；"
                   "场景内容差异会计入 ΔE（局限见研究笔记）"),
    }
    if out_dir is None:
        out_dir = ROOT / "outputs" / "research" / "roll_consistency"
    out_dir.mkdir(parents=True, exist_ok=True)
    # 审查 §2.3：此前直接 json.dumps 会写出裸 NaN，不是合法 JSON，
    # 任何严格解析器（含前端）都会炸。现在非有限值一律变 null。
    (out_dir / f"{roll_name}.json").write_text(
        json.dumps(_jsonable(summary), ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(json.dumps({k: summary[k] for k in
                      ("roll", "n_frames", "de_mean", "de_p50", "de_p95",
                       "outliers", "total_sec")}, ensure_ascii=False))
    print(f"写入 {out_dir / (roll_name + '.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
