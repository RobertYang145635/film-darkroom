"""真实样片批量分析（Phase 2 验证 + §8.3 学习产物 analysis-report.json）。

对每卷抽样（默认每卷前 N 张）：rawpy 解码 -> 线性化 -> 自动分析 -> 指标汇总。
原始文件只读；输出 docs/real-samples-analysis.md 与
fixtures/provided/analysis-report.json。

运行：.venv\\Scripts\\python.exe scripts/analyze_real_samples.py [--per-roll 4]
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

import rawpy  # noqa: E402

from negative_lab import __version__  # noqa: E402
from negative_lab.pipeline.auto import auto_analyze  # noqa: E402
from negative_lab.pipeline.density import invert  # noqa: E402

SAMPLE_ROOT = ROOT / "20260709_SAMPLE-ID(1)"


def decode_nef(path: Path) -> tuple[np.ndarray, dict]:
    """NEF -> 归一化线性 RGB float32 [0,1]。

    翻拍负片的透射率正比于到达传感器的光量，即 RAW 线性像素值；
    黑电平扣除 + 白电平归一；不使用相机白平衡（那是显示端增益，
    会污染通道密度测量）。
    """
    with rawpy.imread(str(path)) as r:
        lens, other, sizes = r.lens, r.other, r.sizes
        params = rawpy.Params(use_camera_wb=False, no_auto_bright=True,
                              half_size=False, output_bps=16)
        params.gamm = [1.0, 1.0, 1.0, 1.0]  # 线性输出（rawpy 0.27 属性名）
        rgb = r.postprocess(params).astype(np.float32) / 65535.0
        meta = {"make": lens.make or "unknown", "lens": lens.model,
                "iso": float(other.iso_speed), "aperture": float(other.aperture),
                "shutter": float(other.shutter_speed),
                "width": int(sizes.width), "height": int(sizes.height),
                "black": [int(v) for v in r.black_level_per_channel],
                "white": int(r.white_level)}
    rgb = np.clip(rgb.astype(np.float32), 0.0, None)
    mx = float(np.nanmax(rgb)) if rgb.size else 1.0
    if mx > 1.0:
        rgb = rgb / mx
    return rgb, meta


def analyze(path: Path) -> dict:
    t0 = time.perf_counter()
    t, meta = decode_nef(path)
    rep = auto_analyze(t)  # 外框启发式片基 ROI
    res = invert(t, rep.params)
    lum = res.positive_linear.mean(axis=2)
    dt = time.perf_counter() - t0
    return {
        "file": path.relative_to(ROOT).as_posix(),
        "camera": f"{meta['make']} {meta['lens']}".strip(),
        "exposure": {"iso": meta["iso"], "aperture": meta["aperture"],
                     "shutter": meta["shutter"]},
        "shape": list(t.shape),
        "seconds": round(dt, 2),
        "params": rep.params.to_parameters_dict(),
        "confidence": rep.confidence,
        "warnings": sorted(set(rep.warnings)),
        "diagnostics": rep.diagnostics,
        "output_mean": float(lum.mean()),
        "output_p99": float(np.quantile(lum, 0.99)),
        "input_t_p01": float(np.quantile(t, 0.01)),
        "input_t_p99": float(np.quantile(t, 0.99)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-roll", type=int, default=4)
    args = ap.parse_args()

    rolls = sorted([d for d in SAMPLE_ROOT.iterdir() if d.is_dir()])
    results = []
    for roll in rolls:
        files = sorted(roll.rglob("*.NEF"))[: args.per_roll]
        for f in files:
            print(f"analyzing {f.name} ({roll.name}) ...", flush=True)
            try:
                results.append(analyze(f))
            except Exception as e:  # noqa: BLE001 —— 单张失败不影响其余
                results.append({"file": f.relative_to(ROOT).as_posix(),
                                "error": f"{type(e).__name__}: {e}"})

    doc = {"software_version": __version__,
           "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "note": "外框启发式片基 ROI；无中性参考点；参数为自动初值，仅供标定参考",
           "images": results}
    out_json = ROOT / "fixtures" / "provided" / "analysis-report.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# 真实样片自动分析报告（Phase 2 验证）", "",
             f"每卷前 {args.per_roll} 张；软件 v{__version__}；"
             "片基 ROI=外框启发式；无中性参考点（§4.6.1 兜底路径）。", "",
             "| 文件 | 相机 | Dmin(R/G/B) | Dmax | offset | 置信dmin | 输出均值 | 输入t p99 | 警告 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        if "error" in r:
            lines.append(f"| {r['file']} | — | — | — | — | — | — | — | ERROR: {r['error']} |")
            continue
        p = r["params"]
        warn = "; ".join(w[:40] for w in r["warnings"][:2])
        lines.append(
            f"| {Path(r['file']).name} ({r['file'].split('/')[1]}) | {r['camera']} | "
            f"{p['dmin'][0]:.3f}/{p['dmin'][1]:.3f}/{p['dmin'][2]:.3f} | "
            f"{p['dmax']:.2f} | {p['offset']:.2f} | {r['confidence'].get('dmin', 0)} | "
            f"{r['output_mean']:.3f} | {r['input_t_p99']:.3f} | {warn} |")
    out_md = ROOT / "docs" / "real-samples-analysis.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n写入 {out_json}\n写入 {out_md}")
    n_err = sum(1 for r in results if "error" in r)
    print(f"完成：{len(results)} 张，失败 {n_err} 张")
    return 1 if n_err else 0


if __name__ == "__main__":
    raise SystemExit(main())
