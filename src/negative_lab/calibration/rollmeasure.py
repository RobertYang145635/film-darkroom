"""卷级测量（整改 v3 P0-C.2）。

对每一卷：片基空间曲面统计、密度分位、逐通道曲线采样、饱和/剪裁、
噪声、候选置信度 -> fixtures/measurements/<roll>.json。
测量只读样片；结果供标定（P0-E）与报告使用，不含任何单张万能参数。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..calibration.base import robust_dmin
from ..calibration.border import detect_regions
from ..calibration.flatfield import fit_base_surface

MEASUREMENTS_DIR = Path(__file__).resolve().parents[3] / "fixtures" / "measurements"


def measure_roll(roll: str, frames: list[tuple[str, np.ndarray]],
                 version: int = 1) -> dict:
    per_frame = []
    for name, t in frames:
        det = detect_regions(t)
        rec: dict = {"file": name}
        if det.get("base") is None:
            rec["status"] = "no-base"
            per_frame.append(rec)
            continue
        y0, x0, y1, x1 = det["base"]
        strip = t[y0:y1, x0:x1]
        b = robust_dmin(strip)
        rec["base_dmin"] = [round(float(v), 5) for v in b["dmin"]]
        rec["base_confidence"] = b["confidence"]
        # 噪声：条带内残差标准差（去均值后）
        rec["base_noise_sigma"] = [round(float(strip[..., c].std()), 5)
                                   for c in range(3)]
        fit = fit_base_surface(t, base_roi=det["base"])
        rec["flat_coverage"] = fit["coverage"]
        # 画面密度分位（z 域，相对片基）
        ia = det.get("image_area")
        if ia:
            img = t[ia[0]:ia[2], ia[1]:ia[3]]
            z = np.log10(np.maximum(img, 1e-6) /
                         np.maximum(np.array(b["dmin"]), 1e-6))
            zl = z.mean(axis=2)
            rec["z_quantiles"] = [round(float(q), 4)
                                  for q in np.quantile(zl, [0.001, 0.02, 0.5, 0.98])]
            rec["scene_above_base_frac"] = round(float((zl > 0.04).mean()), 5)
            # 饱和：接近传感器黑下限的像素（透射率 < 0.5% 基色）
            rec["saturated_frac"] = round(
                float((img < 0.005).mean()), 5)
        rec["status"] = "ok"
        per_frame.append(rec)
    ok = [r for r in per_frame if r["status"] == "ok"]
    dmaxes = [-r["z_quantiles"][0] for r in ok if "z_quantiles" in r]
    return {
        "schema_version": version,
        "roll": roll,
        "n_frames": len(frames),
        "n_measured": len(ok),
        "frames": per_frame,
        "roll_summary": {
            "dmax_p90": round(float(np.quantile(dmaxes, 0.9)), 4) if dmaxes else None,
            "dmax_median": round(float(np.median(dmaxes)), 4) if dmaxes else None,
            "base_dmin_median": [round(float(v), 5) for v in
                                 np.median([r["base_dmin"] for r in ok], axis=0)]
                                 if ok else None,
            "flat_coverage": sorted({r.get("flat_coverage", "?") for r in ok}),
        },
        "note": "测量值非标定：标定需 verified 中性参考（v3 P0-E）",
    }


def write_measurement(doc: dict) -> Path:
    MEASUREMENTS_DIR.mkdir(parents=True, exist_ok=True)
    p = MEASUREMENTS_DIR / f"{doc['roll']}.measure.json"
    p.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    return p
