"""从标定卷学习胶片/设备先验（整改 P0-C.1；留出卷绝不参与）。

产物：fixtures/priors/<stock>.profile.json（FilmProfile，version 递增）。
方法：
- Dmax 先验：每帧画面区域内 z_lum 的 0.1% 分位（内容最大密度），
  卷级取 p90（多数帧可达到的密度上界）——逐帧内容分位会低估胶片能力
  （亮场景帧估 1.15 vs 实际 ~2.0，"冲淡"根因）。
- 幂指数比例先验：各帧统计中性候选的 red/blue ratio 中位数。
- 记录样本清单与方法版本，供 docs/sample-calibration.md 引用。

运行：.venv\\Scripts\\python.exe scripts/learn_priors.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from negative_lab.calibration.base import robust_dmin  # noqa: E402
from negative_lab.calibration.border import detect_regions  # noqa: E402
from negative_lab.calibration.neutral import estimate_exponent_ratios  # noqa: E402
from negative_lab.io.images import load_linear_rgb  # noqa: E402
from negative_lab.pipeline.density import transmittance_to_z  # noqa: E402
from negative_lab.pipeline.neutral_stats import find_neutral_z_samples  # noqa: E402
from negative_lab.pipeline.params import InvertParams  # noqa: E402
from negative_lab.profiles.profile import FilmProfile, save_profile  # noqa: E402

MANIFEST = ROOT / "fixtures" / "manifest.json"
PRIORS_DIR = ROOT / "fixtures" / "priors"
MAX_FRAMES_PER_ROLL = 8
SCALE = 6


def learn_for_files(files: list[Path]) -> dict:
    dmaxes, reds, blues = [], [], []
    for f in files:
        try:
            t = load_linear_rgb(f).rgb[::SCALE, ::SCALE, :].copy()
        except Exception as e:  # noqa: BLE001
            print(f"  skip {f.name}: {e}")
            continue
        det = detect_regions(t)
        if not det.get("base") or not det.get("image_area"):
            print(f"  skip {f.name}: 检测失败")
            continue
        by0, bx0, by1, bx1 = det["base"]
        dmin_res = robust_dmin(t[by0:by1, bx0:bx1])
        if dmin_res["confidence"] < 0.5:
            continue
        dmin = dmin_res["dmin"]
        iy0, ix0, iy1, ix1 = det["image_area"]
        img = t[iy0:iy1, ix0:ix1]
        z = transmittance_to_z(img, dmin)
        zl = z.mean(axis=2)
        valid = (img > 1e-5).all(axis=-1)
        if valid.mean() < 0.5:
            continue
        dmaxes.append(float(-np.quantile(zl[valid], 0.001)))
        p = InvertParams(dmin=tuple(float(v) for v in dmin),
                         dmax=max(np.percentile(dmaxes, 50), 0.5))
        ns = find_neutral_z_samples(img, p)
        if len(ns["samples"]) >= 2:
            est = estimate_exponent_ratios(ns["samples"], weights=ns["weights"])
            reds.append(est["red_ratio"])
            blues.append(est["blue_ratio"])
    out = {"dmax_p90": float(np.percentile(dmaxes, 90)) if dmaxes else None,
           "dmax_frames": len(dmaxes),
           "red_ratio_med": float(np.median(reds)) if reds else None,
           "blue_ratio_med": float(np.median(blues)) if blues else None,
           "n_neutral_frames": len(reds)}
    return out


def main() -> int:
    doc = json.loads(MANIFEST.read_text(encoding="utf-8"))
    calib = [e for e in doc["images"]
             if e["split"] == "calibration" and e["decode_status"] == "ok"]
    by_group: dict[str, list] = {}
    for e in calib:
        key = e["film_stock"] if e["film_stock"] != "unknown" else e["roll"]
        by_group.setdefault(key, []).append(ROOT / e["file"])

    PRIORS_DIR.mkdir(parents=True, exist_ok=True)
    report = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "method": "dmax=p90 of per-frame -z(0.1%) in image area; "
                        "ratios=median of statistical-neutral estimates",
              "groups": {}}
    for key, files in sorted(by_group.items()):
        use = files[:MAX_FRAMES_PER_ROLL]
        print(f"{key}: {len(use)} 帧（标定卷内）")
        stats = learn_for_files(use)
        report["groups"][key] = {"stats": stats,
                                 "files": [f.name for f in use]}
        if stats["dmax_p90"] is None:
            continue
        prof = FilmProfile(
            name=f"{key.replace(' ', ' ')}-Z8-led".replace(" ", ""),
            film_type=("c41_color" if "Gold" in key else "unknown"),
            device="NIKON Z 8", light_source="unknown",
            tone_prior={"dmax": round(stats["dmax_p90"], 3)},
            exponent_prior=({"reference_exponent": 1.0,
                             "red_ratio": round(stats["red_ratio_med"], 4),
                             "blue_ratio": round(stats["blue_ratio_med"], 4)}
                            if stats["red_ratio_med"] else None),
            notes=f"P0-C 先验：dmax p90 over {stats['dmax_frames']} calibration "
                  f"frames; neutral frames={stats['n_neutral_frames']}. "
                  f"holdout 未参与。method={report['method']}",
        )
        out = save_profile(prof, PRIORS_DIR)
        print(f"  -> {out}  dmax={stats['dmax_p90']:.2f} "
              f"ratios={stats['red_ratio_med']},{stats['blue_ratio_med']}")
    (PRIORS_DIR / "learn-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("写入", PRIORS_DIR / "learn-report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
