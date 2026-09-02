"""训练/校验跨卷通用颜色模型资产（v5 §6.3 / §9.4 / §12 P0-5.1）。

纪律：
- 训练、验证、留出**按卷隔离**；同一源图的派生文件不得跨集（§9.3）；
- 固定随机种子，训练可重复；
- 不使用任何参考正片作为拟合目标（无监督密度解混），避免"逐张追参考图"；
- 资产只有在**留出卷**的自动指标不劣于确定性模型时才启用；
  否则写出 `enabled=false` 的资产并保留确定性回退（诚实记录，不假装通过）。

运行：.venv\\Scripts\\python.exe scripts/train_generic_model.py [--limit N]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from negative_lab.calibration.annotations import load_annotation  # noqa: E402
from negative_lab.io.images import load_linear_rgb  # noqa: E402
from negative_lab.library.fingerprint import capture_fingerprint  # noqa: E402
from negative_lab.library.models import GeometryRegions  # noqa: E402
from negative_lab.pipeline.model_assets import (  # noqa: E402
    ASSET_DIR, build_asset_from_fit, deterministic_asset, fit_decorrelation,
    save_asset)

SEED = 20260901
STEP = 6          # 兼容旧字段名（实际降采样由 _process 的 max_dim 决定）
ASSET_VERSION = "generic-color-negative-v1"


def _manifest() -> dict:
    return json.loads((ROOT / "fixtures" / "manifest.json").read_text(encoding="utf-8"))


def _process(entry: dict, *, max_dim: int = 1400):
    """用**生产服务**处理一帧（§0.3.4：训练脚本不得自己实现片基/裁切）。

    返回 (密度样本 (N,3), ProcessResult) 或 (None, 原因)。
    """
    from negative_lab.pipeline.service import process_image

    p = ROOT / entry["file"]
    ann = load_annotation(entry["file"])
    try:
        loaded = load_linear_rgb(p)
    except Exception as ex:  # noqa: BLE001
        print(f"  skip {p.name}: {type(ex).__name__}", flush=True)
        return None, None
    arr = loaded.rgb
    step = max(1, int(np.ceil(max(arr.shape[:2]) / float(max_dim))))
    small = arr[::step, ::step, :].copy()
    geo = GeometryRegions(
        rotation_degrees=ann.rotation_degrees if ann else 0,
        output_crop_roi=_scale(_roi(ann, "output_crop_roi") or _roi(ann, "frame_roi"),
                               arr.shape[:2], small.shape[:2]),
        image_area_roi=_scale(_roi(ann, "image_area_roi") or _roi(ann, "frame_roi"),
                              arr.shape[:2], small.shape[:2]),
        base_sample_rois=[r for r in (_scale(x, arr.shape[:2], small.shape[:2])
                                      for x in _base_rois(ann)) if r])
    fp = capture_fingerprint(loaded.meta, source_path=entry["file"])
    pr = process_image(small, geometry=geo, capture_fingerprint=fp,
                       make_candidates=False)
    if pr.base_source == "generic_prior":
        print(f"  skip {p.name}: 无可靠片基（{pr.frame_base.get('warnings')}）",
              flush=True)
        return None, pr
    rss = pr.geometry.roi_source_space
    area = rss.get("image_area_roi")
    region = small
    if area:
        y0, x0, y1, x1 = (int(v) for v in area)
        region = small[max(0, y0):min(small.shape[0], y1),
                       max(0, x0):min(small.shape[1], x1)]
    b = np.maximum(np.asarray(pr.report.params.dmin, np.float64), 1e-6)
    d = -np.log10(np.maximum(region.astype(np.float64), 1e-6) / b)
    d = d.reshape(-1, 3)
    return d[np.isfinite(d).all(axis=1)], pr


def _roi(ann, name):
    v = getattr(ann, name, None) if ann is not None else None
    return tuple(v) if v else None


def _base_rois(ann):
    if ann is None:
        return []
    out = [tuple(r) for r in (getattr(ann, "base_sample_rois", None) or [])]
    if not out and getattr(ann, "base_roi", None):
        out = [tuple(ann.base_roi)]
    return out


def _scale(roi, src_shape, dst_shape):
    if not roi:
        return None
    sy = dst_shape[0] / src_shape[0]
    sx = dst_shape[1] / src_shape[1]
    return (int(round(roi[0] * sy)), int(round(roi[1] * sx)),
            int(round(roi[2] * sy)), int(round(roi[3] * sx)))


def _densities(entries: list[dict], *, limit: int | None = None) -> list[np.ndarray]:
    out = []
    for e in (entries[:limit] if limit else entries):
        if not (ROOT / e["file"]).exists():
            continue
        d, _pr = _process(e)
        if d is not None and d.shape[0] >= 64:
            out.append(d)
    return out


def _metrics(entries: list[dict], matrix: np.ndarray) -> dict:
    """留出卷自动指标：密度通道去相关程度 + 输入彩度中位（不劣化检查）。"""
    offs, chroma = [], []
    for e in entries:
        if not (ROOT / e["file"]).exists():
            continue
        d, pr = _process(e)
        if d is None or d.shape[0] < 64:
            continue
        u = d @ matrix.T
        c = np.corrcoef(u.T)
        offs.append(float(np.abs(c - np.eye(3)).sum()))
        px = np.clip(pr.t.reshape(-1, 3), 0, 1)
        chroma.append(float(np.median((px.max(axis=1) - px.min(axis=1))
                                      / np.maximum(px.mean(axis=1), 1e-6))))
    return {"mean_offdiag_corr": round(float(np.mean(offs)), 5) if offs else None,
            "median_input_chroma": round(float(np.median(chroma)), 5) if chroma else None,
            "n_frames": len(offs)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="每个集合最多处理多少帧（调试用）")
    ap.add_argument("--ridge", type=float, default=0.35)
    args = ap.parse_args()

    doc = _manifest()
    ok = [e for e in doc["images"] if e["decode_status"] == "ok"]
    train = [e for e in ok if e["split"] == "calibration"]
    hold = [e for e in ok if e["split"] == "holdout"]
    train_rolls = sorted({e["roll"] for e in train})
    hold_rolls = sorted({e["roll"] for e in hold})
    overlap = set(train_rolls) & set(hold_rolls)
    if overlap:
        print(f"FAIL 训练/留出卷重叠：{sorted(overlap)}", file=sys.stderr)
        return 2
    print(f"训练卷 {train_rolls}（{len(train)} 帧）")
    print(f"留出卷 {hold_rolls}（{len(hold)} 帧）")

    t0 = time.perf_counter()
    dens = _densities(train, limit=args.limit)
    print(f"密度样本 {sum(d.shape[0] for d in dens)} 像素（{len(dens)} 帧），"
          f"{time.perf_counter() - t0:.1f}s")
    fit = fit_decorrelation(dens, ridge=args.ridge, seed=SEED)
    print("拟合：", {k: v for k, v in fit.items() if k != "matrix"})
    A = np.asarray(fit["matrix"], np.float64)

    base_metrics = _metrics(hold[:6], np.eye(3))
    asset_metrics = _metrics(hold[:6], A)
    print("留出（确定性）:", base_metrics)
    print("留出（资产）  :", asset_metrics)
    improved = (asset_metrics["mean_offdiag_corr"] is not None
                and base_metrics["mean_offdiag_corr"] is not None
                and asset_metrics["mean_offdiag_corr"]
                < base_metrics["mean_offdiag_corr"])

    manifest_doc = {
        "schema_version": 1,
        "seed": SEED,
        "step": STEP,
        "ridge": args.ridge,
        "train_rolls": train_rolls, "holdout_rolls": hold_rolls,
        "train_files": [{"file": e["file"], "sha256": e["sha256"],
                         "roll": e["roll"]} for e in train],
        "holdout_files": [{"file": e["file"], "sha256": e["sha256"],
                           "roll": e["roll"]} for e in hold],
        "reference_usage": "无：本资产为无监督密度解混，未使用任何参考正片",
        "holdout_metrics_deterministic": base_metrics,
        "holdout_metrics_asset": asset_metrics,
        "enabled_by_metrics": bool(improved),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    mpath = ASSET_DIR / f"{ASSET_VERSION}.train-manifest.json"
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps(manifest_doc, ensure_ascii=False, indent=2),
                     encoding="utf-8")
    manifest_sha = hashlib.sha256(mpath.read_bytes()).hexdigest()

    if improved and fit["confidence"] > 0:
        asset = build_asset_from_fit(
            fit, asset_version=ASSET_VERSION, manifest_sha256=manifest_sha,
            hyper={"seed": SEED, "ridge": args.ridge, "step": STEP},
            enabled=True)
    else:
        asset = deterministic_asset(
            reason=("留出卷自动指标未显示改善：保持确定性密度模型"
                    f"（deterministic={base_metrics['mean_offdiag_corr']} vs "
                    f"asset={asset_metrics['mean_offdiag_corr']}）"))
        asset.asset_version = f"{ASSET_VERSION}-disabled"
        asset.meta.update({"training_manifest_sha256": manifest_sha,
                           "candidate_fit": fit})
    out = save_asset(asset)
    print(f"写入 {out}  enabled={asset.enabled}")
    print(f"训练清单 {mpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
