"""为什么 74/169 帧退回通用先验？把片基估计器的现场原样打出来。

`fixtures/verdicts/metrics.json` 量到：所有 generic_prior 帧的
`contamination` **恰好等于 1.000**、`base_confidence` **恰好 0.000**。
这不是"片基被污染了 100%"，而是 `roll_analysis._base_stats()` 的
**两个提前返回**留下的初始值（`contamination=1.0, confidence=0.0`）：
"有效像素过少" 或 "没有可用片基区域"。也就是说 44% 的帧根本没有
量到片基，而报表把"没数据"和"全脏"写成同一个数字。

本脚本对指定帧把 `frame_base` 的全部字段与警告打出来，用来区分到底是
哪个提前返回、以及自动片基区域到底检出了什么。只读，不改任何行为。

运行：
    .venv\\Scripts\\python.exe scripts/diagnose_base_fallback.py [--roll ECN2_5219] [--limit 4]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from negative_lab.io.images import ReadError, load_linear_rgb  # noqa: E402
from negative_lab.library.fingerprint import capture_fingerprint  # noqa: E402
from negative_lab.library.models import GeometryRegions  # noqa: E402
from negative_lab.pipeline.service import process_image  # noqa: E402

from build_visual_contact_sheet import _downscale_geometry, _geometry  # noqa: E402

ACCEPTANCE = ROOT / "fixtures" / "acceptance" / "manifest.json"
METRICS = ROOT / "fixtures" / "verdicts" / "metrics.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--roll", default=None)
    ap.add_argument("--source", default="generic_prior",
                    help="按 metrics.json 的 base_source 挑帧（generic_prior/frame/all）")
    ap.add_argument("--limit", type=int, default=4)
    ap.add_argument("--max-dim", type=int, default=1200)
    a = ap.parse_args()

    manifest = {f["id"]: f for f in
                json.loads(ACCEPTANCE.read_text(encoding="utf-8"))["frames"]
                if f.get("required")}
    want = None
    if METRICS.exists():
        m = json.loads(METRICS.read_text(encoding="utf-8"))
        want = {f["id"]: f.get("base_source") for f in m["frames"]}

    picked = []
    for fid, fr in sorted(manifest.items()):
        if a.roll and not str(fr["roll"]).startswith(a.roll):
            continue
        if a.source != "all" and want is not None and want.get(fid) != a.source:
            continue
        picked.append((fid, fr))
        if len(picked) >= a.limit:
            break

    for fid, fr in picked:
        try:
            loaded = load_linear_rgb(ROOT / fr["file"])
        except ReadError as exc:
            print(f"{fid}: 读取失败 {exc}")
            continue
        arr = loaded.rgb
        step = max(1, int(np.ceil(max(arr.shape[:2]) / float(a.max_dim))))
        small = arr[::step, ::step, :].copy()
        geo = _downscale_geometry(_geometry(fr), arr.shape[:2], small.shape[:2])
        if not geo.is_complete():
            geo = GeometryRegions(rotation_degrees=geo.rotation_degrees)
        fp = capture_fingerprint(loaded.meta, source_path=fr["file"])
        pr = process_image(small, geometry=geo, ann=None, frame_id=fid,
                           capture_fingerprint=fp.value, make_candidates=False)
        fb = dict(pr.frame_base or {})
        print(f"\n=== {fid}  base_source={pr.base_source} conf={pr.base_confidence}")
        print(f"    shape={small.shape}  base_roi={getattr(geo, 'base_sample_rois', None)}")
        for k in ("n", "n_nonfinite", "kept_frac", "contamination", "confidence",
                  "median", "mad", "rel_spread", "black_level", "spatial_trend"):
            if k in fb:
                print(f"    {k} = {fb[k]}")
        for w in fb.get("warnings") or []:
            print(f"    ! {w}")
        for k in sorted(set(fb) - {"n", "n_nonfinite", "kept_frac", "contamination",
                                   "confidence", "median", "mad", "rel_spread",
                                   "black_level", "spatial_trend", "warnings", "dmin"}):
            print(f"    {k} = {fb[k]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
