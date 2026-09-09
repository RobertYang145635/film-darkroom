"""运行卷级测量（v3 P0-C.2）：标定卷 + ECN-2 转换卷。"""
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from negative_lab.calibration.rollmeasure import measure_roll, write_measurement  # noqa: E402
from negative_lab.io.images import load_linear_rgb  # noqa: E402

import json  # noqa: E402

man = json.loads((ROOT / "fixtures" / "manifest.json").read_text(encoding="utf-8"))
by_roll: dict[str, list] = {}
for e in man["images"]:
    if e["decode_status"] == "ok":
        by_roll.setdefault(e["roll"], []).append(e["file"])

# ECN-2 转换卷（用户 2026-09 提供）：抽样 6 张（全 72 张太慢，测量代表帧）
ecn2 = sorted((ROOT / "20260506_SAMPLE-ID" / "ECN2").glob("*.tif"))
if ecn2:
    by_roll["ECN2-converted"] = [
        str(p.relative_to(ROOT)).replace("\\", "/")
        for p in ecn2[::max(1, len(ecn2) // 6)][:6]]

for roll, files in sorted(by_roll.items()):
    frames = []
    for f in files[:12]:
        p = ROOT / f
        if not p.exists():
            continue
        try:
            t = load_linear_rgb(p).rgb      # TIFF 走 classify_tiff（含 ICC）
            frames.append((f, t[::4, ::4, :].copy()))
        except Exception as e:  # noqa: BLE001
            print("  skip", f, type(e).__name__, str(e)[:80])
    if not frames:
        continue
    doc = measure_roll(roll, frames)
    path = write_measurement(doc)
    s = doc["roll_summary"]
    print(f"{roll}: {doc['n_measured']}/{doc['n_frames']} 帧  "
          f"dmax_med={s['dmax_median']}  base={s['base_dmin_median']}  "
          f"-> {path.name}")
