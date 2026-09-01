"""色彩管线三路径 bake-off（v4 §11.3 重写）。

三路径都必须：先进入明确工作空间 -> 用同一技术反相 backend（invert()）
-> 在同一裁切上统计。禁止反相前 [0,1] 剪裁掩盖越界（越界进诊断）。

  A: sensor-linear（RAW，未乘矩阵/未 WB；矩阵在反相后渲染阶段才用——
     v4 §4.2 决定性发现：橙色片基在相机矩阵负瓣域外，反相前变换裁密度）
  B: sRGB-linear D65（LibRaw 内部 WB+相机矩阵，gamm=1；当前解码默认）
  C: ACES + 完整 AP1(D60)->sRGB(D65) Bradford 变换

输出 outputs/quality/<run-id>/：metrics.json + contact-sheet.jpg +
environment.json。A/B/C 无实际差异 -> 退出 1（防"名字不同计算相同"）。

运行：scripts/benchmark_color_pipeline.py [--file NEF]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import rawpy  # noqa: E402

from negative_lab.calibration.base import robust_dmin  # noqa: E402
from negative_lab.calibration.border import detect_regions  # noqa: E402
from negative_lab.pipeline.density import invert  # noqa: E402
from negative_lab.pipeline.params import InvertParams  # noqa: E402

XYZ_TO_SRGB_D65 = np.array([
    [3.2404542, -1.5371385, -0.4985314],
    [-0.9692660, 1.8760108, 0.0415560],
    [0.0556434, -0.2040259, 1.0572252]])
ACES_TO_SRGB_LINEAR = np.array([
    [1.7050510, -0.6217921, -0.0832582],
    [-0.1302568, 1.1408047, -0.0105479],
    [-0.0240033, -0.1289690, 1.1529723]])


def _decode(path: Path, output_color: int) -> tuple[np.ndarray, dict]:
    with rawpy.imread(str(path)) as r:
        params = rawpy.Params()
        params.use_camera_wb = False
        params.use_auto_wb = False
        params.output_color = output_color
        params.gamm = [1.0, 1.0, 1.0, 1.0]
        params.no_auto_bright = True
        params.output_bps = 16
        params.user_flip = -1
        arr = r.postprocess(params).astype(np.float32) / 65535.0
        info = {"wb_daylight": [float(v) for v in r.daylight_whitebalance],
                "rgb_xyz_matrix": np.asarray(r.rgb_xyz_matrix, float).tolist()}
    return arr, info   # 不做 [0,1] 剪裁：越界进诊断（v4 §11.3.3）


def path_a(path: Path) -> tuple[np.ndarray, dict]:
    raw, info = _decode(path, int(rawpy.ColorSpace.raw.value))
    return raw, {"space": "sensor-linear (pre-matrix, pre-WB)",
                 "matrix_applied": "none (post-inversion render stage)",
                 "oob_frac": float(((raw < 0) | (raw > 1)).mean())}


def path_b(path: Path) -> tuple[np.ndarray, dict]:
    t, info = _decode(path, int(rawpy.ColorSpace.sRGB.value))
    return t, {"space": "srgb-linear-d65 (LibRaw camera matrix + daylight WB)",
               "oob_frac": float(((t < 0) | (t > 1)).mean())}


def path_c(path: Path) -> tuple[np.ndarray, dict]:
    t, info = _decode(path, int(rawpy.ColorSpace.ACES.value))
    srgb = t @ ACES_TO_SRGB_LINEAR.T.astype(np.float32)
    return srgb, {"space": "ACES AP1 -> sRGB D65 (Bradford, full transform)",
                  "oob_frac": float(((srgb < 0) | (srgb > 1)).mean())}


def stats(t: np.ndarray, base_roi) -> dict:
    y0, x0, y1, x1 = base_roi
    dmin = robust_dmin(t[y0:y1, x0:x1])["dmin"]
    det = detect_regions(t)
    ia = det.get("image_area")
    reg = t[ia[0]:ia[2], ia[1]:ia[3]] if ia else t
    p = InvertParams(dmin=tuple(float(v) for v in dmin), dmax=2.0)
    res = invert(reg, p)
    pos = res.positive_linear
    lum = pos @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    mx, mn = pos.max(axis=2), pos.min(axis=2)
    chroma = (mx - mn) / np.maximum(pos.mean(axis=2), 1e-6)
    reg_lum = reg @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    sat = (reg.max(axis=2) - reg.min(axis=2)) / np.maximum(reg_lum, 1e-6)
    return {
        "base_dmin": [round(float(v), 5) for v in dmin],
        "lum_p50": round(float(np.quantile(lum, 0.5)), 4),
        "sat_p50_p90": [round(float(np.quantile(sat, q)), 4)
                        for q in (0.5, 0.9)],
        "chroma_p90": round(float(np.quantile(chroma, 0.9)), 4),
        "tech_clip_low": [round(float(v), 5) for v in (pos <= 1e-6).mean(axis=(0, 1))],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=None)
    ap.add_argument("--manifest", default=None)
    args = ap.parse_args()
    src = Path(args.file) if args.file else \
        ROOT / "20260709_SAMPLE-ID(1)/1298_G200/翻拍/G200-1.NEF"
    if not src.exists():
        print("样片不存在", src)
        return 1
    run_id = time.strftime("%Y%m%d-%H%M%S")
    outdir = ROOT / "outputs" / "quality" / run_id
    outdir.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True, cwd=ROOT
                            ).stdout.strip() or "unknown"
    ref = _decode(src, int(rawpy.ColorSpace.sRGB.value))[0]
    base_roi = detect_regions(ref)["base"]

    results = {}
    for name, fn in (("A-sensor-linear", path_a),
                     ("B-srgb-linear", path_b),
                     ("C-aces-full", path_c)):
        t0 = time.perf_counter()
        t, extra = fn(src)
        r = stats(t, base_roi)
        r["workspace"] = extra
        r["seconds"] = round(time.perf_counter() - t0, 1)
        results[name] = r
        print(name, json.dumps(r, ensure_ascii=False)[:180])

    # 差异断言（v4 §11.3.6：防"名字不同、计算相同"）
    keys = [("A-sensor-linear", "B-srgb-linear"),
            ("B-srgb-linear", "C-aces-full")]
    for a, b in keys:
        if results[a]["base_dmin"] == results[b]["base_dmin"]:
            print(f"[FAIL] 路径 {a} 与 {b} 无实际差异")
            return 1

    doc = {"run_id": run_id, "git_commit": commit,
           "source": src.as_posix(),
           "source_sha256": __import__("hashlib").sha256(
               src.read_bytes()).hexdigest(),
           "environment": {"python": sys.version.split()[0],
                           "rawpy": rawpy.__version__,
                           "LibRaw": rawpy.libraw_version,
                           "platform": sys.platform},
           "note": "NLP 非真值；无色卡：DeltaE 未测量（如实缺省）",
           "paths": results}
    (outdir / "metrics.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    (outdir / "environment.json").write_text(
        json.dumps(doc["environment"], ensure_ascii=False, indent=2),
        encoding="utf-8")
    print("写入", outdir / "metrics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
