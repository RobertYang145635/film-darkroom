"""色彩管线三路径 bake-off（v3 P0-A.4）。

同一 NEF 上对照：
  A: rawpy.ColorSpace.raw（传感器线性）+ 本项目自行应用 daylight WB +
     LibRaw rgb_xyz_matrix -> XYZ -> sRGB 矩阵（RawTherapee 类路径）
  B: rawpy.ColorSpace.sRGB + gamm=1（当前默认：LibRaw 内部完成 WB+矩阵）
  C: rawpy.ColorSpace.ACES + 完整 AP1(D60)->sRGB(D65) 变换（Bradford 色适应，
     公开 Academy 矩阵）——禁止只加 sRGB TRC 的假变换

记录：片基 RGB、有效画面逐通道输入剪裁、中性候选残差（仅诊断，非标定）、
亮度/色度分位、耗时。输出 outputs/benchmarks/<run-id>/。

运行：.venv\\Scripts\\python.exe scripts/benchmark_color_pipeline.py
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
from negative_lab.calibration.neutral_stats import find_neutral_z_samples  # noqa: E402
from negative_lab.pipeline.density import transmittance_to_z  # noqa: E402
from negative_lab.pipeline.params import InvertParams  # noqa: E402

# 公开常数（Academy ACES 技术文档）
XYZ_TO_SRGB_D65 = np.array([
    [3.2404542, -1.5371385, -0.4985314],
    [-0.9692660, 1.8760108, 0.0415560],
    [0.0556434, -0.2040259, 1.0572252]])
# ACES(AP1, D60) -> sRGB(Rec709, D65) 复合矩阵（含 Bradford D60->D65）
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
    return np.clip(arr, 0.0, None), info


def path_a(path: Path) -> tuple[np.ndarray, dict]:
    """A 路径（RawTherapee 式）：在传感器线性空间测量与反相，矩阵在反相
    之后的渲染阶段才应用。

    bake-off 决定性发现（2026-09-01，两轮修正）：
    1. 相机 rgb_xyz_matrix 期望 WB 缩放后的场景数据；对橙色片基应用
       daylight WB 会产生 32% 负 XYZ（矩阵负瓣截断）；
    2. 不应用 WB 直接乘矩阵同样得到负值——片基色在相机色域外。
    结论：负片测量的合法工作空间是传感器线性空间本身（片基=白参考），
    任何"先转场景 RGB 再反相"的路径都在裁剪密度信息。
    """
    raw, info = _decode(path, int(rawpy.ColorSpace.raw.value))
    return raw, {"space": "sensor-linear (pre-matrix, pre-WB)",
                 "note": "matrix applied post-inversion in render stage"}


def path_b(path: Path) -> tuple[np.ndarray, dict]:
    t, info = _decode(path, int(rawpy.ColorSpace.sRGB.value))
    return t, {"note": "LibRaw 内部 WB+矩阵，线性输出"}


def path_c(path: Path) -> tuple[np.ndarray, dict]:
    t, info = _decode(path, int(rawpy.ColorSpace.ACES.value))
    srgb = t @ ACES_TO_SRGB_LINEAR.T.astype(np.float32)
    return np.clip(srgb, 0.0, None), {"transform": "AP1(D60)->sRGB(D65) Bradford"}


def analyze(name: str, t: np.ndarray, extra: dict, base_roi) -> dict:
    by0, bx0, by1, bx1 = base_roi
    dmin = robust_dmin(t[by0:by1, bx0:bx1])["dmin"]
    det = detect_regions(t)
    ia = det.get("image_area")
    valid = np.zeros(t.shape[:2], bool)
    if ia:
        valid[ia[0]:ia[2], ia[1]:ia[3]] = True
    else:
        valid[:] = True
    vpx = t[valid]
    clip_low = [float((vpx[:, c] <= 1e-5).mean()) for c in range(3)]
    p = InvertParams(dmin=tuple(float(v) for v in dmin), dmax=1.8)
    z = transmittance_to_z(t, dmin)
    ns = find_neutral_z_samples(np.stack([t[valid]]), p)  # (N,1,3)
    lum = (t @ np.array([0.2126, 0.7152, 0.0722], np.float32))[valid]
    mx, mn = t[valid].max(axis=1), t[valid].min(axis=1)
    chroma = ((mx - mn) / np.maximum(lum, 1e-6))
    return {
        "path": name, "workspace_transform": extra,
        "base_dmin": [round(float(v), 5) for v in dmin],
        "input_clip_low": [round(c, 5) for c in clip_low],
        "lum_p01_p50_p99": [round(float(np.quantile(lum, q)), 4)
                            for q in (0.01, 0.5, 0.99)],
        "chroma_median_p90": [round(float(np.quantile(chroma, q)), 4)
                              for q in (0.5, 0.9)],
        "neutral_candidates": len(ns["samples"]),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=None)
    args = ap.parse_args()
    src = Path(args.file) if args.file else \
        ROOT / "20260709_SAMPLE-ID(1)/1298_G200/翻拍/G200-1.NEF"
    if not src.exists():
        print("样片不存在", src)
        return 1
    run_id = time.strftime("%Y%m%d-%H%M%S")
    outdir = ROOT / "outputs" / "benchmarks" / run_id
    outdir.mkdir(parents=True, exist_ok=True)
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True,
                                cwd=ROOT).stdout.strip()
    except Exception:  # noqa: BLE001
        commit = "unknown"
    ref = _decode(src, int(rawpy.ColorSpace.sRGB.value))[0]
    base_roi = detect_regions(ref)["base"]
    results = []
    for name, fn in (("A-raw+own-matrix", path_a),
                     ("B-srgb-linear(default)", path_b),
                     ("C-aces+full-transform", path_c)):
        t0 = time.perf_counter()
        t, extra = fn(src)
        r = analyze(name, t, extra, base_roi)
        r["seconds"] = round(time.perf_counter() - t0, 1)
        results.append(r)
        print(json.dumps(r, ensure_ascii=False)[:200])
    doc = {
        "run_id": run_id, "git_commit": commit,
        "environment": {"python": sys.version.split()[0],
                        "rawpy": rawpy.__version__,
                        "LibRaw": rawpy.libraw_version,
                        "platform": sys.platform},
        "source": src.as_posix(),
        "note": "NLP 非真值；本表用于选择默认工作空间与暴露差异，"
                "中性候选仅诊断不参与标定（v3 P0-E）",
        "paths": results,
    }
    (outdir / "report.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print("写入", outdir / "report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
