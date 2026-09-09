"""P1 一次性诊断（v6.1 §5 第一步）：G200-4 自动裁切 vs 人工标注。

输出（只写 outputs/v6_ui/auto_crop_diagnostics/，不复制原片）：
- diagnostics.json：输入哈希、尺寸、step、frame/base/image_area/edge/
  confidence/warnings、人工框与 IoU、行/列亮度 profile、平滑 profile、
  阈值、候选端点与最终端点；
- overlay.png：缩略图上叠加自动 image_area（红）与人工框（绿）。
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from negative_lab.calibration.annotations import load_annotation  # noqa: E402
from negative_lab.calibration.border import (  # noqa: E402
    _downsample_lum, detect_frame, detect_base_strip, detect_image_area, iou)
from negative_lab.io.images import load_linear_rgb  # noqa: E402

REL = "20260709_SAMPLE-ID(1)/1299_G200/翻拍/G200-4.NEF"
OUT = ROOT / "outputs" / "v6_ui" / "auto_crop_diagnostics"


def _smooth(a: np.ndarray, k: int = 9) -> np.ndarray:
    if a.size < k:
        return a
    pad = k // 2
    ap = np.pad(a, pad, mode="edge")
    c = np.cumsum(np.insert(ap, 0, 0.0))
    return (c[k:] - c[:-k]) / k


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    path = ROOT / REL
    data = path.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    t = load_linear_rgb(path).rgb
    h, w = t.shape[:2]

    lum, step = _downsample_lum(t, 800)
    frame = detect_frame(lum)
    det = detect_base_strip(lum, frame)
    area = detect_image_area(t, det["frame"], det.get("edge"),
                             base_band=det.get("base"), max_dim=800)

    ann = load_annotation(REL)
    ref = ann.image_area_roi
    auto_val = iou(tuple(area), tuple(ref)) if area is not None else 0.0

    fy0, fx0, fy1, fx1 = frame
    sub = lum[fy0:fy1, fx0:fx1]
    rowmed = np.median(sub, axis=1)
    colmed = np.median(sub, axis=0)

    # 帧内相对坐标：把自动结果拆回条带轴端点
    rel = {}
    if area is not None:
        ay0, ax0, ay1, ax1 = (v // step for v in area)
        rel["area_y"] = [ay0 - fy0, ay1 - fy0]
        rel["area_x"] = [ax0 - fx0, ax1 - fx0]
        rel["sub_shape"] = [int(sub.shape[0]), int(sub.shape[1])]
        ry0, rx0, ry1, rx1 = (v // step for v in ref)
        rel["ref_y"] = [ry0 - fy0, ry1 - fy0]
        rel["ref_x"] = [rx0 - fx0, rx1 - fx0]

    doc = {
        "file": REL,
        "source_sha256": sha,
        "source_shape": [h, w],
        "step": step,
        "frame": [v * step for v in frame],
        "base": ([v * step for v in det["base"]]
                 if det.get("base") else None),
        "edge": det.get("edge"),
        "contrast": det.get("contrast"),
        "confidence": det.get("confidence"),
        "warnings": list(det.get("warnings", [])),
        "edge_contrasts": det.get("edge_contrasts"),
        "image_area_auto": list(area) if area else None,
        "image_area_manual": list(ref),
        "iou": round(auto_val, 6),
        "frame_relative": rel,
        "rowmed": [round(float(v), 5) for v in rowmed],
        "colmed": [round(float(v), 5) for v in colmed],
        "rowmed_smooth": [round(float(v), 5) for v in _smooth(rowmed)],
        "colmed_smooth": [round(float(v), 5) for v in _smooth(colmed)],
        "rowmed_full": [round(float(v), 5)
                        for v in np.median(lum, axis=1)],
        "colmed_full": [round(float(v), 5)
                        for v in np.median(lum, axis=0)],
    }
    (OUT / "diagnostics.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")

    # overlay：降采样亮度缩略图 + 自动框（红）+ 人工框（绿）
    try:
        from PIL import Image, ImageDraw
        img = np.clip(lum / max(float(lum.max()), 1e-6), 0, 1)
        im = Image.fromarray((img * 255).astype(np.uint8)).convert("RGB")
        d = ImageDraw.Draw(im)

        def rect(roi, color):
            y0, x0, y1, x1 = (v // step for v in roi)
            d.rectangle([x0, y0, x1, y1], outline=color, width=2)

        if area is not None:
            rect(area, (255, 60, 60))
        rect(ref, (60, 255, 60))
        im.save(OUT / "overlay.png")
    except ImportError:
        print("PIL 不可用，跳过 overlay")

    print(json.dumps({k: doc[k] for k in
                      ("source_shape", "step", "frame", "base", "edge",
                       "contrast", "confidence", "warnings",
                       "image_area_auto", "image_area_manual", "iou",
                       "frame_relative")}, ensure_ascii=False, indent=1))
    print(f"\n输出：{OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
