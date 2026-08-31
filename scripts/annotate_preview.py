"""标注辅助：渲染样片 + 候选画面区域，供人工核对 frame_roi / rotation。

用法：.venv\\Scripts\\python.exe scripts/annotate_preview.py <样片> [-o outputs/ann]
输出：全览图（含候选框）与方向判定图（0/90/180/270 四格）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from negative_lab.calibration.border import detect_regions  # noqa: E402
from negative_lab.io.images import load_linear_rgb  # noqa: E402


def image_area_candidate(t: np.ndarray, frame: tuple[int, int, int, int]) -> tuple:
    """帧内找"画面区域"：列亮度方差高的连续区段（片边文字/齿孔/片基方差模式不同）。

    返回 (y0,x0,y1,x1) 原图坐标。
    """
    fy0, fx0, fy1, fx1 = frame
    sub = t[fy0:fy1, fx0:fx1].mean(axis=2)
    H, W = sub.shape
    step = max(1, max(H, W) // 600)
    s = sub[::step, ::step]
    col_var = s.var(axis=0)
    col_med = np.median(s, axis=0)
    # 画面 = 高方差列的连续段；片基条带方差低且亮；黑边方差低且暗
    thr = np.median(col_var[col_var > 0]) * 0.5 if (col_var > 0).any() else 0
    ok = col_var > thr
    best, cur = (0, 0), None
    for i, v in enumerate(ok):
        if v and cur is None:
            cur = i
        elif not v and cur is not None:
            if i - cur > best[1] - best[0]:
                best = (cur, i)
            cur = None
    if cur is not None and len(ok) - cur > best[1] - best[0]:
        best = (cur, len(ok))
    x0, x1 = fx0 + best[0] * step, fx0 + best[1] * step
    # 行方向：排除片基条带（亮且平的行段）与黑边
    row_med = np.median(t[:, x0:x1, :].mean(axis=2), axis=1)
    gmed = np.median(row_med)
    rok = (row_med > gmed * 0.25) & (row_med < gmed * 4.0)
    best_r, cur_r = (0, 0), None
    for i, v in enumerate(rok):
        if v and cur_r is None:
            cur_r = i
        elif not v and cur_r is not None:
            if i - cur_r > best_r[1] - best_r[0]:
                best_r = (cur_r, i)
            cur_r = None
    if cur_r is not None and len(rok) - cur_r > best_r[1] - best_r[0]:
        best_r = (cur_r, len(rok))
    return (best_r[0], x0, best_r[1], x1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sample", type=Path)
    ap.add_argument("-o", type=Path, default=ROOT / "outputs" / "ann")
    args = ap.parse_args()
    args.o.mkdir(parents=True, exist_ok=True)

    loaded = load_linear_rgb(args.sample)
    t = loaded.rgb
    det = detect_regions(t)
    frame = det["frame"]
    cand = image_area_candidate(t, tuple(frame))
    print("frame(film):", frame)
    print("image-area candidate:", cand)

    from PIL import Image, ImageDraw

    def render(arr, scale=0.18, boxes=()):
        x = np.clip(arr, 0, 1)
        x = (x / max(x.max(), 1e-6) * 255).astype(np.uint8)
        im = Image.fromarray(x).resize((int(x.shape[1] * scale), int(x.shape[0] * scale)))
        im = im.convert("RGB")
        d = ImageDraw.Draw(im)
        for (box, color) in boxes:
            y0, x0, y1, x1 = box
            d.rectangle([x0 * scale, y0 * scale, x1 * scale, y1 * scale],
                        outline=color, width=2)
        return im

    full = render(t, boxes=[(frame, (80, 160, 255)), (cand, (80, 255, 80))])
    full.save(args.o / (args.sample.stem + "_overview.jpg"), quality=88)

    # 方向四格（用候选画面区域）
    crop = t[cand[0]:cand[2], cand[1]:cand[3]]
    tiles = [render(np.rot90(crop, k), scale=0.3) for k in range(4)]
    w = sum(im.width for im in tiles)
    h = max(im.height for im in tiles)
    sheet = Image.new("RGB", (w, h + 22), (20, 20, 20))
    x = 0
    for k, im in enumerate(tiles):
        sheet.paste(im, (x, 22))
        d = ImageDraw.Draw(sheet)
        d.text((x + 4, 4), f"rot90x{k} ({k*90} CCW)", fill=(255, 255, 0))
        x += im.width
    sheet.save(args.o / (args.sample.stem + "_orient.jpg"), quality=88)
    print("saved:", args.o / (args.sample.stem + "_overview.jpg"))
    print("saved:", args.o / (args.sample.stem + "_orient.jpg"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
