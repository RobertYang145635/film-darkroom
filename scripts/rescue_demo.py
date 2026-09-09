"""R2 验证：真实负片端到端救援演示（拾点中性救援在管线输出上可用）。

流程：
1. 真实样片 -> 默认管线 -> 技术正片 -> 渲染正片（恒等 intent）；
2. 注入已知色偏（模拟"自动结果翻车"）；
3. 在注入后的图上拾点（画面中部 1/3 区域，模拟用户点"这里该是灰"）；
4. neutral_pick_intent -> apply_render -> 验证拾点区色偏收敛。

输出：outputs/research/rescue_demo/<stem>.json
（不做"哪个点该是灰"的语义判断——那是用户的事；只验证数学与链路。）
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402


def main() -> int:
    rel = (sys.argv[1] if len(sys.argv) > 1
           else "20260709_SAMPLE-ID(1)/1298_G200/翻拍/G200-1.NEF")
    from negative_lab.io.images import load_linear_rgb
    from negative_lab.pipeline.render import apply_render, RenderIntent
    from negative_lab.pipeline.rescue import (neutral_pick_delta)
    from negative_lab.pipeline.service import process_image

    path = ROOT / rel
    loaded = load_linear_rgb(path)
    pr = process_image(loaded.rgb, film_type="unknown")
    pos = pr.result.positive_linear
    h, w = pos.shape[:2]
    # 取输出裁切内的画面
    crop = pr.geometry.roi_source_space.get("output_crop_roi")
    y0, x0, y1, x1 = 0, 0, h, w
    if crop:
        y0, x0, y1, x1 = (int(v) for v in crop)
        y1, x1 = min(y1, h), min(x1, w)
    base = np.asarray(apply_render(pos, RenderIntent()))

    # 拾点：画面中心偏上（研究演示用固定点；真实使用由用户点击）
    py, px = (y0 + 2 * (y1 - y0)) // 3, (x0 + (x1 - x0)) // 2

    def patch_med(img, r=5):
        p = img[py - r:py + r + 1, px - r:px + r + 1].reshape(-1, 3)
        return np.median(p, axis=0)

    results = {}
    for name, cast in (("blue_cast", (0.0, 0.0, 0.10)),
                       ("warm_cast", (0.08, 0.02, -0.06))):
        casted = np.clip(base + np.array(cast, np.float32), 0, 1).astype(np.float32)
        d_temp, d_tint = neutral_pick_delta(casted, py, px)
        intent = RenderIntent(temperature=d_temp, tint=d_tint,
                              origin="neutral-pick")
        fixed = np.asarray(apply_render(casted, intent))
        before, after = patch_med(casted), patch_med(fixed)
        results[name] = {
            "cast": list(cast),
            "pick": [int(py), int(px)],
            "d_temp": round(d_temp, 2), "d_tint": round(d_tint, 2),
            "patch_before": [round(float(v), 4) for v in before],
            "patch_after": [round(float(v), 4) for v in after],
            "spread_before": round(float(before.max() - before.min()), 4),
            "spread_after": round(float(after.max() - after.min()), 4),
        }
        print(f"{name}: d_temp={d_temp:+.1f} d_tint={d_tint:+.1f} "
              f"spread {results[name]['spread_before']:.3f} -> "
              f"{results[name]['spread_after']:.3f}")

    out_dir = ROOT / "outputs" / "research" / "rescue_demo"
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = {"file": rel, "results": results,
           "note": "拾点位置为演示固定点；真实使用由用户点击。注入色偏是"
                   "模拟自动翻车，验证救援数学与链路，不代表真实翻车分布。"}
    (out_dir / f"{path.stem}.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"写入 {out_dir / (path.stem + '.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
