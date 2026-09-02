"""v5 P0-0.4：记录"before"证据（当前 v4 代码的真实样片输出与联系表）。

只读样片；输出写入 outputs/quality/<run-id>/before/。
用途：v5 改造前后的对照基线（§15 证据要求）。

运行：.venv\\Scripts\\python.exe scripts/capture_v5_before_evidence.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from PIL import Image, ImageDraw  # noqa: E402

from negative_lab.calibration.annotations import load_annotation  # noqa: E402
from negative_lab.io.images import load_linear_rgb  # noqa: E402
from negative_lab.pipeline.service import process_image  # noqa: E402

TARGETS = [
    "20260709_SAMPLE-ID(1)/1298_G200/翻拍/G200-1.NEF",
    "20260709_SAMPLE-ID(1)/1298_G200/翻拍/G200-5.NEF",
    "20260709_SAMPLE-ID(1)/1299_G200/翻拍/G200-1.NEF",
    "20260709_SAMPLE-ID(1)/1300_CN400/翻拍/CN400-1.NEF",
    "20260506_SAMPLE-ID/ECN2/5207-1.tif",
    "20260506_SAMPLE-ID/ECN2/5219-1.tif",
]
STEP = 4          # before 证据用 1/4 分辨率控制时长（明确标注，不作为质量结论）


def to_img(a01: np.ndarray) -> Image.Image:
    x = np.clip(a01, 0, 1)
    return Image.fromarray((x * 255 + 0.5).astype(np.uint8)).convert("RGB")


def norm_neg(a: np.ndarray) -> np.ndarray:
    x = np.clip(a, 0, None)
    return x / max(float(x.max()), 1e-6)


def main() -> int:
    run_id = time.strftime("%Y-%m-%d-%H%M%S")
    out = ROOT / "outputs" / "quality" / run_id / "before"
    out.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True).stdout.strip()
    rows = []
    tiles = []
    for rel in TARGETS:
        p = ROOT / rel
        if not p.exists():
            rows.append({"file": rel, "status": "missing"})
            continue
        t0 = time.perf_counter()
        loaded = load_linear_rgb(p)
        t = loaded.rgb[::STEP, ::STEP, :].copy()
        ann = load_annotation(rel)
        pr = process_image(t, rotation_degrees=ann.rotation_degrees if ann else 0,
                           ann=ann, film_type="unknown")
        rec = {
            "file": rel,
            "input_encoding": loaded.meta.get("input_encoding"),
            "working_space": loaded.meta.get("working_space"),
            "gate_status": pr.gate.status,
            "gate_failures": pr.gate.failures,
            "warnings": pr.gate.warnings[:8],
            "confidence": pr.report.confidence,
            "technical_parameters": pr.report.params.to_parameters_dict(),
            "roi_source_space": pr.geometry.roi_source_space,
            "output_size": list(pr.geometry.output_size),
            "seconds": round(time.perf_counter() - t0, 2),
            "note": f"before 证据，{STEP}x 降采样，不作为质量门禁结论",
        }
        rows.append(rec)
        name = Path(rel).stem
        (out / f"{name}_technical.jpg").write_bytes(_jpeg(to_img(pr.result.preview_srgb)))
        tiles.append((f"{name} before", to_img(norm_neg(t))))
        tiles.append((f"{name} v4输出", to_img(pr.result.preview_srgb)))
        print(f"  {name}: gate={pr.gate.status} {rec['seconds']}s", flush=True)

    # 联系表（两列一组：原负片 | v4 输出）
    H = 420
    ims = []
    for label, im in tiles:
        w = int(im.width * H / im.height)
        ims.append((label, im.resize((w, H))))
    per_row = 2
    W = max(i.width for _, i in ims) * per_row + 12
    sheet = Image.new("RGB", (W, (H + 22) * (len(ims) // per_row) + 8), (16, 16, 16))
    d = ImageDraw.Draw(sheet)
    for idx, (label, im) in enumerate(ims):
        r, c = divmod(idx, per_row)
        x = c * (W // per_row) + 4
        y = r * (H + 22) + 4
        sheet.paste(im, (x, y + 18))
        d.text((x + 2, y + 4), label, fill=(255, 255, 140))
    sheet.save(out / "before-contact-sheet.jpg", quality=86)
    (out / "before-metrics.json").write_text(
        json.dumps({"baseline_commit": commit, "run_id": run_id, "step": STEP,
                    "samples": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print("before 证据：", out)
    return 0


def _jpeg(im: Image.Image, q: int = 88) -> bytes:
    import io

    b = io.BytesIO()
    im.save(b, format="JPEG", quality=q)
    return b.getvalue()


if __name__ == "__main__":
    raise SystemExit(main())
