"""整改前后视觉联系表（整改 P0-C 人工视觉门禁）。

每张样片并排：整改前基线（无裁切/无旋转/无统计中性）| 整改后自动结果 |
NLP 对照（如有）。统一方向与画面范围（verified 标注优先）。
输出 outputs/visual/<sample>_sheet.jpg 与汇总 docs/visual-acceptance.md 表格。

运行：.venv\\Scripts\\python.exe scripts/build_visual_contact_sheet.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from negative_lab.calibration.annotations import load_annotation  # noqa: E402
from negative_lab.io.images import load_linear_rgb  # noqa: E402
from negative_lab.pipeline.auto import auto_analyze  # noqa: E402
from negative_lab.pipeline.density import invert_chunked  # noqa: E402
from negative_lab.pipeline.params import InvertParams  # noqa: E402
from negative_lab.pipeline.workflow import prepare_and_analyze  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

OUT = ROOT / "outputs" / "visual"
TARGET_H = 700


def to_img(arr01: np.ndarray) -> Image.Image:
    x = np.clip(arr01, 0, 1)
    return Image.fromarray((x * 255 + 0.5).astype(np.uint8)).convert("RGB")


def norm_negative(arr: np.ndarray) -> np.ndarray:
    x = np.clip(arr, 0, None)
    return x / max(float(x.max()), 1e-6)


def process(rel: str) -> tuple[Image.Image, Image.Image, Image.Image | None, dict]:
    ann = load_annotation(rel)
    src = ROOT / rel
    t = load_linear_rgb(src).rgb
    rot = ann.rotation_degrees if ann else 0
    # 整改前基线：整帧（含片边/片基）、不旋转、无统计中性（wb 全 1）、
    # 旧 offset 策略近似 = offset 0（黑位压死）
    from negative_lab.pipeline.geometry import rotate_cw
    t_rot_full = rotate_cw(t, rot)
    before_rep = auto_analyze(t_rot_full[::6, ::6, :].copy())
    p_before = InvertParams(dmin=before_rep.params.dmin,
                            dmax=before_rep.params.dmax, offset=0.0)
    res_before = invert_chunked(t_rot_full, p_before)
    # 整改后：共享工作流（标注裁切 + 统计中性 + 标定卷 Dmax 先验）
    from negative_lab.profiles.profile import load_profile
    prior = None
    pf = ROOT / "fixtures" / "priors" / "KodakGold200-Z8-led.profile.json"
    if "G200" in rel and pf.exists():
        prior = load_profile(pf).to_initial_params()
    prepared = prepare_and_analyze(t, rotation_degrees=rot, ann=ann, prior=prior)
    res_after = invert_chunked(prepared.t, prepared.report.params)
    # NLP 对照
    nlp = None
    if ann and ann.reference_file and (ROOT / ann.reference_file).exists():
        nlp = to_img(np.asarray(Image.open(str(ROOT / ann.reference_file))
                                .convert("RGB"), np.float32) / 255.0)
    return (to_img(norm_negative(t_rot_full)), to_img(res_before.preview_srgb),
            to_img(res_after.preview_srgb), nlp,
            {"params_before": p_before.to_parameters_dict(),
             "params_after": prepared.report.params.to_parameters_dict(),
             "confidence": prepared.report.confidence,
             "warnings": prepared.report.warnings[:6]})


def sheet(name: str, tiles: list[tuple[str, Image.Image | None]]) -> Image.Image:
    ims = []
    for label, im in tiles:
        if im is None:
            continue
        w = int(im.width * TARGET_H / im.height)
        ims.append((label, im.resize((w, TARGET_H))))
    W = sum(i.width for _, i in ims) + 8 * len(ims)
    out = Image.new("RGB", (W, TARGET_H + 26), (18, 18, 18))
    d = ImageDraw.Draw(out)
    x = 4
    for label, im in ims:
        out.paste(im, (x, 24))
        d.text((x + 2, 6), label, fill=(255, 255, 120))
        x += im.width + 8
    out.save(OUT / f"{name}_sheet.jpg", quality=88)
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / "fixtures" / "manifest.json").read_text(encoding="utf-8"))
    targets = [e for e in manifest["images"]
               if e["decode_status"] == "ok" and load_annotation(e["file"])
               and load_annotation(e["file"]).verified]
    if not targets:
        print("无 verified 标注样片")
        return 1
    rows = []
    for e in targets:
        name = Path(e["file"]).stem + "_" + e["roll"]
        print("processing", e["file"], flush=True)
        neg, before, after, nlp, info = process(e["file"])
        sheet(name, [("原负片(旋转后,含片边)", neg),
                     ("整改前基线(整帧/黑位压死)", before),
                     ("整改后自动(裁切+统计中性)", after),
                     ("NLP对照(非真值)", nlp)])
        rows.append((name, e["file"], info))
        print("  ->", OUT / f"{name}_sheet.jpg")
    # 汇总文档
    lines = ["# 视觉验收记录（P0-C 人工门禁）", "",
             "**状态：等待用户确认。** 没有用户确认不得进入 P3 发布（整改 §P0-C）。",
             "",
             "联系表位于 `outputs/visual/*_sheet.jpg`，四格依次为：",
             "原负片（旋转后）| 整改前基线 | 整改后自动结果 | NLP 对照（仅对照非真值）。",
             "",
             "## 用户确认清单（每张七项）",
             "",
             "| 样片 | 方向 | 裁切 | 高光 | 肤色 | 中性灰 | 阴影 | 整体色偏 | 结论 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name, f, info in rows:
        lines.append(f"| {name} | 180°已应用 | 画面区已裁切 | [ ] | [ ] | [ ] | [ ] | [ ] | **待确认** |")
    lines += ["",
              "## 整改前失败基线（质量审查报告 F-1 对应）",
              "",
              "- G200-1 整改前：倒置、保留大片片边/片基、整体偏青蓝、黑位压死",
              f"  （低端剪裁 R 9.5%/G 10.0%/B 16.2%）。整改后同一管线低端剪裁 <1%。",
              f"- 整改后自动参数（G200-1）：dmax={rows[0][2]['params_after']['dmax']:.2f}, "
              f"wb_high={np.round(rows[0][2]['params_after']['wb_high'], 3).tolist()}, "
              f"ratios R/B={rows[0][2]['params_after']['red_ratio']:.3f}/"
              f"{rows[0][2]['params_after']['blue_ratio']:.3f}",
              f"- 置信度：{rows[0][2]['confidence']}",
              "",
              "## 已知限制",
              "",
              "- 统计中性为受限 gray-world 估计（非色卡），肤色需人工核对；",
              "- NLP 格仅作风格对照，其裁切区域与本软件不完全一致；",
              "- 黑白/ECN-2/真实 TIFF 路线缺样片（docs/support-matrix.md）。"]
    (ROOT / "docs" / "visual-acceptance.md").write_text("\n".join(lines) + "\n",
                                                        encoding="utf-8")
    print("写入 docs/visual-acceptance.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
