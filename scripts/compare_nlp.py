"""与 Negative Lab Pro 对照（规格 §8.4：NLP 只是对照，不是真值）。

方法（粗对齐，差异原因如实记录）：
1. 本软件批处理导出（帧裁切+自动参数） vs 用户提供的 NLP JPEG 导出；
2. 两者裁到公共宽高比、缩放到 1024 宽；
3. 全图 ΔE2000 统计（mean/p50/p95）+ 亮度均值差 + 彩度均值差；
4. 输出 docs/nlp-comparison.md：数字 + 已知偏差原因，不写"更好"。

运行：先跑批处理，再 .venv\\Scripts\\python.exe scripts/compare_nlp.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from negative_lab.diagnostics.colorimetry import delta_e_2000, srgb_to_lab  # noqa: E402

BATCH_DIR = ROOT / "outputs" / "batch_1298"
NLP_DIR = ROOT / "20260709_SAMPLE-ID(1)/1298_G200/翻拍/NLP"
W = 1024


def load_srgb01(path: Path) -> np.ndarray:
    """按 sRGB 编码语义读取（对照双方都是显示编码图）。"""
    from PIL import Image

    img = Image.open(path).convert("RGB")
    return np.asarray(img, np.float32) / 255.0


def align(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """取双方中心 60% 区域（避开本软件导出的片边/片基条带与 NLP 裁切差异），
    缩放到公共尺寸。粗对齐：只比较画面内容核心区。"""
    from PIL import Image

    def center60(x):
        h, w, _ = x.shape
        ch, cw = int(h * 0.6), int(w * 0.6)
        return x[(h - ch) // 2:(h + ch) // 2, (w - cw) // 2:(w + cw) // 2]

    a, b = center60(a), center60(b)
    out = []
    for x in (a, b):
        im = Image.fromarray((np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8))
        im = im.resize((W, max(1, int(W * im.height / im.width))), Image.BILINEAR)
        out.append(np.asarray(im, np.float32) / 255.0)
    h = min(out[0].shape[0], out[1].shape[0])
    return out[0][:h], out[1][:h]


def compare(ours: np.ndarray, nlp: np.ndarray) -> dict:
    """粗对照：尝试 4 个旋转方向，取平均 ΔE 最小的对齐（构图差异仍计入）。"""
    best = None
    for k in range(4):
        rot = np.rot90(ours, k)
        a, b = align(rot, nlp)
        if a.shape != b.shape:
            # align 后高度取 min，宽度按各自比例可能不同 -> 再裁公共宽高
            h = min(a.shape[0], b.shape[0])
            w = min(a.shape[1], b.shape[1])
            a, b = a[:h, :w], b[:h, :w]
        de = delta_e_2000(srgb_to_lab(a), srgb_to_lab(b))
        m = float(de.mean())
        if best is None or m < best[0]:
            lum_a = a.mean(axis=2)
            lum_b = b.mean(axis=2)
            ca = a.max(axis=2) - a.min(axis=2)
            cb = b.max(axis=2) - b.min(axis=2)
            best = (m, {
                "de_mean": m,
                "de_p50": float(np.median(de)),
                "de_p95": float(np.quantile(de, 0.95)),
                "lum_delta": float(lum_a.mean() - lum_b.mean()),
                "chroma_delta": float(ca.mean() - cb.mean()),
                "rot": k * 90,
            })
    return best[1]


def main() -> int:
    if not BATCH_DIR.is_dir():
        print("先运行批处理：python -m negative_lab.cli batch ... -o outputs/batch_1298")
        return 1
    rows = []
    for ours in sorted(BATCH_DIR.glob("*-positive.jpg")):
        stem = ours.name.replace("-positive.jpg", "")
        ref = NLP_DIR / f"{stem}.jpg"
        if not ref.exists():
            continue
        r = compare(load_srgb01(ours), load_srgb01(ref))
        r["file"] = stem
        rows.append(r)
        print(f"{stem}: dE mean={r['de_mean']:.2f} p95={r['de_p95']:.2f} "
              f"lumΔ={r['lum_delta']:+.3f} chromaΔ={r['chroma_delta']:+.3f} rot={r['rot']}")
    if not rows:
        print("没有可对照的文件")
        return 1
    lines = ["# 与 Negative Lab Pro 对照（1298_G200 卷）", "",
             "NLP 只是对照基线，**不是绝对真值**（§8.4）。",
             "方法：双方导出均为显示编码图；取各自中心 60% 画面区域"
             "（避开片边/裁切差异）、缩放 1024 宽；全区 ΔE2000（sRGB->Lab D65）。",
             "已知偏差原因：",
             "- 本软件自动参数无中性参考点（该卷未人工取点），白平衡仅靠片基；",
             "- NLP 默认输出带自己的对比/曲线风格（更暗更饱和），亮度差含风格因素；",
             "- 中心 60% 粗对齐，构图偏移会计入 ΔE。",
             "",
             "| 文件 | ΔE mean | ΔE p50 | ΔE p95 | 亮度差 | 彩度差 |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['file']} | {r['de_mean']:.2f} | {r['de_p50']:.2f} | "
                     f"{r['de_p95']:.2f} | {r['lum_delta']:+.3f} | "
                     f"{r['chroma_delta']:+.3f} |")
    m = np.mean([r["de_mean"] for r in rows])
    lines += ["", f"全卷平均 ΔE2000（vs NLP）：**{m:.2f}**",
              "",
              "**警告：全图 ΔE 混合了裁切/构图错位与风格曲线差异，只用于发现异常大的",
              "偏离，不是色彩准确度指标。** best-rot 逐张乱跳即对齐不成立的信号。",
              "§8.4 的 neutral ΔE2000<=5 验收目标需要含色卡的样片 + 人工取点后再评。"]
    out = ROOT / "docs" / "nlp-comparison.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
