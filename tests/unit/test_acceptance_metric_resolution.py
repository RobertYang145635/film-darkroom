"""锁：**验收那把尺只有 384 像素**，它稳的是什么、不稳的是什么（㉟）.

用户问"密度域、全分辨率统计、浮点计算这些我们做了吗"。对着代码核完是三条事实：

1. **决策侧在密度域**（`picture_density` 把技术正片 luma 按 `−dmax·log10(1−y)` 反解，
   锚点取 p3/p35/p95），但那是"重映射自己的刻度"，不等于逐通道物理密度；
   **打分侧在显示 sRGB**（`g200_manual_look.json` 的 `method.domain`）。
2. **统计不是全分辨率**：`method.resize = 384` ⇒ 从他 5309×5309 的参考缩到 384²
   （面积 0.5%）再取中心 70%。生产管线本身不缩（`cli/batch/roll_export` 里没有 resize）。
3. **处理是浮点**（整帧 float32、归约 float64），但**打分前两边都量化成 8 位**，
   而他一侧的"真值"是 **8 位 JPEG**。

第 2 条本轮当场量了（`tmp_metric_resolution.py` → `outputs/metricres35.txt`，
11 帧、同一批渲染、四把尺子 {384, 768, 1536, 原生}）：

| 量 | 384 档绝对值 | 原生档 | 漂移 | 谁受影响 |
|---|---|---|---|---|
| mean | 0.5879（我） | 0.5881 | **0.0003** | 亮度结论**与尺子无关** |
| p05  | 0.1766 | 0.1472 | 0.0294 | 雾/黑位 |
| std  | 0.2206 | 0.2374 | 0.0168 | 对比 |
| sat95| 0.3715 | 0.4350 | **0.0635** | 饱和 |

四统计平均距离：384 档 **0.0470** → 原生档 **0.0529**（真实差距比 384 报的大 12%）。
**逐帧更会被尺子翻转**：`1298::G200-11` 的 `|Δsat95|` 384 档 0.1137、原生档只有
0.0235（尺子**凭空造出** 0.09）；`1298::G200-7` 反过来 384 档 0.1216、原生档 0.2392
（尺子**吃掉一半**）。⇒ **任何逐帧饱和的结论都不许只在 384 档说**。

所以这条锁做两件事：① 把"384"这个事实钉在断言里（不许有人以为它是全分辨率）；
② 钉住**哪一根轴可以信 384**（亮度：漂移 <0.002）、**哪几根不可以**
（p05/std/sat95：漂移 >0.01 ⇒ 必须另量原生档）。
参考大图不在库里时 `skip` 并明说是降级环境（与 `test_manual_reference_integrity` 同规矩）。
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

ROOT = pathlib.Path(__file__).resolve().parents[2]
MAP = ROOT / "fixtures" / "references" / "g200_manual_look.json"
LUM_W = np.array([0.2126, 0.7152, 0.0722])
KEYS = ("mean", "p05", "std", "sat95")


def _doc() -> dict:
    return json.loads(MAP.read_text(encoding="utf-8"))


def _stats(a01: np.ndarray, size: int, frac: float) -> dict:
    im = Image.fromarray((np.clip(a01, 0, 1) * 255).round().astype(np.uint8))
    if size:
        im = im.resize((size, size), Image.BILINEAR)
    a = np.asarray(im, np.float64) / 255.0
    c = int(a.shape[0] * (1 - frac) / 2)
    r = int(a.shape[1] * (1 - frac) / 2)
    d = a[c:a.shape[0] - c, r:a.shape[1] - r, :]
    lum = d @ LUM_W
    sat = d.max(2) - d.min(2)
    return {"mean": float(lum.mean()), "p05": float(np.percentile(lum, 5)),
            "std": float(lum.std()), "sat95": float(np.percentile(sat, 95))}


def _refs():
    out = []
    for e in _doc()["frames"]:
        if e["variant_of"]:
            continue
        p = ROOT / e["reference_file"]
        if p.is_file():
            out.append((e["frame_id"], p))
    return out


def test_the_acceptance_ruler_is_384_and_that_is_declared_not_implied():
    """口径本身必须写明：**我们不是在比全分辨率统计**。"""
    m = _doc()["method"]
    assert (m["center_crop_frac"], m["resize"], m["domain"]) == (0.70, 384, "display-sRGB"), (
        f"验收口径被改了：{m}。改口径不是不行，但它会让**全部**已钉的预算常数"
        f"（HAZE_BUDGET / SAT_BUDGET / WHITE_GIVEBACK_BUDGET / TRAIN_DIST 等）"
        f"失去可比性——必须连它们一起重钉，并回来更新本文件")


@pytest.mark.skipif(not _refs(), reason="用户参考大图不在库里（gitignore），降级环境")
def test_brightness_is_ruler_invariant_but_shadow_and_saturation_are_not():
    """**亮度可以信 384，黑位/对比/饱和不能**——当场重量的边界（他一侧 11 张）。

    判据（跑之前写下）：每张参考图在 384 与原生两档各量一次，取**档间漂移的均值**：
    ① `mean` 漂移必须 **< 0.002** ⇒ 这就是 ⑮/ 那些亮度结论与尺子无关的依据
      （实测两侧都是 **0.0003**）；
    ② `p05` 与 `sat95` 漂移必须 **> 0.01** ⇒ 这两根轴**不许只在 384 档下逐帧结论**
      （实测他一侧 p05 **0.0416**、std 0.0229、sat95 **0.0182**）。
    ⚠ 真正危险的是**两侧漂移不对称**：我方渲染的 sat95 档间漂移是 **0.0635**
    （他只有 0.0182），于是**同一批图的饱和结论会随尺子翻符号**——
    384 档 我 0.3715 对 他 0.3929（我方**欠**饱和 0.021），
    原生档 我 0.4350 对 他 0.4111（我方**过**饱和 0.024）。
    ⇒ "饱和成了最大一项、修完 ㉝ 反而欠饱和"这句话**只在 384 档成立**，
    原生档要重说（出处 `outputs/metricres35.txt`）。
    """
    drift = {k: [] for k in KEYS}
    for fid, p in _refs():
        a = np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.0
        s384 = _stats(a, 384, 0.70)
        snat = _stats(a, 0, 0.70)
        for k in KEYS:
            drift[k].append(abs(s384[k] - snat[k]))
    md = {k: float(np.mean(v)) for k, v in drift.items()}
    print("  档间漂移均值（他一侧 " + str(len(drift["mean"])) + " 张参考）："
          + "  ".join(f"{k}={md[k]:.4f}" for k in KEYS))
    assert md["mean"] < 0.002, (
        f"亮度不再与尺子无关（漂移 {md['mean']:.4f}）⇒ 本轮所有亮度结论要重估")
    for k in ("p05", "sat95"):
        assert md[k] > 0.01, (
            f"{k} 的档间漂移掉到 {md[k]:.4f}：尺子的影响消失了，该讨论把 "
            f"method.resize=384 升到全分辨率（连全部预算常数一起重钉），而不是删这条锁")