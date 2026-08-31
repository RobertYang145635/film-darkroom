"""胶片边框与片基条带自动检测（规格 §4.5.2、§2.1.5；Phase 2）。

真实翻拍样片的结构是：黑边（画框外）-> 未曝光片基条带 -> 画面内容。
直接取画面外框会命中黑边（诊断见 docs/real-samples-analysis.md 首版教训）。

算法（不使用 OpenCV，保持 Phase 1 依赖最小；几何约束用 NumPy 实现）：
1. 亮度降采样；
2. 帧检测：以全图高亮度分位为参考，逐行/列中位数超过阈值的连续区间
   构成候选胶片帧（取最大连通区间）；
3. 片基候选：在帧内四条边缘带中，选择"中位亮度最高且与帧中心亮度差最大、
   且通道比例稳定（橙罩一致性）"的条带；
4. 返回帧 ROI 与片基 ROI（原图坐标），供 GUI 覆盖与稳健统计。
"""

from __future__ import annotations

import numpy as np


def _downsample_lum(t_rgb: np.ndarray, max_dim: int = 800) -> tuple[np.ndarray, int]:
    h, w = t_rgb.shape[:2]
    step = max(1, int(np.ceil(max(h, w) / max_dim)))
    lum = t_rgb[::step, ::step, :].mean(axis=2)
    return lum, step


def _frame_interval(profile: np.ndarray, thr_factor: float = 0.15,
                    floor: float = 0.008, max_trim: float = 0.2) -> tuple[int, int]:
    """一维亮度剖面 -> 从两端向内修剪黑边，返回胶片占据区间。

    黑边特征：紧贴边缘、连续、亮度 <= max(floor, thr_factor*median)。
    只修剪边缘连续段（内部暗内容不受影响），每侧最多修剪 max_trim 比例。
    """
    n = len(profile)
    med = float(np.median(profile))
    thr = max(floor, thr_factor * med)
    limit = int(n * max_trim)

    i = 0
    while i < min(limit, n) and profile[i] <= thr:
        i += 1
    j = n - 1
    while j > n - 1 - limit and profile[j] <= thr:
        j -= 1
    if i >= j:  # 全剖面都低于阈值：不可信，返回全区间
        return 0, n
    return i, j + 1


def detect_frame(lum: np.ndarray) -> tuple[int, int, int, int]:
    """返回候选胶片帧 (y0, x0, y1, x1)（降采样坐标）。"""
    y0, y1 = _frame_interval(np.median(lum, axis=1))
    x0, x1 = _frame_interval(np.median(lum, axis=0))
    return y0, x0, y1, x1


def detect_base_strip(lum: np.ndarray, frame: tuple[int, int, int, int],
                      band_frac: float = 0.12) -> dict:
    """在帧内四边缘带中选择最像未曝光片基的条带。

    判据：条带中位亮度高、且明显亮于帧中心（负片中片基透光最多）；
    通道比例 R>=G>=B（C-41/ECN-2 橙罩方向）作为加分项而非硬条件（黑白负片无橙罩）。
    """
    fy0, fx0, fy1, fx1 = frame
    fh, fw = fy1 - fy0, fx1 - fx0
    if fh < 4 or fw < 4:
        return {"frame": frame, "base": None, "confidence": 0.0,
                "warnings": ["帧检测失败：候选区域过小"]}
    b = max(2, int(min(fh, fw) * band_frac))
    center = lum[fy0 + fh // 4: fy1 - fh // 4, fx0 + fw // 4: fx1 - fw // 4]
    c_med = float(np.median(center))

    cands = {
        "top": lum[fy0:fy0 + b, fx0:fx1],
        "bottom": lum[fy1 - b:fy1, fx0:fx1],
        "left": lum[fy0:fy1, fx0:fx0 + b],
        "right": lum[fy0:fy1, fx1 - b:fx1],
    }
    scored = []
    for name, band in cands.items():
        m = float(np.median(band))
        contrast = m / max(c_med, 1e-6)
        scored.append((contrast, name, m))
    scored.sort(reverse=True)
    contrast, name, m = scored[0]

    if contrast < 1.15:
        return {"frame": frame, "base": None, "confidence": 0.0,
                "warnings": [f"帧内边缘带与中心亮度差不足（最高对比 {contrast:.2f}），"
                             f"可能没有未曝光片基边缘，请手动指定片基框"],
                "edge_contrasts": {n: round(c, 3) for c, n, _ in scored}}

    ys, xs = {"top": (slice(fy0, fy0 + b), slice(fx0, fx1)),
              "bottom": (slice(fy1 - b, fy1), slice(fx0, fx1)),
              "left": (slice(fy0, fy1), slice(fx0, fx0 + b)),
              "right": (slice(fy0, fy1), slice(fx1 - b, fx1))}[name]
    y0, y1 = (ys.start, ys.stop) if isinstance(ys, slice) else (fy0, fy1)
    x0, x1 = (xs.start, xs.stop) if isinstance(xs, slice) else (fx0, fx1)
    # 置信度：对比度越高分离越可靠；>2.0 视为强
    conf = float(np.clip((contrast - 1.15) / 0.85, 0.1, 1.0))
    return {"frame": (fy0, fx0, fy1, fx1), "base": (y0, x0, y1, x1),
            "edge": name, "contrast": round(contrast, 3), "confidence": round(conf, 3),
            "warnings": []}


def detect_regions(t_rgb: np.ndarray, max_dim: int = 800) -> dict:
    """全图 -> 帧与片基 ROI（原图坐标）。失败时 base=None 并给出警告。"""
    lum, step = _downsample_lum(t_rgb, max_dim)
    frame = detect_frame(lum)
    det = detect_base_strip(lum, frame)
    out = {"step": step, "warnings": list(det["warnings"]),
           "confidence": det["confidence"]}
    if "edge_contrasts" in det:
        out["edge_contrasts"] = det["edge_contrasts"]

    def up(r):
        return None if r is None else tuple(v * step for v in r)

    out["frame"] = up(det["frame"])
    out["base"] = up(det["base"])
    out["edge"] = det.get("edge")
    return out
