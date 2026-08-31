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


def detect_image_area(t_rgb: np.ndarray, frame: tuple[int, int, int, int],
                      base_edge: str | None,
                      base_band: tuple[int, int, int, int] | None = None,
                      max_dim: int = 800) -> tuple[int, int, int, int] | None:
    """帧内定位"画面区域"（排除齿孔/字样片边与片基条带）。

    结构模型（真实样片目检）：35mm 条带 = [片边 | 画面 | 片边] 沿宽度方向，
    条带两端为片基/引带。base_edge 指明片基条带在哪条边，即条带轴方向。

    1) 沿条带轴：从片基侧向内找亮度谷底（画面与片基之间的不透明印边带），
       谷底即画面端点；另一端用帧边。
    2) 垂直条带轴：行限定后按列中位亮度 > 0.4x 全局中位取最长连续段
       （片边为不透光暗印，明显暗于画面）。
    返回 SOURCE 坐标 (y0,x0,y1,x1)；检测失败返回 None（调用方必须显式处理，
    不得静默回退整帧）。
    """
    if base_edge not in ("top", "bottom", "left", "right"):
        return None
    lum, step = _downsample_lum(t_rgb, max_dim)
    # frame 为同一 max_dim 降采样坐标（detect_regions 内传入 det["frame"]）
    fy0, fx0, fy1, fx1 = frame
    sub = lum[fy0:fy1, fx0:fx1]
    H, W = sub.shape
    if H < 16 or W < 16:
        return None
    y0, y1, x0, x1 = 0, H, 0, W

    rowmed = np.median(sub, axis=1)
    colmed = np.median(sub, axis=0)

    def _smooth(a: np.ndarray, k: int = 9) -> np.ndarray:
        if a.size < k:
            return a
        pad = k // 2
        ap = np.pad(a, pad, mode="edge")
        c = np.cumsum(np.insert(ap, 0, 0.0))
        return (c[k:] - c[:-k]) / k   # 长度恰为 len(a)，窗口中心对齐

    def _strip_end(profile: np.ndarray, from_end: bool) -> int:
        """结构先验：[帧边暗隙] [片基亮带] [暗印边带] [画面...]（自片基端向内）。

        1) 从帧端向内进入亮带（>0.6*外侧半区峰值）；
        2) 亮带内缘 = 亮度跌回 <0.4*峰值处；
        3) 继续向内穿过暗印边带（<0.1*峰值，不透光印字近黑），
           画面端点 = 首个不再低于阈值的行。
        无暗印带（片基直接邻接画面）时端点=亮带内缘。
        """
        n = len(profile)
        ps = _smooth(profile)
        outer = ps[n // 2:] if from_end else ps[:n // 2]
        peak = max(float(outer.max()), 1e-6)
        rng_in = range(n - 1, n // 2, -1) if from_end else range(0, n // 2)
        base_inner = None
        in_base = False
        base_start = None
        for i in rng_in:
            if not in_base:
                if ps[i] > 0.6 * peak:
                    in_base = True
                    base_start = i
            elif ps[i] < 0.4 * peak:
                base_inner = i
                break
        if base_inner is None:
            return n // 2
        # 平坦度守卫：真片基是均匀透光带（变异系数低）；亮沙地/天空等
        # 场景高光带方差大，不得当作片基来裁（CN400 目检教训）。
        band = profile[min(base_start, base_inner):max(base_start, base_inner) + 1] \
            if base_start is not None else profile[base_inner:base_inner + 1]
        if band.size >= 4:
            m = float(np.mean(band))
            cv = float(np.std(band)) / max(m, 1e-6)
            if cv > 0.35:
                return n - 1 if from_end else 0   # 无可信片基带：不裁这条
        rng_dark = (range(base_inner - 1, n // 2, -1) if from_end
                    else range(base_inner + 1, n // 2))
        img_end = base_inner
        dark_thr = 0.1 * peak
        for i in rng_dark:
            if ps[i] >= dark_thr:
                img_end = i + (1 if from_end else -1)
                break
            img_end = i
        return int(np.clip(img_end, 2, n - 2))

    if base_band is not None:
        # base_band 为绝对降采样坐标 -> 转帧内相对
        base_band = (base_band[0] - fy0, base_band[1] - fx0,
                     base_band[2] - fy0, base_band[3] - fx0)

    # 结构约束：画面端点不得进入片基带；也不得越过带端 +6% 帧长
    # （暗云合法地比 0.1*peak 更暗时，暗带扫描会切过头 -> 上限保护）。
    slack = max(16, int(0.06 * max(H, W)))
    if base_edge == "bottom":
        y1 = _strip_end(rowmed, from_end=True)
        if base_band:
            y1 = int(np.clip(y1, base_band[0] - slack, base_band[0]))
    elif base_edge == "top":
        y0 = _strip_end(rowmed, from_end=False)
        if base_band:
            y0 = int(np.clip(y0, base_band[2], base_band[2] + slack))
    elif base_edge == "right":
        x1 = _strip_end(colmed, from_end=True)
        if base_band:
            x1 = int(np.clip(x1, base_band[1] - slack, base_band[1]))
    elif base_edge == "left":
        x0 = _strip_end(colmed, from_end=False)
        if base_band:
            x0 = int(np.clip(x0, base_band[3], base_band[3] + slack))

    # 垂直条带轴：从两端向内修剪"片基亮度"的列——齿孔边带是亮片基+暗印字，
    # 整体亮度接近片基本身（实测 colmed≈0.13 vs 片基 0.15 vs 画面 0.05），
    # 只能靠"比 0.7*片基亮"从边缘向内判定；内部列永不修剪（亮雪景安全）。
    region = sub[y0:y1, :]
    if base_edge == "bottom":
        base_level = float(np.max(rowmed[y1:])) if y1 < H else 0.0
    elif base_edge == "top":
        base_level = float(np.max(rowmed[:y0])) if y0 > 0 else 0.0
    elif base_edge == "right":
        base_level = float(np.max(colmed[x1:])) if x1 < W else 0.0
    else:
        base_level = float(np.max(colmed[:x0])) if x0 > 0 else 0.0
    cm = np.median(region, axis=0)
    max_edge = max(2, int(0.15 * len(cm)))   # 片边带从不宽于15%；更内的亮列视为场景
    if base_level > 1e-6 and base_edge in ("top", "bottom"):
        bright = cm > 0.7 * base_level
        # 帧最外列可能是暗隙（不透光边框）：先跳过暗列再进入齿孔亮带
        xa = 0
        while xa < max_edge and not bright[xa]:
            xa += 1
        while xa < max_edge and bright[xa]:
            xa += 1
        xb = len(bright)
        lim = len(bright) - max_edge
        while xb > lim and not bright[xb - 1]:
            xb -= 1
        while xb > lim and bright[xb - 1]:
            xb -= 1
        if xb - xa >= 0.3 * len(bright):
            x0, x1 = xa, xb
    elif base_level > 1e-6:      # left/right：修剪行（片边在上下）
        rm = np.median(region[:, x0:x1], axis=1)
        bright = rm > 0.7 * base_level
        max_edge = max(2, int(0.15 * len(rm)))
        ya = 0
        while ya < max_edge and not bright[ya]:
            ya += 1
        while ya < max_edge and bright[ya]:
            ya += 1
        yb = len(bright)
        lim = len(bright) - max_edge
        while yb > lim and not bright[yb - 1]:
            yb -= 1
        while yb > lim and bright[yb - 1]:
            yb -= 1
        if yb - ya >= 0.3 * len(bright):
            y0, y1 = ya, yb
    if x1 - x0 < 0.3 * W or y1 - y0 < 0.3 * H:
        return None

    out = (fy0 + y0) * step, (fx0 + x0) * step, (fy0 + y1) * step, (fx0 + x1) * step
    if out[2] - out[0] < 32 or out[3] - out[1] < 32:
        return None
    return out


def iou(a: tuple, b: tuple) -> float:
    ay0, ax0, ay1, ax1 = a
    by0, bx0, by1, bx1 = b
    iy0, ix0 = max(ay0, by0), max(ax0, bx0)
    iy1, ix1 = min(ay1, by1), min(ax1, bx1)
    inter = max(0, iy1 - iy0) * max(0, ix1 - ix0)
    union = ((ay1 - ay0) * (ax1 - ax0) + (by1 - by0) * (bx1 - bx0) - inter)
    return inter / union if union > 0 else 0.0


def detect_regions(t_rgb: np.ndarray, max_dim: int = 800) -> dict:
    """全图 -> 帧、片基与画面区域 ROI（原图坐标）。

    失败时对应键为 None 并给出警告；image_area=None 时调用方不得静默
    用整帧冒充画面（整改 P0-C）。
    """
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
    out["image_area"] = detect_image_area(
        t_rgb, det["frame"], det.get("edge"), base_band=det.get("base"),
        max_dim=max_dim)
    if out["image_area"] is None and det["base"] is not None:
        out["warnings"].append("画面区域检测失败：默认输出将保留片边（可手动裁切）")
    return out
