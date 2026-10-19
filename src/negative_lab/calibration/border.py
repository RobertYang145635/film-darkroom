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


# ------------------------------------------------------------- 端点来源标记
# v6.1 P1 用一个 bool（used_fallback）表达「是否跳过 base_band 钳制」，但实现里
# 有三条性质完全不同的路径都返回 True，其中「什么都没找到」的退化端点 n//2
# 因此绕过了原有保护 —— 画面中点可被当作端点，最多裁掉 50% 帧高/帧宽。
# 改成语义明确的来源标记：只有 STRUCTURAL / NO_BASE 允许跳过钳制。
ENDPOINT_STRUCTURAL = "structural"      # 锚定帧缘亮带内缘：比粗 base 带更准
ENDPOINT_DARK_SCAN = "dark_scan"        # 正常暗印带扫描结果：仍受 base_band 约束
ENDPOINT_NO_BASE = "no_base"            # 判为无可信片基带：这一条整条不裁
ENDPOINT_DEGENERATE = "degenerate"      # 结构完全没找到：必须回落到 base_band 钳制
SKIP_BAND_CLIP = (ENDPOINT_STRUCTURAL, ENDPOINT_NO_BASE)


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

    def _strip_end(profile: np.ndarray, from_end: bool) -> tuple[int, str]:
        """结构先验：[帧边暗隙] [片基亮带] [暗印边带] [画面...]（自片基端向内）。

        1) 从帧端向内进入亮带（>0.6*外侧半区峰值）；
        2) 亮带内缘 = 亮度跌回 <0.4*峰值处；
        3) 继续向内穿过暗印边带（<0.1*峰值，不透光印字近黑），
           画面端点 = 首个不再低于阈值的行。
        无暗印带（片基直接邻接画面）时端点=亮带内缘。
        返回 (端点, 来源标记)：STRUCTURAL=锚定帧缘亮带内缘（比粗 base 带更准，
        不得再被拉回带缘）；NO_BASE=判为无可信片基带（整条不裁）；
        DARK_SCAN=正常暗印带扫描；DEGENERATE=结构没找到（回落钳制保护）。
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
            # v6.1 P1（G200-4 诊断）：亮度跌不回 0.4*peak = 片基后没有
            # 暗印边带（片基直接邻接画面）。此时端点取**锚定帧缘**的
            # >=0.5*peak 亮带（起点距帧缘 <=6% 帧长——片基带贴帧缘，
            # 场景亮带/齿孔亮带浮在中间，不锚定帧缘）的内缘。
            th = ps > 0.5 * peak
            runs: list[tuple[int, int]] = []
            s = None
            for i, v in enumerate(th):
                if v and s is None:
                    s = i
                elif not v and s is not None:
                    runs.append((s, i))
                    s = None
            if s is not None:
                runs.append((s, len(th)))
            runs = [(a, b) for a, b in runs if a >= 2 or b - a > 2]
            edge_anchored = max(4, int(0.06 * n))
            for a, b in runs:
                # 锚定端 = 靠帧缘的游程外端；画面端点 = 游程内缘
                if from_end:
                    anchored, inner = (n - b) <= edge_anchored, a
                else:
                    anchored, inner = a <= edge_anchored, b - 1
                if not anchored:
                    continue
                return int(np.clip(inner, 2, n - 2)), ENDPOINT_STRUCTURAL
            return n # 2, ENDPOINT_DEGENERATE
        # 平坦度守卫：真片基是均匀透光带（变异系数低）；亮沙地/天空等
        # 场景高光带方差大，不得当作片基来裁（CN400 目检教训）。
        band = profile[min(base_start, base_inner):max(base_start, base_inner) + 1] \
            if base_start is not None else profile[base_inner:base_inner + 1]
        if band.size >= 4:
            m = float(np.mean(band))
            cv = float(np.std(band)) / max(m, 1e-6)
            if cv > 0.35:
                return (n - 1 if from_end else 0), ENDPOINT_NO_BASE   # 无可信片基带：不裁这条
        rng_dark = (range(base_inner - 1, n // 2, -1) if from_end
                    else range(base_inner + 1, n // 2))
        img_end = base_inner
        dark_thr = 0.1 * peak
        for i in rng_dark:
            if ps[i] >= dark_thr:
                img_end = i + (1 if from_end else -1)
                break
            img_end = i
        return int(np.clip(img_end, 2, n - 2)), ENDPOINT_DARK_SCAN

    if base_band is not None:
        # base_band 为绝对降采样坐标 -> 转帧内相对
        base_band = (base_band[0] - fy0, base_band[1] - fx0,
                     base_band[2] - fy0, base_band[3] - fx0)

    # 结构约束：画面端点不得进入片基带；也不得越过带端 +6% 帧长
    # （暗云合法地比 0.1*peak 更暗时，暗带扫描会切过头 -> 上限保护）。
    # 只有 STRUCTURAL（已落在片基亮带内缘）与 NO_BASE（整条不裁）可以跳过
    # 这个钳制；DEGENERATE / DARK_SCAN 必须受约束（审查修复 2026-09-08：
    # 原 bool 把三种情况混为一谈，退化端点 n//2 因此能裁掉半幅）。
    slack = max(16, int(0.06 * max(H, W)))
    y1_src = y0_src = x1_src = x0_src = ENDPOINT_DARK_SCAN
    if base_edge == "bottom":
        y1, y1_src = _strip_end(rowmed, from_end=True)
        if base_band and y1_src not in SKIP_BAND_CLIP:
            y1 = int(np.clip(y1, base_band[0] - slack, base_band[0]))
    elif base_edge == "top":
        y0, y0_src = _strip_end(rowmed, from_end=False)
        if base_band and y0_src not in SKIP_BAND_CLIP:
            y0 = int(np.clip(y0, base_band[2], base_band[2] + slack))
    elif base_edge == "right":
        x1, x1_src = _strip_end(colmed, from_end=True)
        if base_band and x1_src not in SKIP_BAND_CLIP:
            x1 = int(np.clip(x1, base_band[1] - slack, base_band[1]))
    elif base_edge == "left":
        x0, x0_src = _strip_end(colmed, from_end=False)
        if base_band and x0_src not in SKIP_BAND_CLIP:
            x0 = int(np.clip(x0, base_band[3], base_band[3] + slack))

    # v6.1 P1（G200-4 诊断）：**片基对侧**的端点同样会带进齿孔/边印
    # 亮带（该侧没有 base_band 参照，之前从不修剪）。片边亮带的判据
    # 与片基同源：>=0.7*片基亮度（切的是"亮"结构，暗场景安全）、
    # 自帧缘向内找"暗隙->亮带->跌落"结构，且从不越过 15% 帧长。
    def _trim_opposite_end(pm: np.ndarray, from_end: bool,
                           base_ref: float) -> int | None:
        """片基对侧端点：自帧缘向内找 [暗隙 -> >=0.7*片基亮带 -> 跌落]。

        返回亮带内缘（跌落处），无结构或亮带越过 15% 帧长时返回 None
        （不裁）。暗场景列永不修剪（只对"亮"结构下手）。
        """
        n = len(pm)
        lim = max(2, int(0.15 * n))
        thr = 0.7 * base_ref
        idx = range(n - 1, -1, -1) if from_end else range(n)
        seen_bright = False
        for i in idx:
            if abs(i - (n - 1 if from_end else 0)) >= lim:
                break
            if pm[i] > thr:
                seen_bright = True
                continue
            if seen_bright:
                return i      # 亮带之后的首个暗位 = 亮带内缘
        return None

    # 审查修复（2026-09-08）：扫描剖面改用平滑值（兄弟函数 _strip_end 一直用
    # _smooth 并注明原因，这里此前用原始中位线，单个尘点/高光噪声即可造出假
    # "亮带内缘"）；片基参考位由 max 改 p95 —— 带区里最亮的一个像素不该抬高
    # 整条阈值，那会让对侧修剪在噪声样本上随机失灵或随机误裁。
    colmed_s, rowmed_s = _smooth(colmed), _smooth(rowmed)
    if base_edge == "left" and x0 < W - 8:
        base_level_opp = (float(np.percentile(colmed[:x0 + 1], 95))
                          if x0 > 0 else 0.0)
        if base_level_opp > 1e-6:
            e = _trim_opposite_end(colmed_s, from_end=True,
                                   base_ref=base_level_opp)
            if e is not None and e > x0 + 0.3 * W:
                x1 = min(x1, e)
    elif base_edge == "right" and x1 > 8:
        base_level_opp = (float(np.percentile(colmed[x1:], 95))
                          if x1 < W else 0.0)
        if base_level_opp > 1e-6:
            e = _trim_opposite_end(colmed_s, from_end=False,
                                   base_ref=base_level_opp)
            if e is not None and e < x1 - 0.3 * W:
                x0 = max(x0, e)
    if base_edge == "top" and y0 < H - 8:
        base_level_opp = (float(np.percentile(rowmed[:y0 + 1], 95))
                          if y0 > 0 else 0.0)
        if base_level_opp > 1e-6:
            e = _trim_opposite_end(rowmed_s, from_end=True,
                                   base_ref=base_level_opp)
            if e is not None and e > y0 + 0.3 * H:
                y1 = min(y1, e)
    elif base_edge == "bottom" and y1 > 8:
        base_level_opp = (float(np.percentile(rowmed[y1:], 95))
                          if y1 < H else 0.0)
        if base_level_opp > 1e-6:
            e = _trim_opposite_end(rowmed_s, from_end=False,
                                   base_ref=base_level_opp)
            if e is not None and e < y1 - 0.3 * H:
                y0 = max(y0, e)

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


def refine_base_roi(t_rgb: np.ndarray, roi, *, min_frac: float = 0.10,
                    window: float = 0.14) -> tuple[int, int, int, int] | None:
    """把自动检测的片基条带收窄到"最亮的稳定子带"（v5 §5.3.6）。

    自动检测给出的条带常包含齿孔、边码和暗画框边。这里沿条带轴滑动窗口，
    选择"中位亮度高且变异系数低"的子窗口作为未曝光片基取样区。
    用户/标注给出的取样区不做此改写（用户为准）。
    """
    a = np.asarray(t_rgb, np.float32)
    h, w = a.shape[:2]
    y0, x0, y1, x1 = (int(v) for v in roi)
    y0, x0 = max(0, y0), max(0, x0)
    y1, x1 = min(h, y1), min(w, x1)
    if y1 - y0 < 8 or x1 - x0 < 8:
        return None
    seg = a[y0:y1, x0:x1]
    vertical = seg.shape[0] >= seg.shape[1]      # 高>宽：条带沿行方向
    n = seg.shape[0] if vertical else seg.shape[1]
    win = max(8, int(n * window))
    lum = seg.mean(axis=2)
    best, best_score = None, -1.0
    for start in range(0, max(1, n - win + 1), max(1, win // 2)):
        part = lum[start:start + win] if vertical else lum[:, start:start + win]
        m = float(np.mean(part))
        cv = float(np.std(part)) / max(m, 1e-6)
        score = m * (1.0 - min(cv, 0.6))
        if score > best_score:
            best, best_score = start, score
    if best is None:
        return None
    end = min(n, best + win)
    if (end - best) < max(8, int(n * min_frac)):
        return None
    if vertical:
        return (y0 + best, x0, y0 + end, x1)
    return (y0, x0 + best, y1, x0 + end)


def iou(a: tuple, b: tuple) -> float:
    ay0, ax0, ay1, ax1 = a
    by0, bx0, by1, bx1 = b
    iy0, ix0 = max(ay0, by0), max(ax0, bx0)
    iy1, ix1 = min(ay1, by1), min(ax1, bx1)
    inter = max(0, iy1 - iy0) * max(0, ix1 - ix0)
    union = ((ay1 - ay0) * (ax1 - ax0) + (by1 - by0) * (bx1 - bx0) - inter)
    return inter / union if union > 0 else 0.0


REBATE_MIN_MEAN = 0.25          # 比这更暗的是黑边框，不是片基
REBATE_RB_RANGE = (1.25, 2.8)   # 橙罩区间；白边 R/B<1、黑边无定义
REBATE_PEGGED = 0.995           # 任一通道贴顶 ⇒ 扫描仪白边/过曝，不是胶片
REBATE_MAX_ABOVE_BASE = 0.01    # 用它归一后画面"比片基还亮"的比例上限（物理：应 ≈0）


def detect_rebate_strips(t_rgb: np.ndarray, *, band_frac: float = 0.012,
                         max_dim: int = 1600) -> tuple[list, list]:
    """扫描输入里贴在图像四边的**未曝光片基 (rebate)** → 片基取样条。

    为什么需要它（2026-10-19 实测，ECN-2 两卷 70 帧）：现有 `detect_base_strip`
    只在**检出的画框内部**找最亮边带，而扫描文件的片基在画框**外面**的边上；
    于是 ECN-2 有 18/37 帧退到"画面内橙色实测"，采用的片基比真实片基暗
    **R 0.17 / G 0.34 / B 0.49 decade**，把彩度整体带偏（喂真片基后
    `Cab` 误差 12.27 → 5.43）。

    三道闸缺一不可（每一道都是被一次真实误判逼出来的）：
    ① 亮度均值 ≥ 0.25：5219 的四条边实测 t≈0.0003，是**黑边框**，不是片基；
    ② R/B ∈ [1.25, 2.8] 且任一通道未贴顶：5207 的**右边**测出 [0.86, 0.94, 1.00]，
       那是扫描仪白边——第一版没有这道闸就把它当片基用了；
    ③ 用它归一后画面中心 `above_base ≤ 1%`：负片内不可能有比未曝光片基更亮的像素。
       5207 上/下/左三条边实测 0.02~0.06%，5219 全边 98% ⇒ 这道闸直接把没有片边的卷拒掉。

    返回 `(roi 列表, 每条的 (边名, R/B, 均值))`；没有合格条带时返回 `([], [])`，
    调用方必须回退到原有检测，**不许**因为"找不到"就退回通用先验。
    """
    h, w = t_rgb.shape[:2]
    step = max(1, int(np.ceil(max(h, w) / float(max_dim))))
    a = t_rgb[::step, ::step, :].astype(np.float64)
    ah, aw = a.shape[:2]
    bh, bw = max(4, int(ah * band_frac)), max(4, int(aw * band_frac))
    x0, x1 = int(aw * 0.15), int(aw * 0.85)      # 避开齿孔与四角印字
    y0, y1 = int(ah * 0.15), int(ah * 0.85)
    cands = {"上": (0, bh, x0, x1), "下": (ah - bh, ah, x0, x1),
             "左": (y0, y1, 0, bw), "右": (y0, y1, aw - bw, aw)}
    cen = a[y0:y1, x0:x1].reshape(-1, 3)
    scored = []
    for name, (ry0, ry1, rx0, rx1) in cands.items():
        band = a[ry0:ry1, rx0:rx1].reshape(-1, 3)
        lum = band.mean(axis=1)
        sel = band[lum >= np.quantile(lum, 0.90)]        # 带内最亮的一档
        m = sel.mean(axis=0)
        rb = float(m[0] / max(m[2], 1e-4))
        if (float(m.mean()) < REBATE_MIN_MEAN or not
                (REBATE_RB_RANGE[0] <= rb <= REBATE_RB_RANGE[1]) or float(m.max()) >= REBATE_PEGGED):
            continue
        if float((cen > m[None, :]).mean()) > REBATE_MAX_ABOVE_BASE:
            continue
        spread = float(np.std(sel.mean(axis=1)) / max(np.mean(sel.mean(axis=1)), 1e-3))
        scored.append(((float(m.mean()) * min(rb, 3.0)) / (1.0 + 3.0 * spread),
                       name, (ry0 * step, rx0 * step, ry1 * step, rx1 * step),
                       rb, float(m.mean())))
    scored.sort(key=lambda x: -x[0])
    keep = [s for s in scored[:2] if s[0] >= 0.35 * scored[0][0]]
    return ([s[2] for s in keep],
            [(s[1], round(s[3], 2), round(s[4], 3)) for s in keep])


#: **单帧平铺扫描**的判据：检出的画框占整幅 ≥ 此比例，说明扫描仪已经把扫描
#: 裁到胶片画框本身——此后任何内缩都必须有**物理片基证据**才允许。
#: 实测（`tmp_ecn2_frame_gate.py`，2026-10-20）：ECN-2 两卷 0.939~1.001，
#: G200 条带扫描 0.760~0.764 ⇒ 阈值 0.85 两侧各留 ≥0.09 余量。
FRAME_FLUSH_FRAC = 0.85


def image_area_from_rebate(frame: tuple[int, int, int, int],
                           rebate: tuple[list, list]) -> tuple[tuple, dict]:
    """平铺单帧扫描的画面上边界：**四条边各自只在"该边有合格片基条带"时才内缩**。

    为什么需要它（2026-10-20 看图 + 实测，`outputs/ref_look_probe/
    ecn2_frame_boxes_*.png`）：`detect_image_area` 的结构模型是为**条带扫描**写的
    （`[帧边暗隙][片基亮带][暗印边带][画面]`），单帧平铺扫描里没有这套结构，它会在
    **画面内部**找到一条"亮带"并把带外全裁掉：5207-6 的 `image_area` 只剩整幅的
    35.8%（把 LAWSON 招牌整个切在框外），5219 四帧 52~57%。
    与他的正片做尺度×位移互相关（`tmp_ecn2_rebate_frame.py`，8 帧两卷）：
      不裁（整幅） s=1.05，|r|=0.90~0.98 **八帧全中**；
      现 `image_area` s=0.60~1.05，|r|=0.63~0.98。
    ⇒ 取景权交给物理证据：`detect_rebate_strips` 的三道闸（亮度、R/B、
    `above_base≤1%`）在该边接受了条带，才从条带内缘往里裁；没有证据的边一律不动。

    ⚠ 与 v6.1 整改 P0-C（"检测失败时不得静默用整帧冒充画面"）**不冲突**：
    这里不是"检测失败后的兜底"，而是**平铺扫描的正判**——帧已铺满扫描这一条本身就是
    边界证据；结果写进 `prov["image_area"]="auto-rebate-bounded"` 与逐边内缩量，
    且置信度低时 `needs_crop_review` 照旧上报，不静默。
    返回 `(roi, {边名: 内缩比例})`。
    """
    fy0, fx0, fy1, fx1 = frame
    y0, x0, y1, x1 = fy0, fx0, fy1, fx1
    fh, fw = max(1, fy1 - fy0), max(1, fx1 - fx0)
    inset = {"上": 0.0, "下": 0.0, "左": 0.0, "右": 0.0}
    rois, info = rebate or ([], [])
    for roi, (side, _rb, _m) in zip(rois, info):
        ry0, rx0, ry1, rx1 = roi
        if side == "上" and ry1 > y0:
            y0, inset["上"] = min(ry1, y1), (min(ry1, y1) - fy0) / fh
        elif side == "下" and ry0 < y1:
            y1, inset["下"] = max(ry0, y0), (fy1 - max(ry0, y0)) / fh
        elif side == "左" and rx1 > x0:
            x0, inset["左"] = min(rx1, x1), (min(rx1, x1) - fx0) / fw
        elif side == "右" and rx0 < x1:
            x1, inset["右"] = max(rx0, x0), (fx1 - max(rx0, x0)) / fw
    return (int(y0), int(x0), int(y1), int(x1)), inset



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
