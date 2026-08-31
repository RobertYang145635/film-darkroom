"""基础除尘/划痕（规格 §7，P1）。

原则：
- 手动蒙版是必需路径；自动检测只做保守候选（小孤立斑点、水平/垂直高置信直线划痕）。
- 修复发生在**负片透射率域**（反相之前），尘点表现为局部透射率骤降。
- 修复可整体关闭（完全回退原图）；修复失败绝不阻断核心去色罩导出（safe_inpaint）。
- 不得把胶片颗粒/细线/高反差边缘误当尘点：候选密度过高时直接放弃自动模式
  （§7"误检率高时应只提供手动模式"）。斜划痕不做自动，留给手动。

实现全部基于积分图/移位求和的向量化运算：45MP 全分辨率可行
（真实样片教训：纯 Python BFS 连通域在 8280×5520 上超时）。
"""

from __future__ import annotations

import numpy as np


def detect_dust(t_rgb: np.ndarray,
                spot_max_area: int = 80,
                min_contrast: float = 0.35,
                scratch_min_length: int = 120,
                scratch_max_width: int = 3,
                use_scratches: bool = True,
                max_candidate_frac: float = 0.02,
                region: np.ndarray | None = None) -> dict:
    """保守自动检测。返回 dict: mask(H,W bool), n_spots, n_scratches, warnings。

    n_spots / n_scratches 是候选**像素数**（近似量，用于报告与告警阈值）。
    region: (H,W) bool，仅在该区域内检测（如片基条带）。真实画面内容密度
    高，11x11 max 背景滤波会把内容暗区整体判为候选（实测 21%），自动除尘
    应限定在片基/边缘条带等均匀区；内容区尘点走手动蒙版。

    步骤：
    1. 背景 = 局部极大值滤波（3x3 五遍 ≈ 11x11 max）。尘点比背景暗，max 滤波
       不被斑点自身污染；中值滤波会把斑点"吸收"进背景导致漏检（实测教训）。
    2. 候选 = 比背景暗 min_contrast 以上。候选占比 > max_candidate_frac 时
       放弃自动模式（颗粒/边缘/脏污误检风险高）。
    3. 斑点：9x9 窗口候选数 <= spot_max_area 且 15x15 内仍孤立（<=1.6 倍），
       即"小且周围回到背景"。
    4. 划痕：以像素为中心的 (scratch_min_length x 1) 竖窗（或 1 x 横窗）候选
       占比 >= 0.85（连续长 run），且 3x(2w+1) 邻域占比 <= 0.75（细）。
    """
    t = np.asarray(t_rgb, dtype=np.float32)
    warnings: list[str] = []
    lum = t.mean(axis=2)

    # region 限定：裁到包围盒并内缩，避免区域边界的内容暗带被误判为划痕，
    # 同时把 45MP 全图滤波降为条带规模（实测 51s -> ~4s）。
    off_y = off_x = 0
    work = lum
    if region is not None:
        ys, xs = np.nonzero(region.astype(bool))
        if ys.size == 0:
            return {"mask": np.zeros(lum.shape, bool), "n_spots": 0,
                    "n_scratches": 0, "warnings": ["region 为空"]}
        margin = max(8, scratch_min_length // 4)
        y0 = max(0, int(ys.min()) + margin)
        y1 = min(lum.shape[0], int(ys.max()) + 1 - margin)
        x0 = max(0, int(xs.min()) + margin)
        x1 = min(lum.shape[1], int(xs.max()) + 1 - margin)
        if y1 - y0 < 16 or x1 - x0 < 16:
            return {"mask": np.zeros(lum.shape, bool), "n_spots": 0,
                    "n_scratches": 0,
                    "warnings": ["region 内缩后过小，跳过自动检测"]}
        off_y, off_x = y0, x0
        work = lum[y0:y1, x0:x1]
        reg_sub = region[y0:y1, x0:x1].astype(bool)

    bg = work
    for _ in range(5):        # 3x3 max 五遍 ≈ 11x11：必须大于最大尘点尺寸，
        bg = _max3x3(bg)      # 否则斑点自身污染背景估计导致中心 rel≈0 漏检
    rel = (bg - work) / np.maximum(bg, 1e-6)
    cand = rel > min_contrast
    if region is not None:
        cand &= reg_sub

    # 均匀性门控（目检教训）：region 内背景亮度相对散布过大说明包含
    # 片基->黑边过渡带/齿孔/片边字样，边界会被整圈误判为"划痕"。
    if region is not None:
        med = float(np.median(work))
        q20, q80 = np.quantile(work, [0.2, 0.8])
        spread = (q80 - q20) / max(med, 1e-6)
        if spread > 0.35:
            warnings.append(
                f"检测区域背景不均（散布 {spread:.2f}，含过渡带/齿孔/字样），"
                "自动除尘不可靠，只提供手动蒙版（§7）")
            return {"mask": np.zeros(lum.shape, bool), "n_spots": 0,
                    "n_scratches": 0, "warnings": warnings}

    if cand.mean() > max_candidate_frac:
        warnings.append(
            f"暗点候选占比 {cand.mean():.1%} 过高（颗粒/边缘/脏污），"
            "自动除尘不可靠，只提供手动蒙版（§7）")
        return {"mask": np.zeros(lum.shape, bool), "n_spots": 0,
                "n_scratches": 0, "warnings": warnings}

    # 逐像素均匀性门控（region 模式）：局部 21x21 均值偏离区域中位数 >25%
    # 的像素（过渡带、齿孔阴影、乳剂字样）不得进入任何自动候选。
    uniform = np.ones(work.shape, bool)
    if region is not None:
        med_u = float(np.median(work))
        lm21 = _box_mean(work, 21, 21)
        uniform = np.abs(lm21 - med_u) / max(med_u, 1e-6) <= 0.15

    # 斑点：小且孤立，且各向同性（长条带不是斑点）
    ci = _integral(cand)                      # 积分图只算一次，全部窗口复用
    s9 = _frac_from_ii(ci, 9, 9) * 81.0
    s15 = _frac_from_ii(ci, 15, 15) * 225.0
    col_f = _frac_from_ii(ci, 21, 1)
    row_f = _frac_from_ii(ci, 1, 21)
    spots = (cand & (s9 >= 1) & (s9 <= spot_max_area)
             & (s15 <= spot_max_area * 1.6)
             & (col_f <= 0.5) & (row_f <= 0.5) & uniform)

    # 划痕：长而细（仅水平/垂直），且两侧背景亮度相似（排除高反差直边暗带）
    scratches = np.zeros(work.shape, bool)
    if use_scratches:
        L = int(scratch_min_length)
        w = int(scratch_max_width)
        lm = _box_mean(work, L, 3)
        left = np.roll(lm, 3, axis=1)    # 窗口中心左移 3 列的背景均值
        right = np.roll(lm, -3, axis=1)
        lo = np.minimum(left, right)
        hi = np.maximum(left, right)
        sides_v = (lo > 1e-6) & (hi / np.maximum(lo, 1e-9) <= 2.0)
        lm_h = _box_mean(work, 3, L)
        up = np.roll(lm_h, 3, axis=0)
        dn = np.roll(lm_h, -3, axis=0)
        lo = np.minimum(up, dn)
        hi = np.maximum(up, dn)
        sides_h = (lo > 1e-6) & (hi / np.maximum(lo, 1e-9) <= 2.0)
        vert = ((_frac_from_ii(ci, L, 1) >= 0.85)
                & (_frac_from_ii(ci, 3, 2 * w + 1) <= 0.75) & sides_v)
        horiz = ((_frac_from_ii(ci, 1, L) >= 0.85)
                 & (_frac_from_ii(ci, 2 * w + 1, 3) <= 0.75) & sides_h)
        # 居中窗口在划痕端部覆盖率不足 -> 只标中段；把核心标记沿轴
        # 向两端传播（限 cand 内、距离 <= L/2），向量化实现。
        vert = _propagate_along(vert, cand, L, axis=0)
        horiz = _propagate_along(horiz, cand, L, axis=1)
        scratches = (vert | horiz) & uniform

    sub_mask = spots | scratches
    if region is None:
        mask = sub_mask
    else:
        mask = np.zeros(lum.shape, bool)
        h, w_ = sub_mask.shape
        mask[off_y:off_y + h, off_x:off_x + w_] = sub_mask
    if mask.any():
        warnings.append("自动候选为保守估计，请检查红色蒙版区域后再启用修复")
    return {"mask": mask, "n_spots": int(spots.sum()),
            "n_scratches": int(scratches.sum()), "warnings": warnings}


def safe_inpaint(t_rgb: np.ndarray, mask: np.ndarray | None,
                 radius: int = 3) -> tuple[np.ndarray, str | None]:
    """修复失败时回退原图并返回错误信息——核心导出绝不因修复中断（§7）。"""
    try:
        return inpaint(t_rgb, mask, radius=radius), None
    except Exception as e:  # noqa: BLE001
        return np.asarray(t_rgb, dtype=np.float32).copy(), \
            f"除尘修复失败，已回退未修复数据：{type(e).__name__}: {e}"


def inpaint(t_rgb: np.ndarray, mask: np.ndarray, radius: int = 3) -> np.ndarray:
    """蒙版内用邻域均值迭代回填（负片域修复，保持线性语义）。

    只在蒙版包围盒（外扩迭代半径）内计算——45MP 全图逐轮拷贝不可接受
    （实测教训：全图迭代 38s）。蒙版为空时逐比特恒等。
    """
    t = np.asarray(t_rgb, dtype=np.float32)
    if mask is None or not np.any(mask):
        return t.copy()
    out = t.copy()
    m = mask.astype(bool)
    pad = max(2, radius * 3) + 1
    ys, xs = np.nonzero(m)
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(t.shape[0], int(ys.max()) + 1 + pad)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(t.shape[1], int(xs.max()) + 1 + pad)
    sub = out[y0:y1, x0:x1]
    subm = m[y0:y1, x0:x1]
    bad = subm & ~np.isfinite(sub).all(axis=2)
    if bad.any():                       # 蒙版内非有限值先兜底
        sub[bad] = bg_mean(sub, ~subm)
    filled = ~subm
    for _ in range(max(2, radius * 3)):
        if filled.all():
            break
        kernel_sum, val_sum = _neighbor_sums(sub, filled)
        newly = subm & ~filled & (kernel_sum > 0)
        if not newly.any():
            break
        sub[newly] = val_sum[newly] / kernel_sum[newly][..., None]
        filled = filled | newly
    if not filled.all():                # 极端情况（整幅蒙版）用全局背景
        sub[~filled] = bg_mean(sub, filled)
    out[y0:y1, x0:x1] = sub
    return out.astype(np.float32)


def merge_masks(*masks: np.ndarray | None) -> np.ndarray:
    """手动蒙版 ∪ 自动候选（自动只是建议，手动是必需路径）。"""
    out = None
    for m in masks:
        if m is None:
            continue
        out = m.astype(bool).copy() if out is None else (out | m)
    return np.zeros((0, 0), bool) if out is None else out


def bg_mean(img: np.ndarray, valid: np.ndarray) -> float:
    v = img[valid]
    v = v[np.isfinite(v)]
    return float(np.mean(v)) if v.size else 0.0


# ---------------------------------------------------------------- 向量化内部

def _integral(mask: np.ndarray) -> np.ndarray:
    ii = np.zeros((mask.shape[0] + 1, mask.shape[1] + 1), np.int64)
    ii[1:, 1:] = np.cumsum(np.cumsum(mask.astype(np.int64), axis=0), axis=1)
    return ii


def _frac_from_ii(ii: np.ndarray, kh: int, kw: int) -> np.ndarray:
    """由预计算积分图求 (kh x kw) 中心窗口候选占比（边缘截断）。"""
    H, W = ii.shape[0] - 1, ii.shape[1] - 1
    ry, rx = kh // 2, kw // 2
    y0 = np.clip(np.arange(H) - ry, 0, H)
    y1 = np.clip(np.arange(H) + ry + 1, 0, H)
    x0 = np.clip(np.arange(W) - rx, 0, W)
    x1 = np.clip(np.arange(W) + rx + 1, 0, W)
    cnt = ii[np.ix_(y1, x1)] - ii[np.ix_(y0, x1)] - ii[np.ix_(y1, x0)] + ii[np.ix_(y0, x0)]
    area = (y1 - y0)[:, None] * (x1 - x0)[None, :]
    return cnt / np.maximum(area, 1)


def _box_frac(mask: np.ndarray, kh: int, kw: int) -> np.ndarray:
    return _frac_from_ii(_integral(mask), kh, kw)


def _box_mean(a: np.ndarray, kh: int, kw: int) -> np.ndarray:
    """以像素为中心的 (kh x kw) 窗口均值（边缘截断窗口）。"""
    ii = np.zeros((a.shape[0] + 1, a.shape[1] + 1), np.float64)
    ii[1:, 1:] = np.cumsum(np.cumsum(a.astype(np.float64), axis=0), axis=1)
    H, W = a.shape
    ry, rx = kh // 2, kw // 2
    y0 = np.clip(np.arange(H) - ry, 0, H)
    y1 = np.clip(np.arange(H) + ry + 1, 0, H)
    x0 = np.clip(np.arange(W) - rx, 0, W)
    x1 = np.clip(np.arange(W) + rx + 1, 0, W)
    cnt = ii[np.ix_(y1, x1)] - ii[np.ix_(y0, x1)] - ii[np.ix_(y1, x0)] + ii[np.ix_(y0, x0)]
    area = (y1 - y0)[:, None] * (x1 - x0)[None, :]
    return cnt / np.maximum(area, 1)


def _propagate_along(core: np.ndarray, cand: np.ndarray, L: int,
                     axis: int) -> np.ndarray:
    """把核心标记沿 axis 向两端扩展到距离 <= L/2 的 cand 像素（向量化）。"""
    if not core.any():
        return core
    n = core.shape[axis]
    idx = np.arange(n).reshape([-1 if axis == 0 else 1, -1 if axis == 1 else 1])
    pos = np.where(core, idx, 0)
    above = np.maximum.accumulate(pos, axis=axis)            # 上方/左侧最近核心
    d_down = idx - above
    pos2 = np.where(core, idx, n)
    rev = np.flip(pos2, axis=axis)
    below = np.flip(np.minimum.accumulate(rev, axis=axis), axis=axis)
    d_up = below - idx
    dist = np.minimum(d_down, d_up)
    return cand & (dist <= L // 2)


def _max3x3(a: np.ndarray) -> np.ndarray:
    p = np.pad(a, 1, mode="edge")
    out = p[1:-1, 1:-1].copy()
    for j in range(3):
        for i in range(3):
            np.maximum(out, p[j:j + a.shape[0], i:i + a.shape[1]], out=out)
    return out


def _neighbor_sums(img: np.ndarray, filled: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """4 邻域中已填充像素数量与像素值之和。"""
    h, w, c = img.shape
    ks = np.zeros((h, w), np.float32)
    vs = np.zeros((h, w, c), np.float32)
    f4 = np.zeros((h + 2, w + 2), bool)
    i4 = np.zeros((h + 2, w + 2, c), np.float32)
    f4[1:-1, 1:-1] = filled
    i4[1:-1, 1:-1] = img
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        ks += f4[1 + dy:1 + dy + h, 1 + dx:1 + dx + w].astype(np.float32)
        vs += i4[1 + dy:1 + dy + h, 1 + dx:1 + dx + w] * \
            f4[1 + dy:1 + dy + h, 1 + dx:1 + dx + w][..., None]
    return ks, vs
