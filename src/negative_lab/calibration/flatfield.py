"""Blank-light flat-field 与片基空间曲面（整改 v3 P0-C）。

暗角/导光不均的正当解法：估计乘性增益场 G(x,y,c)，使
T_corrected = T / G，G 由以下来源（优先级从高到低）：
1. blank frame（同装置同几何的未放胶片曝光帧）-> 完整 2D 曲面；
2. 片基条带 -> 沿条带轴的一维低阶多项式（横向分量不可观测，
   必须如实记录"部分覆盖"而不是假装全图已校正）。

曲面归一化到区域中位数=1，保证不改变 Dmin 的物理量级。
"""

from __future__ import annotations

import numpy as np


def _poly_design(yx: np.ndarray, order: int, dim2d: bool) -> np.ndarray:
    if dim2d:
        x, y = yx[:, 0], yx[:, 1]
        cols = []
        for i in range(order + 1):
            for j in range(order + 1 - i):
                cols.append((x ** i) * (y ** j))
        return np.stack(cols, axis=1)
    t = yx[:, 0]
    return np.stack([t ** i for i in range(order + 1)], axis=1)


def fit_base_surface(t_rgb: np.ndarray,
                     base_roi: tuple[int, int, int, int] | None = None,
                     blank: np.ndarray | None = None,
                     order: int = 2) -> dict:
    """返回 {"field": (h,w,3) 归一化增益场, "coverage": "2d"|"strip-1d"|"none",
    "residual": float, "warnings": [...]}。"""
    t = np.asarray(t_rgb, dtype=np.float32)
    h, w, _ = t.shape
    warnings: list[str] = []
    yy, xx = np.mgrid[0:h, 0:w]

    if blank is not None:
        b = np.asarray(blank, dtype=np.float64)
        ref = np.median(b[b > 1e-6]) if (b > 1e-6).any() else 1.0
        field = np.empty((h, w, 3), np.float32)
        resid = []
        for c in range(3):
            m = b[..., c] > max(1e-6, 0.05 * ref)
            A = _poly_design(np.stack([xx[m] / w, yy[m] / h], axis=1), order, True)
            coef, *_ = np.linalg.lstsq(A, np.log(np.maximum(b[m, c], 1e-9)), rcond=None)
            full = _poly_design(np.stack([xx.ravel() / w, yy.ravel() / h], axis=1),
                                order, True) @ coef
            g = np.exp(full.reshape(h, w))
            field[..., c] = (g / np.median(g)).astype(np.float32)
            resid.append(float(np.std(b[m, c] / g[m]) ))
        return {"field": field, "coverage": "2d",
                "residual": float(np.mean(resid)), "warnings": warnings}

    if base_roi is not None:
        y0, x0, y1, x1 = base_roi
        seg0 = t[y0:y1, x0:x1]                 # 先裁剪，坐标轴与实际切片一致
        strip_h, strip_w = seg0.shape[:2]
        field = np.ones((h, w, 3), np.float32)
        # 判定条带方向：高>宽 -> 条带沿行方向（1D 拟合 t=行坐标）
        vertical = strip_h >= strip_w
        coef_all = []
        for c in range(3):
            seg = t[y0:y1, x0:x1, c].astype(np.float64)
            vals = seg[seg > 1e-6]
            if vals.size < 32:
                warnings.append(f"通道 {c} 片基样本不足，曲面未拟合")
                coef_all.append(None)
                continue
            med = float(np.median(vals))
            flat_seg = seg.ravel()
            good = flat_seg > 1e-6
            if vertical:
                tt_full = np.repeat((np.arange(y0, y1) / h), strip_w)
            else:
                tt_full = np.tile((np.arange(x0, x1) / w), strip_h)
            vv = np.log(np.maximum(flat_seg[good], 1e-9))
            tt = tt_full[good]
            A = _poly_design(tt[:, None], min(order, 2), False)
            coef, *_ = np.linalg.lstsq(A, vv, rcond=None)
            coef_all.append(coef)
        if all(c is not None for c in coef_all):
            for c, coef in enumerate(coef_all):
                deg = len(coef) - 1
                if vertical:
                    tt = np.repeat((np.arange(h) / h)[:, None], w, axis=1)
                    g = np.exp(_poly_design(tt.ravel()[:, None], deg, False)
                               @ coef).reshape(h, w)
                    g = np.repeat(g[:, :1], w, axis=1)
                else:
                    tt = np.tile((np.arange(w) / w)[None, :], (h, 1))
                    g = np.exp(_poly_design(tt.ravel()[:, None], deg, False)
                               @ coef).reshape(h, w)
                    g = np.repeat(g[:1, :], h, axis=0)
                field[..., c] = (g / np.median(g)).astype(np.float32)
            warnings.append(
                "片基曲面仅沿条带轴一维拟合：横向（垂直条带方向）增益不可观测，"
                "flat-field 为部分校正；如需完整校正请提供 blank frame（v3 P0-C）")
            return {"field": field, "coverage": "strip-1d",
                    "residual": float("nan"), "warnings": warnings}
        return {"field": np.ones((h, w, 3), np.float32), "coverage": "none",
                "residual": float("nan"), "warnings": warnings + ["曲面拟合失败"]}

    return {"field": np.ones((h, w, 3), np.float32), "coverage": "none",
            "residual": float("nan"),
            "warnings": ["无 blank frame 也无片基条带：未做 flat-field"]}


def apply_flat(t_rgb: np.ndarray, surface: dict | np.ndarray) -> np.ndarray:
    """除以增益场。surface 可为 fit_base_surface 返回的 dict 或 (h,w,3) 数组。"""
    f = surface["field"] if isinstance(surface, dict) else np.asarray(surface)
    t = np.asarray(t_rgb, dtype=np.float32)
    if f.shape != t.shape:
        raise ValueError(f"flat 场形状 {f.shape} 与图像 {t.shape} 不一致")
    out = t / np.maximum(f, 1e-6)
    return np.clip(out, 0.0, None).astype(np.float32)
