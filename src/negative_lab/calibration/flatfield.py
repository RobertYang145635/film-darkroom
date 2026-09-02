"""片基低阶平滑场（flat-field）与背光不均修正（v5 §6.1.2）。

约束（v5 原文）：当片基 ROI 覆盖多个边缘时，允许用**受约束的低阶平滑场**
修正背光/镜头不均；**只能在实际可观测方向拟合**，不能从画面内容臆造二维
flat field。因此本模块按取样区的实际几何覆盖决定 coverage 等级，并如实
声明不可观测方向：

| coverage        | 取样条件                          | 拟合                |
|-----------------|-----------------------------------|---------------------|
| `2d`            | blank frame                       | 完整二阶二维多项式  |
| `2d-multi-edge` | ≥2 条边缘片基带，x/y 方向都有跨度 | 一阶二维（可分离）  |
| `strip-1d`      | 单条片基带                        | 沿条带轴一维        |
| `none`          | 无可用取样                         | 恒等（并警告）      |

场归一化到全局中位=1，保证不改变 Dmin 的物理量级。
"""

from __future__ import annotations

import numpy as np

_MIN_SPAN = 0.25        # 观测点在某方向跨度低于画幅的 25% 视为不可观测
_MIN_PIXELS = 64


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


def _roi_slices(rois, shape):
    h, w = shape
    out = []
    for roi in rois or []:
        y0, x0, y1, x1 = (int(v) for v in roi)
        y0, x0 = max(0, y0), max(0, x0)
        y1, x1 = min(h, y1), min(w, x1)
        if (y1 - y0) * (x1 - x0) >= _MIN_PIXELS:
            out.append((y0, x0, y1, x1))
    return out


_FIELD_MIN, _FIELD_MAX = 0.5, 2.0        # 增益场物理合理范围（背光/暗角）

def _finalize(field: np.ndarray, ref_mask: np.ndarray | None,
              warnings: list[str]) -> np.ndarray:
    """把场变成**纯梯度增益**：片基取样区内每通道中位恰为 1。

    这是唯一能同时满足两条硬约束的归一化（v5 §6.1）：
    - `b(x)` 只来自未曝光片基 -> Dmin 是物理基准，flat-field 不得缩放它
      （实测教训：二阶多项式外推把 Dmin 抬高 1.44 倍）；
    - 暗角/导光不均是**乘性照明**问题 -> 用增益场去除。
    因此场只保留相对梯度，量级信息全部留给 Dmin。最后钳制到物理合理范围，
    防止外推发散把画面改写成负值/爆炸值。
    """
    if ref_mask is not None and ref_mask.any():
        ref = np.median(field[ref_mask], axis=0)
    else:
        ref = np.median(field.reshape(-1, 3), axis=0)
    ref = np.maximum(ref, 1e-6)
    field = field / ref
    return np.clip(field, _FIELD_MIN, _FIELD_MAX).astype(np.float32)


def fit_base_surface(t_rgb: np.ndarray,
                     base_roi: tuple[int, int, int, int] | None = None,
                     base_rois: list | None = None,
                     blank: np.ndarray | None = None,
                     order: int = 2) -> dict:
    """返回 {"field": (h,w,3) 归一化增益场, "coverage": str,
    "residual": float, "warnings": [...], "observed": {...}}。"""
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
            X = np.stack([xx[m] / w, yy[m] / h], axis=1)
            A = _poly_design(X, order, True)
            coef, *_ = np.linalg.lstsq(A, np.log(np.maximum(b[m, c], 1e-9)),
                                       rcond=None)
            # 锚点：观测中位位置。场只保留相对梯度（在锚点处=1），
            # log 量级（含片基截距）留给 Dmin——flat-field 不改写物理基准。
            anchor = np.median(X, axis=0)
            full = (_poly_design(np.stack([xx.ravel() / w, yy.ravel() / h], axis=1),
                                 order, True) @ coef
                    - _poly_design(anchor[None, :], order, True) @ coef)
            g = np.exp(full.reshape(h, w))
            field[..., c] = g.astype(np.float32)
            # 残差：blank 与"梯度×锚点量级"之间的相对偏差（诊断用）
            model = np.exp(_poly_design(X, order, True) @ coef).reshape(-1)
            resid.append(float(np.std(b[m, c] / model) / max(np.median(b[m, c]), 1e-9)))
        field = _finalize(field, None, warnings)
        return {"field": field, "coverage": "2d",
                "residual": float(np.mean(resid)), "warnings": warnings,
                "observed": {"spans": [1.0, 1.0], "n_regions": 1}}

    rois = _roi_slices(base_rois or ([base_roi] if base_roi else []), (h, w))
    if not rois:
        return {"field": np.ones((h, w, 3), np.float32), "coverage": "none",
                "residual": float("nan"), "warnings":
                    ["无 blank frame 也无片基取样区：未做 flat-field"],
                "observed": {"spans": [0.0, 0.0], "n_regions": 0}}

    # 观测点集合（归一化坐标）
    pts_y, pts_x, vals = [], [], []
    ref_mask = np.zeros((h, w), bool)
    for (y0, x0, y1, x1) in rois:
        seg = t[y0:y1, x0:x1]
        gy, gx = np.mgrid[y0:y1, x0:x1]
        pts_y.append(gy.ravel() / h)
        pts_x.append(gx.ravel() / w)
        vals.append(seg.reshape(-1, 3))
        ref_mask[y0:y1, x0:x1] = True
    oy = np.concatenate(pts_y)
    ox = np.concatenate(pts_x)
    ov = np.concatenate(vals)
    span_x = float(np.ptp(ox)) if ox.size else 0.0
    span_y = float(np.ptp(oy)) if oy.size else 0.0
    observed = {"spans": [round(span_x, 3), round(span_y, 3)],
                "n_regions": len(rois)}

    two_d = (len(rois) >= 2 and span_x >= _MIN_SPAN and span_y >= _MIN_SPAN)
    field = np.ones((h, w, 3), np.float32)
    if two_d:
        # 一阶二维：只修正实际可观测的整体梯度与常数项，不外推高阶结构
        good = np.isfinite(ov).all(axis=1) & (ov.max(axis=1) > 1e-6)
        A = np.column_stack([np.ones(good.sum()), ox[good], oy[good]])
        for c in range(3):
            y = np.log(np.maximum(ov[good, c], 1e-9))
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
            full = np.column_stack([np.ones(h * w), xx.ravel() / w, yy.ravel() / h]) @ coef
            field[..., c] = np.exp(full.reshape(h, w)).astype(np.float32)
        field = _finalize(field, ref_mask, warnings)
        warnings.append(
            f"片基场由 {len(rois)} 个取样区拟合（一阶二维，x/y 跨度 "
            f"{span_x:.2f}/{span_y:.2f}）；更高频的不均在可观测方向之外，未修正")
        return {"field": field, "coverage": "2d-multi-edge",
                "residual": float("nan"), "warnings": warnings,
                "observed": observed}

    # 单条带：沿条带轴**一阶**拟合（二阶从窄条带外推到全幅会严重失真，
    # 实测把 Dmin 抬高 1.44 倍——线性外推有界且可解释）
    y0, x0, y1, x1 = rois[0]
    strip_h, strip_w = y1 - y0, x1 - x0
    vertical = strip_h >= strip_w
    coef_all = []
    for c in range(3):
        seg = t[y0:y1, x0:x1, c].astype(np.float64)
        vals_c = seg[seg > 1e-6]
        if vals_c.size < 32:
            warnings.append(f"通道 {c} 片基样本不足，曲面未拟合")
            coef_all.append(None)
            continue
        flat = seg.ravel()
        good = flat > 1e-6
        if vertical:
            tt_full = np.repeat((np.arange(y0, y1) / h), strip_w)
        else:
            tt_full = np.tile((np.arange(x0, x1) / w), strip_h)
        vv = np.log(np.maximum(flat[good], 1e-9))
        tt = tt_full[good]
        A = _poly_design(tt[:, None], 1, False)
        coef, *_ = np.linalg.lstsq(A, vv, rcond=None)
        coef_all.append(coef)
    if all(c is not None for c in coef_all):
        ref_mask = np.zeros((h, w), bool)
        ref_mask[y0:y1, x0:x1] = True
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
            field[..., c] = g.astype(np.float32)
        field = _finalize(field, ref_mask, warnings)
        warnings.append(
            "片基场仅沿单条取样带的一维方向拟合：横向增益不可观测，flat-field 为"
            "部分校正（v5 §6.1.2：不从画面内容臆造二维 flat field）")
        return {"field": field, "coverage": "strip-1d",
                "residual": float("nan"), "warnings": warnings,
                "observed": observed}
    return {"field": np.ones((h, w, 3), np.float32), "coverage": "none",
            "residual": float("nan"),
            "warnings": warnings + ["片基曲面拟合失败"], "observed": observed}


def apply_flat(t_rgb: np.ndarray, surface: dict | np.ndarray) -> np.ndarray:
    """除以增益场。surface 可为 fit_base_surface 返回的 dict 或 (h,w,3) 数组。"""
    f = surface["field"] if isinstance(surface, dict) else np.asarray(surface)
    t = np.asarray(t_rgb, dtype=np.float32)
    if f.shape != t.shape:
        raise ValueError(f"flat 场形状 {f.shape} 与图像 {t.shape} 不一致")
    out = t / np.maximum(f, 1e-6)
    return np.clip(out, 0.0, None).astype(np.float32)
