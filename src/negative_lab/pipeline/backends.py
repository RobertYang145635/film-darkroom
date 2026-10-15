"""技术反相 backend（v5 §6.1-§6.3，数学契约见 docs/algorithm-notes.md）。

统一接口：`backend(t, params, mask=None, diag=None) -> technical positive`。

- power-reciprocal: A=I + 逐通道斜率（可解释基线，旧兼容名）
- density-affine: verified 3x3 A + 明确通道斜率（无隐藏重参数化）
- density-affine-curves: B + 受约束单调通道曲线（通用模型主路线）

v5 §8.2 要求"按最终 image_area_roi、逐通道、**clamp 前**统计"输出剪裁，
因此 backend 接受 `mask`（与 t 同形状的 (H,W) bool）并在 diag 中累计
pre-clamp 越界计数；调用方分块时传入对应行切片，聚合后即为全图精确值。
"""

from __future__ import annotations

import numpy as np

from .params import InvertParams

_EPS = 1e-6
#: 黑位解析软趾（v4 §12.2 实测结论，v5 沿用）：低于黑点的像素渐近逼近 0
#: 而非硬剪为 0——保留"多黑"信息，剪裁指标从此只计真硬剪裁。
SMOOTH_TOE = 0.004


def _validate_matrix(A) -> np.ndarray:
    A = np.asarray(A, np.float64)
    if A.shape == (9,):
        A = A.reshape(3, 3)
    if A.shape != (3, 3):
        raise ValueError(f"density_matrix 必须是 3x3（或 9 元素），得到 shape={A.shape}")
    if not np.isfinite(A).all():
        raise ValueError("density_matrix 含非有限值")
    if float(np.linalg.cond(A)) > 1e6:
        raise ValueError(f"density_matrix 病态（cond={float(np.linalg.cond(A)):.3g}）")
    return A


def _slopes(params: InvertParams) -> np.ndarray:
    """s_c：v_c = s_c * u_c。真实生效（v4 §1.2.9 回归锁）。"""
    return np.asarray([params.red_ratio * params.reference_exponent,
                       params.reference_exponent,
                       params.blue_ratio * params.reference_exponent],
                      np.float64)


def _accumulate_diag(diag: dict | None, paper: np.ndarray, mask,
                     t: np.ndarray, b: np.ndarray) -> None:
    """clamp 前逐通道越界统计（v5 §8.2）+ t>b 与负密度诊断（§6.1.3）。

    mask 与 `paper`/`t>b` 形状不符时**抛错**，不许退化成"整块不套 mask"：
    ㊴ 之前这里是 `paper[m] if paper.shape[:2] == m.shape else paper`，
    调用方传错一个切片就静默把片边/越界像素计进画面统计，而 v5 §8.2 的口径
    宣称"只覆盖 image_area"（`test_rebate_independence` 的负控制
    `test_rebate_leak_would_be_visible` 量到这种泄漏值 3.8e-02 ≈ 5836×容差）。
    """
    if diag is None:
        return
    if mask is None:
        px = paper
        above = t > b                       # 透射率超过片基（超基/漏光）
    else:
        m = np.asarray(mask, bool)
        if paper.shape[:2] != m.shape:
            raise ValueError(
                f"preclip 统计 mask 形状 {m.shape} 与反相输出 "
                f"{paper.shape[:2]} 不符（分块调用须传同一行切片）")
        px = paper[m]
        tb = t > b
        if tb.shape[:2] != m.shape:
            raise ValueError(
                f"above_base 统计 mask 形状 {m.shape} 与 t>b {tb.shape[:2]} 不符")
        above = tb[m]
        if px.ndim == 3:
            px = px.reshape(-1, 3)
    if px.ndim == 3:
        px = px.reshape(-1, 3)
    n = int(px.shape[0])
    if n == 0:
        return
    d = diag.setdefault("preclip", {"low": [0, 0, 0], "high": [0, 0, 0],
                                    "n": 0, "above_base": 0, "negative_density": 0})
    for c in range(3):
        d["low"][c] += int((px[:, c] < 0.0).sum())
        d["high"][c] += int((px[:, c] > 1.0).sum())
    d["n"] += n
    d["above_base"] += int(np.count_nonzero(above))


def _core(t: np.ndarray, params: InvertParams, A: np.ndarray,
          curves=None, mask=None, diag: dict | None = None) -> np.ndarray:
    """唯一技术反相核心（docs/algorithm-notes.md §1 符号）。

    t: (H,W,3) 线性透射率 -> technical positive (0-1)。
    """
    b = np.maximum(np.asarray(params.dmin, np.float64), _EPS)
    d = -np.log10(np.maximum(t, _EPS) / b)              # 正密度
    u = d @ A.T                                          # u = A d (+a，当前 a=0)
    s = _slopes(params)
    v = u * s[None, None, :]                             # 通道响应
    if curves is not None:
        v = _apply_channel_curves(v, curves)             # 通用模型曲线（单调）
    wh = np.asarray(params.wb_high, np.float64)
    wl = np.asarray(params.wb_low, np.float64)
    q = -(wh / params.dmax)[None, None, :] * v \
        + (wh * params.offset * wl)[None, None, :]
    paper = (1.0 + params.black) - np.power(10.0, q)
    # 黑位解析软趾（technical_endpoints 映射的一部分，v4 §12.2 实测结论）：
    # 低于黑点的像素渐近逼近 0 而非硬剪为 0——保留"多黑"的信息（C∞、单调、
    # above-3toe 恒等）。因此"clamp 前"统计应在软趾之后、最终 clip 之前。
    paper = (paper + np.sqrt(paper * paper + (2.0 * SMOOTH_TOE) ** 2)) / 2.0
    _accumulate_diag(diag, paper, mask, t, b)
    return np.clip(paper, 0.0, 1.0).astype(np.float32)


def _apply_channel_curves(v: np.ndarray, curves) -> np.ndarray:
    """单调逐通道曲线（受限节点，分段线性）。curves: [None|[(x,y)...]]×3"""
    out = v.copy()
    for c, knots in enumerate(curves or (None, None, None)):
        if not knots:
            continue
        xs = np.asarray([k[0] for k in knots], np.float64)
        ys = np.asarray([k[1] for k in knots], np.float64)
        order = np.argsort(xs)
        xs, ys = xs[order], ys[order]
        if (np.diff(ys) < -1e-12).any():
            raise ValueError(f"通道 {c} 曲线非单调（v5 §6.3：f_c 必须单调）")
        out[..., c] = np.interp(v[..., c], xs, ys)
    return out


def power_reciprocal(t: np.ndarray, params: InvertParams, mask=None,
                     diag: dict | None = None) -> np.ndarray:
    """路线 A：A=I 基线。"""
    return _core(t, params, np.eye(3), mask=mask, diag=diag)


def density_affine(t: np.ndarray, params: InvertParams, mask=None,
                   diag: dict | None = None) -> np.ndarray:
    """路线 B：verified 3x3 A + 明确 s_c。矩阵就是运行时矩阵（无重参数化）。"""
    A = _validate_matrix(params.density_matrix)
    return _core(t, params, A, mask=mask, diag=diag)


def density_affine_curves(t: np.ndarray, params: InvertParams, mask=None,
                          diag: dict | None = None) -> np.ndarray:
    """路线 C：B + 每通道单调曲线（params.channel_curves）。"""
    A = _validate_matrix(params.density_matrix)
    return _core(t, params, A, curves=getattr(params, "channel_curves", None),
                 mask=mask, diag=diag)


BACKENDS = {"power-reciprocal": power_reciprocal,
            "density-affine": density_affine,
            "density-affine-curves": density_affine_curves}


def get_backend(name: str):
    if name not in BACKENDS:
        raise ValueError(f"未知 backend: {name}（可选 {sorted(BACKENDS)}）")
    return BACKENDS[name]


# 监督解混拟合是纯 numpy 线性代数，已下沉到叶子 metrics.py
# （消除 calibration/rollcalib -> pipeline 的反向依赖，2026-09-08）。
# 显式自别名重导出：ruff 在 ignore=[] 下会把普通 import 当未用而 --fix 删掉。
from ..metrics import fit_decrosstalk_supervised as fit_decrosstalk_supervised  # noqa: E402,F401
