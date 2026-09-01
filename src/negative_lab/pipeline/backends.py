"""技术反相 backend（v4 §4，数学契约见 docs/algorithm-notes.md）。

统一接口：backend(t, params) -> technical positive (0-1)。
- power-reciprocal: A=I + 逐通道斜率（可解释基线，旧兼容名）
- density-affine: verified 3x3 A + 明确通道斜率（无隐藏重参数化）
- density-affine-curves: B + 受约束单调通道曲线（路线 C）

v4 实锤修复：red/blue_ratio（s_c）真实生效（旧代码算 k 不用）；
exposure 从技术链路移除（→ RenderIntent.exposure）；density_affine
不再做行归一化/斜率重置的隐藏重参数化。
"""

from __future__ import annotations

import numpy as np

from .params import InvertParams

_EPS = 1e-6


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


def _core(t: np.ndarray, params: InvertParams, A: np.ndarray,
          curves=None) -> np.ndarray:
    """唯一技术反相核心（docs/algorithm-notes.md §1 符号）。

    t: (H,W,3) 线性透射率 -> technical positive (0-1)。
    """
    b = np.maximum(np.asarray(params.dmin, np.float64), _EPS)
    d = -np.log10(np.maximum(t, _EPS) / b)              # 正密度
    u = d @ A.T                                          # u = A d (+a，当前 a=0)
    s = _slopes(params)
    v = u * s[None, None, :]                             # 通道响应
    if curves is not None:
        v = _apply_channel_curves(v, curves)             # 路线 C（单调）
    wh = np.asarray(params.wb_high, np.float64)
    wl = np.asarray(params.wb_low, np.float64)
    q = -(wh / params.dmax)[None, None, :] * v \
        + (wh * params.offset * wl)[None, None, :]
    paper = (1.0 + params.black) - np.power(10.0, q)
    return np.clip(paper, 0.0, 1.0).astype(np.float32)


def _apply_channel_curves(v: np.ndarray, curves) -> np.ndarray:
    """单调逐通道曲线（受限节点，PCHIP 式线性插值）。curves: [None|[(x,y)...]]×3"""
    out = v.copy()
    for c, knots in enumerate(curves or (None, None, None)):
        if not knots:
            continue
        xs = np.asarray([k[0] for k in knots], np.float64)
        ys = np.asarray([k[1] for k in knots], np.float64)
        order = np.argsort(xs)
        xs, ys = xs[order], ys[order]
        if (np.diff(ys) < -1e-12).any():
            raise ValueError(f"通道 {c} 曲线非单调（v4 §4.1：f_c 必须单调）")
        out[..., c] = np.interp(v[..., c], xs, ys)
    return out


def power_reciprocal(t: np.ndarray, params: InvertParams) -> np.ndarray:
    """路线 A：A=I 基线。"""
    return _core(t, params, np.eye(3))


def density_affine(t: np.ndarray, params: InvertParams) -> np.ndarray:
    """路线 B：verified 3x3 A + 明确 s_c。矩阵就是运行时矩阵（无重参数化）。"""
    A = _validate_matrix(params.density_matrix)
    return _core(t, params, A)


def density_affine_curves(t: np.ndarray, params: InvertParams) -> np.ndarray:
    """路线 C：B + 每通道单调曲线（params.channel_curves）。"""
    A = _validate_matrix(params.density_matrix)
    return _core(t, params, A, curves=getattr(params, "channel_curves", None))


BACKENDS = {"power-reciprocal": power_reciprocal,
            "density-affine": density_affine,
            "density-affine-curves": density_affine_curves}


def get_backend(name: str):
    if name not in BACKENDS:
        raise ValueError(f"未知 backend: {name}（可选 {sorted(BACKENDS)}）")
    return BACKENDS[name]


def fit_decrosstalk_supervised(measured_z: np.ndarray,
                               target_density: np.ndarray,
                               ridge: float = 1e-3) -> dict:
    """监督解混拟合（色卡等价物：已知目标密度）。闭式岭回归。

    min_A ||Z A^T - T||² + λ||A - I||²。
    样本须跨 ≥3 个独立方向（多色相×多密度）；秩不足 -> 单位阵+警告
    （可辨识性见 docs/algorithm-notes.md §2.3）。
    """
    Z = np.asarray(measured_z, np.float64)
    T = np.asarray(target_density, np.float64)
    out = {"matrix": np.eye(3, dtype=np.float32).ravel().tolist(),
           "confidence": 0.0, "n_samples": int(Z.shape[0]), "warnings": []}
    if Z.ndim != 2 or Z.shape[0] < 6 or Z.shape != T.shape:
        out["warnings"].append("监督样本不足（<6 或形状不匹配）：保持单位阵")
        return out
    S = np.linalg.svd(Z - Z.mean(axis=0), compute_uv=False)
    if S[0] <= 0 or S[2] / S[0] < 0.05:
        out["warnings"].append("样本方向秩不足：保持单位阵")
        return out
    G = Z.T @ Z + ridge * np.eye(3)
    A = np.linalg.solve(G, Z.T @ T + ridge * np.eye(3))
    out.update({"matrix": A.T.astype(np.float32).ravel().tolist(),
                "confidence": 1.0, "warnings": []})
    return out
