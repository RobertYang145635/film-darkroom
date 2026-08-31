"""可插拔反相 backend（整改 v3 P0-D）。

统一接口：backend(t, params) -> positive_linear (0-1)。
- power_reciprocal: 现行默认（逐通道对角，无串扰建模）
- density_affine:  密度域 3x3 解混矩阵 + 逐通道幂（F-ALG-6 的染料串扰）
- density_curves:  逐通道单调曲线（P0-D 后续项）

矩阵拟合 fit_decrosstalk(neutral_z_samples)：
  min_A  Σ_i ||A z_i - m_i·1||² + λ||A - I||²   （m_i 每样本自由）
中性样本不足以支撑时 λ 增大 -> 收敛回单位阵（不假装解混）。
"""

from __future__ import annotations

import numpy as np

from .params import InvertParams


def _z(t: np.ndarray, params: InvertParams) -> np.ndarray:
    dmin = np.asarray(params.dmin, np.float32)
    return np.log10(np.maximum(t, 1e-6) / np.maximum(dmin, 1e-6))


def _paper_from_q(q: np.ndarray, params: InvertParams) -> np.ndarray:
    paper = (1.0 + params.black) - np.power(10.0, q)
    return np.clip(paper, 0.0, None)


def power_reciprocal(t: np.ndarray, params: InvertParams) -> np.ndarray:
    """现行默认：逐通道 z -> q = k_c·z_c/dmax + offset_c -> paper。"""
    z = _z(t, params)
    k = np.asarray([params.red_ratio, 1.0, params.blue_ratio], np.float32) \
        * params.reference_exponent
    wh = np.asarray(params.wb_high, np.float32)
    wl = np.asarray(params.wb_low, np.float32)
    q = (wh / np.float32(params.dmax))[None, None, :] * z \
        + (wh * params.offset * wl)[None, None, :]
    paper = _paper_from_q(q, params)
    return np.power(np.clip(paper, 0.0, None), params.gamma).astype(np.float32)


def density_affine(t: np.ndarray, params: InvertParams) -> np.ndarray:
    """密度域仿射：z' = A·z（解混），再走 power_reciprocal 的对角路径。

    A = params.density_matrix（行主 3x3，默认单位阵 = 与 power_reciprocal 等价）。
    """
    A = np.asarray(params.density_matrix, np.float32).reshape(3, 3)
    if np.allclose(A, np.eye(3), atol=1e-6):
        return power_reciprocal(t, params)
    z = _z(t, params)
    zp = z @ A.T
    p2 = InvertParams(dmin=params.dmin, dmax=params.dmax, offset=params.offset,
                      wb_low=params.wb_low, wb_high=params.wb_high,
                      reference_exponent=1.0, red_ratio=1.0, blue_ratio=1.0,
                      black=params.black, gamma=params.gamma,
                      exposure=params.exposure)
    # 串扰校正后的等效 k 已并入 A 的行范数：保持对角参数为 1
    k = np.linalg.norm(A, axis=1)
    zp = zp / k[None, None, :]
    wh = np.asarray(p2.wb_high, np.float32)
    q = (wh / np.float32(p2.dmax)) * zp + (wh * p2.offset)
    paper = _paper_from_q(q, p2)
    return np.power(np.clip(paper, 0.0, None), p2.gamma).astype(np.float32)


BACKENDS = {"power-reciprocal": power_reciprocal,
            "density-affine": density_affine}


def get_backend(name: str):
    if name not in BACKENDS:
        raise ValueError(f"未知 backend: {name}（可选 {sorted(BACKENDS)}）")
    return BACKENDS[name]


def fit_decrosstalk(neutral_z: np.ndarray, ridge: float | None = None,
                    iters: int = 12) -> dict:
    """neutral_z: (n,3) 中性样本的 z 向量（同密度组越多越稳）。

    交替最小二乘：给定 m 解各行 ridge；给定 A 更新 m。
    样本 <3 或方差不足 -> 返回单位阵 + 低置信度（不假装解混）。
    """
    Z = np.asarray(neutral_z, np.float64)
    Z = Z[np.isfinite(Z).all(axis=1)]
    n = Z.shape[0]
    out = {"matrix": np.eye(3, dtype=np.float32).ravel().tolist(),
           "confidence": 0.0, "n_samples": int(n), "warnings": []}
    if n < 3:
        out["warnings"].append("中性样本 <3：解混矩阵不可辨识，保持单位阵")
        return out
    spread = float(np.ptp(Z.mean(axis=1)))
    if spread < 0.05:
        out["warnings"].append(
            f"中性样本密度跨度仅 {spread:.3f}：跨密度串扰不可辨识，保持单位阵")
        return out
    # 可识别性硬检查（v3 F-ALG-5）：中性物体的 z 向量只张成一维射线
    # z(e) = -f(e)·(γ C)——任何把该射线映到等通道的 A 都满足中性约束，
    # 串扰矩阵从纯中性样本【原理上不可辨识】。秩不足必须返回单位阵，
    # 解混需要已知目标密度的多色标定样本（色卡，监督拟合见
    # fit_decrosstalk_supervised）。
    U, S, Vt = np.linalg.svd(Z - Z.mean(axis=0), full_matrices=False)
    if S[0] <= 0 or S[1] / S[0] < 0.15:
        out["warnings"].append(
            "中性样本只张成一维密度射线：解混矩阵不可辨识（v3 F-ALG-5），"
            "保持单位阵；如需启用 density-affine 请提供多色标定目标")
        return out
    lam = ridge if ridge is not None else max(0.05, 12.0 / n)
    A = np.eye(3)
    m = Z.mean(axis=1)
    eye = np.eye(3)
    for _ in range(iters):
        for c in range(3):
            # min ||Z a - m||² + lam||a - e_c||²
            G = Z.T @ Z + lam * eye
            rhs = Z.T @ m + lam * eye[c]
            A[c] = np.linalg.solve(G, rhs)
        m = (Z @ A.T).mean(axis=1)
    # 行归一到对角=1：对角缩放是逐通道指数(red/blue_ratio)的职责，
    # 解混矩阵只保留非对角串扰项（否则两者重复计数，实测 conf 被误杀）。
    A = A / np.abs(np.diag(A))[:, None]
    maxoff = float(np.abs(A - eye).max())
    conf = float(np.clip(1.0 - maxoff / 0.5, 0.0, 1.0)) * min(1.0, n / 24.0)
    if conf <= 0.0:
        # 拟合结果偏离单位阵过大：样本被非中性内容污染，不可信 -> 诚实回退
        out["warnings"].append(
            f"解混拟合不可信（最大偏离 {maxoff:.2f}）：保持单位阵")
        return out
    out.update({"matrix": A.astype(np.float32).ravel().tolist(),
                "confidence": round(conf, 3), "warnings": []})
    return out


def fit_decrosstalk_supervised(measured_z: np.ndarray,
                               target_density: np.ndarray,
                               ridge: float = 1e-3) -> dict:
    """监督解混拟合：已知标定目标（色卡）的测量 z -> 期望密度 t。

    min_A ||Z A^T - T||² + λ||A - I||²  —— 闭式岭回归。
    需要样本跨 ≥3 个独立方向（多色相×多密度）；秩不足 -> 单位阵。
    """
    Z = np.asarray(measured_z, np.float64)
    T = np.asarray(target_density, np.float64)
    out = {"matrix": np.eye(3, dtype=np.float32).ravel().tolist(),
           "confidence": 0.0, "n_samples": int(Z.shape[0]), "warnings": []}
    if Z.shape[0] < 6 or Z.shape != T.shape:
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
