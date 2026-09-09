"""叶子度量（只依赖 numpy）。

为什么单独一个模块：diagnostics/quality_gate.py 要算微调后输出的逐通道剪裁
比例，而 channel_clip_ratios 原先住在 pipeline/density.py —— 构成
diagnostics(下层) -> pipeline(上层) 的反向依赖。该函数是纯数组统计、与反相
管线无关，下沉后两条依赖都指向叶子。pipeline.density.channel_clip_ratios 仍
可用（委托），既有调用方与测试不受影响。
"""

from __future__ import annotations

import numpy as np


def channel_clip_ratios(img: np.ndarray, mask: np.ndarray | None = None,
                        tol: float = 1e-6) -> dict:
    """每通道 [0,1] 之外的剪裁比例（规格 §8.4 剪裁指标）。"""
    px = img[mask] if mask is not None else img.reshape(-1, img.shape[-1])
    lo = (px <= tol).mean(axis=0)
    hi = (px >= 1.0 - tol).mean(axis=0)
    return {
        "low": [float(v) for v in lo],
        "high": [float(v) for v in hi],
    }


#: 密度域换算的最小值保护。原定义在 pipeline/density.py，但 calibration 也要用；
#: 留在上层就形成 calibration -> pipeline 的反向依赖，故下沉到本叶子模块。
EPS = 1e-6


def transmittance_to_z(t: np.ndarray, dmin: np.ndarray | tuple, eps: float = EPS) -> np.ndarray:
    """z_c = log10(max(t_c,eps)/max(Dmin_c,eps))：纯数学，无管线依赖。"""
    t = np.maximum(np.asarray(t, dtype=np.float32), eps)
    d = np.maximum(np.asarray(dmin, dtype=np.float32), eps)
    return np.log10(t / d).astype(np.float32)


def fit_decrosstalk_supervised(measured_z: np.ndarray,
                               target_density: np.ndarray,
                               ridge: float = 1e-3) -> dict:
    """监督解混拟合（研发用：已知目标密度的色卡等价物）。闭式岭回归。

    min_A ||Z A^T - T||² + λ||A - I||²。
    样本须跨 ≥3 个独立方向（多色相×多密度）；秩不足 -> 单位阵+警告
    （可辨识性见 docs/algorithm-notes.md §2.3）。

    为什么在叶子模块：原实现住在 pipeline/backends.py，而
    calibration/rollcalib.build_from_neutral_samples 需要它，构成
    calibration(下层) -> pipeline(上层) 的反向依赖（分层门禁 ALLOW 待办）。
    它是纯 numpy 线性代数、与管线编排无关，下沉后两层都指向叶子。
    pipeline.backends.fit_decrosstalk_supervised 仍可用（委托重导出）。
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
