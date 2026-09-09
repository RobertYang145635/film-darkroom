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
