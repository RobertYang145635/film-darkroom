"""中性参考点估计：幂指数比例、高光白平衡、阴影色偏（规格 §4.4、§4.5.5）。

参考点样本以 z 坐标给出（transmittance_to_z 的输出，见 density.py 符号约定：
片基 z=0，场景越亮 z 越负）。要求样本“不剪裁”：|z| 足够大且各通道有限。
"""

from __future__ import annotations

import numpy as np

MIN_ABS_Z = 0.05  # 太接近片基的样本对通道比例不敏感，视为剪裁/无效


def _valid_samples(samples: list[np.ndarray], weights: list[float] | None = None
                   ) -> tuple[np.ndarray, np.ndarray, list[str]]:
    warnings: list[str] = []
    good, wts = [], []
    for i, s in enumerate(samples):
        v = np.asarray(s, dtype=np.float64).reshape(-1)
        if v.size != 3 or not np.isfinite(v).all():
            warnings.append("忽略无效参考点样本（非有限或通道数不为 3）")
            continue
        if np.abs(v).max() < MIN_ABS_Z:
            warnings.append("忽略过浅的参考点（接近片基，无法约束通道比例）")
            continue
        good.append(v)
        wts.append(float(weights[i]) if weights is not None and i < len(weights) else 1.0)
    S = np.stack(good) if good else np.zeros((0, 3))
    W = np.asarray(wts, dtype=np.float64) if wts else np.zeros(0)
    return S, W, warnings


def _weighted_median(x: np.ndarray, w: np.ndarray) -> float:
    if x.size == 0:
        return 1.0
    order = np.argsort(x)
    cw = np.cumsum(w[order]) / w.sum()
    return float(x[order][np.searchsorted(cw, 0.5)])


def estimate_exponent_ratios(z_samples: list[np.ndarray],
                             reference_exponent: float = 1.0,
                             weights: list[float] | None = None) -> dict:
    """用亮、暗中性点估计 red_ratio / blue_ratio（§4.4）。

    模型：z'_c = k_c * z_c，中性场景要求 k_c * z_c 跨通道相等，
    故 ratio_c = k_c / k_G = 加权中位( z_G / z_c )，k_G = reference_exponent。

    weights 为逐样本信噪权重（透射率低的样本 z 域噪声大，应降权；
    由调用方按 (t_mean·|z|)^2 给出）。缺省等权。
    样本不足时不猜测：返回 ratio=1 并降低置信度（规格 §4.4 末段）。
    """
    S, W, warnings = _valid_samples(z_samples, weights)
    if S.shape[0] == 0:
        return {"red_ratio": 1.0, "blue_ratio": 1.0,
                "reference_exponent": float(reference_exponent),
                "confidence": 0.0,
                "warnings": warnings + ["无有效中性参考点：幂指数保持默认，需手动调整"]}

    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = []
        for c in range(3):
            mask = np.abs(S[:, c]) >= MIN_ABS_Z
            if mask.sum() == 0:
                ratios.append(1.0)
                warnings.append(f"通道 {c} 无有效样本，比例保持 1.0")
                continue
            ratios.append(_weighted_median(S[mask, 1] / S[mask, c], W[mask]))
    r_g = ratios[1] if abs(ratios[1]) > 1e-9 else 1.0
    red_ratio = ratios[0] / r_g
    blue_ratio = ratios[2] / r_g

    # 置信度：样本数量 + 比例合理性（远离 1 过远通常意味着误选参考点）
    n_score = float(min(1.0, S.shape[0] / 2.0))
    sane = max(abs(red_ratio - 1.0), abs(blue_ratio - 1.0))
    s_score = float(np.clip(1.0 - (sane - 0.35) / 0.35, 0.2, 1.0)) if sane > 0.35 else 1.0
    confidence = round(n_score * s_score, 3)
    if S.shape[0] < 2:
        warnings.append("中性参考点少于 2 个，幂指数估计置信度受限")
    if sane > 0.5:
        warnings.append(f"通道比例偏离较大（red={red_ratio:.3f}, blue={blue_ratio:.3f}），请确认参考点为中性")

    return {
        "red_ratio": round(red_ratio, 6),
        "blue_ratio": round(blue_ratio, 6),
        "reference_exponent": float(reference_exponent),
        "confidence": confidence,
        "warnings": warnings,
    }


def estimate_wb_high(z_samples: list[np.ndarray], dmax: float, offset: float,
                     wb_low: tuple = (1.0, 1.0, 1.0),
                     k: tuple | np.ndarray = (1.0, 1.0, 1.0),
                     weights: list[float] | None = None) -> dict:
    """高光白平衡：令亮中性点在各通道 q 相等（§4.3 wb_high 语义）。

    q_c = (wb_high_c / Dmax) * k_c * z_c + wb_high_c * offset * wb_low_c
    解 wb_high_c = q_target / ( k_c*z_c/Dmax + offset*wb_low_c )。
    分母接近零（该点无法约束此通道）时保持 1.0 并警告。
    """
    S, W, warnings = _valid_samples(z_samples, weights)
    if S.shape[0] == 0:
        return {"wb_high": [1.0, 1.0, 1.0], "confidence": 0.0,
                "warnings": warnings + ["无亮部中性参考点：wb_high 保持默认"]}
    k = np.asarray(k, dtype=np.float64)
    wb_low_a = np.asarray(wb_low, dtype=np.float64)
    den = (k[None, :] * S / max(float(dmax), 1e-6)) + float(offset) * wb_low_a[None, :]
    q = den  # 每样本每通道的 q（wb_high=1 时）
    q_target = _weighted_median(q.mean(axis=1), W)
    out = []
    for c in range(3):
        d = den[:, c]
        mask = np.abs(d) > 1e-3
        if not mask.any():
            out.append(1.0)
            warnings.append(f"通道 {c} 参考点无法约束 wb_high（分母≈0），保持 1.0")
            continue
        out.append(_weighted_median(q_target / d[mask], W[mask]))
    wb_high = [float(np.clip(v, 0.2, 5.0)) for v in out]
    return {"wb_high": wb_high, "confidence": 0.6 if S.shape[0] >= 1 else 0.0,
            "warnings": warnings}


def estimate_wb_low(z_samples: list[np.ndarray], dmax: float, offset: float,
                    wb_high: tuple = (1.0, 1.0, 1.0),
                    k: tuple | np.ndarray = (1.0, 1.0, 1.0),
                    weights: list[float] | None = None) -> dict:
    """阴影色偏：暗中性点上解 wb_low_c（§4.3 offset_c = wb_high*offset*wb_low）。

    注意：wb_low 只通过 offset 起作用；offset≈0 时该自由度不存在，
    必须如实报告而不是静默塞进其他参数（执行契约 §6）。
    """
    if abs(float(offset)) < 1e-3:
        return {"wb_low": [1.0, 1.0, 1.0], "confidence": 0.0,
                "warnings": ["offset≈0 时阴影色偏校正无自由度，wb_low 保持 1.0；"
                             "如需自动阴影校正请先给出非零扫描曝光偏置或手动调整"]}
    S, W, warnings = _valid_samples(z_samples, weights)
    if S.shape[0] == 0:
        return {"wb_low": [1.0, 1.0, 1.0], "confidence": 0.0,
                "warnings": warnings + ["无暗部中性参考点：wb_low 保持默认"]}
    k = np.asarray(k, dtype=np.float64)
    wh = np.asarray(wb_high, dtype=np.float64)
    q_fixed = wh[None, :] * k[None, :] * S / max(float(dmax), 1e-6)
    # 目标：q_c 相等 -> wb_low_c = (q_target - q_fixed_c) / (wb_high_c * offset)
    q_total_at_default = q_fixed + wh[None, :] * float(offset)
    q_target = _weighted_median(q_total_at_default.mean(axis=1), W)
    out = []
    for c in range(3):
        denom = wh[c] * float(offset)
        if abs(denom) < 1e-6:
            out.append(1.0)
            continue
        out.append(_weighted_median((q_target - q_fixed[:, c]) / denom, W))
    wb_low = [float(np.clip(v, 0.05, 20.0)) for v in out]
    return {"wb_low": wb_low, "confidence": 0.5, "warnings": warnings}
