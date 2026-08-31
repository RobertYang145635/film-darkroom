"""统计中性候选（整改 P0-C 升级链第 2 级：先验 -> 统计中性 -> 用户取点 -> 警告导出）。

原理：用当前参数反相后，画面中"接近中性且曝光居中"的像素大概率是
真实中性物体（灰岩、云、路面）。它们的通道残差即整体色偏方向。
这是 gray-world 的受限版（只取近中性+中调，排除饱和色/剪裁/高光/暗部），
置信度必须低于人工取点，且警告必须可见——不得伪装成正式标定结果。
"""

from __future__ import annotations

import numpy as np

from ..pipeline.density import invert, transmittance_to_z
from ..pipeline.params import InvertParams


def find_neutral_z_samples(t_rgb: np.ndarray, params: InvertParams,
                           lum_bands: tuple = ((0.15, 0.40), (0.40, 0.70)),
                           max_spread: float = 0.15,
                           min_frac: float = 0.002) -> dict:
    """返回 dict: samples(list[z]), weights(list), frac, warnings。

    samples 按亮度带分组（>=2 带时可用于估计幂指数比例；1 带只够 wb_high）。
    """
    warnings: list[str] = []
    res = invert(t_rgb, params)
    pos = res.positive_linear
    lum = pos.mean(axis=2)
    mx = pos.max(axis=2)
    mn = pos.min(axis=2)
    mean = np.maximum(lum, 1e-6)
    spread = (mx - mn) / mean                      # 通道离散度
    samples, weights = [], []
    frac_total = 0.0
    for (lo, hi) in lum_bands:
        band = (lum >= lo) & (lum <= hi) & (spread < max_spread) \
            & np.isfinite(lum)
        frac = float(band.mean())
        frac_total += frac
        if frac < 0.0005:
            continue
        med_c = np.median(pos[band], axis=0)       # 该带中性候选的正片中位
        # 反解回 z：正片相等要求 k_c*z_c 相等；用中位通道值构造等效 z 样本：
        # 直接取该带像素的真实 z 中位数（逐通道）
        z = transmittance_to_z(t_rgb, np.asarray(params.dmin, np.float32))
        zmed = np.median(z[band], axis=0).astype(np.float64)
        if np.abs(zmed).max() < 0.05:
            continue
        samples.append(zmed)
        weights.append(frac * 1e4)
    if len(samples) < 1 or frac_total < min_frac:
        return {"samples": [], "weights": [], "frac": frac_total,
                "warnings": ["统计中性候选不足（场景无明显中性面或色偏过大）"]}
    if len(samples) == 1:
        warnings.append("仅 1 个统计中性带：只估计高光白平衡，幂指数保持先验")
    return {"samples": samples, "weights": weights, "frac": round(frac_total, 4),
            "warnings": warnings}
