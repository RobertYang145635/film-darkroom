"""比色学：sRGB -> CIELAB(D65) 与 CIEDE2000（规格 §8.4 色彩指标）。

参考：Sharma et al. 2005 CIEDE2000 公式（独立实现，未复制 GPL 代码）。
"""

from __future__ import annotations

import numpy as np

# D65 白点（2° 视场）
_XN, _YN, _ZN = 0.95047, 1.00000, 1.08883


def linear_srgb_to_lab(lin: np.ndarray) -> np.ndarray:
    """线性 sRGB [0,1] -> CIELAB D65（v4 §7.4：矩阵直接作用在线性值上）。

    v4 §1.2.6 实锤修复：不得先加 sRGB TRC 再乘线性矩阵
    （错误实现给 L*[0.5]=88.70；本实现锁定 76.069）。
    """
    x = np.asarray(lin, dtype=np.float64).reshape(-1, 3)
    m = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = x @ m.T
    return _xyz_to_lab(xyz)


def _xyz_to_lab(xyz: np.ndarray) -> np.ndarray:
    xyz = xyz / np.array([_XN, _YN, _ZN])
    eps, kap = 216 / 24389, 24389 / 27          # v4 §7.4.3 标准形式
    f = np.where(xyz > eps, np.cbrt(xyz), kap * xyz / 116 + 16 / 116)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB 编码值 [0,1] -> Lab（先 TRC 解码为线性再走 linear_srgb_to_lab）。"""
    x = np.asarray(rgb, dtype=np.float64).reshape(-1, 3)
    lin = np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    return linear_srgb_to_lab(lin)


def delta_e_2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """CIEDE2000。lab1/lab2: (N,3) 或 (3,)。返回逐点 ΔE00。"""
    lab1 = np.atleast_2d(np.asarray(lab1, dtype=np.float64))
    lab2 = np.atleast_2d(np.asarray(lab2, dtype=np.float64))
    l1, a1, b1 = lab1[:, 0], lab1[:, 1], lab1[:, 2]
    l2, a2, b2 = lab2[:, 0], lab2[:, 1], lab2[:, 2]

    c1 = np.hypot(a1, b1)
    c2 = np.hypot(a2, b2)
    c_bar = (c1 + c2) / 2
    g = 0.5 * (1 - np.sqrt(c_bar ** 7 / (c_bar ** 7 + 25 ** 7)))
    a1p, a2p = a1 * (1 + g), a2 * (1 + g)
    c1p, c2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360

    dLp = l2 - l1
    dCp = c2p - c1p
    dhp = np.where(c1p * c2p == 0, 0.0,
                   np.where(np.abs(h2p - h1p) <= 180, h2p - h1p,
                            np.where(h2p - h1p > 180, h2p - h1p - 360,
                                     h2p - h1p + 360)))
    dHp = 2 * np.sqrt(c1p * c2p) * np.sin(np.radians(dhp) / 2)

    Lbp = (l1 + l2) / 2
    Cbp = (c1p + c2p) / 2
    s = h1p + h2p
    hbp = np.where(np.abs(h1p - h2p) > 180,
                   np.where(s < 360, (s + 360) / 2, (s - 360) / 2), s / 2)

    T = (1 - 0.17 * np.cos(np.radians(hbp - 30))
         + 0.24 * np.cos(np.radians(2 * hbp))
         + 0.32 * np.cos(np.radians(3 * hbp + 6))
         - 0.20 * np.cos(np.radians(4 * hbp - 63)))
    d_theta = 30 * np.exp(-(((hbp - 275) / 25) ** 2))
    rc = 2 * np.sqrt(Cbp ** 7 / (Cbp ** 7 + 25 ** 7))
    sl = 1 + 0.015 * (Lbp - 50) ** 2 / np.sqrt(20 + (Lbp - 50) ** 2)
    sc = 1 + 0.045 * Cbp
    sh = 1 + 0.015 * Cbp * T
    rt = -np.sin(np.radians(2 * d_theta)) * rc
    return np.sqrt((dLp / sl) ** 2 + (dCp / sc) ** 2 + (dHp / sh) ** 2
                   + rt * (dCp / sc) * (dHp / sh))


def neutral_delta_e(img_a: np.ndarray, img_b: np.ndarray,
                    mask: np.ndarray | None = None) -> dict:
    """两幅 sRGB 图在掩码区域的平均/分位 ΔE2000（§8.4 色彩指标）。"""
    a = img_a[mask] if mask is not None else img_a.reshape(-1, 3)
    b = img_b[mask] if mask is not None else img_b.reshape(-1, 3)
    de = delta_e_2000(srgb_to_lab(a), srgb_to_lab(b))
    return {
        "mean": float(de.mean()),
        "p50": float(np.median(de)),
        "p95": float(np.quantile(de, 0.95)),
        "max": float(de.max()),
        "n": int(de.size),
    }
