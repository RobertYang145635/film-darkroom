"""合成负片生成器（规格 §10 Phase 1、§13.3）。

物理前向模型与 density.py 的符号约定严格一致：
    场景亮度 L in [0,1] -> 胶片密度 -> 归一化透射率
        z_true_c = -dmax_true * k_c * f(L),  f 单调, f(0)=0, f(1)=1
        t_c      = dmin_true_c * 10^(z_true_c)
    因此：L=0（最暗场景）-> t = dmin（片基，最亮）；
          L=1（最亮场景）-> t = dmin*10^(-dmax*k)（最密，最暗）。
若本文件与 pipeline 的方向不一致，tests/synthetic/test_direction.py 会失败。
"""

from __future__ import annotations

import numpy as np

# 橙片基典型线性透射率（C-41 mask：透过红/绿、吸收蓝）
ORANGE_BASE = (0.62, 0.45, 0.30)


def _scene_linear_rgb(h: int, w: int) -> tuple[np.ndarray, dict]:
    """构造场景：灰度斜坡 + 彩色块 + 亮/暗中性块。返回 (scene, truth)。"""
    scene = np.zeros((h, w, 3), dtype=np.float32)
    truth: dict = {}

    ramp_w = int(w * 0.45)
    L = np.linspace(0.0, 1.0, ramp_w, dtype=np.float32)[None, :]
    scene[:, :ramp_w, :] = L[..., None]
    truth["ramp_L"] = L[0]
    truth["ramp_cols"] = (0, ramp_w)

    # 彩色块（强色场景）
    x0 = ramp_w + 4
    bw = max(8, int(w * 0.08))
    colors = {"red": (0.9, 0.05, 0.05), "green": (0.05, 0.7, 0.05),
              "blue": (0.05, 0.05, 0.85), "yellow": (0.85, 0.8, 0.05)}
    for i, (name, c) in enumerate(colors.items()):
        xa, xb = x0 + i * (bw + 4), x0 + i * (bw + 4) + bw
        if xb > w:
            break
        scene[:, xa:xb, :] = np.array(c, dtype=np.float32)
        truth[f"patch_{name}"] = (slice(0, h), slice(xa, xb))

    # 中性参考块：亮（L=0.85）与暗（L=0.10），放在右侧
    nb_x = min(w - 12, x0 + 4 * (bw + 4) + 6)
    scene[:, nb_x:nb_x + 8, :] = 0.85
    nb2_x = min(w - 4, nb_x + 12)
    scene[:, nb2_x:nb2_x + 8, :] = 0.10
    truth["neutral_bright"] = (h // 2, nb_x + 4)
    truth["neutral_dark"] = (h // 2, nb2_x + 4)
    return scene, truth


def make_synthetic_negative(h: int = 256, w: int = 512,
                            dmin_true=tuple(ORANGE_BASE),
                            dmax_true: float = 2.0,
                            exponents=(1.0, 1.0, 1.0),
                            f_gamma: float = 0.7,
                            noise: float = 0.0015,
                            dust: int = 60,
                            seed: int = 7) -> dict:
    """生成一张带片基边缘、尘点与噪声的合成负片扫描。

    画面右侧附加宽度 base_width 的未曝光片基条带（L=0 -> t=dmin 精确值），
    用于测试 robust Dmin 与污染剔除。
    """
    rng = np.random.default_rng(seed)
    base_width = max(16, w // 8)
    img_w = w - base_width

    scene, truth = _scene_linear_rgb(h, img_w)
    L = scene.mean(axis=2)  # 场景亮度（灰度语义）
    L = np.clip(L, 0.0, 1.0)
    f = np.power(L, f_gamma)  # 单调 [0,1]->[0,1]

    k = np.asarray(exponents, dtype=np.float32)
    dmin = np.asarray(dmin_true, dtype=np.float32)
    z_true = (-float(dmax_true) * f[..., None] * k[None, None, :]).astype(np.float32)
    t = (dmin[None, None, :] * np.power(10.0, z_true)).astype(np.float32)

    # 片基条带：精确 t = dmin（含噪声与尘点，测试稳健统计）
    base = np.tile(dmin[None, None, :], (h, base_width, 1))
    neg = np.concatenate([t, base], axis=1)

    # 噪声（乘性+加性，保持有限）
    neg *= (1.0 + rng.normal(0.0, noise, size=neg.shape)).astype(np.float32)
    neg += rng.normal(0.0, noise * 0.5, size=neg.shape).astype(np.float32)

    # 尘点：片基条带上的暗色孤立斑点（应被 robust 统计剔除）
    for _ in range(dust):
        cy = int(rng.integers(2, h - 2))
        cx = int(rng.integers(img_w + 2, w - 2))
        r = int(rng.integers(1, 3))
        neg[max(0, cy - r):cy + r + 1, max(0, cx - r):cx + r + 1, :] *= 0.35

    neg = np.clip(neg, 0.0, 1.0).astype(np.float32)

    truth.update({
        "dmin": tuple(float(v) for v in dmin_true),
        "dmax": float(dmax_true),
        "exponents": tuple(float(v) for v in exponents),
        "base_roi": neg[:, img_w:, :],          # 未曝光片基条带
        "film_roi": neg[:, :img_w, :],          # 画面区域
        "base_cols": (img_w, w),
        "shape": neg.shape,
    })
    return {"negative": neg, "truth": truth}
