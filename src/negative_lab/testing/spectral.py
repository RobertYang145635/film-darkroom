"""非循环光谱合成负片（整改 v3 P0-B）。

与生产逆算刻意不同构的前向模型：
  场景曝光 E(x,y,c) --每通道 H-D 曲线(不同 gamma + toe/shoulder 软饱和)-->
  染料密度向量 d --3x3 串扰矩阵 C(染料次要吸收/扫描器混色)--> d_meas
  透射率 T_c = B_c * 10^(-d_meas_c) --空间增益(暗角)-- 颗粒+读出噪声 --> 剪裁

用途：色相次序、色度不塌缩、中性跨密度漂移等**色度**验证——旧生成器
把彩色块 mean 成灰度，只能验证亮度方向（v3 F-ALG-2）。
"""

from __future__ import annotations

import numpy as np

# 近似 ColorChecker 的色块（场景线性 RGB，0-1）
PATCHES = {
    "red":     (0.65, 0.06, 0.05),
    "orange":  (0.70, 0.28, 0.05),
    "yellow":  (0.72, 0.68, 0.06),
    "green":   (0.10, 0.50, 0.12),
    "cyan":    (0.10, 0.55, 0.65),
    "blue":    (0.08, 0.10, 0.60),
    "purple":  (0.40, 0.08, 0.50),
    "gray35":  (0.35, 0.35, 0.35),
    "gray65":  (0.65, 0.65, 0.65),
    "gray18":  (0.18, 0.18, 0.18),
}
PATCH_ORDER = ["red", "orange", "yellow", "green", "cyan", "blue", "purple"]


def _soft_saturation(e: np.ndarray, toe: float = 0.02) -> np.ndarray:
    """H-D 曲线：低曝光趾部压缩 + 高曝光肩部软饱和（与逆算的对数幂不同构）。"""
    e = np.clip(e, 0.0, 1.2)
    toe_c = toe * np.tanh(e / toe)           # 趾部：亚线性
    e2 = toe_c + 0.15 * e * e                # 轻微上凸增益
    return e2 / (1.0 + 0.35 * e2)            # 肩部：软饱和


def make_spectral_negative(h: int = 240, w: int = 640, seed: int = 7,
                           base=(0.62, 0.45, 0.30),
                           gammas=(1.9, 1.6, 2.3),
                           crosstalk: bool = True,
                           vignetting: bool = True,
                           grain: float = 0.01):
    rng = np.random.default_rng(seed)
    g = np.asarray(gammas, np.float32)
    # 3x3 串扰：行=测量通道，列=染料密度；对角占优 + 次要吸收
    if crosstalk:
        C = np.array([[0.92, 0.09, 0.05],
                      [0.06, 0.90, 0.07],
                      [0.04, 0.08, 0.94]], np.float32)
    else:
        C = np.eye(3, dtype=np.float32)

    scene = np.zeros((h, w, 3), np.float32)
    # 左半：灰阶斜坡（跨密度中性）
    ramp = np.linspace(0.06, 0.95, w // 2, dtype=np.float32)[None, :]
    scene[:, :w // 2, :] = ramp[..., None]
    # 右半：色块列
    px = w // 2
    pw = max(8, (w - px) // (len(PATCHES) + 1))
    patch_roi = {}
    for i, name in enumerate(PATCHES):
        x0 = px + i * pw
        if x0 + pw > w:
            break
        scene[:, x0:x0 + pw, :] = np.array(PATCHES[name], np.float32)
        patch_roi[name] = (slice(h // 4, 3 * h // 4), slice(x0 + 2, x0 + pw - 2))

    # 曝光 -> 每通道 H-D 曲线 -> 染料密度
    # Hurter-Driffield：密度随曝光增加（above-base）：d = gamma_c * log10(1+e/e0)
    # （首版误写成 log10(1/e) 方向反了——色相测试当场抓住）
    e = np.clip(scene * 1.15, 0.0, None)
    e_sat = _soft_saturation(e)
    e0 = 0.05
    d = np.stack([g[c] * np.log10(1.0 + e_sat[..., c] / e0)
                  for c in range(3)], axis=-1)      # (h,w,3) 染料密度
    d_meas = d @ C.T                                  # 串扰
    t = np.asarray(base, np.float32)[None, None, :] * np.power(10.0, -d_meas)

    # 空间增益（暗角：中心亮边缘暗）与灯箱漏光带
    if vignetting:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        r2 = ((yy - h / 2) / h) ** 2 + ((xx - w / 2) / w) ** 2
        t *= (1.0 - 0.25 * r2)[..., None]

    # 颗粒 + 读出噪声
    t *= np.exp(rng.normal(0, grain, size=t.shape)).astype(np.float32)
    t += rng.normal(0, 0.0015, size=t.shape).astype(np.float32)

    # 未曝光片基条带（右侧 64px，含少量更暗尘点）
    base_w = 64
    neg = np.concatenate([t, np.tile(np.asarray(base, np.float32),
                                     (h, base_w, 1))], axis=1)
    dust = []
    for _ in range(25):
        cy, cx = int(rng.integers(2, h - 2)), int(rng.integers(t.shape[1] + 2,
                                                              neg.shape[1] - 2))
        neg[cy - 1:cy + 2, cx - 1:cx + 2] *= 0.4
        dust.append((cy, cx))
    neg = np.clip(neg, 0.0, 1.0).astype(np.float32)

    return {
        "negative": neg,
        "truth": {
            "base": tuple(float(v) for v in base),
            "gammas": tuple(float(v) for v in gammas),
            "crosstalk": C.tolist(),
            "patch_roi": patch_roi,
            "patch_rgb": {k: PATCHES[k] for k in PATCHES},
            "ramp_cols": (0, w // 2),
            "base_cols": (t.shape[1], neg.shape[1]),
            "dust": dust,
            "scene": scene,
        },
    }


def hue_angle(rgb: np.ndarray) -> np.ndarray:
    """sRGB(0-1) -> 感知近似 hue 角（度），用于次序断言而非绝对色度。"""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    d = np.maximum(mx - mn, 1e-9)
    hr = ((g - b) / d) % 6
    hg = (b - r) / d + 2
    hb = (r - g) / d + 4
    hsel = np.where(mx == r, hr, np.where(mx == g, hg, hb))
    return (hsel * 60.0) % 360.0   # hsel 已是 0-6 扇区单位，×60 即度（勿再 radians->degrees）
