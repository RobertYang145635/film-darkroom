"""自动打印尺度（B-11）：把这一帧的密度分布映到相纸的反射率范围。

**要解决的问题（实测）**：`params.dmax` 是常数 2.0，而扣掉橙色片基后画面内容
只占约 1.0 decade，自动出片显示域阴影 `p05=0.71`、对比 `std=0.059`
（用户手工成品分别是 0.35 / 0.15）——一层奶白雾。

**为什么必须是一条曲线而不是一个指数**：纸级形式 `paper = 1 − 10^(−v)` 要求
    paper=0 → v=0；  paper=AIM 0.18 → v=0.0862；  paper=WHITE 0.80375 → v=0.699
而实测画面密度里"黑→中灰"与"中灰→高光"几乎等宽（0.479 vs 0.512 decade），
纸级要的比却是 **1 : 8.1**。`(dmax, offset)` 只能整体缩放，**表达不了这个比例**，
所以必须给密度域一条单调特性曲线（先压趾部、后展高光）。
外部实现同理：NegPy 纸 `d_max=2.3/d_min=0.06`、horizon `dMax=(2.66,2.53,2.48)`、
`color-neg-resources` 直接给一张实测 Endura 相纸 LUT（反射率 0.00278 / **0.80375**）。

**两个纸级常数都不是拟合量**：`AIM=0.18` 是 horizon 的 `Paper.printAim`
与 filmeon 的 `log10(0.18)` 共同取值；`WHITE=0.80375` 是上面那张实测相纸的最大反射率。
控制点的三个横坐标来自**这一帧自己的密度分位数**（p5/p50/p95，三通道共用 luma）。

**落点**：反相之后、候选之前，对**技术正片的亮度**做保色度重映射
（`out = x · 新亮度/旧亮度`，逐像素统一增益）。
所以预览、导出、门禁统计看到的是同一张图，而**色度比例不变**——
第一版把曲线逐通道施加到密度上，对比是上来了，天空却变成深青蓝，
那是颜色改动，`service.py:383-388` 不允许在这条路径上静默发生。
"""

from __future__ import annotations

import math

import numpy as np

#: 相纸中灰 aim（线性反射率）。外部两家同值，非对样片拟合。
PRINT_AIM = 0.18
#: 相纸最大反射率。实测自 Endura 相纸（`alchemy-color/color-neg-resources`）。
PRINT_WHITE = 0.80375
#: 画面密度的三个锚点分位（三通道共用 luma 域；逐通道会引入未验证色偏，
#: horizon 实测放宽端点让中间调 R-G 漂了 14.4 codes）。
P_LOW, P_MID, P_HIGH = 5.0, 50.0, 95.0
#: 跨度不足时**不套曲线**：薄负片/纯色卡帧用三点解出的曲线会近乎阶跃。
MIN_SPAN = 0.15
_EPS = 1e-6
#: Rec.709 luma 权重（与 `library/render.py` 同一组，不另起一份）。
_LUM_W = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)


def positive_density(t_rgb: np.ndarray, base: np.ndarray,
                     mask: np.ndarray | None = None) -> np.ndarray | None:
    """画面内**正密度** `d = −log10(t / base)` 的 luma 一维样本。

    片基非有限或非法时返回 None（调用方必须回退到"不套曲线"，
    而不是拿一个假 base 去算分位数）。
    """
    b = np.asarray(base, dtype=np.float64).reshape(-1)
    if b.size < 3 or not np.isfinite(b[:3]).all() or (b[:3] <= 0).any():
        return None
    x = np.asarray(t_rgb, dtype=np.float64)
    if x.ndim != 3 or x.shape[-1] < 3:
        return None
    with np.errstate(divide="ignore", invalid="ignore"):
        d = -np.log10(np.clip(x[..., :3], _EPS, None) / b[:3])
    dl = 0.2126 * d[..., 0] + 0.7152 * d[..., 1] + 0.0722 * d[..., 2]
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        if m.shape == dl.shape:
            dl = dl[m]
    dl = dl[np.isfinite(dl)]
    return dl if dl.size else None


def density_anchor(samples: np.ndarray | None) -> tuple[float, float, float] | None:
    """(d_lo, d_mid, d_hi)；样本不足或跨度太窄时返回 None。"""
    if samples is None or samples.size < 256:
        return None
    lo = float(np.percentile(samples, P_LOW))
    mid = float(np.percentile(samples, P_MID))
    hi = float(np.percentile(samples, P_HIGH))
    if not (math.isfinite(lo) and math.isfinite(mid) and math.isfinite(hi)):
        return None
    if hi - lo < MIN_SPAN or mid <= lo or hi <= mid:
        return None
    return lo, mid, hi


def print_curve(anchor: tuple[float, float, float], *,
                aim: float = PRINT_AIM, white: float = PRINT_WHITE):
    """(密度控制点 x, 纸级曝光量 y)；要求斜率递增（先压趾部、后展高光）。

    纵坐标由 `paper = 1 − 10^(−v)` 反解：`v = −log10(1 − paper)`。
    """
    lo, mid, hi = anchor
    v_mid = -math.log10(max(1e-9, 1.0 - aim))
    v_hi = -math.log10(max(1e-9, 1.0 - white))
    s1 = v_mid / (mid - lo)
    s2 = (v_hi - v_mid) / (hi - mid)
    if not (math.isfinite(s1) and math.isfinite(s2)) or s2 <= s1:
        raise ValueError(f"打印曲线斜率未递增（趾部 {s1:.3f} ≥ 高光 {s2:.3f}）")
    return (np.array([lo, mid, hi], dtype=np.float64),
            np.array([0.0, v_mid, v_hi], dtype=np.float64))


def remap_luma(y_old: np.ndarray, anchor: tuple[float, float, float],
               dmax_old: float, *, aim: float = PRINT_AIM,
               white: float = PRINT_WHITE) -> np.ndarray:
    """把**旧纸级映射下的亮度**重映到锚定的新映射上（一维，供测试与复用）。

    旧映射 `y = 1 − 10^(−d/dmax)` 可逆：`d = −dmax·log10(1 − y)`，
    所以不需要重跑反相就能拿到"这像素的密度是多少"，再按曲线求新亮度。
    """
    xs, ys = print_curve(anchor, aim=aim, white=white)
    y = np.clip(np.asarray(y_old, dtype=np.float64), 0.0, 1.0 - 1e-6)
    d = -float(dmax_old) * np.log10(np.maximum(1e-9, 1.0 - y))
    v = np.interp(d, xs, ys)
    return 1.0 - np.power(10.0, -v)


def apply_print_remap(res, anchor: tuple[float, float, float] | None,
                      dmax_old: float, mask: np.ndarray | None = None):
    """对技术正片做**保色度**的打印重映射，返回 (新 InvertResult, 采用记录)。

    为什么必须保色度：第一版把同一条曲线**逐通道**施加到密度上，
    结果对比上来了但天空变成深青蓝——因为 B 通道在橙色片基扣除后
    落在共享曲线的另一段上，通道差被 8 倍斜率放大，那是**颜色改动**，
    而 `service.py:383-388` 明令禁止在这条路径上静默改颜色。
    现在只对**亮度**施加曲线，再按逐像素统一增益缩放三通道，
    色度比例逐比特保持（`out = x · (y_new / y)`）。

    无锚点 / 曲线退化 / 增益不可靠 ⇒ **原样返回**，不猜也不抛错。
    """
    if anchor is None:
        return res, None
    x = np.asarray(res.positive_linear, dtype=np.float64)
    if x.ndim != 3 or x.shape[-1] < 3:
        return res, None
    lum = x[..., :3] @ _LUM_W
    try:
        new_lum = remap_luma(lum, anchor, dmax_old)
    except ValueError:
        return res, None
    floor = 1e-5
    safe = np.where(lum > floor, lum, 1.0)
    gain = np.where(lum > floor, new_lum / safe, 1.0)
    # 增益上限是**相纸最大反射率**，不是 1.0：纸不可能反射超过 WHITE，
    # 而封到 1.0 会让像素正好堆在 1.0 上，被门禁的 `_clip`（按 ≥1−tol 计数）
    # 判成削顶——实测那样仍是 2.29% 的 B 通道高光越界，等于白封。
    top = x[..., :3].max(axis=2)
    cap = np.where(top > 1e-6, PRINT_WHITE / np.maximum(top, 1e-6), 1.0)
    gain = np.where(gain > 1.0, np.minimum(gain, cap), gain)
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        if m.shape == gain.shape:
            gain = np.where(m, gain, 1.0)   # 片边不是画面内容，不该被"打印"
    out = np.clip(x[..., :3] * gain[..., None], 0.0, 1.0)
    if out.shape[-1] == 3 and x.shape[-1] > 3:
        out = np.concatenate([out, x[..., 3:]], axis=2)
    from .density import encode_srgb
    new = _copy_result(res, out.astype(np.float32), encode_srgb(out))
    info = {"d_lo": round(anchor[0], 5), "d_mid": round(anchor[1], 5),
            "d_hi": round(anchor[2], 5), "aim": PRINT_AIM, "white": PRINT_WHITE,
            "dmax_old": round(float(dmax_old), 5)}
    return new, info


def _copy_result(res, positive: np.ndarray, preview: np.ndarray):
    import dataclasses
    return dataclasses.replace(res, positive_linear=positive, preview_srgb=preview)
