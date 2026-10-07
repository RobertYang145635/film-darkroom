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

**落点**：技术层的 `InvertParams`（route C：A 不变 + 单调曲线），
所以预览、导出、门禁统计看到的是同一张图；不改后端公式、不动 3×3 矩阵。
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


def print_knots(anchor: tuple[float, float, float],
                *, aim: float = PRINT_AIM,
                white: float = PRINT_WHITE) -> tuple:
    """密度域单调折线控制点（**绝对密度 d → 纸级曝光量 v**）。

    纵坐标由 `paper = 1 − 10^(−v)` 反解：`v = −log10(1 − paper)`。
    横坐标必须是**绝对密度**——`_apply_channel_curves` 喂进来的是 `v = A·d·s`，
    用相对坐标会在 `d_lo` 不为 0 的帧上把整条曲线平移错位。
    `np.interp` 在两端自动钳位：`d < d_lo → v = 0`（印成黑），
    `d > d_hi → v = v_hi`（印成相纸最大反射率），这正是我们要的硬端点。
    斜率必须**递增**（先压趾部、后展高光），否则不是"加对比"而是扭曲。
    """
    lo, mid, hi = anchor
    v_mid = -math.log10(max(1e-9, 1.0 - aim))
    v_hi = -math.log10(max(1e-9, 1.0 - white))
    knots = ((lo, 0.0), (mid, v_mid), (hi, v_hi))
    s1 = v_mid / (mid - lo)
    s2 = (v_hi - v_mid) / (hi - mid)
    if not (math.isfinite(s1) and math.isfinite(s2)) or s2 <= s1:
        raise ValueError(f"打印曲线斜率未递增（趾部 {s1:.3f} ≥ 高光 {s2:.3f}）")
    return knots


def apply_print_scale(params, anchor: tuple[float, float, float] | None):
    """把打印尺度写进 `InvertParams`；无锚点或曲线不合法时**原样返回**。

    只改 `dmax/offset/black/channel_curves/backend_name` 四项；
    `density_matrix` 与色温/色调**一律不碰**——曲线是影调，矩阵是颜色。
    **带指数先验（`reference_exponent`/`red_ratio`/`blue_ratio` 非 1）时不套**：
    那时曲线自变量是 `d·s` 且逐通道不同，共用一条绝对密度曲线会
    悄悄改掉通道比例（= 色偏）。宁可退回"没有自动打印尺度"，
    也不能让一整卷导出因为某帧曲线退化而中断。
    """
    if anchor is None:
        return params, None
    slopes = (float(getattr(params, "reference_exponent", 1.0)),
              float(getattr(params, "red_ratio", 1.0)),
              float(getattr(params, "blue_ratio", 1.0)))
    if any(abs(s - 1.0) > 1e-9 for s in slopes):
        return params, None
    try:
        knots = print_knots(anchor)
    except ValueError:
        return params, None
    curves = (knots, knots, knots)      # 三通道共用同一条：不引入通道间比例变化
    out = _replace(params, dmax=1.0, offset=0.0, black=0.0,
                   channel_curves=curves,
                   backend_name="density-affine-curves")
    return out, {"d_lo": round(anchor[0], 5), "d_mid": round(anchor[1], 5),
                 "d_hi": round(anchor[2], 5),
                 "knots": [[round(x, 5), round(y, 5)] for x, y in knots],
                 "aim": PRINT_AIM, "white": PRINT_WHITE}


def _replace(params, **kw):
    import dataclasses
    return dataclasses.replace(params, **kw)
