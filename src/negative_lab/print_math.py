"""自动打印尺度的**纯数学**（叶子模块：只依赖 numpy / math）。

**为什么单独成一个根目录叶子模块**：`tests/unit/test_layering.py` 规定
下层不得 import 上层（`library` 不许 import `pipeline`），而施加打印曲线的
`apply_render` 就在 `library/render.py` 里。本模块与 `constants.py`、
`metrics.py` 同族——那两处反向依赖当初就是这么消除的。

**要解决的问题（实测）**：`pipeline/params.py:18` 把密度跨度写成常数 `dmax=2.0`，
而扣掉橙色片基后画面内容只占约 1.0 decade，自动出片显示域阴影 `p05=0.71`、
对比 `std=0.059`（用户手工成品分别是 0.35 / 0.15）——一层奶白雾。

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

**施加方式**：对亮度做保色度重映射（`out = x · 新亮度/旧亮度`，逐像素统一增益），
所以**色度比例不变**——第一版把曲线逐通道施加到密度上，对比上来了但天空变成
深青蓝，那是颜色改动，`pipeline/service.py` 里"未验证的颜色矩阵不得静默启用"
的同一条纪律不允许它发生。
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
#: 色度增益。**这是唯一一个来自用户手工参考的量，属于"口味"不是"测量"**：
#: 1.897 由 `1298_G200` 9 帧参考反解（中位数 `目标 sat_p95 / 软件 sat_p95`），
#: 而留出卷 `1299_G200` **没有参考可独立验证风格**，所以它的地位与
#: `PRINT_AIM`/`PRINT_WHITE`（外部实测相纸常数）不同，见 blockers B-11 追加⑧。
#: 设成 1.0 即完全关闭这一步，退回"只印亮度"。
LOOK_CHROMA_GAIN = 1.897
_EPS = 1e-6
#: Rec.709 luma 权重（与 `library/render.py` 同一组，不另起一份）。
_LUM_W = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)
#: 同一组权重的 float32 版：整帧运算用它，别把 (H,W,3) 抬到 float64
#: （45MP 帧上每个 float64 数组就是 1.08GB）。
_LUM_W32 = _LUM_W.astype(np.float32)


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
    """(密度控制点 x, 纸级曝光量 y)；要求斜率**递增**（先压趾部、后展高光）。

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


def _remap_band(rgb: np.ndarray, anchor: tuple[float, float, float],
                dmax_old: float, mask_band: np.ndarray | None) -> np.ndarray:
    """若干行的保色度重映射。

    **逐像素、无空间操作**，所以分块与整帧结果逐比特一致
    （`rgb @ _LUM_W32` 只在最后一轴归约，与行数无关）——这条由
    `tests/unit/test_print_scale.py::test_chunked_matches_whole_frame_bit_exact` 钉住。
    """
    lum = rgb @ _LUM_W32
    top = rgb.max(axis=2)
    new_lum = remap_luma(lum, anchor, dmax_old).astype(np.float32)
    floor = np.float32(1e-5)
    bright = lum > floor
    gain = np.ones(lum.shape, dtype=np.float32)
    np.divide(new_lum, np.where(bright, lum, np.float32(1.0)),
              out=gain, where=bright)
    # 上限是**相纸最大反射率**而非 1.0：纸到不了 1.0，
    # 而封到 1.0 会让像素正好堆在 1.0 上，被门禁 `_clip`（按 ≥1−tol 计数）
    # 判成削顶——实测那样仍有 2.29% 的 B 通道高光越界，等于白封。
    cap = np.where(top > 1e-6,
                   np.float32(PRINT_WHITE) / np.maximum(top, np.float32(1e-6)),
                   np.float32(1.0)).astype(np.float32)
    np.minimum(gain, cap, out=gain, where=gain > np.float32(1.0))
    if mask_band is not None:
        gain = np.where(mask_band, gain, np.float32(1.0))  # 片边不被"打印"
    out = rgb * gain[..., None]
    # ---- 色度增益：亮度已经由曲线定好了，这里只把"偏离亮度的部分"放大。
    # `out = y + k(x − y)` 保持亮度 y 不变（权重和为 1），所以不会推翻上一步的
    # 电平与对比。**为什么必须显式做**：扣完橙色片基仍残留抑制，实测
    # 软件饱和 p95 0.212 对用户参考 0.405（差 −0.193），是成品"不够像"的主因。
    y = out @ _LUM_W32
    out = y[..., None] + np.float32(LOOK_CHROMA_GAIN) * (out - y[..., None])
    # 放大色度会把最亮通道顶过天花板，所以按**整像素统一缩放**收回：
    # 同除一个系数不改通道比例（不重演上次"逐通道钳位造成青蓝偏"的错误）。
    top2 = out.max(axis=2)
    over = top2 > np.float32(PRINT_WHITE)
    shrink = np.where(over, np.float32(PRINT_WHITE) / np.maximum(top2, np.float32(1e-6)),
                      np.float32(1.0)).astype(np.float32)
    out = out * shrink[..., None]
    return np.clip(out, 0.0, 1.0, dtype=np.float32)


def print_remap_array(positive_linear: np.ndarray,
                      anchor: tuple[float, float, float] | None,
                      dmax_old: float, mask: np.ndarray | None = None,
                      max_chunk_pixels: int = 4_000_000):
    """**按行分块**的打印重映射，返回 (新数组, 采用记录或 None)。

    分块是必须的，不是优化：`apply_render` 作用在**全分辨率**技术正片上，
    未分块时 45MP 帧会一次新建六个同尺寸数组（约 1.6GB 峰值），
    实测把全量测试打成 4/4 段错误（栈顶 `theme.py:189`，Qt 分配处）；
    见 `docs/blockers.md` B-11 追加⑦。全程 float32。
    """
    if anchor is None:
        return positive_linear, None
    x = np.asarray(positive_linear, dtype=np.float32)
    if x.ndim != 3 or x.shape[-1] < 3:
        return positive_linear, None
    try:
        print_curve(anchor)                      # 先验曲线合法，避免半路抛错
    except ValueError:
        return positive_linear, None
    m = np.asarray(mask, dtype=bool) if mask is not None else None
    if m is not None and m.shape != x.shape[:2]:
        m = None
    rgb = x[..., :3]
    h = x.shape[0]
    row_px = max(1, x.shape[1] * 3)
    rows = max(1, int(max_chunk_pixels) // row_px)
    if h <= rows:
        out = _remap_band(rgb, anchor, float(dmax_old),
                          None if m is None else m)
    else:
        out = np.empty(rgb.shape, dtype=np.float32)
        for y0 in range(0, h, rows):
            y1 = min(y0 + rows, h)
            out[y0:y1] = _remap_band(rgb[y0:y1], anchor, float(dmax_old),
                                     None if m is None else m[y0:y1])
    if x.shape[-1] > 3:
        out = np.concatenate([out, x[..., 3:]], axis=2)
    info = {"d_lo": round(anchor[0], 5), "d_mid": round(anchor[1], 5),
            "d_hi": round(anchor[2], 5), "aim": PRINT_AIM, "white": PRINT_WHITE,
            "dmax_old": round(float(dmax_old), 5)}
    return out, info
