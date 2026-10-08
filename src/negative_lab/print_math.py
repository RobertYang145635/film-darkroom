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
控制点的三个横坐标来自**这一帧自己的分位数**（p5/p50/p95），
且必须取在**重映射自己会读到的那个域**（`picture_density`）——
早期版本取逐通道密度的 luma 分位，两个域隔着 0 截断并不等价，
实测黑点只咬住了名义像素的 1/50，趾部等于没压（见 `picture_density` 文档）。
取样区也必须落在**会被看到的那块**（`centre_region`：画面区包围盒的中心 70%）——
整幅画面区里最暗的 5% 实测有 7/9 帧落在外沿（翻拍架照度 falloff、边缘灰尘），
让边角决定主体的黑点，中间调就永远压不实。

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
#: 打印锚点只看**会被看到的那块**：画面区包围盒的中心这个比例。
#: 实测依据见 `centre_region`。
PRINT_REGION_CENTRE_FRAC = 0.70
#: 色度增益。**这是唯一一个来自用户手工参考的量，属于"口味"不是"测量"**：
#: 1.897 由 `1298_G200` 9 帧参考反解（中位数 `目标 sat_p95 / 软件 sat_p95`），
#: 而留出卷 `1299_G200` **没有参考可独立验证风格**，所以它的地位与
#: `PRINT_AIM`/`PRINT_WHITE`（外部实测相纸常数）不同，见 blockers B-11 追加⑧。
#: 设成 1.0 即完全关闭这一步，退回"只印亮度"。
LOOK_CHROMA_GAIN = 1.897
#: Rec.709 luma 权重（与 `library/render.py` 同一组，不另起一份）。
_LUM_W = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)
#: 同一组权重的 float32 版：整帧运算用它，别把 (H,W,3) 抬到 float64
#: （45MP 帧上每个 float64 数组就是 1.08GB）。
_LUM_W32 = _LUM_W.astype(np.float32)


def anchor_refusal_reason(samples: np.ndarray | None) -> str:
    """为什么定不出打印曲线——**拒绝必须说得出原因**。

    没有这句话，被拒的帧会安静地按未印相的技术正片导出，而用户看到的
    就是"一键出图还是糊了一层奶"，且没有任何地方告诉他软件放弃了。
    实测样本：标定卷 `1298_G200::G200-10`（中心 p5/p50/p95 =
    0.048 / 0.128 / 0.889 decade，趾部斜率 1.413 > 高光 0.795）。
    曲线本来可解时返回空串（调用方不该拿它当"永远有话要说"的装饰）。
    """
    if samples is None or samples.size < 256:
        n = 0 if samples is None else int(samples.size)
        return f"画面中心取样像素不足（{n} < 256）"
    lo = float(np.percentile(samples, P_LOW))
    mid = float(np.percentile(samples, P_MID))
    hi = float(np.percentile(samples, P_HIGH))
    if hi - lo < MIN_SPAN:
        return (f"画面中心密度跨度仅 {hi - lo:.2f} decade"
                f"（阈值 MIN_SPAN={MIN_SPAN}），三点曲线会近乎阶跃")
    if mid <= lo or hi <= mid:
        return (f"画面中心分位退化（lo {lo:.3f} / mid {mid:.3f} / hi {hi:.3f}），"
                "黑-中灰-白次序不成立")
    try:
        print_curve((lo, mid, hi))
    except ValueError as e:
        return str(e)
    return ""


def centre_region(mask: np.ndarray | None,
                  frac: float = PRINT_REGION_CENTRE_FRAC,
                  shape: tuple | None = None) -> np.ndarray | None:
    """把**取样**区域限制到"会被看到的那块"：画面区包围盒的中心 `frac`。

    为什么锚点不能由整幅画面区决定（2026-10-08 实测，标定卷 9 帧，
    `picture_density` 域里的 p5，整幅 → 中心）：
    G200-8 `0.014 → 0.369`（差 0.355 decade）、G200-11 `0.087 → 0.247`、
    G200-6 `0.132 → 0.255`、G200-1 `0.041 → 0.117`；G200-7/G200-10 无差。
    也就是说**最暗的那 5% 基本落在画面外沿**（翻拍架的照度 falloff、
    边缘灰尘、片基不均），拿它当黑点就等于让边角决定主体的影调：
    黑点被拖到主体暗部之下，中间调整段压不实——实测中心构图
    `p05=0.433` 而用户手工参考是 `0.193`，同一帧改成中心取样后是 `0.190`。

    **用包围盒而不是数组中心**是刻意的：包围盒只由画面区决定，
    所以"保留片边"开关（会改变数组尺寸）不会移动取样区，
    这条由 `tests/unit/test_print_scale.py::test_centre_region_ignores_rebate_padding`
    钉住。
    """
    if mask is None:
        if shape is None or len(shape) < 2:
            return None
        h, w = int(shape[0]), int(shape[1])
        out = np.zeros((h, w), dtype=bool)
        cy, cx = int(round(h * (1 - frac) / 2)), int(round(w * (1 - frac) / 2))
        out[cy:h - cy, cx:w - cx] = True
        return out
    m = np.asarray(mask, dtype=bool)
    if m.ndim != 2 or not m.any():
        return m
    ys, xs = np.nonzero(m)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    bh, bw = y1 - y0, x1 - x0
    cy, cx = int(round(bh * (1 - frac) / 2)), int(round(bw * (1 - frac) / 2))
    if bh - 2 * cy < 2 or bw - 2 * cx < 2:      # 包围盒太小时不再收中心
        return m
    out = np.zeros_like(m)
    out[y0 + cy:y1 - cy, x0 + cx:x1 - cx] = True
    return out & m


def picture_density(positive_linear: np.ndarray, dmax: float,
                    mask: np.ndarray | None = None) -> np.ndarray | None:
    """锚点取样域 = **`remap_luma` 自己会反解出来的那个密度**。

    这一版是实测逼出来的修正。原先的分位取在 `−log10(t/base)` 的**逐通道密度的
    luma** 上，但重映射读的是**技术正片的 luma**并用
    `d = −dmax·log10(1−y)` 反解密度。两者不等价：逐通道先做纸级形式
    （再对负值截断到 0）后取 luma，与先取密度 luma 再做纸级形式，中间隔着
    截断，而橙色片基让三通道密度天然不等。

    实测后果（九帧标定卷，`outputs/ref_look_probe/print_reach.txt`）：
    G200-1 名义 p5 以下按真实密度应有 **5.00%** 的像素，按反解域只有 **0.10%**；
    G200-8 是 5.00% 对 **0.00%**——黑点根本没咬住，趾部整段没被压，
    这就是接入打印曲线后暗部偏差仍高达 `p05 +0.110` 的原因。
    锚点取在哪个域，就必须和施加域是同一个域，否则分位数只是名义数字。
    代价是这个刻度**不等于物理密度**：`density.py:132` 的链条还带
    `exposure/black/gamma/逐通道 wb_high`，所以它是"重映射自己的刻度"。
    对印相来说这正是想要的东西——"这一帧最暗的 5% 印成纯黑"，
    而不必假装在读密度计。
    """
    d = float(dmax)
    if not math.isfinite(d) or d <= 0:
        return None
    x = np.asarray(positive_linear, dtype=np.float32)
    if x.ndim != 3 or x.shape[-1] < 3:
        return None
    # 与 `_remap_band` 逐比特同式：float32 归约 → float64 反解
    lum = np.clip(x[..., :3] @ _LUM_W32, 0.0, 1.0 - 1e-6)
    dl = -d * np.log10(np.maximum(1e-9, 1.0 - lum)).astype(np.float64)
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
                dmax_old: float, mask_band: np.ndarray | None,
                chroma_gain: float = LOOK_CHROMA_GAIN) -> np.ndarray:
    """若干行的保色度重映射。

    `chroma_gain` 是**逐帧可换**的：片基没实测的帧不该拿 1.897 去放大一片
    不知道错在哪的颜色（见 `print_remap_array` 与 `service` 第 9b 步）。
    参数名不能叫 `gain`——函数体里那个名字是逐像素亮度增益数组。

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
    out = y[..., None] + np.float32(chroma_gain) * (out - y[..., None])
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
                      max_chunk_pixels: int = 4_000_000,
                      chroma_gain: float | None = None):
    """**按行分块**的打印重映射，返回 (新数组, 采用记录或 None)。

    `chroma_gain=None` 表示"用模块默认值"（`LOOK_CHROMA_GAIN`）；
    生产路径会**按片基来源**传 1.0（未实测就不放大色彩），见 `service` 第 9b 步。

    分块是必须的，不是优化：`apply_render` 作用在**全分辨率**技术正片上，
    未分块时 45MP 帧会一次新建六个同尺寸数组（约 1.6GB 峰值），
    实测把全量测试打成 4/4 段错误（栈顶 `theme.py:189`，Qt 分配处）；
    见 `docs/blockers.md` B-11 追加⑦。全程 float32。
    """
    if anchor is None:
        return positive_linear, None
    k = LOOK_CHROMA_GAIN if chroma_gain is None else float(chroma_gain)
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
                          None if m is None else m, k)
    else:
        out = np.empty(rgb.shape, dtype=np.float32)
        for y0 in range(0, h, rows):
            y1 = min(y0 + rows, h)
            out[y0:y1] = _remap_band(rgb[y0:y1], anchor, float(dmax_old),
                                     None if m is None else m[y0:y1], k)
    if x.shape[-1] > 3:
        out = np.concatenate([out, x[..., 3:]], axis=2)
    info = {"d_lo": round(anchor[0], 5), "d_mid": round(anchor[1], 5),
            "d_hi": round(anchor[2], 5), "aim": PRINT_AIM, "white": PRINT_WHITE,
            "dmax_old": round(float(dmax_old), 5), "chroma_gain": round(k, 5)}
    return out, info
