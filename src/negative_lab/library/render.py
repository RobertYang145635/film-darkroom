"""v5 §2.2 / §7.2：唯一微调层 `RenderIntent`。

TechnicalPositive（技术正片）之后、显示/导出编码之前的**单一**渲染意图。
顺序固定（§7.2），不得在技术 backend 与本层重复应用曝光、gamma、白平衡
或饱和度；主界面不再直接改写 `InvertParams` 的技术参数。

```
TechnicalPositive
  -> temp/tint chromatic adaptation
  -> exposure
  -> contrast around protected mid pivot
  -> highlights/shadows
  -> whites/blacks
  -> vibrance
  -> saturation
  -> gamut/finite guard
  -> display/export encoding
```

数值安全：影调曲线由**非负斜率累积**构造，因此对亮度严格单调；
所有控件在 0 处为恒等（默认值 = 不改变技术正片的一个比特）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# v5 §2.2 P0 微调面板的全部控件（顺序即 §7.2 的应用顺序）
P0_FIELDS = ("temperature", "tint", "exposure", "contrast", "highlights",
             "shadows", "whites", "blacks", "vibrance", "saturation")

#: 控件范围与单位（GUI 与测试共用，避免各处硬编码）
RANGES: dict[str, tuple[float, float, float]] = {
    "temperature": (-100.0, 100.0, 1.0),     # 归一化色温（>0 更暖）
    "tint": (-100.0, 100.0, 1.0),            # >0 偏绿，<0 偏洋红
    "exposure": (-5.0, 5.0, 0.01),           # EV 档
    "contrast": (-100.0, 100.0, 1.0),
    "highlights": (-100.0, 100.0, 1.0),
    "shadows": (-100.0, 100.0, 1.0),
    "whites": (-100.0, 100.0, 1.0),
    "blacks": (-100.0, 100.0, 1.0),
    "vibrance": (-100.0, 100.0, 1.0),
    "saturation": (-100.0, 100.0, 1.0),
}

#: 候选可施加的色温/色调硬上限（v5 §6.6：必须有上限并记录触发情况）
CANDIDATE_TEMP_CAP = 45.0
CANDIDATE_TINT_CAP = 25.0

RENDER_ORDER = ["temperature", "tint", "exposure", "contrast", "highlights",
                "shadows", "whites", "blacks", "vibrance", "saturation",
                "gamut_guard", "display_encoding"]

_LUM_W = np.array([0.2126, 0.7152, 0.0722], np.float32)

# Bradford（von Kries）线性 sRGB <-> LMS
_BRADFORD = np.array([
    [0.8951000, 0.2664000, -0.1614000],
    [-0.7502000, 1.7135000, 0.0367000],
    [0.0389000, -0.0685000, 1.0616000]], np.float64)
_BRADFORD_INV = np.linalg.inv(_BRADFORD)
_D65_XYZ = np.array([0.95047, 1.00000, 1.08883], np.float64)


@dataclass
class RenderIntent:
    """P0 基础微调。默认全 0 = 恒等。"""

    temperature: float = 0.0
    tint: float = 0.0
    exposure: float = 0.0
    contrast: float = 0.0
    highlights: float = 0.0
    shadows: float = 0.0
    whites: float = 0.0
    blacks: float = 0.0
    vibrance: float = 0.0
    saturation: float = 0.0
    # ---- 自动打印尺度（B-11）：**不是控件**，与 origin/notes 同类，随候选存储。
    # 为什么放这里而不是在 service 里插一步：v4 契约要求"恒等 intent 与技术正片
    # 逐像素等价"（在技术与渲染之间藏处理是违规的），而 v5 §8.2 的门禁只读技术正片。
    # 放进 intent 同时满足两边：技术层与门禁口径一字不动，
    # 印相决定可见、进 sidecar、可复算，预览与导出同源。
    # 空 tuple = 不施加（默认，保持旧行为）。
    print_anchor: tuple = ()           # (d_lo, d_mid, d_hi) 画面密度分位
    print_dmax: float = 0.0            # 施加时所依据的技术 dmax（反解密度用）
    #: 逐帧色度增益。`0.0` = 用模块默认 `LOOK_CHROMA_GAIN`；
    #: 片基**未实测**（generic_prior）的帧由 `service` 显式写 1.0——
    #: 拿一个对着已标定帧拟合出来的口味系数去放大一片不知道错在哪的颜色，
    #: 实测把 1299 两帧的饱和 p95 从 0.537/0.341 推到 0.910/0.769。
    print_chroma_gain: float = 0.0
    #: 印相曲线的**中灰输出目标**（纸级反射率）。`0.0` = 未施加 = 用模块默认 `PRINT_AIM`。
    #: 为什么逐帧可换而不是全局常数：ECN-2 两卷一致偏亮 ~0.30 EV 而 G200 两卷不偏，
    #: 全局改会把 G200 推过头（`docs/blockers.md` B-16 追加⑯）⇒ 按片种给。
    #: 与 `print_anchor` 同进同出：只在施加过印相时才写进 schema。
    print_aim: float = 0.0
    #: 高光带白点蓝黄校正的**片种强度**（`white_balance.apply_white_balance`）。
    #: `0.0` = 不施加 = 交付逐比特不变；方向与大小由本帧自己解，强度按片种给。
    #: 为什么在片种表里而不是全局常数：两卷量到的**内点都是 1.30**，但可用区间宽度差很多
    #: （5207 在 1.30~2.80 都合格；5219 只有 1.30 一格，更大的 `k` 把白的 `b*` 推过零、
    #: 撞回 T1）⇒ 入口必须能按片种分开调（`docs/blockers.md` B-16 追加㉙/㉚）。
    #: 与 `print_anchor` 同进同出：只在施加过印相时才写进 schema。
    white_balance: float = 0.0
    #: 纸白端彩度**收回到几成**（`print_math.chroma_gain_at` 的 `highlight_gain`）。
    #: `0.0` = 不施加 = 用模块默认 `PRINT_CHROMA_HIGHLIGHT_GAIN`（现产 0.80）。
    #: 为什么按片种：㉞ 量到 5219 在 0.55 上 `J` 赢 2.1% 且 `sat_p95` 只动 +0.0002，
    #: 而 5207 整条网格只动 0.3% ⇒ 与 `aim`/`white_balance` 同属"一卷一个方向"的那一类。
    print_chroma_highlight_gain: float = 0.0
    # 候选来源说明（不是控件；进入 sidecar）
    origin: str = "user"
    notes: list = field(default_factory=list)

    FIELD_NAMES = P0_FIELDS
    ORDER = RENDER_ORDER

    # ------------------------------------------------------------ 构造
    @classmethod
    def identity(cls) -> "RenderIntent":
        return cls()

    def is_identity(self) -> bool:
        """恒等 = 没有控件改动，**也没有施加自动打印尺度**。

        把 `print_anchor` 算进来是必须的：否则一个"只改了印相"的 intent 会自称
        恒等，`apply_render` 的恒等短路就把印相悄悄丢了，预览与导出随即分叉。
        `print_aim` / `white_balance` 同理：它们是两个"可以脱离锚点单独施加"的处理，
        漏进来就会重演同一个短路丢处理（生产里它们随锚点同进同出，但**契约不该靠巧合**）。
        """
        return (not self.print_anchor
                and abs(float(self.print_aim or 0.0)) < 1e-12
                and abs(float(self.white_balance or 0.0)) < 1e-12
                and abs(float(self.print_chroma_highlight_gain or 0.0)) < 1e-12
                and all(abs(float(getattr(self, k))) < 1e-12 for k in P0_FIELDS))

    def copy(self) -> "RenderIntent":
        return RenderIntent(**{k: getattr(self, k) for k in P0_FIELDS},
                            print_anchor=tuple(self.print_anchor or ()),
                            print_dmax=float(self.print_dmax or 0.0),
                            print_chroma_gain=float(self.print_chroma_gain or 0.0),
                            print_aim=float(self.print_aim or 0.0),
                            white_balance=float(self.white_balance or 0.0),
                            print_chroma_highlight_gain=float(
                                self.print_chroma_highlight_gain or 0.0),
                            origin=self.origin, notes=list(self.notes))

    def reset_field(self, name: str) -> None:
        if name not in P0_FIELDS:
            raise KeyError(f"非 P0 控件：{name!r}")
        setattr(self, name, 0.0)

    def reset_all(self) -> None:
        for k in P0_FIELDS:
            setattr(self, k, 0.0)

    def merge(self, other: "RenderIntent", fields=None) -> "RenderIntent":
        """选择性同步：只复制 fields（默认全部）到自身。"""
        for k in (fields or P0_FIELDS):
            if k in P0_FIELDS:
                setattr(self, k, float(getattr(other, k)))
        return self

    def changed_fields(self, other: "RenderIntent") -> list[str]:
        return [k for k in P0_FIELDS
                if abs(float(getattr(self, k)) - float(getattr(other, k))) > 1e-9]

    # ------------------------------------------------------------ 序列化
    def to_dict(self) -> dict:
        d = {k: float(getattr(self, k)) for k in P0_FIELDS}
        if self.print_anchor:               # 只在施加过时才写，不动旧 schema 的形状
            d["print_anchor"] = [float(v) for v in self.print_anchor]
            d["print_dmax"] = float(self.print_dmax)
            if float(self.print_chroma_gain or 0.0) > 0:
                d["print_chroma_gain"] = float(self.print_chroma_gain)
            if float(self.print_aim or 0.0) > 0:
                d["print_aim"] = float(self.print_aim)
            if float(self.white_balance or 0.0) > 0:
                d["white_balance"] = float(self.white_balance)
            if float(self.print_chroma_highlight_gain or 0.0) > 0:
                d["print_chroma_highlight_gain"] = float(self.print_chroma_highlight_gain)
        d["origin"] = self.origin
        d["notes"] = list(self.notes)
        return d

    @classmethod
    def from_dict(cls, d: dict | None) -> "RenderIntent":
        it = cls()
        if not d:
            return it
        for k in P0_FIELDS:
            if k in d and d[k] is not None:
                lo, hi, _ = RANGES[k]
                setattr(it, k, float(np.clip(float(d[k]), lo, hi)))
        anchor = d.get("print_anchor") or ()
        if isinstance(anchor, (list, tuple)) and len(anchor) in (2, 3):
            # 2 = 两点退化印相（三点曲线解不出时），3 = 正常三点
            try:
                it.print_anchor = tuple(float(v) for v in anchor)
                it.print_dmax = float(d.get("print_dmax") or 0.0)
                it.print_chroma_gain = float(d.get("print_chroma_gain") or 0.0)
                it.print_aim = float(d.get("print_aim") or 0.0)
                it.white_balance = float(d.get("white_balance") or 0.0)
                it.print_chroma_highlight_gain = float(
                    d.get("print_chroma_highlight_gain") or 0.0)
            except (TypeError, ValueError):
                it.print_anchor = ()        # 坏数据：退回"不施加"，不猜
        it.origin = str(d.get("origin", "user"))
        it.notes = list(d.get("notes") or [])
        return it

    @classmethod
    def from_legacy_parameters(cls, d: dict) -> "RenderIntent":
        """旧 sidecar 技术字段的一次性**兼容读取**（v5 §7.2；主界面不再写）。"""
        gamma = float(d.get("gamma", 1.0) or 1.0)
        exposure = float(d.get("exposure", 1.0) or 1.0)
        hl = float(d.get("highlight_strength", 0.0) or 0.0)
        it = cls()
        if gamma > 0 and abs(gamma - 1.0) > 1e-9:
            it.contrast = float(np.clip((gamma - 1.0) * 50.0, -100, 100))
        if exposure > 0 and abs(exposure - 1.0) > 1e-9:
            it.exposure = float(np.clip(math.log2(exposure), -5, 5))
        if hl:
            it.highlights = float(np.clip(-hl * 100.0, -100, 100))
        it.origin = "legacy"
        it.notes = [f"由旧技术参数兼容映射（gamma={gamma}, exposure={exposure}, "
                    f"highlight_strength={hl}）"]
        return it

    # ------------------------------------------------------------ 色温/色调
    def adaptation_gains(self) -> np.ndarray:
        """(3,) 线性 sRGB 3x3 适应矩阵（von Kries/Bradford）。"""
        m = adaptation_matrix(self.temperature, self.tint)
        return m

    def chromatic_gain_xy(self) -> tuple[float, float]:
        return white_xy(self.temperature, self.tint)


# ------------------------------------------------------- 色温/色调数学
#
# 白点轨迹用 CIE 日光轨迹（D-locus）：它在 6500K 处恰好经过 D65，
# 因此色温控件回到 0 时连续地趋于恒等（不产生跳变）；向下延伸到 2500K
# 时与普朗克轨迹几乎重合（实测偏差 <0.01），向上到 25000K 单调变冷。
def cct_from_temperature(temp: float) -> float:
    """归一化色温控件 [-100,100] -> CCT(K)。0=6500K，>0 更暖（低色温）。"""
    t = float(np.clip(temp, -100, 100)) / 100.0
    if t >= 0:
        return 6500.0 * (2500.0 / 6500.0) ** t       # 6500 -> 2500K（暖）
    return 6500.0 * (25000.0 / 6500.0) ** (-t)       # 6500 -> 25000K（冷）


def daylight_xy(cct: float) -> tuple[float, float]:
    """CCT -> CIE 日光轨迹 xy（2500-25000K，单调、连续）。"""
    T = float(np.clip(cct, 2500.0, 25000.0))
    k3, k6, k9 = 1e3 / T, 1e6 / T ** 2, 1e9 / T ** 3
    if T <= 7000.0:
        x = 0.244063 + 0.09911 * k3 + 2.9678 * k6 - 4.6048 * k9
    else:
        x = 0.237040 + 0.24748 * k3 + 1.9018 * k6 - 2.0064 * k9
    y = -3.000 * x * x + 2.870 * x - 0.275
    return float(x), float(y)


# 兼容旧命名（研发脚本/测试可能引用）
planckian_xy = daylight_xy


def white_xy(temperature: float, tint: float) -> tuple[float, float]:
    """色温/色调控件 -> 目标白点 xy（tint>0 向绿，即 y 增大）。"""
    x, y = daylight_xy(cct_from_temperature(temperature))
    y += float(np.clip(tint, -100, 100)) * 0.0009
    return x, float(np.clip(y, 0.05, 0.60))


def xy_to_xyz(x: float, y: float) -> np.ndarray:
    y = max(float(y), 1e-6)
    return np.array([x / y, 1.0, (1.0 - x - y) / y], np.float64)


def adaptation_matrix(temperature: float, tint: float) -> np.ndarray:
    """D65 -> 目标白点的 von Kries(Bradford) 适应矩阵（线性 sRGB 域）。

    方向约定（v5 §2.2）：色温控件 >0 让画面**更暖**，色调控件 >0 **偏绿**。
    因此这里施加的是"D65 到中选的暖/绿白点"的正向增益，而不是反向校正。
    增益按 LMS 几何均值归一，色温/色调几乎不改变整体亮度。
    """
    if abs(float(temperature)) < 1e-12 and abs(float(tint)) < 1e-12:
        return np.eye(3)
    wx, wy = white_xy(temperature, tint)
    w_xyz = xy_to_xyz(wx, wy)
    dst = _BRADFORD @ w_xyz
    src = _BRADFORD @ _D65_XYZ
    if np.any(dst <= 0) or not np.isfinite(dst).all():
        return np.eye(3)
    gain = dst / src
    # 归一到几何均值 1：只做色度适应，不做亮度缩放
    gm = float(np.exp(np.mean(np.log(np.maximum(gain, 1e-12)))))
    if gm <= 0 or not np.isfinite(gm):
        return np.eye(3)
    gain = gain / gm
    return _BRADFORD_INV @ np.diag(gain) @ _BRADFORD


# ------------------------------------------------------- 影调曲线
def _bump(t: np.ndarray, center: float, width: float) -> np.ndarray:
    """[0,1] 上的平滑钟形权重（紧支撑，端点为 0）。"""
    d = (t - center) / max(width, 1e-6)
    w = np.clip(1.0 - d * d, 0.0, None)
    return w * w


def tone_curve(contrast: float = 0.0, highlights: float = 0.0,
               shadows: float = 0.0, whites: float = 0.0, blacks: float = 0.0,
               *, pivot: float = 0.18, n: int = 512,
               log_range: tuple[float, float] = (-4.0, 2.0)) -> np.ndarray:
    """返回 n 点**非递减**亮度映射（输入/输出都是 log2 亮度，同一天花板）。

    构造：斜率 = 各控件的指数修正之积（恒正）-> 累积积分 -> 端点归一
    （黑仍黑、白仍白）-> 在 pivot 处与恒等相交（v5 §7.2
    "contrast around protected mid pivot"）-> 单调性兜底。
    """
    lo, hi = log_range
    t = np.linspace(lo, hi, n)                    # log2 亮度坐标
    u = (t - lo) / (hi - lo)                      # 归一化 [0,1]
    pu = _pivot_u(pivot, log_range)
    idx = int(np.clip(round(pu * (n - 1)), 1, n - 2))
    slope = np.ones(n, np.float64)
    c = float(contrast) / 100.0
    if abs(c) > 1e-9:
        # 支点以下压缩、以上扩展（c>0 = 加对比）
        slope *= np.exp(1.35 * c * (u - pu))
    for value, center, width, amp in (
            (shadows, 0.20, 0.30, 1.00),
            (highlights, 0.80, 0.30, 1.00),
            (blacks, 0.04, 0.14, 0.90),
            (whites, 0.96, 0.14, 0.90)):
        a = float(value) / 100.0
        if abs(a) > 1e-9:
            slope *= np.exp(amp * a * _bump(u, center, width))
    slope = np.clip(slope, 1e-4, 1e4)
    y = np.concatenate([[0.0], np.cumsum((slope[1:] + slope[:-1]) * 0.5)])
    y = lo + (y - y[0]) * (hi - lo) / max(y[-1] - y[0], 1e-12)
    # pivot 保护：减去一个两端为 0、pivot 处为 1 的鼓包
    shape = np.sin(np.pi * u) ** 2
    denom = shape[idx]
    if denom > 1e-6:
        y = y - (y[idx] - t[idx]) * (shape / denom)
    # 单调性兜底（宁可产生平台，也不产生反转）
    y = np.maximum.accumulate(y)
    return np.clip(y, lo, hi)


def _pivot_u(pivot: float, log_range: tuple[float, float]) -> float:
    lo, hi = log_range
    return float(np.clip((math.log2(max(pivot, 1e-6)) - lo) / (hi - lo),
                         0.0, 1.0))


_CURVE_CACHE: dict[tuple, np.ndarray] = {}
_GRID_N = 512
_GRID_LO, _GRID_HI = -4.0, 2.0        # log2 亮度网格（0.0625x .. 4x）


def _curve_for(intent: RenderIntent) -> np.ndarray | None:
    key = (round(intent.contrast, 3), round(intent.highlights, 3),
           round(intent.shadows, 3), round(intent.whites, 3),
           round(intent.blacks, 3))
    if not any(abs(v) > 1e-9 for v in key):
        return None
    cached = _CURVE_CACHE.get(key)
    if cached is None:
        if len(_CURVE_CACHE) > 64:
            _CURVE_CACHE.clear()
        cached = tone_curve(*key, log_range=(_GRID_LO, _GRID_HI), n=_GRID_N)
        _CURVE_CACHE[key] = cached
    return cached


def _apply_tone(x: np.ndarray, curve: np.ndarray) -> np.ndarray:
    """按亮度曲线缩放 RGB（保持色度比例），严格单调。

    曲线是自变量为 log2 亮度、值域同为 log2 亮度的映射；这里只取**比值**
    乘回原始像素，因此 0 仍为 0、超出网格的亮部按最后一段斜率外推。
    """
    floor = 2.0 ** _GRID_LO
    lum = np.maximum(x @ _LUM_W, floor)
    t = np.log2(lum)
    step = (_GRID_HI - _GRID_LO) / (_GRID_N - 1)
    pos = np.clip((t - _GRID_LO) / step, 0.0, _GRID_N - 1.0001)
    i0 = pos.astype(np.int32)
    frac = (pos - i0).astype(np.float32)
    flat = i0.ravel()
    y0 = curve[flat].reshape(lum.shape)
    y1 = curve[np.minimum(flat + 1, _GRID_N - 1)].reshape(lum.shape)
    out_log = y0 + (y1 - y0) * frac
    new_lum = np.power(2.0, out_log).astype(np.float32)
    ratio = new_lum / lum.astype(np.float32)
    return (x * ratio[..., None]).astype(np.float32)


def _apply_chroma(x: np.ndarray, vibrance: float, saturation: float) -> np.ndarray:
    """彩度：vibrance 保护已饱和像素，saturation 均匀缩放。"""
    v = float(vibrance) / 100.0
    s = float(saturation) / 100.0
    if abs(v) < 1e-9 and abs(s) < 1e-9:
        return x
    lum = x @ _LUM_W
    mx = x.max(axis=2)
    mn = x.min(axis=2)
    sat = (mx - mn) / np.maximum(x.mean(axis=2), 1e-6)
    gain = (1.0 + s) * (1.0 + v * np.clip(1.0 - np.clip(sat, 0, 1) * 1.2, 0, 1))
    gain = np.maximum(gain.astype(np.float32), 0.0)
    out = lum[..., None] + (x - lum[..., None]) * gain[..., None]
    return out.astype(np.float32)


def apply_render(technical: np.ndarray, intent: RenderIntent | None = None
                 ) -> np.ndarray:
    """(H,W,3) 线性 0-1 技术正片 -> 渲染正片（线性 0-1）。

    恒等 intent 时逐比特返回输入（v4/v5 共同验收：预览与导出一致）。
    """
    x = np.asarray(technical, np.float32)
    it = intent or RenderIntent()
    if it.is_identity():
        return np.clip(x, 0.0, 1.0).astype(np.float32, copy=True)
    # 0) 自动打印尺度（B-11）：先"印相"，再叠加用户控件。
    # 保色度（逐像素统一增益），上限是相纸最大反射率；分块执行，
    # 因为这里是**全分辨率**整帧（未分块实测把全量测试打成 4/4 段错误，
    # 见 docs/blockers.md B-11 追加⑦）。
    if it.print_anchor and it.print_dmax > 0:
        from ..print_math import print_remap_array
        anchor = tuple(float(v) for v in it.print_anchor)
        remapped, applied = print_remap_array(
            x, anchor, it.print_dmax,
            chroma_gain=(float(it.print_chroma_gain)
                         if float(it.print_chroma_gain or 0.0) > 0 else None),
            aim=(float(it.print_aim)
                 if float(it.print_aim or 0.0) > 0 else None),
            highlight_gain=(float(it.print_chroma_highlight_gain)
                            if float(it.print_chroma_highlight_gain or 0.0) > 0
                            else None))
        if applied is not None:
            x = remapped                    # 锚点退化则不动，控件照旧生效
    # 1) 色温/色调（色适应）
    m = adaptation_matrix(it.temperature, it.tint)
    if not np.array_equal(m, np.eye(3)):
        x = (x.astype(np.float64) @ m.T).astype(np.float32)
    # 2) 曝光（线性乘子）
    if abs(it.exposure) > 1e-12:
        x = x * np.float32(2.0 ** float(it.exposure))
    # 3-6) 对比/高光/阴影/白色/黑色：单一单调亮度曲线
    curve = _curve_for(it)
    if curve is not None:
        x = _apply_tone(x, curve)
    # 7-8) 自然饱和度 / 饱和度
    x = _apply_chroma(x, it.vibrance, it.saturation)
    # 8.5) 高光带白点蓝黄校正（B-16 追加㉚）：方向逐帧由本帧自己解，强度按片种给。
    # 放在彩度之后、钳位之前——它改的是"白该不该带蓝"，不该被前面的口味系数决定。
    if abs(float(it.white_balance or 0.0)) > 1e-12:
        from ..print_math import PRINT_WHITE
        from ..white_balance import apply_white_balance
        x, _wb = apply_white_balance(x, float(it.white_balance), PRINT_WHITE)
    # 9) gamut/finite guard
    return np.nan_to_num(np.clip(x, 0.0, 1.0), nan=0.0, posinf=1.0,
                         neginf=0.0).astype(np.float32)


def render_clip_ratios(rendered: np.ndarray, tol: float = 1e-6) -> dict:
    """渲染后剪裁（用户微调造成）——单独提示，不改写技术门禁（v5 §8.2）。"""
    x = np.asarray(rendered, np.float32)
    lo = (x <= tol).mean(axis=tuple(range(x.ndim - 1)))
    hi = (x >= 1.0 - tol).mean(axis=tuple(range(x.ndim - 1)))
    return {"low": [float(v) for v in lo], "high": [float(v) for v in hi]}


def clamp_candidate_intent(intent: RenderIntent) -> tuple[RenderIntent, bool]:
    """候选修正的硬上限（v5 §6.6）。返回 (受限 intent, 是否触发上限)。"""
    capped = False
    out = intent.copy()
    for name, cap in (("temperature", CANDIDATE_TEMP_CAP), ("tint", CANDIDATE_TINT_CAP)):
        v = float(getattr(out, name))
        if abs(v) > cap:
            setattr(out, name, math.copysign(cap, v))
            capped = True
    out.origin = "candidate"
    if capped:
        out.notes = list(out.notes) + [
            f"自动色温/色调修正已截断到硬上限（±{CANDIDATE_TEMP_CAP}/±{CANDIDATE_TINT_CAP}）"]
    return out, capped
