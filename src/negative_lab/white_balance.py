r"""**高光带白点蓝黄校正**（版 D，`docs/blockers.md` B-16 追加㉙/㉚）。

要治的病（㉘ 在正确量程上整卷重量出来的）：成品里"该是中性的那批像素"
（最亮 20% 里彩度最低的一半 = 白大褂、白墙、纸白端）**系统性偏蓝**：
两卷各 35 帧，带符号 `Δb* = −3.214 / −3.105`，逐帧同号 32/35。
物理出处是片基与残余染料色罩在低密度端（=正片的亮部）留下的蓝。

三个形状都被量死过，才知道约束该写成什么：
  A 全帧·单轴：5219 能过，5207 的整幅 `|a|` 破（`+0.27`）；
  B 全帧·双轴（把白的 `a*` 也拉到 0）：两卷都破整幅 `|a|`；
  C 高光带·双轴：5207 首次同时过，5219 仍破 `|a|` **+0.150**——
     而 5219 的白 `a*` 本来就合格（`+0.597`），**那条约束对它纯属多余**。
⇒ 分工写进约束里：**只管蓝黄，且不许动整幅的红绿账**（红绿归 C 层按片种斜率，
见 `library/film_types.py`）。两个方程都是**我方输出的自引用量**，他的样片只参与评分：
  F1 近中性高光的 `b*` → 0；F2 整幅平均 `a*` 保持不动。
未知量 `log gR, log gB` 两个（`G` 为参照），数值雅比可比定点解。

⚠ 两条不能破的口径：
 ① **域**：解与评都在 `encode_srgb` 之后的显示域量 `a*,b*`（线性光上的是另一个数，
    sRGB 传递函数中调提亮、高光压平）。㉗⑤ 与 ㉘ 各踩过一次域/量程 bug。
 ② **逐像素**：权重只依赖该像素自己的亮度与两个常数 ⇒ 与 `print_math.chroma_gain_at`
    同族，分块与整帧逐比特一致。
 ③ 解要读整帧的分位/均值，但它是**一次性的帧级决策**（同打印锚），
    不是逐块统计；大帧必须抽稀（`_SOLVE_CAP`），否则 45MP 会重演 B-11 追加⑦ 的段错误。
"""
from __future__ import annotations

import numpy as np

from .color.management import linear_to_srgb
from .diagnostics.colorimetry import srgb_to_lab

#: 诊断带边缘：`L* = 80` 是"近中性高光"的**绝对**下界，严格反解
#: `Y = ((L*+16)/116)^3`（`L*>8` 走线性立方根分支），再过 sRGB 传递函数得到编码亮度。
NEUTRAL_HIGHLIGHT_L_STAR = 80.0
#: 选取比例：取最亮 20%（`L*` 的 80 分位）里彩度最低的一半。
#: ⚠ 这与上面那个 80 **不是同一个量**（一个是绝对亮度档，一个是帧内分位），
#:    数值相同纯属巧合——合并成一个常数会让下一次改档把两处一起改掉。
NEUTRAL_HIGHLIGHT_PERCENTILE = 80.0
#: 解算用的最大样本像素数。验收尺子本身把图缩到 384，帧级统计不需要全分辨率；
#: 现有语料的单帧 0.9 MP 在阈值内 ⇒ **取全部像素**，与扫描时的读数逐比特同源。
_SOLVE_CAP = 1_000_000
_LUM = np.array([0.2126, 0.7152, 0.0722], np.float32)
_EPS = 1e-3
_ITERS = 9
_TOL = 1e-3


def encode(rgb_linear: np.ndarray) -> np.ndarray:
    """线性光 → 显示域（与导出同一份传递函数，含 [0,1] 钳制）。"""
    return linear_to_srgb(np.clip(rgb_linear, 0.0, 1.0))


def highlight_luma_start(print_white: float) -> float:
    """权重起点：`L* = NEUTRAL_HIGHLIGHT_L_STAR` 对应的编码亮度。"""
    y = ((float(NEUTRAL_HIGHLIGHT_L_STAR) + 16.0) / 116.0) ** 3
    return float(encode(np.array([y], np.float64))[0])


def highlight_weight(positive_linear: np.ndarray, print_white: float) -> np.ndarray:
    """逐像素高光权重：编码亮度从 `highlight_luma_start()` 到纸白线性 0→1。

    起点=诊断带边缘，终点=生产常数 `PRINT_WHITE` 的编码值 ⇒ **不引入新的自由参数**，
    带的上下沿都由已有的量决定。
    """
    x = np.asarray(positive_linear, np.float32)
    disp = encode(x[..., :3])
    y = np.clip(disp @ _LUM, 0.0, 1.0)
    lo = np.float32(highlight_luma_start(print_white))
    hi = np.float32(float(encode(np.array([print_white], np.float64))[0]))
    return np.clip((y - lo) / max(float(hi) - float(lo), 1e-6), 0.0, 1.0)


def _lab(disp: np.ndarray) -> np.ndarray:
    return srgb_to_lab(disp).reshape(*disp.shape[:2], 3)


def neutral_highlight_mask(lab: np.ndarray) -> np.ndarray:
    """近中性高光：最亮 20% 里彩度最低的一半（与扫描用的诊断同一个定义）。"""
    l_star = lab[..., 0]
    m = l_star >= np.percentile(l_star, NEUTRAL_HIGHLIGHT_PERCENTILE)
    if int(m.sum()) < 64:
        return np.zeros(l_star.shape, dtype=bool)
    cab = np.hypot(lab[..., 1], lab[..., 2])
    return m & (cab <= np.percentile(cab[m], 50))


def _subsample(x: np.ndarray) -> tuple[np.ndarray, int]:
    """定步长抽稀（位置固定 ⇒ 与像素顺序无关，也不改变小帧的行为）。"""
    n = int(x.shape[0]) * int(x.shape[1])
    step = max(1, int(np.ceil(n / float(_SOLVE_CAP))))
    return (x if step == 1 else x[::step, ::step]), step


def white_gain_direction(positive_linear: np.ndarray,
                         print_white: float) -> tuple[float, float]:
    """解 `(log gR, log gB)` 使 F1=0 ∧ F2=0。白已经中性 / 无高光的帧解出 ≈ `(0, 0)`。"""
    x = np.asarray(positive_linear, np.float32)
    small, _step = _subsample(x)
    w = highlight_weight(small, print_white)
    lab0 = _lab(encode(small[..., :3]))
    a0 = float(lab0[..., 1].mean())

    def f(lg: np.ndarray) -> np.ndarray:
        g = np.exp(np.clip(w[..., None] * np.array([lg[0], 0.0, lg[1]], np.float32),
                           -0.5, 0.5))
        lab = _lab(encode(np.clip(small * g, 0, 1))[..., :3])
        sel = neutral_highlight_mask(lab)
        b_err = 0.0 if not sel.any() else float(lab[..., 2][sel].mean())
        return np.array([b_err, float(lab[..., 1].mean()) - a0], float)

    lg = np.zeros(2, float)
    for _ in range(_ITERS):
        r = f(lg)
        if float(np.hypot(*r)) < _TOL:
            break
        j = np.empty((2, 2), float)
        for col in range(2):
            p = lg.copy()
            p[col] += _EPS
            j[:, col] = (f(p) - r) / _EPS
        try:
            d = np.linalg.solve(j, r)
        except np.linalg.LinAlgError:
            break
        if not bool(np.all(np.isfinite(d))):
            break
        lg = np.clip(lg - np.clip(d, -0.05, 0.05), -0.35, 0.35)
    return float(lg[0]), float(lg[1])


def apply_white_balance(positive_linear: np.ndarray, strength: float,
                        print_white: float,
                        direction: tuple[float, float] | None = None,
                        max_chunk_pixels: int = 4_000_000) -> tuple[np.ndarray, dict]:
    """按**片种强度** `strength` 把解打折后打到高光带上；`strength == 0` 原样返回。

    `direction` 允许把已解好的 `(log gR, log gB)` 传进来：解要读整帧统计，而应用是
    逐像素的 ⇒ 分块执行时**只许解一次**（`tests/unit/test_white_balance.py` 用这个
    参数钉住"分块与整帧逐比特一致"）。不给就现场解。

    **应用按行分块**（与 `print_remap_array` 同一个理由与同一个阈值）：
    整帧一次性做会同时握住"编码图 + 权重 + 增益 + 结果"四份 45MP×3 的浮点数组，
    正是 B-11 追加⑦ 把全量测试打成段错误的那种形状。分块不改变结果
    （每一步都只看该像素自己），由同型锁钉住。

    返回 `(新线性正片, 记录)`。记录里给方向与权重面积——
    只写"校了多少"不够，下一轮要能判断"这一帧到底有没有白可校"。
    """
    x = np.asarray(positive_linear, np.float32)
    s = float(strength or 0.0)
    if s == 0.0:
        return x, {"applied": False}
    lg = white_gain_direction(x, print_white) if direction is None else (
        float(direction[0]), float(direction[1]))
    if abs(lg[0]) < 1e-4 and abs(lg[1]) < 1e-4:
        return x, {"applied": False, "log_g": list(lg), "reason": "white_already_neutral"}
    out = np.empty_like(x)
    rows = max(1, int(max_chunk_pixels) // max(int(x.shape[1]), 1))
    w_sum, n_pix = 0.0, 0
    for r0 in range(0, int(x.shape[0]), rows):
        blk = x[r0:r0 + rows]
        w = highlight_weight(blk, print_white)
        g = np.exp(np.clip(w[..., None] * np.array([s * lg[0], 0.0, s * lg[1]], np.float32),
                           -0.5, 0.5)).astype(np.float32)
        out[r0:r0 + rows] = np.clip(blk * g, 0.0, 1.0)
        w_sum += float(w.sum())          # 记录只交"和与个数"，不交比例（同 print_math 口径）
        n_pix += int(w.size)
    return out, {"applied": True, "strength": s, "log_g": list(lg),
                 "weight_sum": w_sum, "weight_pixels": n_pix,
                 "weight_mean": (w_sum / n_pix) if n_pix else 0.0}
