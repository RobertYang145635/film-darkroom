"""单点中性救援（去色罩算法研究 R2；痛点调研 #2：自动失败时"点一下灰就能救"）。

研究结论（为什么不是"拾点增益 -> gain_to_temp_tint 投影"）：
- 拾点给出的物理量是线性域对角增益 g_c = gray/point_c；
- 但 (temperature, tint) 控件对应的适应矩阵是 Bradford 域 3x3（RGB 域
  非对角），把 RGB 对角增益投影到该二参数族会**大幅衰减修正**
  （合成实验：蓝偏 0.18 极差只消 22%——投影路线不可用）；
- 正确做法：在 (temp,tint) 族内**直接数值求解**"施加适应矩阵后拾点
  最接近中性"。同族内这是精确解，2D 粗到细网格搜索 <10ms。

两个入口：
- ``neutral_pick_gain``：拾点补丁中位数 + 对角增益 + 可信度警告
  （诊断展示用；对角增益无法被 (temp,tint) 族精确表达，施加请走 delta）；
- ``neutral_pick_delta``：渲染正片 + 拾点 -> 使拾点中性的
  (d_temp, d_tint) 增量。当前 intent 只有色温/色调时，适应矩阵在 LMS
  域是对角乘积可交换，增量直接加到当前控件上（近似条件见 docstring）。

契约：
- **风格只在 RenderIntent**——绝不写 InvertParams（docs/algorithm-notes.md；
  技术正片保持逐比特不变）；施加路径 = 同一个 apply_render（与滑杆/
  候选完全一致，无第二套公式）；
- 与 auto.py 的 z 域中性参考（estimate_exponent_ratios 等，作用于技术
  参数、需要负片上的亮/暗两点）互补不替代：那边是"技术底座标定"，
  这里是"用户可见结果的即时救援"。
"""

from __future__ import annotations

import numpy as np

#: 拾点补丁半径：(2*PICK_RADIUS+1)² 补丁中位数（抗单像素噪声/尘点）
PICK_RADIUS = 3
#: 增益上限：超过即视为拾点不可信（死黑/死白）
GAIN_MAX = 8.0
#: 补丁亮度有效下限/上限（线性域）
PICK_LUM_MIN = 0.02
PICK_LUM_MAX = 0.95

EPS = 1e-6


def _slice_patch(img: np.ndarray, y: int, x: int, radius: int) -> np.ndarray:
    """拾点 (H',W',3) 子补丁：**四边一律钳制**，不依赖 numpy 负索引语义。

    审查修复（2026-09-08）：此前写成 `arr[y-r:y+r+1]`，`y<r` 时负索引按
    「从尾部计数」解释 —— 大图上得到空切片（合法坐标却抛 ValueError），
    小图（H < 2r）上更是**静默取到图像另一侧的像素**（实测 4×600 图取
    y=1 返回第 2/3 行的值）。正边界 numpy 会自动钳制、负边界不会，这个
    不对称就是漏网根因。`neutral_pick_intent_exact` 曾各自钳制过一次，
    现在两条路径共用本函数，杜绝「一处对一处错」。
    """
    arr = np.asarray(img, np.float32)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"需要 (H,W,3) 图像，得到 {arr.shape}")
    h, w = arr.shape[:2]
    if not (0 <= int(y) < h and 0 <= int(x) < w):
        raise ValueError(f"拾点 ({y},{x}) 越界 (H={h},W={w})")
    r = max(0, int(radius))
    y0, x0 = max(0, int(y) - r), max(0, int(x) - r)
    y1, x1 = min(h, int(y) + r + 1), min(w, int(x) + r + 1)
    return arr[y0:y1, x0:x1]


def _patch_median(img: np.ndarray, y: int, x: int, radius: int) -> np.ndarray:
    patch = _slice_patch(img, y, x, radius).reshape(-1, 3)
    patch = patch[np.isfinite(patch).all(axis=1)]
    if patch.shape[0] == 0:
        raise ValueError("拾点补丁内没有有限像素")
    return np.maximum(np.median(patch, axis=0).astype(np.float64), EPS)


def neutral_pick_gain(img: np.ndarray, y: int, x: int, *,
                      radius: int = PICK_RADIUS,
                      return_warnings: bool = False):
    """拾点 -> 把该点拉向中性的线性域对角增益 (3,) + 可信度警告。

    gray = 补丁透射率几何均值（保持整体亮度，与 render.py 的适应增益
    归一策略一致）；g_c = gray / point_c。诊断/展示用。
    """
    pt = _patch_median(img, y, x, radius)
    warnings: list[str] = []
    lum = float(pt.mean())
    if lum < PICK_LUM_MIN:
        warnings.append(f"拾点过暗（线性亮度 {lum:.4f} < {PICK_LUM_MIN}）："
                        "增益不可信，请点在画面里应该呈现中性灰的区域")
    elif lum > PICK_LUM_MAX:
        warnings.append(f"拾点过亮（线性亮度 {lum:.4f} > {PICK_LUM_MAX}）："
                        "疑似死白/剪裁，增益不可信")
    gray = float(np.exp(np.mean(np.log(pt))))
    gain = gray / pt
    gm = float(np.exp(np.mean(np.log(np.maximum(gain, EPS)))))
    gain = gain / max(gm, EPS)
    if float(np.max(gain)) > GAIN_MAX:
        warnings.append(f"拾点色偏过大（增益 {float(np.max(gain)):.1f} > "
                        f"{GAIN_MAX}）：已钳制，请确认拾点不是死黑/强色区域")
        gain = np.minimum(gain, GAIN_MAX)
        gm = float(np.exp(np.mean(np.log(gain))))
        gain = gain / max(gm, EPS)
    gain = np.clip(gain, 1.0 / GAIN_MAX, GAIN_MAX)
    if return_warnings:
        return gain.astype(np.float32), warnings
    return gain.astype(np.float32)


def _spread_after(c: np.ndarray, temp: float, tint: float) -> float:
    """施加适应矩阵后拾点 RGB 的极差（越小越中性）。"""
    from .render import adaptation_matrix

    out = adaptation_matrix(temp, tint) @ c
    return float(out.max() - out.min())


def neutral_pick_delta(img: np.ndarray, y: int, x: int, *,
                       radius: int = PICK_RADIUS,
                       temp_range: tuple[float, float] = (-100.0, 100.0),
                       tint_range: tuple[float, float] = (-100.0, 100.0),
                       return_warnings: bool = False):
    """渲染正片 + 拾点 -> 使拾点中性的 (d_temp, d_tint) 增量。

    在控件族内直接数值求解（粗到细 2D 网格），目标 =
    adaptation_matrix(temp, tint) @ 拾点中位数的 RGB 极差最小化。

    组合语义：当前 intent 只有色温/色调时（救援的典型起点），适应矩阵
    在 LMS 域为对角乘积、可交换，M(delta)·M(base) ≈ M(base+delta)——
    增量直接加到当前控件值即可。当前 intent 还有影调/彩度控件时近似
    仍成立但误差更大（如实记录，后续研究）。
    """
    pt = _patch_median(img, y, x, radius)
    warnings: list[str] = []
    lum = float(pt.mean())
    if lum < PICK_LUM_MIN or lum > PICK_LUM_MAX:
        warnings.append(
            f"拾点线性亮度 {lum:.4f} 超出可信区间 [{PICK_LUM_MIN}, "
            f"{PICK_LUM_MAX}]：结果可能不可靠，请点在应呈中性灰的区域")
    # 拾点可信度：把该点拉平所需的对角增益。真实"该是灰"的点经处理后
    # 通道比通常在 2-3 倍内；所需增益过大说明拾点大概率不是中性场景点
    # （比如点在了 R 通道极暗的树叶/深水上）——救援只能部分收敛。
    gray = float(np.exp(np.mean(np.log(pt))))
    need = float(np.max(gray / pt) if np.all(pt > 0) else float("inf"))
    if need > 3.0:
        warnings.append(
            f"拾点通道比失衡（所需对角增益 {need:.1f}x > 3.0）：该点可能"
            "不是中性场景点，救援只能部分收敛；请改点画面中更接近灰的区域")

    def solve(t_lo, t_hi, n_lo, n_hi, steps):
        best = (float("inf"), 0.0, 0.0)
        for t in np.linspace(t_lo, t_hi, steps[0]):
            for n in np.linspace(n_lo, n_hi, steps[1]):
                s = _spread_after(pt, float(t), float(n))
                if s < best[0]:
                    best = (s, float(t), float(n))
        return best

    # 粗到细：全控件域步长 5 -> 步长 1 -> 步长 0.2
    _, bt, bn = solve(temp_range[0], temp_range[1],
                      tint_range[0], tint_range[1], (41, 41))
    for step, n in ((5.0, 11), (1.0, 11), (0.2, 9)):
        _, bt, bn = solve(bt - step * (n // 2), bt + step * (n // 2),
                          bn - step * (n // 2), bn + step * (n // 2), (n, n))
    # 恒等优先：极差本来就不小多少（已接近中性）时不硬给修正
    if _spread_after(pt, 0.0, 0.0) <= _spread_after(pt, bt, bn) + 1e-9:
        bt, bn = 0.0, 0.0
    d_temp = float(np.clip(bt, -100, 100))
    d_tint = float(np.clip(bn, -100, 100))
    if return_warnings:
        return d_temp, d_tint, warnings
    return d_temp, d_tint


def neutral_pick_intent(img: np.ndarray, y: int, x: int, *,
                        radius: int = PICK_RADIUS,
                        base_temperature: float = 0.0,
                        base_tint: float = 0.0):
    """拾点 -> RenderIntent(temperature, tint)（救援的完整入口）。

    base_* 为当前已施加的色温/色调（增量直接叠加，见
    ``neutral_pick_delta`` 的组合语义）；origin/notes 如实标记来源。
    """
    from .render import RenderIntent

    d_temp, d_tint = neutral_pick_delta(img, y, x, radius=radius)
    temp = float(np.clip(base_temperature + d_temp, -100, 100))
    tint = float(np.clip(base_tint + d_tint, -100, 100))
    return RenderIntent(temperature=temp, tint=tint, origin="neutral-pick",
                        notes=["单点中性救援：拾点 -> 控件族内精确求解"
                               f"（增量 {d_temp:+.1f}/{d_tint:+.1f}）"])


# ---------------------------------------------------------------- 精确求解
# R5a：含影调/彩度控件时的精确救援。
#
# 增量叠加 M(delta)·M(base) ≈ M(base+delta) 只在"当前 intent 仅色温/色调"
# 时精确；影调曲线作用于亮度、彩度作用于色度比例，与色适应不可交换。
# 正确做法：对**技术正片补丁**用完整 apply_render 逐点渲染——影调/彩度
# 对固定控件是常量变换，目标函数只随 (temp,tint) 变化，网格求解结果
# 就是"整套 intent 下拾点最中性"的精确解。补丁小（(2r+1)²），渲染
# 代价可忽略；粗到细三层网格 ~800 次小图渲染 <1s。

def neutral_pick_delta_render(tech_patch: np.ndarray, *,
                              base_temperature: float = 0.0,
                              base_tint: float = 0.0,
                              fixed_fields: dict | None = None,
                              temp_range: tuple[float, float] = (-100.0, 100.0),
                              tint_range: tuple[float, float] = (-100.0, 100.0),
                              return_warnings: bool = False):
    """技术正片补丁 -> 使拾点（完整渲染后）最中性的绝对 (temp, tint)。

    fixed_fields 为 base intent 中除色温/色调外的全部控件（对比/高光/
    阴影/白色/黑色/曝光/自然饱和度/饱和度）——求解时保持不变。
    目标 = 渲染后补丁中位数的 RGB 极差；恒等优先（已中性不硬给修正）。
    可信度警告基于**当前渲染结果**（用户看到的样子）：剪裁通道 /
    所需对角增益 >3x。
    """
    from .render import RenderIntent, apply_render

    patch = np.asarray(tech_patch, np.float32)
    if patch.ndim != 3 or patch.shape[2] != 3 or patch.shape[0] == 0 \
            or patch.shape[1] == 0:
        raise ValueError(f"需要非空 (H,W,3) 技术正片补丁，得到 {patch.shape}")
    fixed = {k: float(v) for k, v in (fixed_fields or {}).items()}

    def rendered_median(temp: float, tint: float) -> np.ndarray:
        it = RenderIntent(temperature=float(temp), tint=float(tint), **fixed)
        out = apply_render(patch, it)
        return np.median(out.reshape(-1, 3), axis=0)

    def spread(temp: float, tint: float) -> float:
        m = rendered_median(temp, tint)
        return float(m.max() - m.min())

    cur = rendered_median(base_temperature, base_tint)
    warnings: list[str] = []
    if float(cur.max()) >= 0.98 or float(cur.min()) <= 0.02:
        warnings.append("拾点已被剪裁（接近纯黑/纯白）：救援无法恢复已丢失"
                        "的信息，请改点中间调的应中性区域")
    pos = np.maximum(cur, EPS)
    need = float(np.max(np.exp(np.mean(np.log(pos))) / pos))
    if need > 3.0:
        warnings.append(f"拾点通道比失衡（所需对角增益 {need:.1f}x > 3.0）："
                        "该点可能不是中性场景点，救援只能部分收敛")

    def solve(t_lo, t_hi, n_lo, n_hi, steps):
        best = (float("inf"), base_temperature, base_tint)
        for t in np.linspace(t_lo, t_hi, steps[0]):
            for n in np.linspace(n_lo, n_hi, steps[1]):
                s = spread(float(t), float(n))
                if s < best[0]:
                    best = (s, float(t), float(n))
        return best

    _, bt, bn = solve(temp_range[0], temp_range[1],
                      tint_range[0], tint_range[1], (21, 21))
    for step, n in ((4.0, 9), (1.0, 7), (0.25, 7)):
        _, bt, bn = solve(bt - step * (n // 2), bt + step * (n // 2),
                          bn - step * (n // 2), bn + step * (n // 2), (n, n))
    bt = float(np.clip(bt, *temp_range))
    bn = float(np.clip(bn, *tint_range))
    # 恒等优先：当前已足够中性（或求解无改进）时保持 base 值
    sp_base = spread(base_temperature, base_tint)
    sp_best = spread(bt, bn)
    if sp_base <= sp_best + 1e-9:
        bt, bn = float(base_temperature), float(base_tint)
        sp_best = sp_base
    # 收敛性如实报告（审查修复 2026-09-08）：此前只报「拾点本身可信度」，
    # 求解顶到控件边界、或极差几乎没消掉时**零警告**——实测偏红点给出
    # d_temp=-100（拉满）残差仍有 68%，UI 却显示「已把该点拉向中性灰」。
    # 控件族只有二自由度，本就只能部分校正，但必须让用户知道没救动。
    if sp_base > 1e-6 and sp_best / sp_base > 0.5:
        warnings.append(
            f"色温/色调未能拉平该点（RGB 极差 {sp_base:.3f} → {sp_best:.3f}，"
            f"仅消除 {(1 - sp_best / sp_base) * 100:.0f}%）：该点大概率不是中性"
            "场景点，或色偏超出二自由度控件族的表达能力")
    if (bt <= temp_range[0] + 1e-9 or bt >= temp_range[1] - 1e-9
            or bn <= tint_range[0] + 1e-9 or bn >= tint_range[1] - 1e-9):
        warnings.append("求解已顶到色温/色调控件边界仍未收敛：继续加大幅度也不会"
                        "更接近中性，请改点画面中更接近灰的区域")
    if return_warnings:
        return bt, bn, warnings
    return bt, bn


def neutral_pick_intent_exact(tech: np.ndarray, y: int, x: int, *,
                              radius: int = PICK_RADIUS,
                              base_intent=None, return_warnings: bool = False):
    """技术正片 + 拾点 -> 完整 RenderIntent（含影调/彩度时的精确救援）。

    与 ``neutral_pick_intent``（渲染后图 + 增量）不同：本入口接收**技术
    正片**，对拾点补丁用完整 apply_render 逐点渲染求解——影调/彩度控件
    与色适应不可交换，绝对求解吸收 base 已有的全部控件。补丁内部截取，
    求解代价 <1s（粗到细三层网格 × 小补丁渲染）。
    """
    from .render import RenderIntent

    arr = np.asarray(tech, np.float32)
    base = base_intent or RenderIntent()
    patch = _slice_patch(arr, y, x, radius)   # 四边钳制，与 _patch_median 同一把尺子
    if not np.isfinite(patch).any():
        raise ValueError("拾点补丁内没有有限像素（NaN/Inf），无法求解")
    fixed = {k: float(getattr(base, k)) for k in RenderIntent.FIELD_NAMES
             if k not in ("temperature", "tint")}
    res = neutral_pick_delta_render(
        patch, base_temperature=float(base.temperature),
        base_tint=float(base.tint), fixed_fields=fixed,
        return_warnings=True)
    bt, bn, warns = res
    it = base.copy()
    it.temperature, it.tint = float(bt), float(bn)
    it.origin = "neutral-pick"
    # notes 会随「整卷同步」扩散到每一帧并持久化进会话/sidecar：不设上限
    # 就是无界增长（反复拾点 50 次即 50 条）。只保留最近 3 条救援注记。
    it.notes = (list(base.notes) + [
        "单点中性救援（精确：整套 intent 下对技术正片补丁渲染求解）"
        f" temp/tint = {bt:+.1f}/{bn:+.1f}"])[-3:]
    if return_warnings:
        return it, warns
    return it
