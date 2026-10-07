"""自动打印曝光（B-11）：把一帧的几何平均亮度放到相纸 aim 上。

**为什么需要**：`pipeline/params.py:18` 把密度跨度写成常数 `dmax = 2.0`，
而扣掉橙色片基后画面内容实测只占 0.4–1.0 decade，
自动出片因此整体偏亮、阴影抬到显示域 0.71（用户手工成品 0.35）。

**为什么不放在技术层**：后端 `paper = (1+black) − 10^q` 无法同时满足
"黑点锚在画面 p5" 与 "中间调印到 aim" 两个约束——两式联立把曲线钉死，
实测要么把暗部整段压碎（p05→0.03、饱和虚高到 0.95），
要么高光只到 0.27。外部实现因此一律把
**端点 / 中灰 / 斜率** 做成三级（horizon `measureEnds` + `autoGamma`；
NegPy 端点 + ISO-R 级数 + 纸级 dMin/dMax）。
本模块只补**中灰那一级**，且落在美学层 `RenderIntent.exposure`，
技术正片逐比特不变。

**常数不是拟合出来的**：`PRINT_AIM = 0.18` 是相纸中灰反射率，
horizon-film-lab `Paper.printAim` 与 filmeon 的 `log10(0.18)` 取同一个值；
EV 由闭式解出。跨卷跨工艺实测（2026-10-07）：
1298_G200(calibration) 达成均值 0.519，
1299_G200(holdout) Δ+0.013、1300_CN400 Δ+0.005、
ECN2_5207 Δ−0.020、ECN2_5219 Δ+0.005 —— **掉得回去**。
"""

from __future__ import annotations

import math

import numpy as np

#: 相纸中灰 aim（线性反射率）。见模块 docstring：外部两家同值，非对样片拟合。
PRINT_AIM = 0.18
#: `RenderIntent.exposure` 的合法范围（`library/render.py:39`）。
EV_LIMIT = 5.0
#: 全黑/无效输入时的保护：亮度低于此值认为"没有可锚的内容"，不动曝光。
_MIN_ANCHOR_LUMA = 1e-5


def _luma(x: np.ndarray) -> np.ndarray:
    return 0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]


def aim_exposure_ev(positive_linear, *, aim: float = PRINT_AIM) -> float:
    """返回把**几何平均亮度**放到 `aim` 所需的 EV 档。

    `log10(gm·2^EV) = log10(aim)` ⇒ `EV = (log10 aim − mean log10 luma)/log10 2`。
    用几何平均而不是算式平均：曝光档是乘性的，算术平均会被高光主导，
    一卷里高调帧会把暗帧一起推亮（实测正是原缺陷的方向）。
    """
    x = np.asarray(positive_linear, dtype=np.float64)
    if x.ndim != 3 or x.shape[-1] < 3:
        return 0.0
    lum = _luma(x[..., :3])
    lum = lum[np.isfinite(lum)]
    if lum.size == 0 or float(lum.max()) <= _MIN_ANCHOR_LUMA:
        return 0.0        # 只有黑/只有无效值：没有可锚的内容，不动曝光
    lum = np.clip(lum, _MIN_ANCHOR_LUMA, None)
    g = float(np.mean(np.log10(lum)))
    if not math.isfinite(g):
        return 0.0
    ev = (math.log10(max(1e-6, float(aim))) - g) / math.log10(2.0)
    if not math.isfinite(ev):
        return 0.0
    return float(np.clip(ev, -EV_LIMIT, EV_LIMIT))


def apply_print_exposure(candidates, positive_linear,
                         *, aim: float = PRINT_AIM) -> float:
    """把 EV 折进**每个候选的起点 intent**，返回实际采用的 EV。

    必须写进候选本身：预览、`candidate_positive()`、sidecar 都从
    `candidate.intent` 取值，只改调用点的局部 intent 就会出现
    "屏幕上看到的"和"导出落盘的"不是一回事。
    三个候选加的是**同一个** EV，所以候选之间的相对关系不变
    （`models.py:317`：候选只在色温/色调与受限影调起点上不同）。
    """
    if candidates is None:
        return 0.0
    ev = aim_exposure_ev(positive_linear, aim=aim)
    if abs(ev) < 1e-9:
        return 0.0
    note = f"自动打印曝光 {ev:+.2f} EV（相纸 aim {aim:.2f}）"
    for cand in candidates:
        intent = dict(cand.intent or {})
        base = float(intent.get("exposure", 0.0) or 0.0)
        moved = float(np.clip(base + ev, -EV_LIMIT, EV_LIMIT))
        intent["exposure"] = moved
        cand.intent = intent
        cand.rationale = f"{cand.rationale}；{note}" if cand.rationale else note
    return ev
