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
#:
#: **`P_LOW` 由 5.0 改成 2.0**（2026-10-09 实测，`outputs/ref_look_probe/
#: black_sweep.json`，11 帧用户手工参考、1600px 生产分辨率、用户亲手片基）：
#: 黑点定在 p5 的含义就是"让最暗 5% 印成纯黑"，而他一侧显示 `p05` 均值
#: 0.182、最低 0.02 —— **他不压黑**。扫 `P_LOW ∈ {0.25,0.5,1,2,5}` 的结果：
#: 阴影偏差 `p05` 从 −0.075(p5) 收到 **+0.007**(p2)，对比偏差 +0.017 收到 **−0.000**，
#: 四统计平均距离 0.0817 → **0.0732**；p2 是这条曲线的**内点最小值**
#: （p1=0.0742、p0.5=0.0763、p0.25=0.0779 都更差，不是往一边推到底）。
#: 逐帧不同意：4 帧要 p0.25、3 帧要 p5（`1298::G200-8` 在 p5 恰好 0.0466），
#: 所以这是**一个折中的全局分位**，不是"每帧都变好"。
#:
#: **`P_MID` 由 50.0 改成 35.0**（同一批数据，`tmp_look_layers.py` 的 B 段）：
#: 黑点修完后剩下的最大缺口是整体亮度偏暗 −0.061，而分层归因显示这一步
#: 全部由**打印层**造成（技术正片均值 0.757，打印压掉 −0.236 到 0.505，
#: 用户参考 0.566）。中灰锚点原先放在"这一帧的中位密度"，扫
#: `P_MID ∈ {20,25,30,35,40,45,50}` 得到一条 30~40 之间的**浅盆**：
#: 平均距离 0.0752 / 0.0715 / 0.0692 / **0.0685** / 0.0688 / 0.0707 / 0.0732，
#: p35 处亮度偏差 −0.061 → **−0.016**、饱和偏差 −0.037 → **−0.014**，
#: 代价是阴影偏差 +0.007 → +0.029（仍然很小）。
#: 物理上讲得通：自然画面的像素直方图偏向暗侧（天空、阴影占面积），
#: 把"主体的中灰"放在像素中位**偏亮一侧**是常规曝光实践，不是拟合出来的手感。
#: 逐帧仍然不同意，且方向可解释：变好的 6 帧正是原本偏暗的（`1299` 三帧
#: 0.0659/0.0841/0.0698 → 0.0426/0.0695/0.0489），变差的 3 帧是原本就偏亮的
#: （`1298::G200-5/+0.074`、`::G200-8/+0.060`、`::G200-11/+0.029`），
#: 另有 2 帧走两点退化锚点、与 `P_MID` 无关（数字全程不变）。
#: ⇒ **逐帧亮度误差另有来源**（候选：`dmax` 估计，见 B-13 追加① 末尾），
#:   全局分位只是在平均意义上补偿它。
P_LOW, P_MID, P_HIGH = 2.0, 35.0, 95.0
#: 跨度不足时**不套曲线**：薄负片/纯色卡帧用三点解出的曲线会近乎阶跃。
MIN_SPAN = 0.15
#: 打印锚点只看**会被看到的那块**：画面区包围盒的中心这个比例。
#: 实测依据见 `centre_region`。
PRINT_REGION_CENTRE_FRAC = 0.70
#: 电平的**绝对项**（"部分固定曲线"）——把锚点的中灰与白按本帧取样总体的
#: **密度中位**平移 `δ = strength·(d_median − reference)`，黑锚不动。
#:
#: **它补的是哪个洞**（`tmp_level_predictor_prod.py` 的"跨帧散布"表，
#: 10 帧有参考、1200px、生产整卷两遍、显示统计按 `MET["resize"]=384` 再取中心 70%）：
#: 不加这一项时我方显示 p35 帧间 std **0.0302**、亮度 std **0.0187**，
#: 而用户参考是 **0.0946** 与 **0.0902**——锚点把中灰钉成常数，
#: 帧间电平自由度等于被归一化掉了。两边的**中位几乎相同**
#: （我方 p35 0.4899 对 他 0.4829）⇒ `PRINT_AIM` 这个常数没有错，
#: 错的是"每帧都归到它"。上线后我方 p35 std 0.0302 → **0.0828**、
#: 亮度 std 0.0187 → **0.0436**（朝他那边长，但还只有一半）。
#:
#: **卷级错位是主要症状**：1298 我方比参考亮 **+0.045**、1299 我方比参考暗 **−0.098**
#: （`tmp_level_carrier.py`）。`d_p50` 与"我方亮度 − 他方亮度"的相关 **−0.595**，
#: 即"越浓（场景越亮）我方越暗" ⇒ 沿密度中位抬电平方向是对的。
#:
#: ⚠ **携带者的量法错过一次，数字留在这里当证据**：第一次扫（
#: `tmp_level_predictor_scan.py`）用的是**整幅画面掩膜**的密度，得到
#: "只有密度中位卷内也有信号（卷内去均值 r=+0.484）"；换成**生产总体**
#: （`centre_region` ∩ `picture_density`，最小样本 114912 px）重扫，
#: 密度中位的卷内 r 其实是 **−0.019**（=没有），卷内有信号的是
#: **暗部面积**（`d ≤ p05+0.10` 的占比，卷内 r **−0.534**）与高光面积带
#: （`d ≥ p99−0.20`，+0.278）。⇒ 本项**主要是把卷间电平放对**，
#: 卷内的 1298 改善（|亮度| 0.0604 → 0.0506）来自"7 帧里 4 帧是'太亮且比标称薄'"
#: 这一条被它一起压住了，不是线性携带。暗部面积那条**没上线**，是下一步的线索。
#:
#: **系数由训练卷选、留出卷只看**（`tests/regression/test_level_absolute.py`，
#: 前提校验：离线重渲与 `candidate_positive` 逐比特一致、复算漂移 0.000007）：
#: 逐帧\|亮度\| **0.0718 → 0.0541**、四统计距离 **0.0600 → 0.0566**、
#: 训练卷 1298 距离 **0.0634 → 0.0586**、留出卷 1299 \|亮度\| **0.0983 → 0.0620**、
#: 饱和偏差 −0.0243 → −0.0204；**反方向（−0.25）0.1019 比"什么都不做"更糟**
#: ⇒ 符号是承重的。变好的 7/10 帧含三个最差的帧，变差的三帧幅度 ≤0.011
#: （`::G200-1`、`::G200-5`、`::G200-9`）。`strength` 更大（0.5/0.75）在 768 口径
#: 的扫描里训练卷选不出来（1298 距离更差）且 0.5 那一格出现过 1 帧曲线非法，不上线。
#: **已知代价照实记**：抬电平让阴影雾略微恶化，Δp05 **+0.0257 → +0.0279**
#: （锁里预算 +0.02），我方 p95 中位 0.8043 对他 0.8578 的**白点偏低**这一项没动。
#:
#: 工业出处：**负片类型用一条固定密度→纸级曲线**（扫描/放印机的胶片型配置文件），
#: 测光只做**限幅**的逐帧修正（NegPy `anchor_meter_strength=0.2` 配 ±0.12 decade 限幅、
#: Noritsu 条件放大、Fuji US5223892A 一卷一次曝光）。原来我们是"每帧无限修正"
#: （p35 硬钉 AIM），那正是把绝对电平抹平的那一步。
#:
#: `reference` **不是拟合量**而是本批 G200 生产总体的密度中位的中位（0.5539），
#: 地位同 `PRINT_AIM`；`max_shift` 取 NegPy 的 ±0.12 decade 护栏——
#: 本批逐帧 δ 实际 **−0.0537 ~ +0.0583**，**限幅没有绑住任何一帧**，它只挡病态帧。
#: ⚠ 两点退化锚（趾部挤成一团）**不加电平**——那条路径没测过，见 `anchor_with_level`。
PRINT_LEVEL_STRENGTH = 0.25
PRINT_LEVEL_REFERENCE_DENSITY = 0.5539
PRINT_LEVEL_MAX_SHIFT = 0.12
#: 色度增益。**来自用户手工参考，属于"口味"不是"测量"**：
#: 原值 1.897 由 `1298_G200` 9 帧参考反解（中位数 `目标 sat_p95 / 软件 sat_p95`），
#: 而留出卷 `1299_G200` **没有参考可独立验证风格**，所以它的地位与
#: `PRINT_AIM`/`PRINT_WHITE`（外部实测相纸常数）不同，见 blockers B-11 追加⑧。
#: 设成 1.0 即完全关闭这一步，退回"只印亮度"。
#:
#: **2.50 是它与 `LOOK_RED_EXPONENT` 联立定出来的**（`tmp_joint_slope_gain.py`，
#: 11 帧参考 × 5 档增益，前提校验：默认增益下离线重渲与 `candidate_positive`
#: **逐比特一致**，所以扫的是产品不是探针）。两个旋钮在色相轴上**互相拉扯**，
#: 不正交：红斜率抬到 1.30 会把通道差压小（更淡），而增益本身又把 `Δ带宽RG` 往负方向推
#: （red 1.30 下增益 1.897→2.700 时 `Δ带宽RG` 从 +0.0360 掉到 +0.0079）。
#: 所以单独扫任何一个都会得到相反结论——上一轮就是只看饱和否掉了红斜率。
#: 15 个联立条件里四统计平均距离在 0.0583~0.0634 之间几乎平，
#: 取**留出卷（三帧，未参与选值）两项都最好**的那格：`red 1.30 × gain 2.50`
#: 留出 |Δ带宽RG| 0.0129（现产 0.0390）、留出 |Δsat95| 0.0353（现产 0.0392），
#: 11 帧距离 0.0593（现产 0.0610）。
LOOK_CHROMA_GAIN = 2.50
#: 红通道密度指数（`InvertParams.red_ratio` 的**产品默认值**）。
#: 与 `LOOK_BLUE_EXPONENT` 同地位：来自用户手工成品的校正量，留出卷逐帧复验过。
#: 实测（`tmp_slope_resweep.py` / `tmp_joint_slope_gain.py`，11 帧参考、1200px、
#: 注入走生产资产通道、矩阵单位阵）：中间带线性比 `R−G` 我方恒为负（一律偏绿），
#: `red_ratio` 从 1.0 抬到 1.30 时逐帧 |Δ带宽RG| 走**内点最小值**
#: 0.0699 → **0.0546**，`Δ带宽RG` 在 1.12 附近过零，再往上（1.5/1.75/2.0）单调变差；
#: 留出卷 `1299` 三帧在 1.15 处 **3/3 全部变好**，在 1.30 处 2/3。
#: **上一轮"red_ratio 方向无效"的说法作废**：那次只看了蓝黄轴 `Δb` 与饱和，
#: 而这条误差一直在 R−G 轴上；且当时若干帧片基错 0.19~0.36 decade，
#: 片基直接进 `d = −log10(t/base)`，等于搬走自变量再测斜率。见 B-14⑧。
#: ⚠ 抬红斜率会把通道差压小（更淡），所以它**必须与上面的 `LOOK_CHROMA_GAIN` 联立定**——
#: 单独看任何一个都会得出相反结论。联立扫描的出处写在那条增益上。
LOOK_RED_EXPONENT = 1.30
#: 蓝通道密度指数（`InvertParams.blue_ratio` 的**产品默认值**）。
#: 与 `LOOK_CHROMA_GAIN` 同一地位：唯一一个来自用户手工成品的口味/校正量，
#: 但比它多一层证据——**留出卷逐帧复验过**，方向也做过负控制。
#: 实测（`tmp_ratio_cast.py` / `tests/regression/test_profile_blue_exponent.py`，
#: 11 帧用户参考、1200px、注入走生产资产通道）：
#: 中间调色度向量差中位 0.0492 → **0.0407**，蓝偏 Δb −0.0288 → **−0.0092**，
#: 饱和偏差基本不动（−0.012 → −0.018）；留出卷 `1299_G200` 三帧
#: 0.0201/0.0197/0.0520 → **0.0170/0.0160/0.0400**（逐帧都变好）；
#: 反方向（2026-08-31 学到的 `blue_ratio=1.0556`）把 \|Δb\| 从 0.0310 推到 0.0323。
#: `red_ratio` 那条路实测无效：1.3/1.6/2.0 时 Δb 只到 −0.0260/−0.0232/−0.0220，
#: 饱和却崩到 −0.070/−0.105/−0.120。
#: **跨片种代价（没装死）**：CN400 两帧 Δb −0.033→−0.001、−0.071→−0.050（变好），
#: ECN-2 五帧里四帧 Δb 朝零，但 `5207-1` 饱和 0.453→0.383（−0.070）——
#: ECN-2 上这是**用饱和换中性**，与 C-41 上"两者都得到"不同。
#: 可选的同值 profile 见 `fixtures/priors/KodakGold200-Z8-userfit.profile.json`。
LOOK_BLUE_EXPONENT = 0.7
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


def density_anchor_linear(samples: np.ndarray | None) -> tuple[float, float] | None:
    """两点锚 `(d_lo, d_hi)`：三点曲线解不出时的**退化印相**。

    仍然要求跨度够（否则直线也等于阶跃），但不要求中灰位置合理。
    """
    if samples is None or samples.size < 256:
        return None
    lo = float(np.percentile(samples, P_LOW))
    hi = float(np.percentile(samples, P_HIGH))
    if not (math.isfinite(lo) and math.isfinite(hi)) or hi - lo < MIN_SPAN:
        return None
    return lo, hi


def print_curve(anchor: tuple[float, float, float], *,
                aim: float = PRINT_AIM, white: float = PRINT_WHITE):
    """(密度控制点 x, 纸级曝光量 y)；要求斜率**递增**（先压趾部、后展高光）。

    纵坐标由 `paper = 1 − 10^(−v)` 反解：`v = −log10(1 − paper)`。

    **两点锚 `(lo, hi)` 是合法的退化形式**：三点解不出单调曲线时
    （画面暗部挤成一团，`mid − lo` 太小 ⇒ 趾部斜率超过高光段），
    用"黑在 lo、白在 hi"的直线仍然把雾压掉，比**完全不印**好得多。
    实测 `1298_G200::G200-10` 三点被拒时该帧显示域 `p05=0.292`
    （用户参考 0.036），是标定后十一帧里最差的一帧。
    """
    if len(anchor) == 2:
        lo, hi = float(anchor[0]), float(anchor[1])
        if not (hi > lo > -6.0) or hi - lo < MIN_SPAN:
            raise ValueError(f"两点锚退化（lo {lo:.3f} / hi {hi:.3f}）")
        return (np.array([lo, hi], dtype=np.float64),
                np.array([0.0, -math.log10(max(1e-9, 1.0 - white))],
                         dtype=np.float64))
    lo, mid, hi = anchor
    v_mid = -math.log10(max(1e-9, 1.0 - aim))
    v_hi = -math.log10(max(1e-9, 1.0 - white))
    s1 = v_mid / (mid - lo)
    s2 = (v_hi - v_mid) / (hi - mid)
    if not (math.isfinite(s1) and math.isfinite(s2)) or s2 <= s1:
        raise ValueError(f"打印曲线斜率未递增（趾部 {s1:.3f} ≥ 高光 {s2:.3f}）")
    return (np.array([lo, mid, hi], dtype=np.float64),
            np.array([0.0, v_mid, v_hi], dtype=np.float64))


def level_shift(density_median: float | None) -> float:
    """电平项 δ（decade）：`strength·(本帧取样总体的密度中位 − 标称中位)`，限幅 ±cap。

    δ>0（本帧比标称更浓 = 场景更亮）⇒ 锚点往下平移 ⇒ **印得更亮**，
    这正是"每帧归一到 AIM"抹掉的那个自由度。输入非有限/缺省时返回 0.0
    （= 现产行为），病态帧不该被电平项二次放大。
    """
    if density_median is None or not math.isfinite(float(density_median)):
        return 0.0
    raw = PRINT_LEVEL_STRENGTH * (float(density_median) - PRINT_LEVEL_REFERENCE_DENSITY)
    return float(np.clip(raw, -PRINT_LEVEL_MAX_SHIFT, PRINT_LEVEL_MAX_SHIFT))


def anchor_with_level(anchor: tuple[float, ...] | None,
                      density_median: float | None):
    """给锚点加电平绝对项，返回 `(锚, 实际δ, 说明)`；**δ=0 时锚逐比特不变**。

    只动**中灰与白**、黑锚不动：黑锚是场景暗部的地板（`P_LOW=2.0` 定它），
    跟着电平一起抬等于把已经偏高的阴影再抬一次（不加这一项时我方阴影偏差
    Δp05 = **+0.0257**，上线后 **+0.0279**，见 `test_level_absolute` 的当场打印）。
    纯曝光式的整体平移（黑锚一起动）在 768 显示统计那一版扫描里训练卷距离更差
    （**0.0681** 对 **0.0620**），所以这不是"没测就选"。

    **两点退化锚原样返回**：那批帧（如 `1298_G200::G200-10`）没有中灰控制点，
    电平项的定义（"抬中灰"）落不了地，而"抬白"未测 ⇒ 不加。
    **平移后三点曲线不合法**（趾部被压没：`mid − δ` 贴近 `lo` 使趾部斜率超过高光段）
    时同样原样返回并说明，绝不让这帧从"印"退成"不印"。
    """
    if anchor is None:
        return None, 0.0, "无锚点：不施加电平项"
    delta = level_shift(density_median)
    if delta == 0.0:
        return anchor, 0.0, ""
    if len(anchor) != 3:
        return anchor, 0.0, f"两点退化锚：不加电平项（δ={delta:+.4f} 未施加）"
    shifted = (anchor[0], anchor[1] - delta, anchor[2] - delta)
    try:
        print_curve(shifted)
    except ValueError as exc:
        return anchor, 0.0, f"平移后曲线不合法，保持原锚（δ={delta:+.4f} 未施加）：{exc}"
    return shifted, delta, ""


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


#: 色度增益在**纸级末端**要收回来（"相纸高光染料自然脱色"）。
#:
#: **它补的是哪个洞**（`tmp_highlight_cast.py`，11 帧，取显示亮度最高 5% 的像素
#: 求三通道均值，分三层量）：
#:
#: | 层 | `1299` 三帧最高 5% 的 R/G/B | 彩度 |
#: |---|---|---|
#: | 技术正片（打印前） | 0.849~0.860 / 0.908~0.918 / 0.971~0.987 | 0.122~0.128 |
#: | 成品（打印后，本项之前） | 0.665~0.688 / 0.808~0.831 / 0.903~0.920 | **0.226~0.249** |
#: | 用户手工参考 | 0.964~0.991 / 0.951~0.990 / 0.933~0.993 | **0.006~0.031** |
#:
#: ⇒ **反相出来的白本来就接近中性**（最低通道只比他差 −0.08~−0.13），
#: 是 `LOOK_CHROMA_GAIN=2.50` 无差别放大把高光通道差翻倍、再被纸级天花板
#: 整像素同除砍掉 R，才变成 −0.25~−0.33。**这就是用户点名的"水是一片死青"。**
#: 而 2.50 当初是用**整帧 `sat_p95`**（一个由高光主导的量）拟合的，
#: 等于拿高光的数去定中间调的口味，再反过来把高光放大一倍——两头都不对：
#: 他的剖面是"中间调浓、高光淡"，我们是"中间调淡、高光浓"。
#:
#: **形式**：`k_eff(y) = k − (k − k_hi)·clip(y/PRINT_WHITE − t0n, 0, 1)/(1 − t0n)`，
#: 其中 `y` 是**线性纸级亮度**（上限就是 `PRINT_WHITE`）。
#: ⚠ `t0n` 归一在纸级刻度上是有原因的：第一版把 `t0` 当显示亮度扫到 0.88，
#: 结果**一个像素都没进斜坡**（三档 `k_hi` 数字完全相同），
#: 而 `t0=0.45` 其实已经咬到纸级 56% 处 = 中间调（`tmp_chroma_profile.py` 当场抓到）。
#:
#: **系数由四条预写判据 + 看图共同定**（`tmp_chroma_profile.py` 扫
#: `t0n{0.70,0.85}×k_mid{2.5,3.0,3.5}×k_hi{0.8,1.3}`）：`0.85 / 0.8` 这一格
#: 高光最低通道差 `0.1623 → 0.1176`（−28%，留出卷 `0.2935 → 0.2172`）、
#: 高光彩度差 `0.1251 → 0.0783`（−37%）、白点缺口 `Δp95 −0.0461 → −0.0185`（闭合 60%）、
#: 中间带色相 `|ΔRG| 0.0533 → 0.0527`。**代价照实记**：
#: `Δsat95 −0.0204 → −0.0271`、**训练卷四统计距离 0.0597 → 0.0614（+0.0017）**、
#: `Δ亮度 +0.0059 → +0.0120`。训练卷变差的两项来源（亮度与整帧饱和）
#: **都不是这条机制声称要修的缺陷**——按纪律第 4 条这时由看图裁决：
#: `outputs/ref_look_probe/highlight_desat_3way.jpg` 三列并排，
#: `1299` 三帧与 `1298::G200-1` 的云/水**明显回到白**，
#: 只有 `1298::G200-5`（他这一帧的高光本来就带暖彩度 0.151，我们 0.116）被抹淡一点。
#:
#: 抬 `k_mid` 去付饱和账是**试过并否掉的**：`k_mid=3.0, k_hi=0.8` 时
#: `Δsat95` 变成 +0.0259（过量）、`|ΔRG|` 从 0.0527 恶化到 0.0701、两卷距离都变差。
#: `k_hi < 1.0` 不是"往中性硬压"：他一侧也有彩度高的高光（`::G200-5` 0.151、
#: `::G200-7` 0.146），0.8 只是"末端略微收"，保留技术正片自己的通道差。
#: ⚠ 片基未实测的帧本来就被闸门设成 `chroma_gain=1.0`（不放大），
#: 这一项仍会在末端把它收到 0.8——那是纸的行为，与放大无关，见 `chroma_gain_at`。
PRINT_CHROMA_HIGHLIGHT_GAIN = 0.80
PRINT_CHROMA_HIGHLIGHT_START = 0.85


def chroma_gain_at(paper_luma: np.ndarray, base_gain: float) -> np.ndarray:
    """逐像素色度增益：纸级 `PRINT_CHROMA_HIGHLIGHT_START` 之前恒等于 `base_gain`，
    之后线性收到 `PRINT_CHROMA_HIGHLIGHT_GAIN`（**亮度不参与运算**，权重和恒为 1）。

    `base_gain == 高光档` 时整帧恒等 ⇒ 这一项可以整体关掉做对照。
    生产里 `base_gain` 只有两种取值（已实测片基 `LOOK_CHROMA_GAIN` /
    闸门帧 1.0），两者都 > 高光档，所以这条斜坡**只会往下收**；
    但公式本身不做钳位——**故意不做**：负控制要靠"把末端反而放大"这一档
    来证明这把尺有区分力（`tests/regression/test_highlight_desaturation.py`），
    加了钳位那条控制就变成空转（本轮实测踩过：钳位版下 `k_hi=4.2` 与 2.5 逐比特相同）。
    """
    y = np.asarray(paper_luma, np.float32)
    k0 = np.float32(base_gain)
    kh = np.float32(PRINT_CHROMA_HIGHLIGHT_GAIN)
    t0 = np.float32(PRINT_CHROMA_HIGHLIGHT_START)
    ramp = np.clip((y / np.float32(PRINT_WHITE) - t0) / np.float32(1.0 - t0), 0.0, 1.0)
    return (k0 - (k0 - kh) * ramp).astype(np.float32, copy=False)


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
    # k 在**纸级末端**收回到 `PRINT_CHROMA_HIGHLIGHT_GAIN`（见那三条常量的依据）。
    y = out @ _LUM_W32
    out = y[..., None] + chroma_gain_at(y, chroma_gain)[..., None] * (out - y[..., None])
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
    info = {"d_lo": round(anchor[0], 5), "d_hi": round(anchor[-1], 5),
            "points": len(anchor), "aim": PRINT_AIM, "white": PRINT_WHITE,
            "dmax_old": round(float(dmax_old), 5), "chroma_gain": round(k, 5),
            "chroma_gain_highlight": PRINT_CHROMA_HIGHLIGHT_GAIN,
            "chroma_ramp_start": PRINT_CHROMA_HIGHLIGHT_START}
    if len(anchor) == 3:
        info["d_mid"] = round(anchor[1], 5)
    return out, info
