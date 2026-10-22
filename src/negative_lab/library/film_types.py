r"""片种参数表（**数据层**）：片种 → 技术级通道斜率。

为什么放在 `library` 而不是 `pipeline`：这张表要被 `library/models.py`（卷声明的校验）
和 `pipeline/`（构造资产）两边读，而分层门禁禁止下层 import 上层
（`tests/unit/test_layering.py`）。表本身是纯数据 + 校验，构造资产那一步
（依赖 `pipeline.model_assets`）留在 `pipeline/film_types.py`。

为什么需要这张表（`docs/blockers.md` B-16 追加⑨/⑫/⑭）：技术级通道斜率按片种给才有解——
在**交付口径（A 层）**上，把 `red_ratio` 从现产 1.30 抬到 1.70 后，
两支 ECN-2 卷的 `cast_bg` 过零分别是 **5207≈0.92、5219≈0.62**（要往**相反**方向调），
而 G200 两支卷的最优格仍落在现产附近（1298 的 rr 最优就是 1.30）。
一卷一个方向 ⇒ 全局常数无解，只能按片种给。

纪律：
 ① `provenance` 为空、`frames<=0`、`roll` 为空 ⇒ 直接抛错（不带出处的常数不进表）；
 ② 值必须是**网格内点最小**（两端更差才算），端点值不进表；
 ③ 未声明 = `resolve()` 返回 `None` = 现产行为逐比特不变。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class FilmType:
    key: str                      # 机器可读，写进 `FrameRecord.film_type` / `RollGroup.film_type`
    label: str                    # 给人看的（GUI 下拉、CLI --film-type 的说明）
    roll: str                     # 值是在**哪一卷样片**上定的（核对与回归都要用）
    red_ratio: float
    blue_ratio: float
    provenance: str               # 在谁的样片上、怎么量的；空字符串直接抛错
    frames: int                   # 定值用的帧数（置信度证据）
    #: 印相中灰输出目标（纸级反射率）。`0.0` = 不施加 = 用全局 `PRINT_AIM`。
    #: 这一维为什么也在片种表里：ECN-2 两卷一致偏亮 ~0.30 EV、G200 两卷不偏，
    #: 全局改会把 G200 推过头（`docs/blockers.md` B-16 追加⑯）。
    print_aim: float = 0.0
    #: 高光带白点蓝黄校正的**强度**（`white_balance.apply_white_balance`）。
    #: `0.0` = 不施加。方向与逐帧大小由我方输出自己解，这一维只是"打到几成"。
    #: 为什么按片种给：㉙ 量到同一套约束在两卷上的可用区间宽度不同
    #: （5207 合格区间 1.30~2.80，5219 只有 1.30 一格）⇒ 不能钉成全局常数。
    white_balance: float = 0.0
    #: 纸白端彩度收回档（`print_math.chroma_gain_at` 的 `highlight_gain`）。
    #: `0.0` = 不施加 = 用全局 `PRINT_CHROMA_HIGHLIGHT_GAIN`。
    #: ㉞：5219 要 0.55（`J` −2.1%、全部护栏干净通过），5207 整条网格只动 0.3% ⇒ 按片种。
    chroma_highlight_gain: float = 0.0
    #: 印相纸白输出目标（纸级反射率）。`0.0` = 不施加 = 用全局 `PRINT_WHITE`。
    #: ㊦㊩㊫：验收尺十根轴全是**我方自己分布**的边缘量，对"绕中灰被压扁"这种
    #: 两端对称错**不敏感**（12 箱条件量上升降两端各错 0.10~0.12，而 `luma_mean` 只差 0.0044）。
    #: 抬纸白 + 降中灰目标这一对把 12 箱形状误差在**留出帧**上打掉 15.8%（两卷同向），
    #: 代价是 `sat_p95`/`luma_std` 两根边缘量离他更远——图上（`toe_ship_wall.png`）
    #: 候选比现产明显更像他的样片，且逐帧黑位抬升 12/12 帧改善 ⇒ 按片种给，不动全局。
    print_white: float = 0.0

    def __post_init__(self):
        if not self.key or not self.label or not self.roll:
            raise ValueError(f"片种 {self.key!r} 缺 key/label/roll")
        if not self.provenance.strip():
            raise ValueError(f"片种 {self.key} 没有出处——不许进表"
                             f"（常数不带出处，下一轮就没人敢动）")
        if int(self.frames) <= 0:
            raise ValueError(f"片种 {self.key} 的定值帧数必须 >0")
        for name in ("red_ratio", "blue_ratio"):
            v = float(getattr(self, name))
            if not (0.2 <= v <= 4.0):
                raise ValueError(f"片种 {self.key}.{name}={v} 出界（0.2~4.0）")
        pa = float(self.print_aim)
        if pa and not (0.02 <= pa <= 0.45):
            raise ValueError(f"片种 {self.key}.print_aim={pa} 出界（0.02~0.45，"
                             f"0 表示不施加）")
        pw = float(self.print_white)
        if pw and not (0.60 <= pw <= 0.98):
            raise ValueError(f"片种 {self.key}.print_white={pw} 出界（0.60~0.98，"
                             f"0 表示不施加；全局默认 0.80375）")
        wb = float(self.white_balance)
        if wb and not (0.05 <= wb <= 4.0):
            raise ValueError(f"片种 {self.key}.white_balance={wb} 出界（0.05~4.0，"
                             f"0 表示不施加）")
        chg = float(self.chroma_highlight_gain)
        if chg and not (0.05 <= chg <= 1.50):
            raise ValueError(f"片种 {self.key}.chroma_highlight_gain={chg} 出界"
                             f"（0.05~1.50，0 表示不施加）")


#: 表。每个值都要是 A 层（交付口径）网格的**内点最小**，出处写在各自字段里。
#: 空表 = 机制存在但没有任何片种被标定 ⇒ 产品行为与今天逐比特相同。
FILM_TYPES: dict[str, FilmType] = {
    "ecn2-5207": FilmType(
        key="ecn2-5207", label="ECN-2 · 5207 卷", roll="ECN2_5207",
        red_ratio=1.70, blue_ratio=0.85, frames=35, print_aim=0.07,
        print_white=0.845, white_balance=1.80,
        provenance="用户手工正片真值 35 帧（`棚拍5207+5219/ECN2`，对齐表 v2）。"
                   "A 层（交付口径，显示域→384→中心 70%）联立网格 `tmp_slope_cast_sweep.py joint`："
                   "rr 轴 Δcast_rg 在 bb=0.85 上 1.50→−0.0872 / 1.70→−0.0094 / 1.85→+0.0401 / "
                   "2.00→+0.0876（内点）；bb 轴在 rr=1.70 上 Jc=|Δrg|+|Δbg| 为 "
                   "0.50→0.4830 / 0.65→0.2435 / 0.85→0.0579 / 1.00→0.0598 / 1.15→0.1350（内点）。"
                   "⚠ 单轴扫时 bb 的最优给到 1.00，抬 rr 之后降到 0.85——两轴不正交，"
                   "0.85 与 1.00 只差 0.0019（Δbg 过零 ≈0.92），取 J 最小且改动更少的 0.85。"
                   "cast 误差 0.2328→0.0579。"
                   "print_aim=0.10 另有一笔账：`tmp_curve_targets_perstock.py` 在 white 不动"
                   "（=现产 0.80375）时扫 aim，本卷 |L*| 走 0.07→5.898 / **0.10→5.742** / "
                   "0.13→6.041（内点最小），带符号 +4.902→**+0.314**、|std| 0.0257→0.0200、"
                   "|Cab| 5.502→4.365 同时变好；G200 两卷在同一格过冲到 −2.09/−2.41 ⇒ 不进全局。"
                   "white_balance=1.80：`tmp_wb5_prod.py` 走**生产代码**（`apply_render` 里的 "
                   "`white_balance` 步）扫，约束=近中性高光 b*→0 ∧ 整幅平均 a* 不动，"
                   "权重带沿 L*=80 严格反解 0.7778 → 纸白 0.9082。"
                   "本卷白 b* 带符号误差 −3.214（35 帧、32/35 帧同号）；"
                   "J=|b|+|Cab| 走 0.90→8.792 / 1.30→8.666 / **1.80→8.561** / 2.40→8.606"
                   "（2.40 已破 T1：Δb*=+1.197）⇒ 内点最小；该格 "
                   "Δb*→+0.639、|b| 4.864→4.390、|Cab| 4.250→4.170、|a| 2.043→2.062"
                   "（红绿账由约束钉住，代价 +0.019 < T2 的 +0.05）。"
                   "print_white=0.845 与 print_aim 0.10→0.07（㊫ 上线，两卷**同一格**）："
                   "搜索目标不是验收尺十轴，而是 ㊦ 那条**按他的亮度分 12 箱的条件传递曲线**"
                   "（现产本卷暗端 +0.0963、亮端 −0.0728，两卷 12 箱全部同号；"
                   "十轴对这种两端对称错不敏感——`luma_mean` 只差 0.0044）。"
                   "三条逐比特门全过（`tmp_white_ship_verify.py` → `outputs/white_ship_verify.log`）："
                   "V0 基线臂与改动前冻结的现产裁块 35/35 帧一致；V1 **探针路（改模块全局）与"
                   "上线路（intent 传参）0 帧不一致** ⇒ 上线的像素就是量过的像素；"
                   "V1b 显式传 `0.80375` 与不传一致 ⇒ 未声明的卷不受影响。"
                   "HOLD（17 帧，不参与定值）：S 0.8673→**0.7380**（本卷 −14.9%，"
                   "两卷均值 −15.7%＝㊩ 预写 J1 的口径），暗端三箱 0.1046→0.0870、"
                   "亮端三箱 0.1023→0.0862；条件黑位抬升 +0.1064→+0.0928；"
                   "12 箱 9 箱变好、3 箱变差（变差的中在 0.35~0.65 的中调拐点）。"
                   "整卷 35 帧十轴：`L` 5.3820→5.3336、`a` 2.0618→1.9562、`b` 4.3904→4.0764、"
                   "`Cab` 4.1702→3.9105、`cast_rg_mid` 0.0745→0.0700 全部变好；"
                   "**代价照实记**：`sat_p95` 0.1103→0.1275、`luma_std` 0.0190→0.0305（都破原 TOL），"
                   "G1 削底 4.2→4.3 倍（基线本就是他 4.2 倍，这条在基线上就红），"
                   "G2 `|p98−他 p98|` 0.04328→**0.02321**（亮端变近，这条变好）。"
                   "⚠ HOLD 那一半的 `Cab` 反而 +0.1167（破 TOL）、`cast_bg_mid` +0.0025，"
                   "与整卷方向相反 ⇒ 上报口径按哪个总体必须写清，两处的账都在 ㊫。"
                   "口径变更理由写在 B-16 追加㊫：不能让一把看不见该缺陷的尺一票否决；"
                   "看图 `toe_ship_wall.png` 12 行里候选的黑位逐帧更贴他（抬升幅度 12/12 帧变小）。"),
    "ecn2-5219": FilmType(
        key="ecn2-5219", label="ECN-2 · 5219 卷", roll="ECN2_5219",
        red_ratio=1.70, blue_ratio=0.65, frames=35, print_aim=0.07,
        print_white=0.845, white_balance=1.80, chroma_highlight_gain=0.45,
        provenance="用户手工正片真值 35 帧（同上一卷的目录，另一个卷号、同一 ECN-2 工艺）。"
                   "A 层联立网格：rr 轴 Δcast_rg 1.50→−0.0477 / 1.70→+0.0283 / 1.85→+0.0785 / "
                   "2.00→+0.1236（内点）；bb 轴在 rr=1.70 上 Jc 为 "
                   "0.50→0.1846 / 0.65→0.0567 / 0.85→0.2252 / 1.00→0.3080 / 1.15→0.3701（内点）。"
                   "与 5207 同工艺却往**相反**方向调（这支 Δbg 过零 ≈0.624），"
                   "这正是参数必须按片种给的证据。cast 误差 0.3059→0.0567。"
                   "print_aim=0.10：本卷 |L*| 走 0.07→4.496 / **0.10→4.499** / 0.13→4.699，"
                   "带符号 +4.853→**+0.757**、|std| 0.0250→0.0211、|Cab| 6.824→5.697；"
                   "与 5207 落在**同一格**（两卷独立同意），出处见上一条的曲线扫。"
                   "white_balance=1.80：同一支扫描 `tmp_wb5_prod.py`，本卷白 b* 误差 −3.105"
                   "（32/35 帧同号），而**白 a* 误差本来就只有 +0.597**——这正是 ㉘/㉙ 前两版"
                   "把整幅 |a| 拖坏的原因（那条约束对本卷纯属多余），故本版改成"
                   "「b*→0 ∧ 整幅 a* 不动」。J 走 0.90→破 T1(−1.436) / 1.30→10.956 / "
                   "**1.80→10.926** / 2.40→10.950（2.40 破 T1：Δb*=+1.514）⇒ 内点最小；"
                   "该格 Δb*→+0.417、|b| 5.567→5.253、|Cab| 5.697→5.673、|a| 4.571→4.549"
                   "（不升反降）、|L*| 4.499→4.496。与 5207 又是同一格。"
                   "chroma_highlight_gain=0.45：`tmp_chg_prod_rescan.py` 走**生产代码**"
                   "（`RenderIntent.print_chroma_highlight_gain` → `print_remap_array` → "
                   "`chroma_gain_at`）扫，基线自检两条都过——不施加那一行逐比特复现声明臂数"
                   "（|Cab|5.6732/|b|5.2531），显式 `kh=0.80` 与不施加差 +0.000000。"
                   "本卷 `J=|Cab|+|b|+|sat_mean|` 走 0.35→10.8481 / **0.45→10.7264** / "
                   "0.55→10.7369 / 0.65→10.7720 / 0.80→10.9657 ⇒ **内点最小 0.45**，"
                   "ΔJ −0.2393（−2.2%）；该格 |b| 5.2531→**5.0245**、|Cab| −0.0106、"
                   "|a| +0.0044、|L*| +0.0028、sat_p95 +0.0002、luma_std 0.0000 ⇒ 六根护栏全过。"
                   "⚠ **只给本卷**：同一格在 5207 上 ΔJ 只有 −0.0098（−0.11%）而 |b| 反而 "
                   "+0.018 ⇒ 那一卷保持不施加（现产 0.80），不为 0.1% 动一个常数。"
                   "print_white=0.845 与 print_aim 0.10→0.07（㊫ 上线，与 5207 **同一格**）："
                   "搜索目标是 ㊦ 的**12 箱条件传递曲线**（本卷现产暗端 +0.0801、亮端 −0.0514，"
                   "与 5207 同号）。三条逐比特门与上一卷同（V0 35/35、V1 探针路==上线路 0 帧差、"
                   "V1b 显式默认==不传）。HOLD（17 帧）：S 0.4951→**0.4100**（−17.2%），"
                   "暗端三箱 0.0801→0.0605、亮端三箱 0.0514→0.0421；"
                   "条件黑位抬升 +0.0702→+0.0547；12 箱 8 箱变好、4 箱变差"
                   "（0.35~0.55 与 0.65~0.85 两段被推过头）。"
                   "整卷 35 帧十轴：`a` 4.5531→4.3580、`b` 5.0245→4.8266、"
                   "`Cab` 5.6626→**5.0916**、`sat_mean` 0.0393→0.0359 变好，"
                   "`sat_p95` 0.1036→0.1080（容差内）；"
                   "**代价比 5207 多一根**：`|L|` 4.4983→4.6517（破 TOL，而 HOLD 那一半只 "
                   "+0.0199 ⇒ 回归集中在 TRAIN 帧）、`luma_std` 0.0210→0.0320、"
                   "`cast_bg_mid` 0.1124→0.1241 都破 TOL；"
                   "G1 削底 5.2→5.6 倍（基线本就是他 5.2 倍），"
                   "G2 `|p98−他 p98|` 0.02331→**0.00390**（亮端几乎贴上）。"
                   "⚠ **上一条 `print_aim=0.10` 的旧扫被本条取代**：那次是 `white` 钉在"
                   "现产 0.80375 上单扫 aim（0.07→4.496 / 0.10→4.499 几乎同值），"
                   "而 ㊧ 量到这两把旋钮**联立才有解、单边只能修一头"
                   "（aim 单独往上 0.13/0.16 时 S 反而 +18%/+40%）**⇒ 规矩 9/11：挪了工作点"
                   "必须联立重扫，旧格子的结论不再当作约束。代价同 5207：sat_p95/luma_std "
                   "超原边缘量护栏、G1 削底变差，理由与看图证据写在 B-16 追加㊫。"),
}


def resolve(key: str | None) -> FilmType | None:
    """`None`/空/`unknown`/未知键 ⇒ `None`（不改变任何行为）。

    未知键也返回 `None` 是**故意的**：宁可退回现产行为，也不要把拼错的片种
    当成一套新参数。要给用户报错的地方用 `known_keys()` 自己校验。
    """
    if not key or key == "unknown":
        return None
    return FILM_TYPES.get(key)


def resolve_keys(keys: Iterable[str | None]) -> FilmType | None:
    """帧级声明优先于卷级：按传入顺序取**第一个已声明**的键。
    混合声明不静默平均——那会造出一个没有任何出处的第三套值。"""
    for k in keys:
        ft = resolve(k)
        if ft is not None:
            return ft
    return None


def known_keys() -> list[str]:
    return sorted(FILM_TYPES)
