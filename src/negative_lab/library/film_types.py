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


#: 表。每个值都要是 A 层（交付口径）网格的**内点最小**，出处写在各自字段里。
#: 空表 = 机制存在但没有任何片种被标定 ⇒ 产品行为与今天逐比特相同。
FILM_TYPES: dict[str, FilmType] = {
    "ecn2-5207": FilmType(
        key="ecn2-5207", label="ECN-2 · 5207 卷", roll="ECN2_5207",
        red_ratio=1.70, blue_ratio=0.85, frames=35,
        provenance="用户手工正片真值 35 帧（`棚拍5207+5219/ECN2`，对齐表 v2）。"
                   "A 层（交付口径，显示域→384→中心 70%）联立网格 `tmp_slope_cast_sweep.py joint`："
                   "rr 轴 Δcast_rg 在 bb=0.85 上 1.50→−0.0872 / 1.70→−0.0094 / 1.85→+0.0401 / "
                   "2.00→+0.0876（内点）；bb 轴在 rr=1.70 上 Jc=|Δrg|+|Δbg| 为 "
                   "0.50→0.4830 / 0.65→0.2435 / 0.85→0.0579 / 1.00→0.0598 / 1.15→0.1350（内点）。"
                   "⚠ 单轴扫时 bb 的最优给到 1.00，抬 rr 之后降到 0.85——两轴不正交，"
                   "0.85 与 1.00 只差 0.0019（Δbg 过零 ≈0.92），取 J 最小且改动更少的 0.85。"
                   "cast 误差 0.2328→0.0579。"),
    "ecn2-5219": FilmType(
        key="ecn2-5219", label="ECN-2 · 5219 卷", roll="ECN2_5219",
        red_ratio=1.70, blue_ratio=0.65, frames=35,
        provenance="用户手工正片真值 35 帧（同上一卷的目录，另一个卷号、同一 ECN-2 工艺）。"
                   "A 层联立网格：rr 轴 Δcast_rg 1.50→−0.0477 / 1.70→+0.0283 / 1.85→+0.0785 / "
                   "2.00→+0.1236（内点）；bb 轴在 rr=1.70 上 Jc 为 "
                   "0.50→0.1846 / 0.65→0.0567 / 0.85→0.2252 / 1.00→0.3080 / 1.15→0.3701（内点）。"
                   "与 5207 同工艺却往**相反**方向调（这支 Δbg 过零 ≈0.624），"
                   "这正是参数必须按片种给的证据。cast 误差 0.3059→0.0567。"),
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
