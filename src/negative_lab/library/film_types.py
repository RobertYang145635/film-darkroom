r"""片种参数表（**数据层**）：片种 → 技术级通道斜率。

为什么放在 `library` 而不是 `pipeline`：这张表要被 `library/models.py`（卷声明的校验）
和 `pipeline/`（构造资产）两边读，而分层门禁禁止下层 import 上层
（`tests/unit/test_layering.py`）。表本身是纯数据 + 校验，构造资产那一步
（依赖 `pipeline.model_assets`）留在 `pipeline/film_types.py`。

为什么需要这张表（`docs/blockers.md` B-16 追加⑨/⑫）：技术级通道斜率按片种给才有解——
在**交付口径（A 层）**上四卷的 `cast_bg` 过零点分别是
`5207≈0.97 / 1298≈0.91 / 1299≈0.85(=现产) / 5219≈0.61`，
其中 ECN-2 的两支卷（同工艺 250D / 500T）要往**相反**方向调。

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
FILM_TYPES: dict[str, FilmType] = {}


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
