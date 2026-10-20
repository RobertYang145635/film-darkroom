r"""**配置文件里的逐通道斜率在生产路径上是死的**（㊶ 追加⑩ 的锁）。

发现过程：下一轮要做"按片种给技术级斜率"（追加⑨），先查现成的通道 ——
`FilmProfile.exponent_prior` 已经支持 `{red_ratio, blue_ratio}`，
`to_initial_params()` 也把它们写进 `InvertParams`，CLI 有 `--profile`。
看起来什么都不用做。但三臂实测（本文件跑的就是这三臂）：

| 臂 | 传入 | 解析后的 `blue_ratio` |
| --- | --- | --- |
| A | `prior(blue=1.30)` + 默认资产 | **0.850**（= 产品默认，先验被覆盖） |
| B | `prior(blue=1.30)` + 斜率为 `None` 的资产 | **1.300**（先验活着） |
| C | 不给先验 + 默认资产 | 0.850（现产） |

根因：`process_image` 在 `auto_analyze` 之后无条件 `asset.apply_to(params)`，
而 `deterministic_asset()` 的 payload 里带着
`red_ratio=LOOK_RED_EXPONENT / blue_ratio=LOOK_BLUE_EXPONENT`
（`model_assets.py:132`），`apply_to` 见 `not None` 就 `setattr`
（`model_assets.py:87-89`）⇒ **默认资产永远赢过用户先验**，
而规格 §2.2.2 写的是"配置文件只能作为初始先验、不得强行覆盖逐张分析结果"——
现在的实际情况恰好相反：一个**没人声明过的内置资产**覆盖了**用户显式声明**的值。

本文件不修行为（修法要在"该谁赢"这个产品问题上先有结论），
它做的是**把这个坑钉住**：
  ① 三臂数字必须复现（A≠B、C=0.85）——行为若被改动，锁会红，逼着改动者回来写出处；
  ② 明确断言"默认资产覆盖用户先验"当前成立，注释里写清这是**已知缺陷不是设计**；
  ③ 反空断言：若哪天 `blue_ratio` 三臂全等（说明资产不再参与），本锁必须换主语而不是删。
"""
from __future__ import annotations

import numpy as np
import pytest

from negative_lab.library.models import GeometryRegions
from negative_lab.pipeline.model_assets import ModelAsset, canonical_hash
from negative_lab.pipeline.params import InvertParams
from negative_lab.pipeline.service import process_image


def _neg(h: int = 160, w: int = 200) -> np.ndarray:
    """合成一张"像 ECN-2 负片"的线性图：整体偏橙、中间一条更亮的带。

    只要求通道结构与取样点数量够（`dmin`/分位需要 ≥256 个样本），不要求像任何真片。
    """
    rng = np.random.default_rng(7)
    base = rng.uniform(0.02, 0.30, (h, w, 3)).astype(np.float32)
    base[..., 0] *= 1.9
    base[..., 2] *= 0.45
    base[h // 3: 2 * h // 3] *= 2.4
    return np.clip(base, 1e-4, 0.98).astype(np.float32)


def _asset(blue_ratio):
    payload = {"density_matrix": np.eye(3).tolist(), "channel_curves": [None, None, None],
               "red_ratio": None, "blue_ratio": blue_ratio,
               "reference_exponent": None, "dmax": None}
    return ModelAsset(asset_version="slope-precedence-probe", payload=payload,
                      sha256=canonical_hash(payload), enabled=True)


def _resolved(prior, asset):
    pr = process_image(_neg(), geometry=GeometryRegions(), orientation_decided=True,
                       prior=prior, model_asset=asset, make_candidates=False)
    return float(pr.report.params.blue_ratio), pr


def test_default_asset_overrides_the_users_profile_slope():
    """A 臂：用户先验 blue=1.30，默认资产（带 0.85）⇒ 解析出来是 0.85。"""
    p = InvertParams()
    p.blue_ratio = 1.30
    got, _ = _resolved(p, None)
    assert got == pytest.approx(0.85, abs=1e-6), (
        f"默认资产不再覆盖用户先验（解析值 {got}）⇒ 本锁换主语："
        f"改成断言'先验赢'并把出处补到 profiles/profile.py 与 model_assets.py")


def test_asset_without_slope_leaves_the_prior_alone():
    """B 臂：同一份先验，资产的 blue_ratio=None ⇒ 1.30 活下来（证明通道本身是通的）。"""
    p = InvertParams()
    p.blue_ratio = 1.30
    got, _ = _resolved(p, _asset(None))
    assert got == pytest.approx(1.30, abs=1e-6)


def test_no_prior_is_the_production_value():
    """C 臂（现产）：不给先验、走默认资产 ⇒ 0.85，与 A 同值 ⇒ A 的先验确实没生效。"""
    a, _ = _resolved(None, None)
    b, _ = _resolved(None, _asset(1.30))
    assert a == pytest.approx(0.85, abs=1e-6)
    assert b == pytest.approx(1.30, abs=1e-6), "资产通道失效，B/A 两臂失去区分力"
