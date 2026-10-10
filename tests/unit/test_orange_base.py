"""画面内片基检测（橙色签名法）的单测：合成真值 + 每条判据的负控制。

这个文件里的**每一道正锁都配一条负控制**（仓库交付铁律）。合成图刻意做出
真实负片的三件事：① 片基是亮区且橙色最纯；② 场景内容只会让透射率**下降**
（负片物理：不可能有像素比片基更亮）；③ 翻拍台的漏光是**中性白**，比片基还亮。
第 ③ 件事就是负控制的靶子——去掉橙色闸，它会把蓝通道片基抬高半个数量级。
"""
from __future__ import annotations

import numpy as np
import pytest

from negative_lab.calibration.base import ORANGE_BASE, base_from_orange_signature
from negative_lab.pipeline.roll_analysis import resolve_base

BASE = np.array([0.300, 0.115, 0.038])          # R/G≈2.61、R/B≈7.9：与实测片基同量级


def _scene(rng: np.ndarray, density: np.ndarray) -> np.ndarray:
    """片基透过一层密度：t = base · 10^(−density)，density≥0 ⇒ 永不比片基亮。"""
    return BASE[None, None, :] * np.power(10.0, -density[..., None] * np.array(
        [1.0, 1.35, 2.10])[None, None, :])


def _frame(with_leak: bool = True, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    h = w = 260
    dens = np.abs(rng.normal(0.55, 0.45, (h, w)))          # 场景密度，含大片低密度区
    img = _scene(rng, dens).astype(np.float32)
    img[dens < 0.03] = BASE.astype(np.float32)             # 未曝光片基/极亮区
    if with_leak:
        img[10:40, 10:120] = np.array([0.92, 0.90, 0.88], np.float32)   # 中性白漏光
    return np.clip(img, 0.0, 1.0)


def test_橙色亮区法在合成真值上量回片基():
    img = _frame()
    out = base_from_orange_signature(img)
    assert out["dmin"] is not None
    err = np.abs(np.log10(out["dmin"] / BASE))
    assert err.max() < 0.02, f"合成真值下误差应远小于留出实测的 0.073：{err}"


def test_漏光被橙色闸挡住_去掉闸就被骗():
    """**负控制**：同一张图，只把比值闸关掉，估计必须显著变差。

    这条锁的意义不是"估计器准"，而是"**准是因为有这道闸**"——
    没有它，读者会以为是"取高分位"在起作用，而高分位恰恰是最容易被漏光骗的。
    """
    img = _frame(with_leak=True)
    gated = base_from_orange_signature(img)["dmin"]
    open_ = base_from_orange_signature(img, params={"rg_min": 0.0, "rb_min": 0.0})["dmin"]
    assert gated is not None and open_ is not None
    err_gated = np.abs(np.log10(gated / BASE)).max()
    err_open = np.abs(np.log10(open_ / BASE)).max()
    assert err_open > err_gated + 0.15, (
        f"去掉橙色闸后误差应显著恶化（漏光被当成片基）：闸内 {err_gated:.3f} "
        f"闸外 {err_open:.3f}")
    assert err_open > 0.2, (
        f"漏光电平 0.90 vs 片基 R 0.30 ⇒ 去掉闸后误差应到 0.48 量级，实得 {err_open:.3f}")


def test_没有橙色亮区时必须拒绝而不是造一个值():
    """纯灰阶（通道比恒为 1）里没有片基可测：必须返回 None 并说得出原因。

    尺寸要够大，否则会先撞上"像素不足"那道闸，量的就不是橙色闸了。
    """
    ramp = np.linspace(0.02, 0.95, 240, dtype=np.float32)
    gray = np.repeat(np.repeat(ramp[:, None, None], 240, axis=1), 3, axis=2)
    out = base_from_orange_signature(gray)
    assert out["dmin"] is None
    assert any("看不到片基" in w for w in out["warnings"]), out["warnings"]
    assert out["stats"].get("n", 0) >= 4096


def test_小图与非法输入不崩也不造值():
    assert base_from_orange_signature(np.zeros((4, 4, 3), np.float32))["dmin"] is None
    assert base_from_orange_signature(np.zeros((8, 8), np.float32))["dmin"] is None
    bad = _frame()
    bad[0, 0] = np.nan
    assert base_from_orange_signature(bad)["dmin"] is not None


def test_采用后仍高于片基的像素比例被如实报出():
    out = base_from_orange_signature(_frame())
    viol = out["stats"]["above_base"]
    assert len(viol) == 3
    assert max(viol) < 0.05, f"漏光面积只占 ~3%，不该把置信度打到一半以下：{viol}"
    assert out["confidence"] == pytest.approx(ORANGE_BASE["confidence"])


def test_分位数取99p9而不是最大值_因为噪声会把最大值顶高():
    """**这条锁钉的是"为什么不是 p100"**。

    留出真值上 p100 与 p99.9 几乎同准（片基区噪声极小），但那是运气：
    片基区一旦有颗粒/照明噪声，最大值就系统性偏高，而偏高方向的后果是
    扣完片基的密度整体偏小 ⇒ 技术正片卡进纸级曲线的直线段 ⇒ 发灰。
    所以这里造一个**带噪声的片基区**，要求默认分位的偏差比 p100 更接近真值。
    """
    rng = np.random.default_rng(11)
    h = w = 300
    dens = np.abs(rng.normal(0.55, 0.45, (h, w)))
    img = _scene(rng, dens)
    plateau = dens < 0.03
    noise = 1.0 + rng.normal(0.0, 0.02, plateau.sum())        # ±2% 的片基区噪声
    img[plateau] = (BASE[None, :] * noise[:, None]).astype(np.float32)
    img = np.clip(img, 0.0, 1.0).astype(np.float32)

    p_max = base_from_orange_signature(img, params={"percentile": 100.0})["dmin"]
    p_def = base_from_orange_signature(img)["dmin"]
    assert p_max is not None and p_def is not None
    assert np.all(p_max >= p_def * (1 - 1e-6)), "分位越高只会越亮，否则参数没接上"
    assert np.log10(p_max / p_def).max() > 0.01, "带噪声片基区下两个分位必须分得开"
    err_max = np.abs(np.log10(p_max / BASE)).mean()
    err_def = np.abs(np.log10(p_def / BASE)).mean()
    assert err_def < err_max, (
        f"默认分位应比最大值更接近真值：默认 {err_def:.4f} vs p100 {err_max:.4f}")
    assert np.all(p_def >= BASE * (1 - 0.02)), "片基区取高分位不许低估（低估=密度被吃掉）"


# ------------------------------------------------------------- 接线（回退顺序）
def test_片边与卷都不可用时走画面内估计而不是先验():
    img = _frame()
    r = resolve_base({}, None, t_oriented=img, image_area_roi=(0, 0, *img.shape[:2]))
    assert r["base_source"] == "content_orange", r["warnings"]
    err = np.abs(np.log10(np.asarray(r["dmin"]) / BASE)).max()
    assert err < 0.02


def test_画面内也测不到时才退回通用先验():
    gray = np.repeat(np.linspace(0.02, 0.95, 300, dtype=np.float32)[:, None, None],
                     3, axis=2)
    r = resolve_base({}, None, t_oriented=gray, image_area_roi=(0, 0, 300, 1))
    assert r["base_source"] == "generic_prior"
    assert r["confidence"] == 0.0


def test_人工片基框永远优先于画面内估计():
    """**这条是纪律不是功能**：用户画的框是绝对真值，新机制不许推翻它。"""
    img = _frame()
    r = resolve_base({"dmin": [0.31, 0.12, 0.040], "confidence": 0.9}, None,
                     t_oriented=img, image_area_roi=(0, 0, *img.shape[:2]))
    assert r["base_source"] == "frame"
    assert np.allclose(r["dmin"], [0.31, 0.12, 0.040])


# ------------------------------------------------- 卷级继承前的物理闸
def _roll_with(dmin):
    from negative_lab.library.models import RollAnalysis
    return RollAnalysis(base_statistics={"dmin": list(map(float, dmin))},
                        base_source="roll", confidence=0.9,
                        source_frames=["A", "B"])


def test_与卷模型不矛盾的帧仍然继承卷模型():
    """**负控制**：物理闸不许把正常帧也改成画面内估计。

    没有这条，读者无法判断"来源变了"是闸在起作用还是代码写错了方向。
    """
    img = _frame()                                  # 真片基就是 BASE
    r = resolve_base({}, _roll_with(BASE), t_oriented=img,
                     image_area_roi=(0, 0, *img.shape[:2]))
    assert r["base_source"] == "roll", r["warnings"]
    assert np.allclose(r["dmin"], BASE, rtol=1e-6)


def test_继承值被本帧内容否掉时改用画面内实测():
    """**正锁**：本帧有大面积像素比继承来的片基还亮 ⇒ 继承值必然估低了。

    实测靶子：`1298::G200-7` 在继承值下 26%/34%/40% 越界，画面内实测高
    +0.19/+0.28/+0.36 decade，而冻结容差 `max_base_rel_drift` 是 0.18。
    """
    true_base = BASE * (10.0 ** 0.30)               # 真片基比卷模型亮 0.30 decade
    img = _frame(with_leak=False, seed=21) * np.float32(10.0 ** 0.30)
    img = np.clip(img, 0.0, 1.0)
    r = resolve_base({}, _roll_with(BASE), t_oriented=img,
                     image_area_roi=(0, 0, *img.shape[:2]))
    assert r["base_source"] == "content_orange", r["warnings"]
    got = np.asarray(r["dmin"], np.float64)
    assert np.all(got > BASE * 1.15), f"片基必须被抬高，实得 {np.round(got, 4)}"
    assert np.abs(np.log10(got / true_base)).max() < 0.05
    assert any("负片物理上不可能" in w for w in r["warnings"])
    prov = r["provenance"]
    assert prov["above_base_under_roll"][0] > 0.15 and prov["drift_decade"][0] > 0.18
