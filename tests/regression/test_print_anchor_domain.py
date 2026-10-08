"""回归：打印锚点的**取样域必须和施加域一致**，在真实底片上量。

单元侧的锁（`tests/unit/test_print_scale.py::test_black_anchor_really_bites_in_its_own_domain`）
在合成帧上只能证明"逻辑自洽"，而域错配这件事**只在真实底片上才现形**：
合成帧两域的 p5 实测只差 0.86 个百分点（5.86% 对 5%），因为它的三通道密度
是对称随机的，而真实负片的密度 luma 有一段**负尾**（片基取的是区域均值，
画面里必然有比片基更亮的像素），那段负尾在纸级形式 `max(0, …)`
（`pipeline/density.py:132`）处被截成 0，两个域就此分道。

实测（`process_roll`，标定卷，`outputs/ref_look_probe/print_reach.txt`）：
G200-1 名义 p5 以下真实密度占 5.00%，重映射反解域只认 **0.10%**；
G200-2 是 5.00% 对 **0.00%**；G200-8 同样是 5.00% 对 **0.00%**。
也就是说曲线接上以后黑点根本没咬住——这正是"一键出片还带一层奶白雾"
在 B-11 修完之后仍残留 `p05 +0.110` 的原因。

本文件跑出来的数（2026-10-08，`-s` 打印后取下的原值，`process_image` 帧级片基）：
修正域后**看得见的中心区**里真印到黑的比例 = G200-1 **5.026%** /
G200-2 **5.040%** / G200-8 **5.006%**（名义 p5=5.0%）；
把旧域三点喂给同一个重映射 = **1.841% / 0.005% / 3.388%**。
中心区 p5 相对整幅画面区 p5 的抬高 = **+0.075 / +0.014 / +0.355 decade**。

第二件事：**黑点由谁决定**。中心构图取样与整幅画面区取样的 p5 差
G200-1 `+0.075`、G200-2 `+0.014`、G200-8 `+0.355` decade（标定卷 9 帧里
7 帧 Δ>0，`outputs/ref_look_probe/`），最暗的 5% 落在画面**外沿**
（翻拍架照度 falloff、边缘灰尘、片基不均）。拿它当黑点就等于让边角决定
主体的影调：G200-8 中心构图 `p05=0.433`（参考 `0.193`），改成中心取样后
实测 `0.190`。所以生产的取样区是 `centre_region(valid_mask)`。
"""

from __future__ import annotations

import importlib.util
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from negative_lab.io.images import load_linear_rgb
from negative_lab.pipeline.print_scale import (P_LOW, centre_region, density_anchor,
                                               picture_density, print_remap_array)
from negative_lab.pipeline.service import process_image

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "fixtures" / "acceptance" / "manifest.json"
_KIT = ROOT / "scripts" / "build_visual_contact_sheet.py"
_LUM = np.array([0.2126, 0.7152, 0.0722])


def _kit():
    """取验收套具自己的几何解释（方向来自标注）——判定不许有第二条口径。"""
    spec = importlib.util.spec_from_file_location("bvcs_for_anchor_domain", _KIT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
#: 这三帧是当初量出域错配的具体样本（不是挑出来的"好看的"帧——
#: 挑的标准恰恰是"错配最严重"，见模块文档）。
CASE_IDS = ["1298_G200::G200-1", "1298_G200::G200-2", "1298_G200::G200-8"]


def _cases():
    if not MANIFEST.exists():
        return []
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    out = []
    for fid in CASE_IDS:
        fr = next((f for f in man["frames"] if f["id"] == fid), None)
        if fr is None or not (ROOT / fr["file"]).exists():
            continue
        out.append(fr)
    return out


CASES = _cases()
pytestmark = pytest.mark.skipif(not CASES, reason="无验收语料（missing-fixture）")


@lru_cache(maxsize=4)
def _run(fid: str):
    fr = next(f for f in CASES if f["id"] == fid)
    kit = _kit()
    t = load_linear_rgb(ROOT / fr["file"]).rgb
    step = max(1, int(np.ceil(max(t.shape[:2]) / 1600)))
    small = t[::step, ::step, :].copy()
    geo = kit._downscale_geometry(kit._geometry(fr), t.shape[:2], small.shape[:2])
    if not geo.is_complete():
        geo = type(geo)(rotation_degrees=geo.rotation_degrees)
    return process_image(small, geometry=geo, ann=None), step


def _region(pr):
    return centre_region(pr.valid_mask, shape=pr.result.positive_linear.shape[:2])


def _black_fraction(pr, anchor, sel) -> float:
    """施加打印后，**sel 区域内**真印到黑（线性亮度 ≤1e−4）的像素占比。"""
    out, info = print_remap_array(pr.result.positive_linear, anchor,
                                  pr.report.params.dmax, pr.valid_mask)
    assert info is not None, "锚点合法却未被采用"
    lum = np.clip(out, 0, 1)[..., :3] @ _LUM.astype(np.float32)
    return float((lum[sel] <= 1e-4).mean())


@pytest.mark.parametrize("fr_id", [c["id"] for c in CASES])
def test_anchor_percentile_really_prints_black_on_real_frames(fr_id):
    """**锁**：锚点说"看得见的这块最暗的 5% 印成纯黑"，那块里就必须真有 5% 是黑的。

    这条不是同义反复：它同时穿过取样域、反解、`np.interp`、统一增益与天花板
    钳位——上一版的错就藏在"取样域 ≠ 反解域"，任何一处域不一致都会让它红。
    取样与计数用同一个区域（生产里取样是中心区、施加是整个画面区）。
    """
    pr, _ = _run(fr_id)
    sel = _region(pr)
    samples = picture_density(pr.result.positive_linear, pr.report.params.dmax, sel)
    a = density_anchor(samples)
    assert a is not None, "该帧应当有合法锚点"
    frac = _black_fraction(pr, a, sel)
    nominal = P_LOW / 100.0
    assert abs(frac - nominal) < 0.005, (
        f"{fr_id}：锚定 p{P_LOW} 只印出 {frac:.3%} 的黑，名义 {nominal:.1%}")


@pytest.mark.parametrize("fr_id", [c["id"] for c in CASES])
def test_old_density_luma_domain_is_the_negative_control(fr_id):
    """**负控制**：把"逐通道密度的 luma"三点（上一版的做法）喂给同一个重映射，
    真印到黑的比例必须**远低于**名义 p5——否则上面那条锁是空转的。

    这个量必须用被测代码自己的公式现场算（`−log10(t/base)` 与 luma 权重），
    不许引用文档里的数字。
    """
    pr, _ = _run(fr_id)
    base = np.asarray(pr.report.params.dmin, dtype=np.float64)
    assert np.isfinite(base).all() and (base > 0).all(), "片基非法，本帧无法比较"
    t = np.asarray(pr.t, dtype=np.float64)[..., :3]
    with np.errstate(divide="ignore", invalid="ignore"):
        d = -np.log10(np.clip(t, 1e-6, None) / base)
    dl = (d @ _LUM)
    sel = _region(pr)
    dl_m = dl[sel]
    dl_m = dl_m[np.isfinite(dl_m)]
    old_triple = tuple(float(np.percentile(dl_m, q)) for q in (P_LOW, 50.0, 95.0))
    frac_old = _black_fraction(pr, old_triple, sel)
    nominal = P_LOW / 100.0
    assert frac_old < 0.8 * nominal, (
        f"{fr_id}：旧域锚点 {np.round(old_triple, 3).tolist()} 仍印出 "
        f"{frac_old:.3%} 的黑（名义 {nominal:.1%}），两域错配已消失——"
        f"正锁失去意义，须重估")


@pytest.mark.parametrize("fr_id", [c["id"] for c in CASES])
def test_visible_region_is_not_the_same_as_the_whole_image_area(fr_id):
    """**锁：黑点该由看得见的构图决定**。中心区 p5 必须不低于整幅 p5，
    且至少在多数帧上高出可感知的量——因为最暗的那 5% 实际落在画面外沿
    （翻拍架照度 falloff / 边缘灰尘 / 片基不均）。

    实测（本文件 `-s` 打印）：Δdecade G200-1 +0.075、G200-2 +0.014、
    G200-8 +0.355；全卷 9 帧里 7 帧 Δ>0。同一帧改成中心取样后
    中心构图 `p05` 从 `0.433` 变 `0.190`，而用户手工参考是 `0.193`。
    """
    pr, _ = _run(fr_id)
    tech, dmax = pr.result.positive_linear, pr.report.params.dmax
    lo_all = density_anchor(picture_density(tech, dmax, pr.valid_mask))
    lo_cen = density_anchor(picture_density(tech, dmax, _region(pr)))
    assert lo_all is not None and lo_cen is not None, (
        f"{fr_id}：两种取样都该有锚点，否则比较无从做起（lo_all={lo_all} "
        f"lo_cen={lo_cen}）——需改用别的帧")
    assert lo_cen[0] >= lo_all[0] - 1e-9, (
        f"{fr_id}：中心区黑点反而更低（{lo_cen[0]:.3f} < {lo_all[0]:.3f}），"
        "说明最暗像素不在外沿，本规则的前提在这帧上不成立，须重估")


def test_outer_ring_holds_the_dark_tail_on_most_frames():
    """多数帧的**最暗 5% 落在外沿**：中心区 p5 比整幅 p5 高出的量必须实测可见，
    否则上一条锁只是"两种算法碰巧一样"。"""
    deltas = []
    for c in CASES:
        pr, _ = _run(c["id"])
        tech, dmax = pr.result.positive_linear, pr.report.params.dmax
        a = density_anchor(picture_density(tech, dmax, pr.valid_mask))
        b = density_anchor(picture_density(tech, dmax, _region(pr)))
        if a is None or b is None:
            continue
        deltas.append(b[0] - a[0])
    assert len(deltas) >= 3, f"可比帧太少：{len(deltas)}"
    big = sum(1 for x in deltas if x > 0.05)
    assert big >= max(2, len(deltas) // 2), (
        f"外沿携带暗部的帧太少：Δ={np.round(deltas, 3).tolist()}")
