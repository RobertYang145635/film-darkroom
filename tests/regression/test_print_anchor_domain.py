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


# ---------------------------------------------------------------- 色彩放大闸门
#: 片基未实测 ⇒ 不放大色彩（B-11 追加⑩）。这两帧是实测挑出来的对照：
#: `1299_G200::G200-2` base_source=generic_prior 置信 0.0，
#: `1298_G200::G200-1` base_source=frame 置信 0.79。
UNCALIBRATED = "1299_G200::G200-2"
CALIBRATED = "1298_G200::G200-1"


@lru_cache(maxsize=4)
def _run_id(fid: str):
    """按 frame_id 跑一帧（清单里的 ROI 不完整时退回自动检测，与套具同口径）。"""
    kit = _kit()
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    fr = next((f for f in man["frames"] if f["id"] == fid), None)
    if fr is None:
        raise AssertionError(f"验收清单里没有 {fid}，本锁的前提已失效")
    t = load_linear_rgb(ROOT / fr["file"]).rgb
    step = max(1, int(np.ceil(max(t.shape[:2]) / 1200)))
    small = t[::step, ::step, :].copy()
    geo = kit._downscale_geometry(kit._geometry(fr), t.shape[:2], small.shape[:2])
    if not geo.is_complete():
        geo = type(geo)(rotation_degrees=geo.rotation_degrees)
    from negative_lab.pipeline.service import process_image
    return process_image(small, geometry=geo, ann=None)


def _sat_p95(linear) -> float:
    """**显示域**饱和 p95（与用户参考同一口径），所以先过 sRGB 编码。"""
    from PIL import Image as _I

    from negative_lab.pipeline.density import encode_srgb
    disp = np.asarray(encode_srgb(linear), np.float32)[..., :3]
    im = _I.fromarray((np.clip(disp, 0, 1) * 255).round().astype(np.uint8))
    d = np.asarray(im.resize((384, 384), _I.BILINEAR), np.float32) / 255.0
    c = int(d.shape[0] * 0.15)
    d = d[c:d.shape[0] - c, c:d.shape[1] - c, :]
    return float(np.percentile(d.max(2) - d.min(2), 95))


def _force_generic(monkeypatch):
    """把"片基没实测"这个**状态**造出来：只改写 `resolve_base` 的出口标签。

    为什么必须这么干：这条锁要钉的是 `service` 第 9b 步那道闸
    （`gain_off = 1.0 if base_source == "generic_prior" else 0.0`），
    而自 2026-10-11 起，本语料里**已经没有一帧会真的落到通用先验**
    （画面内橙色亮区实测接管了它们），合成帧也不行——自动检测在干净的
    渐变图上照样能框出一块"片基"。所以只能闸的输入端造这个状态。
    **这条锁的作用域因此是"闸读到的来源标签"，不是"估计器会不会失败"**；
    后者由 `tests/unit/test_orange_base.py::test_画面内也测不到时才退回通用先验`
    在真实调用栈上钉住。
    """
    import negative_lab.pipeline.service as svc

    real = svc.resolve_base

    def fake(*a, **kw):
        r = real(*a, **kw)
        return {**r, "base_source": "generic_prior", "confidence": 0.0}

    monkeypatch.setattr(svc, "resolve_base", fake)


def test_unmeasured_base_does_not_amplify_chroma(monkeypatch):
    """**锁**：片基没实测的帧，intent 里的色度增益必须是 1.0，且成品饱和
    不得高于用默认 1.897 时的饱和。

    **前提的构造方式在 2026-10-11 换过一次**（断言一个字没松）：原来直接吃
    `1299_G200::G200-2` 单帧路径天然落到 `generic_prior` 这个事实，而本轮上线
    画面内橙色亮区实测（B-14③）之后**这个事实没了**——那帧现在测得到片基。
    见 `_force_generic` 的作用域说明。

    历史依据（`outputs/ref_look_probe/base_and_gain.txt`，2026-10-08）：
    `1299::G200-2` 技术正片饱和 p95 0.902，打印后不放大是 0.537、放大是 **0.910**，
    用户参考 0.322；`1299::G200-7` 是 0.341 对 **0.769**，参考 0.357。
    放大系数是对着**已标定**的帧拟合出来的口味值，乘到一片不知道错在哪的颜色上
    只会把错误放大。
    """
    import negative_lab.print_math as pm

    _force_generic(monkeypatch)
    pr = _run_id.__wrapped__(UNCALIBRATED)
    assert pr.base_source == "generic_prior", (
        f"{UNCALIBRATED} 现在片基来源是 {pr.base_source}，闸门的输入端没被造出来")
    it = pr.render_intent
    assert it.print_anchor, "该帧应有打印锚点，否则增益根本不会施加"
    assert float(it.print_chroma_gain) == 1.0, (
        f"未实测片基的帧仍在放大色彩：{it.print_chroma_gain}")
    assert pr.meta["print_scale"]["chroma_gain"] == 1.0
    assert "未实测" in pr.meta["print_scale"]["chroma_gain_reason"]
    tech = pr.result.positive_linear
    off, i1 = pm.print_remap_array(tech, tuple(it.print_anchor), it.print_dmax,
                                   chroma_gain=1.0)
    on, i2 = pm.print_remap_array(tech, tuple(it.print_anchor), it.print_dmax,
                                  chroma_gain=pm.LOOK_CHROMA_GAIN)
    assert i1 is not None and i2 is not None
    assert _sat_p95(off) <= _sat_p95(on) + 1e-6, (
        "闸门没起作用：不放大反而更饱和")


def test_gate_keys_on_measurement_not_on_manual_box(monkeypatch):
    """**负控制**：闸门只认"测没测到"，不认"有没有人工框"。

    同一帧、同一个 `_run_id`，只把 `_force_generic` 拿掉 ⇒ 来源是实测的
    （`frame`/`roll`/`content_orange` 之一），增益就必须回到默认放大。
    没有这条，上一条锁可以靠"这帧根本没有打印锚点"之类的旁路蒙过去。
    """
    pr_plain = _run_id.__wrapped__(UNCALIBRATED)
    assert pr_plain.base_source != "generic_prior", (
        "对照帧现在也落到通用先验了，上一条锁与这条都失去区分力")
    assert float(pr_plain.render_intent.print_chroma_gain) == 0.0, (
        "0.0 是'用模块默认增益'的写法；实测帧被写成 1.0 说明闸门在误伤")
    assert pr_plain.meta["print_scale"]["chroma_gain"] != 1.0

    _force_generic(monkeypatch)
    pr_forced = _run_id.__wrapped__(UNCALIBRATED)
    assert float(pr_forced.render_intent.print_chroma_gain) == 1.0, (
        "同一帧只改来源标签，闸门却没反应——它读的不是 base_source")


def test_measured_base_still_gets_the_default_gain():
    """**负控制**：片基实测过的帧**不许**被这条规则误伤——
    增益保持默认（intent 里写 0.0 表示"用模块默认"），成品饱和应当高于不放大时。
    """
    import negative_lab.print_math as pm

    pr = _run_id(CALIBRATED)
    assert pr.base_source != "generic_prior", (
        f"{CALIBRATED} 片基来源退化成 {pr.base_source}，对照帧失效")
    assert float(pr.render_intent.print_chroma_gain) == 0.0
    assert float(pr.meta["print_scale"]["chroma_gain"]) == pm.LOOK_CHROMA_GAIN
    it = pr.render_intent
    assert it.print_anchor, "该帧应有锚点"
    off, _i1 = pm.print_remap_array(pr.result.positive_linear,
                                    tuple(it.print_anchor), it.print_dmax,
                                    chroma_gain=1.0)
    on, _i2 = pm.print_remap_array(pr.result.positive_linear,
                                   tuple(it.print_anchor), it.print_dmax,
                                   chroma_gain=pm.LOOK_CHROMA_GAIN)
    assert _sat_p95(on) > _sat_p95(off), "标定帧的增益被削弱了：这条规则越界了"


def test_chroma_gain_survives_the_intent_round_trip():
    """决定必须**进 sidecar 且可复算**：0/1.0/自定义值都要原样回来，
    坏数据退回 0.0（= 用默认），不许猜。"""
    from negative_lab.library.render import RenderIntent

    for v in (0.0, 1.0, 1.4):
        it = RenderIntent(print_anchor=(0.1, 0.5, 1.0), print_dmax=1.2,
                          print_chroma_gain=v)
        back = RenderIntent.from_dict(it.to_dict())
        assert float(back.print_chroma_gain) == v, (v, back.to_dict())
    bad = RenderIntent.from_dict({"print_anchor": [0.1, 0.5, 1.0],
                                  "print_dmax": 1.2,
                                  "print_chroma_gain": "很饱和"})
    assert float(bad.print_chroma_gain) == 0.0
