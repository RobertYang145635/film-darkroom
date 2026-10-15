"""回归：**红通道指数与色度增益是一对**，必须联立复验，且留出卷**逐帧**都要变好。

为什么是一对而不是两个独立口味值：抬 `red_ratio` 会把 R 拉到 G 旁边 ⇒ 通道差变小
⇒ 更淡；而增益本身又把中间带 `R−G` 往负方向推（增益放大"偏离亮度"的部分，
而放大前我们偏绿）。所以**单独扫任何一个都会得到相反结论**：
上一轮就是只看饱和把红斜率否掉的（`red_ratio` 1.3/1.6/2.0 时饱和崩到
−0.070/−0.105/−0.120），而它没看的是 R−G 轴——误差一直在那里。
本轮先修掉片基缺陷（B-14③），再按纪律重测被它污染过的结论（B-14⑧）。

实测（`tmp_holdout_pair.py`，留出卷 `1299_G200` 三帧，未参与选值）：

| 条件 | 逐帧 abs(Δ带宽RG) 均值 | 色相变好帧数 | 逐帧 abs(Δsat95) 均值 |
|---|---|---|---|
| 现产 red 1.00 × gain 1.897 | 0.0377 | — | 0.0405 |
| red 1.15 × gain 2.30 | 0.0328 | 3/3 | 0.0366 |
| **red 1.30 × gain 2.50（上线）** | **0.0197** | **3/3** | 0.0366 |

没修好的也照实写：`G200-6` 的 `Δsat95` 停在 **−0.1020** 一动不动（两个旋钮都救不了它），
`1299` 三帧整体仍偏暗 −0.06~−0.12。逐帧饱和散布仍然是开着的缺陷。
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest
from PIL import Image

from negative_lab.calibration.annotations import load_annotation
from negative_lab.io.images import load_linear_rgb
from negative_lab.library.fingerprint import capture_fingerprint
from negative_lab.library.models import FrameRecord, GeometryRegions
from negative_lab.pipeline.density import encode_srgb
from negative_lab.pipeline.model_assets import ModelAsset, canonical_hash
from negative_lab.pipeline.service import process_roll

ROOT = pathlib.Path(__file__).resolve().parents[2]

HOLDOUT_ROLL = "1299_G200"          # manifest split=holdout
MAX_DIM = 1200
BASE_RED, BASE_GAIN = 1.00, 1.897   # 上线前的生产默认，作为对照基准

_map = json.loads((ROOT / "fixtures/references" / "g200_manual_look.json")
                  .read_text(encoding="utf-8"))
REF = {e["frame_id"]: e for e in _map["frames"]
       if not e["variant_of"] and e["frame_id"].startswith(HOLDOUT_ROLL)}
MET = _map["method"]


@pytest.fixture(scope="module")
def inputs():
    missing = [f["reference_file"] for f in REF.values()
               if not (ROOT / f["reference_file"]).exists()]
    if missing:
        pytest.skip(f"用户手工参考大图不在库里（gitignore），缺 {len(missing)} 张，"
                    f"例如 {missing[0]}")
    man = json.loads((ROOT / "fixtures/acceptance/manifest.json")
                     .read_text(encoding="utf-8"))
    frs = sorted([f for f in man["frames"] if f.get("required")
                  and f["roll"] == HOLDOUT_ROLL], key=lambda f: str(f["id"]))
    recs, imgs, kept = [], [], []
    for fr in frs:
        if fr["id"] not in REF or not (ROOT / fr["file"]).exists():
            continue
        loaded = load_linear_rgb(ROOT / fr["file"])
        arr = loaded.rgb
        step = max(1, int(np.ceil(max(arr.shape[:2]) / float(MAX_DIM))))
        img = arr[::step, ::step, :].copy()
        ann = load_annotation(fr["file"])

        def scale(roi, sy, sx):
            out = (int(round(roi[0] * sy)), int(round(roi[1] * sx)),
                   int(round(roi[2] * sy)), int(round(roi[3] * sx)))
            return out if (out[2] - out[0] >= 4 and out[3] - out[1] >= 4) else None

        sy, sx = img.shape[0] / arr.shape[0], img.shape[1] / arr.shape[1]
        geo = GeometryRegions(
            rotation_degrees=ann.rotation_degrees,
            output_crop_roi=scale(ann.output_crop_roi or ann.image_area_roi, sy, sx),
            image_area_roi=scale(ann.image_area_roi or ann.output_crop_roi, sy, sx),
            base_sample_rois=[r for r in (scale(x, sy, sx)
                                          for x in (ann.base_sample_rois or [])) if r])
        recs.append(FrameRecord(source_path=fr["file"],
                                source_sha256=fr["source_sha256"],
                                input_facts=dict(loaded.meta),
                                capture_fingerprint=capture_fingerprint(
                                    loaded.meta, source_path=fr["file"]).value,
                                geometry=geo, orientation_decided=True))
        imgs.append(img)
        kept.append(fr)
    return list(zip(recs, imgs)), kept


def _asset(rr: float, rb: float = 0.7) -> ModelAsset:
    payload = {"density_matrix": np.eye(3).tolist(),
               "channel_curves": [None, None, None],
               "red_ratio": rr, "blue_ratio": rb,
               "reference_exponent": None, "dmax": None}
    return ModelAsset(asset_version="regression-red-gain", payload=payload,
                      sha256=canonical_hash(payload), enabled=True)


def _lin(d):
    c = np.asarray(d, np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _stats(disp01):
    im = Image.fromarray((np.clip(disp01, 0, 1) * 255).round().astype(np.uint8))
    a = np.asarray(im.resize((MET["resize"],) * 2, Image.BILINEAR), np.float64) / 255.0
    c = int(a.shape[0] * (1 - MET["center_crop_frac"]) / 2)
    r = int(a.shape[1] * (1 - MET["center_crop_frac"]) / 2)
    d = a[c:a.shape[0] - c, r:a.shape[1] - r, :]
    L = np.clip(_lin(d), 1e-6, None)
    luma = 0.2126 * d[..., 0] + 0.7152 * d[..., 1] + 0.0722 * d[..., 2]
    lo, hi = MET["mid_tone_band"]
    band = (luma >= lo) & (luma < hi)
    mb = L[band].mean(axis=0) if band.sum() > 400 else np.full(3, np.nan)
    sat = d.max(2) - d.min(2)
    return {"band": mb, "sat95": float(np.percentile(sat, 95))}


def _gaps(inputs, rr: float, gain: float, rb: float = 0.7) -> dict:
    """在留出卷上逐帧量 `Δ带宽RG` 与 `Δsat95`。

    增益走 `RenderIntent.print_chroma_gain`（生产读它的同一个字段），
    所以这里改的是产品的渲染出口，不是另写一份数学。
    """
    from negative_lab.library.render import RenderIntent, apply_render

    frames, kept = inputs
    _, results = process_roll(frames, model_asset=_asset(rr, rb))
    out = {}
    for fr, pr in zip(kept, results):
        if fr["id"] not in REF:
            continue
        name = pr.chosen_candidate or pr.candidates.default_name
        it = RenderIntent.from_dict(dict(pr.candidates.get(name).intent or {}))
        it.print_chroma_gain = float(gain)
        s = _stats(np.clip(encode_srgb(apply_render(
            pr.result.positive_linear, it))[..., :3], 0, 1))
        r = _stats(np.asarray(Image.open(ROOT / REF[fr["id"]]["reference_file"])
                              .convert("RGB"), np.float32) / 255.0)
        out[fr["id"]] = {
            "grg": float((s["band"][0] - s["band"][1]) - (r["band"][0] - r["band"][1])),
            "gbg": float((s["band"][2] - s["band"][1]) - (r["band"][2] - r["band"][1])),
            "gsat": float(s["sat95"] - r["sat95"])}
    return out


def test_shipped_defaults_are_the_measured_pair():
    """默认值本身钉住：改了常数就必须同时改这里的依据。"""
    from negative_lab.print_math import (LOOK_CHROMA_GAIN, LOOK_BLUE_EXPONENT,
                                         LOOK_RED_EXPONENT)
    from negative_lab.pipeline.model_assets import deterministic_asset

    assert LOOK_RED_EXPONENT == pytest.approx(1.30)
    assert LOOK_BLUE_EXPONENT == pytest.approx(0.85), (
        "㊳/㊴ 上线的值是 0.85（三方尺上两卷同向的中间调缺蓝：训 −0.1076→−0.0228、"
        "留 −0.0861→−0.0068）。换值必须同时重钉 "
        "`test_blue_index_is_not_dominated_at_the_shipped_pair` 与 "
        "`test_highlight_desaturation.py` 的两条 ⑲ 锁——它们的主语是这个工作点。")
    assert LOOK_CHROMA_GAIN == pytest.approx(2.50)
    p = deterministic_asset().payload
    assert p["red_ratio"] == pytest.approx(LOOK_RED_EXPONENT), (
        "指数常量在，但一键默认没带上——与 LOOK_BLUE_EXPONENT 的同一条坑")
    assert p["blue_ratio"] == pytest.approx(LOOK_BLUE_EXPONENT)
    assert np.allclose(np.asarray(p["density_matrix"]).reshape(3, 3), np.eye(3)), (
        "跨通道矩阵不许进默认：它至今没有一份通过留出的证据（B-14①/B-13 追加①）")


def test_pair_improves_holdout_hue_on_every_frame(inputs):
    """**逐帧**都要变好：三帧里任何一帧色相没变好，这条锁就该红。"""
    base = _gaps(inputs, BASE_RED, BASE_GAIN)
    shipped = _gaps(inputs, 1.30, 2.50)
    print()
    for fid in sorted(base):
        print(f"  {fid.split('::')[-1]:<9} Δ带宽RG {base[fid]['grg']:+.4f} → "
              f"{shipped[fid]['grg']:+.4f}   abs(Δsat95) {abs(base[fid]['gsat']):.4f} → "
              f"{abs(shipped[fid]['gsat']):.4f}")
    worse = [f for f in base if abs(shipped[f]["grg"]) >= abs(base[f]["grg"])]
    assert not worse, f"留出卷上这些帧色相没有变好：{worse}"
    assert len(base) == 3, f"留出帧数变了（{len(base)}），本锁的逐帧判据要重估"


def test_gain_alone_does_not_fix_hue_but_pair_pays_for_saturation(inputs, monkeypatch):
    """**负控制 + 分工说明**：两个旋钮各管一条轴，别把它们混成"一个口味值"。

    ⚠ **本锁把逐帧电平项钉回 `strength=0` 再量**（B-14⑮ 上线后必须这么做，理由如下）。
    电平 δ 把每帧工作点挪了 −0.054~+0.058 decade，而"中间带"是
    `(0.30≤lum<0.72)` 的**内容统计量**——带里的像素换了一批，于是本锁的
    **0.005 绝对容差是在 δ=0 的工作点上量出来的**。电平上线后当场重测
    （`tmp_pair_tiebreak.py`，11 帧、384 量法、red 1.30 × blue 0.70）：

    | gain | 有符号 Δsat95 | 有符号 Δ带宽RG | 逐帧 abs(Δ带宽RG) | 四统计距离 |
    |---|---|---|---|---|
    | 1.897 | −0.0848 | +0.0271 | 0.0505 | 0.0600 |
    | 2.200 | −0.0471 | +0.0158 | 0.0522 | **0.0559** |
    | **2.500（现产）** | **−0.0114** | **+0.0041** | 0.0548 | 0.0565 |

    训练卷 1298 在 2.20 与 2.50 之间**分不开**（|RG|+|sat| 0.1597 对 0.1600，差 0.2%），
    而**有符号**的两条轴（平均饱和、平均色相偏）都指着 2.50 离零最近
    ⇒ 不挪已上线的常数（挪到 2.20 会把"我们比他淡"重新拉开 0.036，
    而那正是当初引入增益要修的那个"不够像"）。逐帧 abs 更爱低增益是**散布**，
    散布的最大贡献是 `1298::G200-7` 一帧（Δ带宽RG **+0.2439**，其余帧 ≤0.07）。
    ⇒ 本锁继续在 δ=0 的条件下量它原本要量的东西；
    **δ 开着时的色相账由 `test_level_absolute.py` 负责**。

    实测（δ=0，留出三帧，`-s` 打印）：
    * 只动增益（1.00×2.50）色相 **0.0993**，比现产 0.0377 **更糟**
      ⇒ 增益救不了色相，它只会把已有的偏绿放大。
    * 只动斜率（1.30×1.897）色相 0.0192，与联立 0.0197 同量级
      ⇒ **色相这一项确实是斜率管的**，本锁不假装"联立"在色相上更优。
    * 但只动斜率的饱和更差，联立才把它拉回来
      ⇒ **增益的存在是为了付斜率欠下的饱和账**，这才是"一对"的真实含义。
    上一轮的错误正是只看饱和去评一个管色相的旋钮，于是把它否了。
    """
    monkeypatch.setattr("negative_lab.print_math.PRINT_LEVEL_STRENGTH", 0.0)
    #: ㉓ 之后电平有**两个**携带者（密度中位 + 高调面积），只钉第一个不等于回到 δ=0——
    #: 本锁就是这样变红的（0.0216 → 0.0283 越过 0.005）。两个都钉才算关掉这一项。
    monkeypatch.setattr("negative_lab.print_math.PRINT_LEVEL_AREA_STRENGTH", 0.0)
    base = _gaps(inputs, BASE_RED, BASE_GAIN)
    gain_only = _gaps(inputs, BASE_RED, 2.50)
    slope_only = _gaps(inputs, 1.30, BASE_GAIN)
    pair = _gaps(inputs, 1.30, 2.50)
    m = lambda d, k="grg": float(np.mean([abs(v[k]) for v in d.values()]))  # noqa: E731
    print(f"\n  abs(Δ带宽RG) 均值：现产 {m(base):.4f} / 只动增益 {m(gain_only):.4f} / "
          f"只动斜率 {m(slope_only):.4f} / 联立 {m(pair):.4f}")
    print(f"  abs(Δsat95) 均值：现产 {m(base, 'gsat'):.4f} / 只动斜率 "
          f"{m(slope_only, 'gsat'):.4f} / 联立 {m(pair, 'gsat'):.4f}")
    assert m(gain_only) > m(base), (
        f"只动增益就把色相修好了（{m(gain_only):.4f} ≤ {m(base):.4f}）："
        "分工的判断错了，增益不是饱和旋钮而是色相旋钮")
    assert m(slope_only) < m(base), "只动斜率没修好色相 ⇒ 斜率不管色相，分工判断错"
    assert m(pair) <= m(slope_only) + 0.005, (
        f"联立把色相让掉了（{m(slope_only):.4f} → {m(pair):.4f}）："
        "增益在吃掉斜率的成果，这一对不能一起上")
    assert m(pair, "gsat") < m(slope_only, "gsat"), (
        f"增益没把斜率欠的饱和账付回来（{m(slope_only, 'gsat'):.4f} → "
        f"{m(pair, 'gsat'):.4f}），那就不该动增益")


def test_negative_control_pushing_red_the_other_way_is_worse(inputs):
    """**负控制**：把红斜率往**反方向**压（0.85）必须更糟，
    否则"变好"只是任意扰动。"""
    base = _gaps(inputs, BASE_RED, BASE_GAIN)
    wrong = _gaps(inputs, 0.85, BASE_GAIN)
    b0 = float(np.mean([abs(v["grg"]) for v in base.values()]))
    bw = float(np.mean([abs(v["grg"]) for v in wrong.values()]))
    print(f"\n  反向 red_ratio=0.85：abs(Δ带宽RG) {b0:.4f} → {bw:.4f}（必须更糟）")
    assert bw >= b0 - 1e-4, (
        f"反方向也把它压小了（{b0:.4f} → {bw:.4f}）：该量对任意扰动都反应，"
        "正锁没有区分力")


def test_blue_index_is_not_dominated_at_the_shipped_pair(inputs):
    """**蓝指数不被低值支配**（㊴ 换主语：从"留在 0.70"改成"钉住当前上线值"）。

    旧版本钉的是 0.70，依据是 `tmp_blue_resweep.py`（red 固定 1.30 扫 blue × gain）：
    标定卷 0.55 更好（0.0366 对 0.0425）、留出卷 0.55 更差（0.0276 对 0.0180）
    ⇒ 0.70 当时是"没有被别的值在两个 split 上同时打败"的取值。

    ㊳ 用**三方尺**（我方 / NLP / 他的样片，11 帧）重扫之后换成 0.85，
    这把尺（留出卷逐帧 `abs(Δ带宽RG)+abs(Δ带宽BG)` 均值）当场打印：
    0.55 **0.1246** / 0.70 **0.0645** / 0.85 **0.0327**
    ⇒ 0.55 现在在**两个 split 上都被打败**，"不被支配"这个关系仍然成立而且更硬。
    断言把比较锚在**常量本身**上：谁把常量挪到低值，这一条就红。
    """
    from negative_lab.print_math import LOOK_BLUE_EXPONENT

    at_ship = _gaps(inputs, 1.30, 2.50, rb=float(LOOK_BLUE_EXPONENT))
    at_055 = _gaps(inputs, 1.30, 2.50, rb=0.55)
    at_070 = _gaps(inputs, 1.30, 2.50, rb=0.70)
    m = lambda d: float(np.mean([abs(v["grg"]) + abs(v["gbg"]) for v in d.values()]))  # noqa: E731
    m_ship, m055, m070 = m(at_ship), m(at_055), m(at_070)
    print(f"\n  留出卷 abs(ΔRG)+abs(ΔBG) 均值：blue {float(LOOK_BLUE_EXPONENT):.2f} "
          f"（现产）{m_ship:.4f} / 0.70（它替换掉的值）{m070:.4f} / 0.55 {m055:.4f}")
    assert m055 > m_ship, (
        f"blue 0.55 在留出卷上更好（{m055:.4f} < {m_ship:.4f}）："
        f"'保留 {float(LOOK_BLUE_EXPONENT):.2f}' 的依据没了，重扫 blue 并改默认值，"
        "别改这条锁的阈值")
    #: ㊳ 的选值账：新默认必须打赢**它替换掉的那个值**（0.1246/0.0645/0.0327 三档里
    #: 0.85 最低）。若有人把常量挪回 0.70 或更高，这一条会红——那是要求他重新拿证据，
    #: 不是豁免。
    assert m_ship < m070, (
        f"现产 blue {float(LOOK_BLUE_EXPONENT):.2f} 没有打赢它替换掉的 0.70"
        f"（{m_ship:.4f} vs {m070:.4f}）⇒ 换值的依据没了")


@pytest.mark.xfail(
    strict=True,
    reason="㉝ 之后**不承重**、㊳/㊴ 换成 0.85 之后**理由换了**（下面这组数 2026-10-16 当场重印）："
           "留出三帧逐帧 `abs(Δ带宽RG)+abs(Δ带宽BG)`，blue 0.55/0.70/0.85 = "
           "G200-2 0.2008/0.1188/0.0667、G200-6 0.0742/0.0222/0.0082、"
           "G200-7 0.0988/0.0524/0.0232（均值 0.1246/0.0645/**0.0327**）"
           "⇒ 相对 0.70，0.85 让**三帧全部**朝参考走。但'每帧都比**完全不做蓝校正**更好'"
           "仍然做不到：对照 blue=1.00 时 G200-2 0.0663 → 0.85 的 0.0667（差 4e-4，原地不动）、"
           "G200-6 0.0566 → **0.0082**（大幅变好）、G200-7 0.0022 → **0.0232**（变差）——"
           "`G200-7` 不做校正时本来就几乎正好，任何全局蓝增益都只能把它推离。"
           "⇒ '一个全局常数服务所有帧'在本语料上是假的（**第五次**确认，㉗/㉚/㉛/㊳ 之后）。"
           "strict=True：哪天真做到每帧都变好，这条会 XPASS 报错，必须回来删标记，"
           "不许留着当豁免。")
def test_blue_index_improves_every_holdout_frame_at_the_product_point(inputs):
    """**产品工作点上的蓝指数锁**（电平项开着）。

    ⚠ **当前 xfail**（理由见装饰器）：0.85 让留出三帧全部比 0.70 更接近参考，
    但仍然做不到"比完全不校正更好"——`G200-7` 那一帧是全局常数的结构性受害者。

    上面那条"每帧都变好"的旧锁在 `red 1.00 × gain 1.00` 那个**隔离角**上量，
    B-14⑮ 的电平 δ 一上线它就红了（`1299::G200-2` 0.0141 → 0.0230）——
    那是工作点变了，不是蓝指数错了。所以这里把蓝指数放回**它真正被使用的地方**
    （`red 1.30 × gain 2.50`，电平开着）逐帧重钉：这才是"用户一键出片时蓝指数有没有用"。

    实测（`tmp_pair_refit_after_level.py` 阶段二，red 1.30 × gain 2.20 那一格）：
    blue 1.00 → 0.70 时留出三帧的 `Δ带宽RG` 逐帧为
    `−0.0339 → −0.0279`、`+0.0120 → +0.0113`、`−0.0546 → −0.0452`，
    `|ΔRG|` 均值 0.0335 → **0.0281**、`|ΔBG|` 0.0678 → **0.0413**、距离 0.0514 → **0.0477**。
    """
    from negative_lab.print_math import LOOK_BLUE_EXPONENT

    at_100 = _gaps(inputs, 1.30, 2.50, rb=1.00)
    at_ship = _gaps(inputs, 1.30, 2.50, rb=float(LOOK_BLUE_EXPONENT))
    print()
    for fid in sorted(at_100):
        v1 = abs(at_100[fid]["grg"]) + abs(at_100[fid]["gbg"])
        v2 = abs(at_ship[fid]["grg"]) + abs(at_ship[fid]["gbg"])
        print(f"  {fid.split('::')[-1]:<10} 向量差 blue1.00 {v1:.4f} → "
              f"blue{float(LOOK_BLUE_EXPONENT):.2f} {v2:.4f}")
    worse = [f for f in at_100
             if abs(at_ship[f]["grg"]) + abs(at_ship[f]["gbg"])
             >= abs(at_100[f]["grg"]) + abs(at_100[f]["gbg"])]
    assert not worse, f"产品工作点上蓝指数让这些帧没变好：{worse}"
    assert len(at_100) == 3, f"留出帧数变了（{len(at_100)}），逐帧判据要重估"
