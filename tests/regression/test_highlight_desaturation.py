"""回归：**高光残留色罩**（"水是一片死青"）必须由纸级末端的色度收回来治，
且这条锁要用**对通道敏感**的统计量——上一轮就是拿显示亮度把它错杀的。

分层测量（`tmp_highlight_cast.py`，11 帧，显示亮度最高 5% 的像素三通道均值）：

| 层 | `1299` 三帧 R/G/B | 彩度 |
|---|---|---|
| 技术正片（打印前） | 0.849~0.860 / 0.908~0.918 / 0.971~0.987 | 0.122~0.128 |
| 本项之前（成品） | 0.665~0.688 / 0.808~0.831 / 0.903~0.920 | **0.226~0.249** |
| 用户手工参考 | 0.964~0.991 / 0.951~0.990 / 0.933~0.993 | **0.006~0.031** |

⇒ 反相出来的白本来接近中性，是 `LOOK_CHROMA_GAIN=2.50` 在高光处继续翻倍放大、
再被纸级天花板整像素同除砍掉 R 造成的。上线 `k_eff` 在纸级 85% 之后线性收到 0.80
（依据与代价全部写在 `print_math` 那两条常量上）。

**为什么必须新写一条锁而不是改旧的**：⑰ 判否用的 `|Δp95|` 与四统计距离
**对通道配比不敏感**（luma 由 G/B 主导，R 掉 0.19 它几乎不动）——
那是纪律第 4 条的第二个形态。本文件因此**以"最高 5% 像素的最低通道差"为主判据**，
并把已知代价（训练卷四统计距离 +0.0017、Δsat95 −0.007）钉成**预算**而不是忽略。

harness 与 ⑮ 相同：生产整卷两遍 → 离线重渲逐比特 == `candidate_positive`。
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
from negative_lab.library.render import RenderIntent, apply_render
from negative_lab.pipeline.density import encode_srgb
from negative_lab.pipeline.service import candidate_positive, process_roll
from negative_lab.print_math import (PRINT_CHROMA_HIGHLIGHT_GAIN,
                                     PRINT_CHROMA_HIGHLIGHT_START, LOOK_CHROMA_GAIN)

ROOT = pathlib.Path(__file__).resolve().parents[2]
MAX_DIM = 1200
ROLLS = ("1298_G200", "1299_G200")
HOLDOUT = "1299_G200"
LUM_W = np.array([0.2126, 0.7152, 0.0722])
# 已知代价的预算（量死时写进断言，不许事后放宽）
TRAIN_DIST_BUDGET = 0.0060      # 训练卷四统计距离允许比"关掉这一项"差这么多
SAT_BUDGET = 0.0150             # Δsat95 允许更负这么多

_map = json.loads((ROOT / "fixtures/references" / "g200_manual_look.json")
                  .read_text(encoding="utf-8"))
MET = _map["method"]
REF = {e["frame_id"]: e for e in _map["frames"] if not e["variant_of"]}


def _crop(disp01):
    im = Image.fromarray((np.clip(disp01, 0, 1) * 255).round().astype(np.uint8))
    a = np.asarray(im.resize((MET["resize"],) * 2, Image.BILINEAR), np.float64) / 255.0
    c = int(a.shape[0] * (1 - MET["center_crop_frac"]) / 2)
    r = int(a.shape[1] * (1 - MET["center_crop_frac"]) / 2)
    return a[c:a.shape[0] - c, r:a.shape[1] - r, :]


def _hi(disp01):
    """最高 5% 亮度像素的三通道均值、最低通道与彩度——**对通道配比敏感**的那把尺。"""
    d = _crop(disp01)
    lum = d @ LUM_W
    m = lum >= float(np.quantile(lum, 0.95))
    ch = d[m].mean(axis=0)
    return {"min_ch": float(ch.min()), "R": float(ch[0]), "G": float(ch[1]), "B": float(ch[2]),
            "chroma": float((d[m].max(1) - d[m].min(1)).mean())}


def _full(disp01):
    d = _crop(disp01)
    lum = d @ LUM_W
    sat = d.max(2) - d.min(2)
    return {"mean": float(lum.mean()), "p05": float(np.percentile(lum, 5)),
            "p95": float(np.percentile(lum, 95)), "std": float(lum.std()),
            "sat95": float(np.percentile(sat, 95))}


def _dist(s, r):
    return float(np.mean([abs(s["mean"] - r["mean"]), abs(s["p05"] - r["p05"]),
                          abs(s["std"] - r["std"]), abs(s["sat95"] - r["sat95"])]))


@pytest.fixture(scope="module")
def corpus():
    missing = [f["reference_file"] for f in REF.values()
               if not (ROOT / f["reference_file"]).exists()]
    if missing:
        pytest.skip(f"用户手工参考大图不在库里（gitignore），缺 {len(missing)} 张")
    man = json.loads((ROOT / "fixtures/acceptance" / "manifest.json")
                     .read_text(encoding="utf-8"))
    rows, mismatches = [], []
    for roll in ROLLS:
        frs = sorted([f for f in man["frames"] if f.get("required") and f["roll"] == roll],
                     key=lambda f: str(f["id"]))
        recs, imgs, kept = [], [], []
        for fr in frs:
            if not (ROOT / fr["file"]).exists():
                continue
            loaded = load_linear_rgb(ROOT / fr["file"])
            arr = loaded.rgb
            step = max(1, int(np.ceil(max(arr.shape[:2]) / float(MAX_DIM))))
            img = arr[::step, ::step, :].copy()
            ann = load_annotation(fr["file"])
            if ann is not None and (ann.image_area_roi or ann.base_sample_rois):
                sy, sx = img.shape[0] / arr.shape[0], img.shape[1] / arr.shape[1]

                def scale(roi):
                    o = (int(round(roi[0] * sy)), int(round(roi[1] * sx)),
                         int(round(roi[2] * sy)), int(round(roi[3] * sx)))
                    return o if (o[2] - o[0] >= 4 and o[3] - o[1] >= 4) else None

                geo = GeometryRegions(
                    rotation_degrees=ann.rotation_degrees,
                    output_crop_roi=scale(ann.output_crop_roi or ann.image_area_roi),
                    image_area_roi=scale(ann.image_area_roi or ann.output_crop_roi),
                    base_sample_rois=[q for q in (scale(x)
                                                  for x in (ann.base_sample_rois or [])) if q])
            else:
                geo = GeometryRegions()
            recs.append(FrameRecord(source_path=fr["file"],
                                    source_sha256=fr["source_sha256"],
                                    input_facts=dict(loaded.meta),
                                    capture_fingerprint=capture_fingerprint(
                                        loaded.meta, source_path=fr["file"]).value,
                                    geometry=geo, orientation_decided=ann is not None))
            imgs.append(img)
            kept.append(fr)
            del arr, loaded
        _, results = process_roll(list(zip(recs, imgs)))
        for fr, pr in zip(kept, results):
            if fr["id"] not in REF:
                continue
            name = pr.chosen_candidate or pr.candidates.default_name
            it = RenderIntent.from_dict(dict(pr.candidates.get(name).intent or {}))
            if len(it.print_anchor) != 3:
                continue
            prod = np.clip(encode_srgb(apply_render(pr.result.positive_linear, it))[..., :3],
                           0, 1)
            if not np.array_equal(prod, np.clip(
                    encode_srgb(candidate_positive(pr, name))[..., :3], 0, 1)):
                mismatches.append(fr["id"])
            ref_arr = np.asarray(Image.open(ROOT / REF[fr["id"]]["reference_file"])
                                 .convert("RGB"), np.float32) / 255.0
            rows.append({"id": fr["id"], "tech": pr.result.positive_linear, "it": it,
                         "gain": float(it.print_chroma_gain or LOOK_CHROMA_GAIN),
                         "hi": _hi(prod), "full": _full(prod),
                         "hi_ref": _hi(ref_arr), "full_ref": _full(ref_arr)})
        del results
    return {"rows": rows, "mismatches": mismatches}


def _rerender(corpus, k_hi):
    """把高光档换成 `k_hi` 重渲一遍（`chroma_gain_at` 运行时读模块全局）。"""
    import negative_lab.print_math as pm

    orig = pm.PRINT_CHROMA_HIGHLIGHT_GAIN
    pm.PRINT_CHROMA_HIGHLIGHT_GAIN = float(k_hi)
    try:
        out = {}
        for r in corpus["rows"]:
            d = np.clip(encode_srgb(apply_render(r["tech"], r["it"]))[..., :3], 0, 1)
            out[r["id"]] = {"hi": _hi(d), "full": _full(d)}
        return out
    finally:
        pm.PRINT_CHROMA_HIGHLIGHT_GAIN = orig


def test_offline_rerender_is_the_production_path(corpus):
    """前提校验：不改任何常数时离线重渲必须与 `candidate_positive` 逐比特相同。"""
    assert corpus["mismatches"] == []
    assert len(corpus["rows"]) >= 8, f"参与帧数掉到 {len(corpus['rows'])}"


def test_highlight_ramp_is_recorded_on_every_frame(corpus):
    """来源标签：常量必须就是量出来的那两个值；每帧生效的总增益只能是
    "已实测片基 ⇒ `LOOK_CHROMA_GAIN`"或"闸门 ⇒ 1.0"，不许有第三种状态。"""
    assert PRINT_CHROMA_HIGHLIGHT_GAIN == pytest.approx(0.80)
    assert PRINT_CHROMA_HIGHLIGHT_START == pytest.approx(0.85)
    for r in corpus["rows"]:
        assert r["gain"] == pytest.approx(LOOK_CHROMA_GAIN) or r["gain"] == 1.0, r["id"]
    print(f"\n  {len(corpus['rows'])} 帧，高光档 {PRINT_CHROMA_HIGHLIGHT_GAIN} "
          f"@ 纸级 {PRINT_CHROMA_HIGHLIGHT_START}")


def test_ramp_closes_highlight_cast_more_than_without_it(corpus):
    """**主锁（对通道敏感的尺）**：关掉这一项（高光档钉成与总增益相同）后，
    最高 5% 像素的"最低通道差"与"彩度差"都必须**变差**，留出卷尤其必须变差。"""
    off = _rerender(corpus, 2.50)          # 与 LOOK_CHROMA_GAIN 相同 ⇒ 斜坡恒等
    ids = [r["id"] for r in corpus["rows"]]
    hold = [i for i in ids if i.startswith(HOLDOUT)]
    print("\n  帧              最低通道 关掉/上线/参考      彩度 关掉/上线/参考")
    for r in corpus["rows"]:
        i = r["id"]
        print(f"  {i.replace('_G200::', '/'):<14} "
              f"{off[i]['hi']['min_ch']:.3f}/{r['hi']['min_ch']:.3f}"
              f"/{r['hi_ref']['min_ch']:.3f}    "
              f"{off[i]['hi']['chroma']:.3f}/{r['hi']['chroma']:.3f}"
              f"/{r['hi_ref']['chroma']:.3f}")
    g_off = np.mean([abs(off[i]["hi"]["min_ch"] - next(
        r for r in corpus["rows"] if r["id"] == i)["hi_ref"]["min_ch"]) for i in ids])
    g_up = np.mean([abs(r["hi"]["min_ch"] - r["hi_ref"]["min_ch"]) for r in corpus["rows"]])
    h_off = np.mean([abs(off[i]["hi"]["min_ch"] - next(
        r for r in corpus["rows"] if r["id"] == i)["hi_ref"]["min_ch"]) for i in hold])
    h_up = np.mean([abs(r["hi"]["min_ch"] - r["hi_ref"]["min_ch"])
                    for r in corpus["rows"] if r["id"] in hold])
    c_off = np.mean([abs(off[i]["hi"]["chroma"] - next(
        r for r in corpus["rows"] if r["id"] == i)["hi_ref"]["chroma"]) for i in ids])
    c_up = np.mean([abs(r["hi"]["chroma"] - r["hi_ref"]["chroma"]) for r in corpus["rows"]])
    print(f"  |Δ高光最低通道| 关掉 {g_off:.4f} → 上线 {g_up:.4f}"
          f"（留出 {h_off:.4f} → {h_up:.4f}）；|Δ高光彩度| {c_off:.4f} → {c_up:.4f}")
    assert g_up < g_off, f"关掉反而更好（{g_off:.4f} ≤ {g_up:.4f}）：这一项不承重"
    assert h_up < h_off, "留出卷没变好：这条机制的主要目标就是它"
    assert c_up < c_off


def test_reversed_ramp_is_worse_than_doing_nothing(corpus):
    """**负控制（方向）**：把"末端收回"反过来做成"末端更放大"（高光档 4.2）
    必须比"关掉"（2.50）更差。三档分不开就说明这把尺也没有区分力。"""
    ids = [r["id"] for r in corpus["rows"]]
    off = _rerender(corpus, 2.50)
    wrong = _rerender(corpus, 4.20)

    def gap(v):
        return float(np.mean([abs(v[i]["hi"]["min_ch"] - next(
            r for r in corpus["rows"] if r["id"] == i)["hi_ref"]["min_ch"]) for i in ids]))

    b0, bw = gap(off), gap(wrong)
    print(f"\n  |Δ高光最低通道|：关掉 {b0:.4f} / 反向放大 {bw:.4f}（必须更差）")
    assert bw > b0, f"反方向没变差（{b0:.4f} → {bw:.4f}）：该量对任意扰动都反应"


def test_known_costs_stay_inside_the_registered_budget(corpus):
    """**代价钉成预算**：上线这一项换来的是训练卷四统计距离微涨、整帧饱和微降。
    允许，但**必须钉住**——否则下一轮会在没人注意时把它放大。

    ⚠ 命名一律写全（`ship_` = 生产带这一项，`inert_` = 把高光档钉成基增益关掉它）：
    第一版这里把两个变量名写反，于是"白点朝参考走"那条断言**方向反着也能通过**，
    跑出来才抓到（`assert p_up > p_off` 里 `p_up` 其实是关掉的）。
    """
    inert = _rerender(corpus, 2.50)
    rows = corpus["rows"]
    train = [r for r in rows if r["id"].startswith("1298_G200")]

    def mean(xs):
        return float(np.mean(xs))

    d_inert = mean([_dist(inert[r["id"]]["full"], r["full_ref"]) for r in train])
    d_ship = mean([_dist(r["full"], r["full_ref"]) for r in train])
    s_inert = mean([inert[r["id"]]["full"]["sat95"] - r["full_ref"]["sat95"] for r in rows])
    s_ship = mean([r["full"]["sat95"] - r["full_ref"]["sat95"] for r in rows])
    p_inert = mean([inert[r["id"]]["full"]["p95"] - r["full_ref"]["p95"] for r in rows])
    p_ship = mean([r["full"]["p95"] - r["full_ref"]["p95"] for r in rows])
    print(f"\n  训练卷距离 关掉 {d_inert:.4f} → 上线 {d_ship:.4f}（预算 +{TRAIN_DIST_BUDGET}）；"
          f"Δsat95 {s_inert:+.4f} → {s_ship:+.4f}（预算 {SAT_BUDGET}）；"
          f"Δp95 {p_inert:+.4f} → {p_ship:+.4f}")
    assert d_ship - d_inert <= TRAIN_DIST_BUDGET, (
        f"训练卷距离代价超出登记的预算（{d_inert:.4f} → {d_ship:.4f}）："
        f"重估高光档而不是放宽这一条")
    assert s_inert - s_ship <= SAT_BUDGET, "整帧饱和的代价超出预算：重估而不是放宽"
    assert p_ship > p_inert, (
        f"白点没有朝参考走（关掉 {p_inert:+.4f}，上线 {p_ship:+.4f}）：这一项存在的理由没了")
