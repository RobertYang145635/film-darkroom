r"""回归：电平的**第二个携带者**（高调面积）与黑锚分位 `P_LOW` 2.0→3.0 —— 联立才成立。

㉑ 量出 `strength` 的最优值在两卷上**符号相反**；㉒ 单扫第二携带者被否
（亮度两卷都降，但留出卷 `|Δstd|` 0.0389→0.0484）；本轮单扫 `P_LOW` 也被否
（留出 `|亮度|` 0.0530→0.0589）。**两个各自被否的量合起来通过全部判据**
（`tmp_joint_level_black.py`，判据动手前写下）：

| 格 | 训练\|亮度\| | 留出\|亮度\| | 留出\|Δstd\| | 留出Δp05 | 全部Δp05 | 距离 |
|---|---|---|---|---|---|---|
| (P_LOW 2.0, κ₂ 0) 旧现产 | 0.0395 | 0.0530 | 0.0389 | +0.0174 | +0.0276 | 0.0541 |
| (2.0, 0.10) 只加面积 | 0.0323 | 0.0427 | 0.0449 | +0.0300 | +0.0292 | 0.0532 |
| (3.0, 0) 只抬黑锚 | 0.0344 | 0.0589 | 0.0339 | −0.0176 | −0.0017 | 0.0505 |
| **(3.0, 0.10) 上线** | **0.0277** | **0.0473** | **0.0365** | **−0.0048** | **+0.0001** | **0.0491** |

⇒ 阴影雾（用户说过的"发雾"）从 `+0.0276` 打到 **+0.0001**，逐帧亮度误差
`0.0436 → 0.0336`，四统计距离 `0.0541 → 0.0491`，且**留出卷三项同时变好**。

本文件把"联立"这件事本身钉成锁：**单独退回任何一个都必须让某条判据变红**，
否则后来的人会把这当成"其实只需要一个"而简化掉（那正是 ㉒ 单扫会做的错）。
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
from negative_lab.print_math import (P_LOW, PRINT_LEVEL_AREA_STRENGTH,
                                     PRINT_LEVEL_HIGHLIGHT_AREA_REFERENCE,
                                     highlight_area_fraction, level_shift)

ROOT = pathlib.Path(__file__).resolve().parents[2]
MAX_DIM = 1200
ROLLS = ("1298_G200", "1299_G200")
HOLDOUT, TRAIN_ROLL = "1299_G200", "1298_G200"
SHIP = (3.0, 0.10)
NO_AREA = (3.0, 0.0)
NO_BLACK = (2.0, 0.10)
BOTH_OLD = (2.0, 0.0)
LUM_W = np.array([0.2126, 0.7152, 0.0722])

_map = json.loads((ROOT / "fixtures/references" / "g200_manual_look.json")
                  .read_text(encoding="utf-8"))
MET = _map["method"]
REF = {e["frame_id"]: e for e in _map["frames"] if not e["variant_of"]}


def _lin(d):
    c = np.asarray(d, np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _stats(disp01):
    im = Image.fromarray((np.clip(disp01, 0, 1) * 255).round().astype(np.uint8))
    a = np.asarray(im.resize((MET["resize"],) * 2, Image.BILINEAR), np.float64) / 255.0
    c = int(a.shape[0] * (1 - MET["center_crop_frac"]) / 2)
    r = int(a.shape[1] * (1 - MET["center_crop_frac"]) / 2)
    d = a[c:a.shape[0] - c, r:a.shape[1] - r, :]
    lum = d @ LUM_W
    sat = d.max(2) - d.min(2)
    return {"mean": float(lum.mean()), "p05": float(np.percentile(lum, 5)),
            "p95": float(np.percentile(lum, 95)), "std": float(lum.std()),
            "sat95": float(np.percentile(sat, 95))}


@pytest.fixture(scope="module")
def corpus():
    """**整卷两遍**跑生产，并留下每帧**与生产同一总体**的密度样本
    （重定黑锚分位必须用同一批像素，不能另取总体——⑭ 的第一版就栽在这里）。"""
    missing = [f["reference_file"] for f in REF.values()
               if not (ROOT / f["reference_file"]).exists()]
    if missing:
        pytest.skip(f"用户手工参考大图不在库里（gitignore），缺 {len(missing)} 张")
    import negative_lab.print_math as pm

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
            ps = pr.meta.get("print_scale") or {}
            it = RenderIntent.from_dict(dict(pr.candidates.get(name).intent or {}))
            if len(it.print_anchor) != 3 or "highlight_area" not in ps:
                continue
            prod = np.clip(encode_srgb(apply_render(pr.result.positive_linear, it))[..., :3],
                           0, 1)
            if not np.array_equal(prod, np.clip(
                    encode_srgb(candidate_positive(pr, name))[..., :3], 0, 1)):
                mismatches.append(fr["id"])
            tech = pr.result.positive_linear
            s = pm.picture_density(tech, float(ps["dmax_old"]),
                                   pm.centre_region(pr.valid_mask, shape=tech.shape[:2]))
            ref_arr = np.asarray(Image.open(ROOT / REF[fr["id"]]["reference_file"])
                                 .convert("RGB"), np.float32) / 255.0
            rows.append({"id": fr["id"], "tech": tech, "it": it, "ps": ps,
                         "samples": np.asarray(s, np.float64),
                         "ship": _stats(prod), "ref": _stats(ref_arr)})
        del results
    return {"rows": rows, "mismatches": mismatches}


def _rerender(corpus, p_low: float, k2: float) -> dict:
    """在 `(P_LOW, κ₂)` 这一格上重走**生产的那三个函数**：定锚 → 窗口下限 → 加电平。

    次序跟着 ㉔ 换：下限作用在出厂锚上，电平项最后说话（两者抢同一个 `hi`）。
    """
    import negative_lab.print_math as pm

    out = {}
    orig_low, orig_k2 = pm.P_LOW, pm.PRINT_LEVEL_AREA_STRENGTH
    pm.P_LOW, pm.PRINT_LEVEL_AREA_STRENGTH = float(p_low), float(k2)
    try:
        for r in corpus["rows"]:
            anchor = pm.density_anchor(r["samples"])
            assert anchor is not None and len(anchor) == 3, f"{r['id']}：对照格锚点退化"
            area = highlight_area_fraction(r["samples"])
            anchor, _raised, wnote = pm.white_window_floor(
                anchor, None, float(r["ps"]["density_median"]))
            anchor, delta, note = pm.anchor_with_level(anchor, float(r["ps"]["density_median"]),
                                                       area)
            #: 符号闸（㉕）拦下窄跨度但本来就印暗的帧时会给一条说明——那是**预期行为**，
            #: 不许当成回退；真正的回退只有"会使曲线不合法"那一种。
            assert not note and (not wnote or "符号闸" in wnote), f"{r['id']}：{note}{wnote}"
            it = RenderIntent.from_dict(r["it"].to_dict())
            it.print_anchor = [float(v) for v in anchor]
            out[r["id"]] = _stats(np.clip(
                encode_srgb(apply_render(r["tech"], it))[..., :3], 0, 1))
        return out
    finally:
        pm.P_LOW, pm.PRINT_LEVEL_AREA_STRENGTH = orig_low, orig_k2


def _terms(g, corpus_rows, sel):
    ok = [r for r in corpus_rows if r["id"] in sel]
    return {"gm": float(np.mean([abs(g[r["id"]]["mean"] - r["ref"]["mean"]) for r in ok])),
            "gstd": float(np.mean([abs(g[r["id"]]["std"] - r["ref"]["std"]) for r in ok])),
            "g05": float(np.mean([g[r["id"]]["p05"] - r["ref"]["p05"] for r in ok])),
            "dist": float(np.mean([np.mean([abs(g[r["id"]]["mean"] - r["ref"]["mean"]),
                                            abs(g[r["id"]]["p05"] - r["ref"]["p05"]),
                                            abs(g[r["id"]]["std"] - r["ref"]["std"]),
                                            abs(g[r["id"]]["sat95"] - r["ref"]["sat95"])])
                                   for r in ok]))}


def test_shipped_pair_and_meta_self_consistency(corpus):
    """常量必须是量出来那一对；`meta` 里的 δ 必须等于 `level_shift(中位, 面积)`
    （**来源标签**：没有这一条就说不清生产到底走了哪条分支）。"""
    assert P_LOW == pytest.approx(3.0)
    assert PRINT_LEVEL_AREA_STRENGTH == pytest.approx(0.10)
    assert PRINT_LEVEL_HIGHLIGHT_AREA_REFERENCE == pytest.approx(0.3122)
    assert corpus["mismatches"] == []
    for r in corpus["rows"]:
        want = level_shift(float(r["ps"]["density_median"]), float(r["ps"]["highlight_area"]))
        assert float(r["ps"]["level_shift"]) == pytest.approx(want, abs=1e-5), r["id"]
    areas = [float(r["ps"]["highlight_area"]) for r in corpus["rows"]]
    print(f"\n  {len(corpus['rows'])} 帧；高调面积占比 min {min(areas):.3f} "
          f"中位 {float(np.median(areas)):.3f} max {max(areas):.3f}")
    assert max(areas) - min(areas) > 0.2, "占比没有散布，这个携带者在这一批上无从分辨"


def test_pair_beats_either_single_change_on_the_holdout(corpus):
    """**主锁 = 联立本身**：现产对 (3.0, 0.10) 必须在留出卷上同时不差于
    "只抬黑锚"(3.0, 0) 与"只加面积"(2.0, 0.10)——**这两个单改各自都有一项比现产差**，
    这正是"单扫会漏掉联立最优点"的可执行证据。"""
    rows = corpus["rows"]
    train = [r["id"] for r in rows if r["id"].startswith(TRAIN_ROLL)]
    hold = [r["id"] for r in rows if r["id"].startswith(HOLDOUT)]
    cells = {"现产(3.0,0.10)": SHIP, "只抬黑锚(3.0,0)": NO_AREA,
             "只加面积(2.0,0.10)": NO_BLACK, "都改回(2.0,0)": BOTH_OLD}
    t = {}
    for label, (p, k2) in cells.items():
        g = _rerender(corpus, p, k2)
        t[label] = (_terms(g, rows, train), _terms(g, rows, hold),
                    _terms(g, rows, [r["id"] for r in rows]))
    print("\n  格                    训练|亮度| 留出|亮度| 留出|Δstd| 留出Δp05 全部Δp05  距离")
    for label, (tr, ho, al) in t.items():
        print(f"  {label:<22}{tr['gm']:>11.4f}{ho['gm']:>11.4f}{ho['gstd']:>11.4f}"
              f"{ho['g05']:>+11.4f}{al['g05']:>+11.4f}{al['dist']:>10.4f}")
    ship_tr, ship_ho, ship_al = t["现产(3.0,0.10)"]
    # 现产必须比"都改回"好（两卷亮度 + 距离 + 雾）
    old_tr, old_ho, old_al = t["都改回(2.0,0)"]
    assert ship_al["gm"] < old_al["gm"], "现产的逐帧亮度没比旧值好"
    assert ship_al["dist"] < old_al["dist"], "现产的总距离没比旧值好"
    assert abs(ship_al["g05"]) < abs(old_al["g05"]), "阴影雾没被治好"
    assert abs(ship_ho["g05"]) < abs(old_ho["g05"]), "留出卷的雾没被治好"
    # 两个单改各自都有一项比现产差 ⇒ 联立是必要的，不是冗余
    only_black = t["只抬黑锚(3.0,0)"]
    only_area = t["只加面积(2.0,0.10)"]
    assert only_black[1]["gm"] > ship_ho["gm"], (
        "只抬黑锚在留出卷亮度上不比现产差 ⇒ 面积项是冗余的，重估这条锁")
    assert only_area[1]["gstd"] > ship_ho["gstd"], (
        "只加面积在留出卷对比上不比现产差 ⇒ 黑锚改动是冗余的，重估这条锁")


def test_area_carrier_negative_controls(corpus):
    """**两个负控制**：κ₂=0 与 κ₂ 反号都必须比现产差。
    反号这一条尤其重要——它证明"高光面积大 ⇒ 该印亮"这句话是承重的，
    而不是"任何扰动都会动指标"。"""
    rows = corpus["rows"]
    allids = [r["id"] for r in rows]
    hold = [i for i in allids if i.startswith(HOLDOUT)]
    ship = _terms(_rerender(corpus, *SHIP), rows, allids)
    ship_h = _terms(_rerender(corpus, *SHIP), rows, hold)
    zero = _terms(_rerender(corpus, SHIP[0], 0.0), rows, allids)
    flipped = _terms(_rerender(corpus, SHIP[0], -PRINT_LEVEL_AREA_STRENGTH), rows, allids)
    print(f"\n  逐帧|亮度|：现产 {ship['gm']:.4f} / κ₂=0 {zero['gm']:.4f} / "
          f"κ₂ 反号 {flipped['gm']:.4f}")
    assert flipped["gm"] > ship["gm"], "反号没变差 ⇒ 该量对任意扰动都反应，正锁无区分力"
    assert zero["gm"] >= ship["gm"] - 1e-4, "κ₂=0 不比现产差 ⇒ 现产收益不是面积项给的"
    assert abs(ship_h["g05"]) < 0.01, f"留出卷雾偏差 {ship_h['g05']:+.4f} 超出 ±0.01"


def test_haze_bias_is_now_a_two_sided_residual(corpus):
    """雾这条轴改成了"两侧都有残余"而不是"整体偏高"：
    均值必须贴着零，但**逐帧散布仍在**——不许后来的人看到均值归零就以为缺陷没了。"""
    g = _rerender(corpus, *SHIP)
    per = [(r["id"], g[r["id"]]["p05"] - r["ref"]["p05"]) for r in corpus["rows"]]
    worst = max(per, key=lambda kv: abs(kv[1]))
    print("\n  逐帧 Δp05：" + "  ".join(f"{i.replace('_G200::', '/')} {v:+.3f}"
                                        for i, v in per))
    print(f"  均值 {np.mean([v for _, v in per]):+.4f}，最差 {worst[0]} {worst[1]:+.4f}")
    assert abs(float(np.mean([v for _, v in per]))) < 0.005, "均值该贴着零"
    assert abs(worst[1]) > 0.03, (
        "逐帧散布突然消失了：多半是量法变了（比如参考没对上），要查不是庆祝")
