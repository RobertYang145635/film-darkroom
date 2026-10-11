"""回归：**窄跨度负片不许拉满纸级**（`PRINT_MIN_DENSITY_WINDOW`）。

⑮ 之后最大的两项逐帧亮度误差是 `1298::G200-5` **+0.133** 与 `::G200-8` **+0.098**
（其余 ≤0.06），而这两帧恰好是全语料跨度的两个极小值：
`d_hi − d_lo` = **0.383 / 0.419** decade，其余八帧 0.558 ~ 0.987。
`density_anchor` 把每帧自己的 [p02, p95] 映到 [黑, 纸白] ⇒ **跨度越小放大率越高**；
而他给 `::G200-5` 的 std 只有 **0.153**（十一帧最低），我们做到 0.196。
机制与依据全部写在 `print_math` 那条常量上（含"训练卷会选到 0.70 而 0.70 在留出卷上
明确更差"那次分裂）。

**㉔ 把这一刀挪到了电平项之前**：下限与 ⑮/㉓ 抢的是同一个 `hi` 端点，放在电平项
**之后**时它会把"逐帧限幅平移"刚让出来的窗口收回（`1299::G200-7` δ=+0.0796 抬亮、
被下限抬回 +0.127 压暗，而它本来就印暗了 0.0516 ⇒ 留出卷 `|亮度|` 0.0433→0.0473
全是这一帧）。所以本文件的对照一律按 `定锚 → 下限(出厂锚) → 电平` 重放，
且**调生产函数**而不是手推公式。

**对照怎么来的**：出厂锚的 `hi` 会被下限抬、再被电平项搬，单看 `d_hi` 两个方向混在一起
反推不出来，所以生产把**出厂白锚**记进 `print_scale.window_pre_hi`
（`window_min` / `window_raised` / `level_shift` 同批）。本文件所有对照都由这几个字段重建，
并在下面第一条测试里校验"用出厂 `window_min` 重建 == 出厂锚"。

判据是**动手前**写进 `tmp_min_window.py` / `tmp_window_order.py` 文档的，这里逐条钉住：
① 逐帧与留出卷 \\|亮度\\| 都降；② 逐帧 \\|Δstd\\| 降；③ 两卷四统计距离都不变差；
④ 黑锚与中灰锚只随电平项动、不被下限动。另把已知代价钉成预算：
白点会被推回去一点（实测 `Δp95 −0.0185 → −0.0369`）。
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest
from PIL import Image

from negative_lab import print_math as pm
from negative_lab.calibration.annotations import load_annotation
from negative_lab.io.images import load_linear_rgb
from negative_lab.library.fingerprint import capture_fingerprint
from negative_lab.library.models import FrameRecord, GeometryRegions
from negative_lab.library.render import RenderIntent, apply_render
from negative_lab.pipeline.density import encode_srgb
from negative_lab.pipeline.service import candidate_positive, process_roll
from negative_lab.print_math import PRINT_MIN_DENSITY_WINDOW, print_curve

ROOT = pathlib.Path(__file__).resolve().parents[2]
MAX_DIM = 1200
ROLLS = ("1298_G200", "1299_G200")
HOLDOUT = "1299_G200"
OFF, TOO_BIG = 0.0, 0.85
WHITE_GIVEBACK_BUDGET = 0.030      # 允许从 ⑲ 的白点成果里还回去这么多

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
    lum = 0.2126 * d[..., 0] + 0.7152 * d[..., 1] + 0.0722 * d[..., 2]
    sat = d.max(2) - d.min(2)
    lo, hi = MET["mid_tone_band"]
    band = (lum >= lo) & (lum < hi)
    L = np.clip(_lin(d), 1e-6, None)
    return {"mean": float(lum.mean()), "p05": float(np.percentile(lum, 5)),
            "p95": float(np.percentile(lum, 95)), "std": float(lum.std()),
            "sat95": float(np.percentile(sat, 95)),
            "band": L[band].mean(axis=0) if band.sum() > 400 else np.full(3, np.nan)}


@pytest.fixture(scope="module")
def corpus():
    """**整卷两遍**跑生产（单帧跑的 `base_source` 不同）。只留重渲所需的工件。"""
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
            ps = pr.meta.get("print_scale") or {}
            it = RenderIntent.from_dict(dict(pr.candidates.get(name).intent or {}))
            if len(it.print_anchor) != 3 or "window_pre_hi" not in ps:
                continue
            prod = np.clip(encode_srgb(apply_render(pr.result.positive_linear, it))[..., :3],
                           0, 1)
            if not np.array_equal(prod, np.clip(
                    encode_srgb(candidate_positive(pr, name))[..., :3], 0, 1)):
                mismatches.append(fr["id"])
            ref_arr = np.asarray(Image.open(ROOT / REF[fr["id"]]["reference_file"])
                                 .convert("RGB"), np.float32) / 255.0
            rows.append({"id": fr["id"], "tech": pr.result.positive_linear, "it": it,
                         "ps": ps, "anchor": [float(v) for v in it.print_anchor],
                         "ship": _stats(prod), "ref": _stats(ref_arr)})
        del results
    return {"rows": rows, "mismatches": mismatches}


def _anchor_at(r, w_min: float, order: str = "FL") -> list[float]:
    """由 meta 重建"窗口下限 = `w_min`"时的锚，**两步都调生产函数**。

    ㉔ 之后下限作用在**出厂锚**上、电平项在后（`order="FL"`），所以对照必须按同一条链重放：
    `pre = (lo, mid + δ, window_pre_hi)` → `white_window_floor(pre, w)` → `anchor_with_level`。
    `order="LF"` 重放的是**旧出厂次序**（先电平再下限），只用来在**同一量法**下
    给"这次替换买到了什么、没买到什么"提供对照格。
    手推公式不行：下限与电平项抢的是同一个 `hi`，次序写反就是在比一个产品里
    不存在的锚（⑯ 那条锁也是栽在手推合成上的）。
    """
    ps = r["ps"]
    d = float(ps["level_shift"])
    a = r["anchor"]
    pre = (a[0], a[1] + d, float(ps["window_pre_hi"]))
    area = None if ps["highlight_area"] is None else float(ps["highlight_area"])
    med = float(ps["density_median"])
    if order == "LF":
        lvl = pm.anchor_with_level(pre, med, area)[0]
        out = pm.white_window_floor(lvl, float(w_min))[0]
        return [float(v) for v in out]
    floored = pm.white_window_floor(pre, float(w_min))[0]
    out, _delta, _note = pm.anchor_with_level(floored, med, area)
    return [float(v) for v in out]


def _rerender(corpus, w_min: float, order: str = "FL") -> dict:
    out = {}
    for r in corpus["rows"]:
        cand = _anchor_at(r, w_min, order)
        print_curve(cand)                    # 对照本身必须是合法曲线，否则比的是废锚
        it = RenderIntent.from_dict(r["it"].to_dict())
        it.print_anchor = cand
        out[r["id"]] = _stats(np.clip(
            encode_srgb(apply_render(r["tech"], it))[..., :3], 0, 1))
    return out


def _gap(s, r):
    return {"gm": s["mean"] - r["mean"], "gstd": s["std"] - r["std"],
            "g05": s["p05"] - r["p05"], "gp95": s["p95"] - r["p95"],
            "grg": (s["band"][0] - s["band"][1]) - (r["band"][0] - r["band"][1]),
            "dist": float(np.mean([abs(s["mean"] - r["mean"]), abs(s["p05"] - r["p05"]),
                                   abs(s["std"] - r["std"]), abs(s["sat95"] - r["sat95"])]))}


def test_rebuilding_from_meta_reproduces_the_shipped_anchor(corpus):
    """**前提校验**：用出厂 `window_min` 重建的锚必须逐比特等于出厂锚
    （否则后面所有"关掉对照"比的都不是产品）；离线重渲也必须等于 `candidate_positive`。"""
    assert corpus["mismatches"] == []
    for r in corpus["rows"]:
        rebuilt = _anchor_at(r, float(r["ps"]["window_min"]))
        assert rebuilt == pytest.approx(r["anchor"], abs=1e-5), r["id"]
    assert len(corpus["rows"]) >= 8, f"参与帧数掉到 {len(corpus['rows'])}"


def test_window_is_recorded_and_only_touches_the_white_anchor(corpus):
    """来源标签 + 结构锁：被抬的帧必须满足 `d_hi == d_lo + window_min − δ`（下限作用在
    出厂锚上、电平项最后搬），没被抬的帧出厂跨度本来就够宽；且**不能全抬或全不抬**。"""
    raised = 0
    for r in corpus["rows"]:
        lo, mid, hi = r["anchor"]
        ps = r["ps"]
        delta = float(ps["level_shift"])
        pre_hi = float(ps["window_pre_hi"])
        assert float(ps["window_min"]) == pytest.approx(PRINT_MIN_DENSITY_WINDOW)
        assert mid > lo and hi > mid
        assert lo == pytest.approx(_anchor_at(r, OFF)[0], abs=1e-9)
        #: `abs=1e-5` 而不是逐比特：meta 里的 `level_shift` 按 5 位小数记录，
        #: 而 `_anchor_at` 内部用**未舍入**的 δ 重放，两边最多差 5e-6。
        assert mid == pytest.approx(_anchor_at(r, OFF)[1], abs=1e-5)
        if ps["window_raised"]:
            raised += 1
            assert hi == pytest.approx(lo + PRINT_MIN_DENSITY_WINDOW - delta,
                                       abs=2e-5), r["id"]
            assert pre_hi < lo + PRINT_MIN_DENSITY_WINDOW - 1e-6, "标记为抬升但出厂跨度已够宽"
        else:
            assert pre_hi >= lo + PRINT_MIN_DENSITY_WINDOW - 1e-4, r["id"]
            assert hi == pytest.approx(pre_hi - delta, abs=2e-5), r["id"]
    print(f"\n  {len(corpus['rows'])} 帧里 {raised} 帧被抬白锚，"
          f"其余 {len(corpus['rows']) - raised} 帧逐比特不变")
    assert 1 <= raised < len(corpus["rows"]), "全抬或全不抬都说明这个闸没在区分跨度"


def test_floor_applies_before_the_level_term_not_after(corpus):
    """**次序锁**（㉔）：被抬且 δ≠0 的帧，`hi` 必须是 `lo + W − δ`（下限→电平），
    不是 `max(hi_pre − δ, lo + W)`（电平→下限）。

    这条锁是红灯换来的：LF 次序里下限会把电平项刚让出来的窗口原额收回，
    δ 对亮度的贡献在窄跨度帧上被整笔抹掉——那时 `hi` 恰好等于 `lo + W`，
    老的"`hi == lo + window_min`"断言照样绿，看不出窗口被谁定过。
    """
    checked = 0
    for r in corpus["rows"]:
        ps = r["ps"]
        delta = float(ps["level_shift"])
        lo, hi = r["anchor"][0], r["anchor"][2]
        if not ps["window_raised"] or abs(delta) < 5e-4:
            continue
        checked += 1
        fl = lo + float(ps["window_min"]) - delta          # 下限 → 电平（生产）
        lf = max(float(ps["window_pre_hi"]) - delta,
                 lo + float(ps["window_min"]))             # 电平 → 下限（旧出厂）
        assert hi == pytest.approx(fl, abs=2e-5), r["id"]
        assert abs(hi - lf) > 1e-4, f"{r['id']}：两种次序没区别，锁失去区分力"
    print(f"\n  次序锁覆盖 {checked} 帧（被抬且 δ≠0）")
    assert checked >= 1, "没有一帧同时被抬且有电平项 ⇒ 这条锁是空转的"


def test_floor_beats_turning_it_off_on_both_rolls(corpus):
    """**主锁**（判据①②③）：把窗口下限关掉后，逐帧\\|亮度\\|、逐帧\\|Δstd\\|
    与两卷四统计距离都必须变差。"""
    off = _rerender(corpus, OFF)
    rows = corpus["rows"]
    ids = [r["id"] for r in rows]
    hold = [i for i in ids if i.startswith(HOLDOUT)]
    train = [i for i in ids if i.startswith("1298_G200")]
    ship = {r["id"]: _gap(r["ship"], r["ref"]) for r in rows}
    o = {r["id"]: _gap(off[r["id"]], r["ref"]) for r in rows}

    def mm(d, sel, key, f=lambda x: x):
        return float(np.mean([f(d[i][key]) for i in sel]))

    print("\n  帧              |亮度 关掉/上线     |Δstd 关掉/上线")
    for r in rows:
        i = r["id"]
        print(f"  {i.replace('_G200::', '/'):<14} {abs(o[i]['gm']):.4f}/{abs(ship[i]['gm']):.4f}"
              f"      {abs(o[i]['gstd']):.4f}/{abs(ship[i]['gstd']):.4f}")
    print(f"  逐帧|亮度| {mm(o, ids, 'gm', abs):.4f} → {mm(ship, ids, 'gm', abs):.4f}；"
          f"留出 {mm(o, hold, 'gm', abs):.4f} → {mm(ship, hold, 'gm', abs):.4f}")
    print(f"  逐帧|Δstd| {mm(o, ids, 'gstd', abs):.4f} → {mm(ship, ids, 'gstd', abs):.4f}")
    print(f"  距离 全部 {mm(o, ids, 'dist'):.4f} → {mm(ship, ids, 'dist'):.4f}；"
          f"训练 {mm(o, train, 'dist'):.4f} → {mm(ship, train, 'dist'):.4f}；"
          f"留出 {mm(o, hold, 'dist'):.4f} → {mm(ship, hold, 'dist'):.4f}")
    print(f"  Δp05 {mm(o, ids, 'g05'):+.4f} → {mm(ship, ids, 'g05'):+.4f}；"
          f"|Δ带宽RG| {mm(o, ids, 'grg', abs):.4f} → {mm(ship, ids, 'grg', abs):.4f}")
    assert mm(ship, ids, 'gm', abs) < mm(o, ids, 'gm', abs), "逐帧|亮度|没降"
    assert mm(ship, hold, 'gm', abs) <= mm(o, hold, 'gm', abs) + 1e-4, "留出卷|亮度|变差"
    assert mm(ship, ids, 'gstd', abs) < mm(o, ids, 'gstd', abs), "逐帧|Δstd|没降"
    assert mm(ship, ids, 'dist') < mm(o, ids, 'dist'), "四统计距离没降"
    assert mm(ship, train, 'dist') < mm(o, train, 'dist'), "训练卷距离没降"
    assert mm(ship, hold, 'dist') <= mm(o, hold, 'dist') + 1e-4, "留出卷距离变差"
    # 判据④：这一刀只砍亮部，不许把误差搬到阴影或色相上
    assert mm(ship, ids, 'g05') <= mm(o, ids, 'g05') + 1e-3, "阴影雾被推高"
    assert mm(ship, ids, 'grg', abs) <= mm(o, ids, 'grg', abs) + 3e-3, "色相被推坏"


GRID = (0.0, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70)


def test_window_grid_under_the_shipped_order_is_the_pre_registered_selection(corpus):
    """**选值锁**（㉔）：`W_min` 必须在**本文件的量法**上重扫一遍再认账。

    教训：`tmp_window_order.py` 的网格是在探针几何（无手框的帧走自动框）上量的，
    与本锁（有手框用手框、没手框就**不给框**）不同总体——同一批帧上
    `1298::G200-9` 的 |亮度| 一个给 0.0231 一个给 **0.0126**（差一倍）。
    记忆 `repo-delivery-invariants` 第 2 条："同一件事第二次量化必须用同一量法"，
    这里就是当场补上那一遍。⚠ 两处的**留出卷**数字逐比特相同（1299 三帧都有用户手框），
    所以决定取值的判据②本来就口径一致；差的是训练卷那侧。

    判据沿用 `tmp_window_order.py` 动手前写下的两条，不新加：
    ① 上线格在训练卷上必须比 `W=0`（关掉）好；
    ② 上线格在留出卷的 |亮度| 与四统计距离都不许比 `W=0` 差；
    ③ **比上线格更大的每一格都必须被 ② 否掉**——否则"0.55 是过判据的最大值"这句话
       没有区分力，值就该往上改。
    """
    rows = corpus["rows"]
    ids = [r["id"] for r in rows]
    hold = [i for i in ids if i.startswith(HOLDOUT)]
    train = [i for i in ids if i.startswith("1298_G200")]
    ship = {r["id"]: _gap(r["ship"], r["ref"]) for r in rows}

    def mm(d, sel, key, f=lambda x: x):
        return float(np.mean([f(d[i][key]) for i in sel]))

    w_ship = float(PRINT_MIN_DENSITY_WINDOW)
    assert w_ship in GRID, f"上线值 {w_ship} 不在网格里，选值锁无法对照"

    cells = {}
    for w in GRID:
        g = _rerender(corpus, w)
        cells[w] = {r["id"]: _gap(g[r["id"]], r["ref"]) for r in rows if r["id"] in g}

    #: **自检**：网格里"上线值"那一格必须复现产品渲染（重建次序写错时锚仍然合法、
    #: 数字却不对 —— ⑯ 那轮就是栽在重建与产品不一致上）。
    drift = max(abs(cells[w_ship][i]["gm"] - ship[i]["gm"]) for i in ids)
    assert drift < 1e-4, f"网格在 w={w_ship} 格不复现产品（最大 {drift:.6f}）⇒ 重建写错了"

    print("\n  W_min   训练|亮度|  训练|Δstd|  训练分和  留出|亮度|  留出距离   全部距离  判定")
    base = cells[0.0]
    base_sum = mm(base, train, "gm", abs) + mm(base, train, "gstd", abs)
    passing = []
    for w in GRID:
        c = cells[w]
        tsum = mm(c, train, "gm", abs) + mm(c, train, "gstd", abs)
        ok = (tsum < base_sum and mm(c, hold, "gm", abs) <= mm(base, hold, "gm", abs) + 1e-4
              and mm(c, hold, "dist") <= mm(base, hold, "dist") + 1e-4)
        if w > 0 and ok:
            passing.append(w)
        mark = "← 关掉" if w == 0 else ("← 上线值" if w == w_ship else "")
        print(f"  {w:>6.2f}  {mm(c, train, 'gm', abs):>11.4f}  {mm(c, train, 'gstd', abs):>11.4f}"
              f"  {tsum:>9.4f}  {mm(c, hold, 'gm', abs):>11.4f}"
              f"  {mm(c, hold, 'dist'):>9.4f}  {mm(c, ids, 'dist'):>9.4f}"
              f"  {'过' if ok else '不过'}{mark}", flush=True)

    c_ship = cells[w_ship]
    assert mm(c_ship, train, "gm", abs) + mm(c_ship, train, "gstd", abs) < base_sum, (
        "判据①：上线格在训练卷不比关掉好")
    assert mm(c_ship, hold, "gm", abs) <= mm(base, hold, "gm", abs) + 1e-4, (
        "判据②：上线格把留出卷亮度压暗了（这正是 ㉔ 要修的红灯）")
    assert mm(c_ship, hold, "dist") <= mm(base, hold, "dist") + 1e-4, "判据②：留出卷距离变差"
    bigger = [w for w in GRID if w > w_ship
              and mm(cells[w], hold, "gm", abs) <= mm(base, hold, "gm", abs) + 1e-4
              and mm(cells[w], hold, "dist") <= mm(base, hold, "dist") + 1e-4]
    assert not bigger, (
        f"判据③：{bigger} 在本量法下也过留出卷判据 ⇒ 上线值该往上改，不是守住旧数")
    print(f"  本量法下过①②的 w>0：{passing}")

    #: **旧格（`LF@0.60`）在同一量法下的对照**——"这次替换是停止伤害、不是全面变好"
    #: 这句话不许引用探针口径的数字来支撑，必须在验收口径上成立。
    g_old = _rerender(corpus, 0.60, "LF")
    old = {r["id"]: _gap(g_old[r["id"]], r["ref"]) for r in rows if r["id"] in g_old}
    print(f"  旧次序 LF@0.60：训练分和 {mm(old, train, 'gm', abs) + mm(old, train, 'gstd', abs):.4f}"
          f"（上线 {mm(c_ship, train, 'gm', abs) + mm(c_ship, train, 'gstd', abs):.4f}）"
          f" 留出|亮度| {mm(old, hold, 'gm', abs):.4f}（上线 {mm(c_ship, hold, 'gm', abs):.4f}）"
          f" 留出距离 {mm(old, hold, 'dist'):.4f}（上线 {mm(c_ship, hold, 'dist'):.4f}）"
          f" 全部距离 {mm(old, ids, 'dist'):.4f}（上线 {mm(c_ship, ids, 'dist'):.4f}）")
    assert mm(old, hold, 'gm', abs) > mm(c_ship, hold, 'gm', abs) + 1e-4, (
        "旧次序在同一量法下不再把留出卷压暗 ⇒ ㉔ 换次序的理由没了，该重估这一整轮")
    assert mm(old, train, 'gm', abs) < mm(c_ship, train, 'gm', abs), (
        "旧次序在训练卷上不再更好 ⇒ 记在账上的'代价'不成立，重述这一轮")


def test_bigger_floor_is_worse_than_the_shipped_one(corpus):
    """**方向负控制**：窗口不是越宽越好——抬到 0.85 必须在留出卷上比上线值更差，
    否则"0.55 是判据①②同时成立的最大值"这句话没有区分力。"""
    big = _rerender(corpus, TOO_BIG)
    rows = corpus["rows"]
    hold = [r["id"] for r in rows if r["id"].startswith(HOLDOUT)]
    ship = {r["id"]: _gap(r["ship"], r["ref"]) for r in rows}
    b = {r["id"]: _gap(big[r["id"]], r["ref"]) for r in rows}

    def mm(d, sel, key, f=lambda x: x):
        return float(np.mean([f(d[i][key]) for i in sel]))

    print(f"\n  留出卷|亮度| 上线 {mm(ship, hold, 'gm', abs):.4f} / "
          f"w_min={TOO_BIG} {mm(b, hold, 'gm', abs):.4f}（必须更差）；"
          f"留出距离 {mm(ship, hold, 'dist'):.4f} → {mm(b, hold, 'dist'):.4f}")
    assert mm(b, hold, 'gm', abs) > mm(ship, hold, 'gm', abs), (
        "更宽的窗口没变差 ⇒ 0.60 只是任意一点，重扫而不是保留")
    assert mm(b, hold, 'dist') > mm(ship, hold, 'dist')


def test_white_giveback_stays_inside_budget(corpus):
    """**代价钉成预算**：抬白锚会把 ⑲ 刚找回来的高光推回去一点
    （实测 `Δp95 −0.0185 → −0.0369`）。允许，但必须钉住，
    超了要重估窗口下限**或与 ⑲ 的高光档联立**，不许放宽这一条。"""
    off = _rerender(corpus, OFF)
    rows = corpus["rows"]
    ids = [r["id"] for r in rows]
    ship = {r["id"]: _gap(r["ship"], r["ref"]) for r in rows}
    o = {r["id"]: _gap(off[r["id"]], r["ref"]) for r in rows}
    p_ship = float(np.mean([ship[i]["gp95"] for i in ids]))
    p_off = float(np.mean([o[i]["gp95"] for i in ids]))
    print(f"\n  Δp95：关掉 {p_off:+.4f} → 上线 {p_ship:+.4f}"
          f"（允许还回去 {WHITE_GIVEBACK_BUDGET}）")
    assert p_off - p_ship <= WHITE_GIVEBACK_BUDGET, (
        f"白点还回去超过预算（{p_off:+.4f} → {p_ship:+.4f}）")
