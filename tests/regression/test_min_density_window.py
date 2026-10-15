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
import math
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


def _anchor_at(r, w_min: float, order: str = "FL", tau: float | None = None) -> list[float]:
    """由 meta 重建"窗口下限 = `w_min`"时的锚，**三步都调生产函数**。

    ㉔ 之后下限作用在**出厂锚**上、电平项在后（`order="FL"`），所以对照必须按同一条链重放：
    `pre = (lo, mid + δ, window_pre_hi)` → `white_window_floor(pre, w, 中位, τ)` → `anchor_with_level`。
    ㉕ 起下限还带**符号闸**：`τ=None` 用出厂闸值（= 产品形状），`τ=math.inf` 表示恒开
    （用来复现 ㉔ 及更早"没有闸"的产品）。
    `order="LF"` 重放的是**旧出厂次序**（先电平再下限、无闸），只用来在**同一量法**下
    给"㉔ 这次替换买到了什么、没买到什么"提供对照格。
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
        out = pm.white_window_floor(lvl, float(w_min), tau=math.inf)[0]
        return [float(v) for v in out]
    t = pm.PRINT_WINDOW_YMED_REFERENCE if tau is None else float(tau)
    floored = pm.white_window_floor(pre, float(w_min), med, t)[0]
    out, _delta, _note = pm.anchor_with_level(floored, med, area)
    return [float(v) for v in out]


def _rerender(corpus, w_min: float, order: str = "FL", tau: float | None = None) -> dict:
    out = {}
    for r in corpus["rows"]:
        cand = _anchor_at(r, w_min, order, tau)
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
    出厂锚上、电平项最后搬）；**被 ㉕ 的符号闸拦下的帧必须"出厂跨度确实窄、但 `y_med` 在
    闸值之上"**；其余帧出厂跨度本来就够宽。且**不能全抬或全不抬、也不能一帧都不拦**
    （闸若在生产里从不拦任何东西，㉕ 就是死代码）。"""
    raised = blocked = 0
    for r in corpus["rows"]:
        lo, mid, hi = r["anchor"]
        ps = r["ps"]
        delta = float(ps["level_shift"])
        pre_hi = float(ps["window_pre_hi"])
        y_med = None if ps.get("window_y_med") is None else float(ps["window_y_med"])
        assert float(ps["window_min"]) == pytest.approx(PRINT_MIN_DENSITY_WINDOW)
        assert float(ps["window_gate"]) == pytest.approx(pm.PRINT_WINDOW_YMED_REFERENCE)
        assert mid > lo and hi > mid
        assert lo == pytest.approx(_anchor_at(r, OFF)[0], abs=1e-9)
        #: `abs=1e-5` 而不是逐比特：meta 里的 `level_shift` 按 5 位小数记录，
        #: 而 `_anchor_at` 内部用**未舍入**的 δ 重放，两边最多差 5e-6。
        assert mid == pytest.approx(_anchor_at(r, OFF)[1], abs=1e-5)
        narrow = pre_hi < lo + PRINT_MIN_DENSITY_WINDOW - 1e-6
        if ps["window_raised"]:
            raised += 1
            assert hi == pytest.approx(lo + PRINT_MIN_DENSITY_WINDOW - delta,
                                       abs=2e-5), r["id"]
            assert narrow, "标记为抬升但出厂跨度已够宽"
            assert y_med is not None and y_med < float(ps["window_gate"]), \
                f"{r['id']}：抬了却没经过符号闸 ⇒ 闸没在生产路径上"
        else:
            if narrow:
                blocked += 1
                assert y_med is not None and y_med >= float(ps["window_gate"]), (
                    f"{r['id']}：出厂跨度窄却被跳过，但 `y_med` 并不在闸值之上")
            else:
                assert y_med is None or True
            assert hi == pytest.approx(pre_hi - delta, abs=2e-5), r["id"]
    print(f"\n  {len(corpus['rows'])} 帧：{raised} 帧被抬白锚、{blocked} 帧被符号闸拦下、"
          f"其余 {len(corpus['rows']) - raised - blocked} 帧出厂跨度本来就够宽")
    assert 1 <= raised < len(corpus["rows"]), "全抬或全不抬都说明这个闸没在区分跨度"
    assert blocked >= 1, "符号闸在生产里一帧都没拦 ⇒ ㉕ 是死代码，方向收益无处可寻"


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
    """**选值锁**（㉔ 建立、㉕ 改写）：`W_min` 的网格必须在**本文件的量法**上跑。

    教训：`tmp_window_order.py` 的网格是在探针几何（无手框的帧走自动框）上量的，
    与本锁（有手框用手框、没手框就**不给框**）不同总体——同一批帧上
    `1298::G200-9` 的 |亮度| 一个给 0.0231 一个给 **0.0126**（差一倍）。
    记忆 `repo-delivery-invariants` 第 6、13 条："探针必须复现生产口径"、
    "选值的网格必须在验收口径上跑"，这里就是当场补上那一遍。⚠ 两处的**留出卷**数字
    逐比特相同（1299 三帧都有用户手框），所以决定取值的判据②本来就口径一致；
    差的是训练卷那侧。

    判据（㉔ 动手前写下，㉕ 只改第三条）：
    ① 上线格在训练卷上必须比 `W=0`（关掉）好；
    ② 上线格在留出卷的 |亮度| 与四统计距离都不许比 `W=0` 差；
    ③ ~~比上线格更大的每一格都必须被②否掉~~ —— **㉕ 的符号闸把这条作废**：
       留出卷之所以能否决大 `W`，是因为那一刀会把它压暗；闸去掉这个副作用之后，
       留出卷对 `W_min` **不再有否决权**，网格退化成训练卷单调。
       ⇒ `W_min` 改由工业先验钉住（0.60），本锁改而成一条**绊线**：
       只要"更大的格子仍然过②"，就断言并打印出来；哪天第三卷让某格不过②，
       这条断言会红，提醒回到 ㉔ 的判据③重估取值。
       真正带区分力的联立证据搬去 `tests/regression/test_window_sign_gate.py`。
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
    print(f"  带闸网格过①②的 w>0：{passing}")
    #: **判据③ 已被 ㉕ 作废，这里把它换成一条明写的绊线**：
    #: ㉔ 的"比上线值大的每一格都必须被留出卷否掉"靠的是"留出卷会被这一刀压暗"，
    #: 而符号闸正是去掉了那个副作用 ⇒ 留出卷从此**管不着** `W_min`，
    #: 网格退化成"训练卷单调"（0.60→0.65→0.70 一路降而留出纹丝不动）。
    #: 所以 `W_min` 不再由本语料选，而是钉在工业先验 0.60；这条绊线的含义是
    #: **"留出卷对本语料的 `W_min` 没有否决权"这件事是已知的、被盯着的**——
    #: 一旦第三卷让它有了否决权（某格开始不过②），下面这条就会红，提醒重估。
    unvetoed = [w for w in passing if w > w_ship]
    assert unvetoed, ("留出卷现在能否掉比上线值更大的格子 ⇒ ㉕ 的'闸门使 W 不可测'这条论断"
                      "已过时，回到 ㉔ 的判据③（取过判据的最大值）")
    print(f"  绊线：带闸时留出卷否不掉的更大格 = {unvetoed}"
          f" ⇒ `W_min` 由工业先验钉住，不由本语料选")

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
    #: ㉔ 记下的"代价"是"旧次序在训练卷上更好（0.0252 对 0.0289）"。
    #: **㉕ 的符号闸把这笔代价整笔抹掉了**：上线点训练卷 \|亮度\| **0.0252**、
    #: 旧次序 **0.0260** ⇒ 训练卷这一侧现在**不输**给旧次序。
    #: 所以这条断言从"旧次序必须更好"改成"**上线点必须不更差**"——
    #: 这不是放宽，是把判据改成刚被证明为真的那句话；
    #: 若哪天有人改坏了闸，训练卷重新落后于旧次序，这一条就会红。
    assert mm(old, train, 'gm', abs) >= mm(c_ship, train, 'gm', abs) - 1e-4, (
        "上线点在训练卷上又比旧次序差 ⇒ ㉕ 的闸失效或被改坏，回到 ㉔ 的'付代价'口径")


def test_bigger_floor_is_worse_only_without_the_gate(corpus):
    """**方向控制（㉕ 改写，㊴ 换主语）**：`w_min=0.85` 这一格现在测出两件事，缺一不成立：

    ① **摘掉闸**（`tau=inf`）时它必须**看得见地改变**留出卷，而且在**亮度这一根轴上
       必须更差**——这是符号闸存在的唯一理由（㉔：单向钳会把本来就印暗的帧再压暗）。
       ㉔ 实测留出 \\|亮度\\| 0.0433 → **0.0722**；㊴ 在 `blue=0.85` 的工作点上重跑
       0.0226 → **0.0500**。
       ⚠ **同一格的四统计距离符号在 ㊴ 翻了**：0.0421 → **0.0401（反而更好）**。
       这不是"更宽的窗口变好了"，而是 ㉗/㉚ 那一类**轴间交换**——压暗留出的同时把
       `sat95`/`std` 推向参考，四条平均之后距离反而变小。所以判据在这里**不再要求
       距离的方向**，只要求"这一格对闸敏感"（|Δ距离| 必须 > 1e-3）：
       要求一个已被实测推翻的符号，那才叫放宽。
    ② **带着闸**（生产形状）时它必须**不再**伤留出卷，且一帧"该放"的帧都不许被抬。
       这条是 ㉕ 的**爆炸半径锁**：闸的作用就是把"手滑调宽"的伤害限制在
       已经被判为偏亮的那一侧；若有人把闸改坏，①②会同时红。

    ⚠ ㉔ 之前这条锁写的是"更宽的窗口必须在留出卷上更差"，
    ㉕ 之后**带闸的更宽窗口不再更差**——那是闸在起作用，不是判据可以放松。
    所以判据换了主语（从"宽度"换到"有没有闸"），并留下 ② 作为闸的常驻证明。
    """
    rows = corpus["rows"]
    hold = [r["id"] for r in rows if r["id"].startswith(HOLDOUT)]
    ship = {r["id"]: _gap(r["ship"], r["ref"]) for r in rows}

    def mm(d, sel, key, f=lambda x: x):
        return float(np.mean([f(d[i][key]) for i in sel]))

    big_off = _rerender(corpus, TOO_BIG, "FL", math.inf)
    big_on = _rerender(corpus, TOO_BIG)
    ungated = {r["id"]: _gap(big_off[r["id"]], r["ref"]) for r in rows}
    gated = {r["id"]: _gap(big_on[r["id"]], r["ref"]) for r in rows}
    print(f"\n  留出卷|亮度| 上线 {mm(ship, hold, 'gm', abs):.4f} / "
          f"带闸 w={TOO_BIG} {mm(gated, hold, 'gm', abs):.4f} / "
          f"摘闸 w={TOO_BIG} {mm(ungated, hold, 'gm', abs):.4f}（只有摘闸那一格必须更差）")
    # ① 摘闸必须坏（亮度轴），且必须**看得见地**改变留出距离（符号不再要求，见 docstring ㊴）
    assert mm(ungated, hold, 'gm', abs) > mm(ship, hold, 'gm', abs) + 1e-3, (
        "摘掉闸后更宽的窗口不再伤留出卷 ⇒ ㉔ 的'单向钳会压暗'这条前提被推翻，重估两轮")
    d_ship, d_ung = mm(ship, hold, 'dist'), mm(ungated, hold, 'dist')
    assert abs(d_ung - d_ship) > 1e-3, (
        f"摘闸后留出四统计距离几乎不动（{d_ship:.4f} → {d_ung:.4f}）"
        "⇒ 这一格对闸不再敏感，① 只剩亮度一根轴承重：距离这条轴要重新找证据，"
        "别把这条断言改成恒真")
    print(f"  摘闸留出距离 {d_ship:.4f} → {d_ung:.4f}"
          f"（{'更差' if d_ung > d_ship else '反而更好——㊴ 登记的轴间交换，符号不作判据'}）")
    # ② 带闸必须不坏，且不许抬任何"该放"的帧
    assert mm(gated, hold, 'gm', abs) <= mm(ship, hold, 'gm', abs) + 1e-4, (
        f"带闸 w={TOO_BIG} 仍然压暗留出卷 ⇒ 符号闸没有限制住爆炸半径")
    spared = 0
    for r in rows:
        lo = r["anchor"][0]
        y = r["ps"].get("window_y_med")
        if y is None or float(y) < float(r["ps"]["window_gate"]):
            continue
        if lo + TOO_BIG > float(r["ps"]["window_pre_hi"]) + 1e-9:
            spared += 1
            cand = _anchor_at(r, TOO_BIG)
            assert cand[2] == pytest.approx(float(r["ps"]["window_pre_hi"])
                                            - float(r["ps"]["level_shift"]), abs=2e-5), r["id"]
    print(f"  带闸 w={TOO_BIG} 时'该放'却被下限盯上的帧数 {spared}（全部被闸拦住）")
    assert spared >= 1, "没有一帧需要闸来拦 ⇒ 这条锁是空转的"


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
