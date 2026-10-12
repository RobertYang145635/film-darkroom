"""回归：**窗口下限的符号闸 × `W_min` 必须联立**（B-14㉕）。

**为什么要有这一把锁**：⑳ 的窗口下限是**单向钳**（抬 `hi` ⇒ 只会压暗），
所以它必然害"本来就印暗了"的帧。㉔ 的处置是把 `W_min` 从 0.60 降到 0.55，
让这一刀在留出卷上**一帧都不触发**——统计上干净，机制上是在躲：
`0.55` 恰好卡在留出最窄帧的出厂跨度（≈0.553）之下，换个卷就失效。

㉕ 换成**读方向**：`y_med` = 未加下限的出厂曲线上、这帧密度中位落到的纸级亮度
（只用底片自身，运行时不碰参考）。本批实测 `y_med ≤ 0.404` 的六帧 Δmean 全 ≥ 0
（我方印亮了，该收），`≥ 0.458` 的四帧全 < 0（印暗了，该放）⇒ 闸取 0.43。

**四格联立表**（`tmp_gate_joint.py` 与下面的当场打印同形，判据动手前写下）：

| 格 | 训练分和 | 留出\\|亮度\\| | 留出距离 | 全部距离 |
|---|---|---|---|---|
| 无闸 @0.55（㉔ 出厂） | 0.0504 | 0.0433 | 0.0456 | 0.0494 |
| 无闸 @0.60 | 0.0475 | **0.0449 ✗** | 0.0460 ✗ | 0.0490 |
| 有闸 @0.55 | 0.0504 | 0.0433 | 0.0456 | 0.0494 |
| **有闸 @0.60（上线）** | **0.0475** | **0.0433** | **0.0456** | **0.0489** |

⇒ **两半各买一样东西，缺一就不成立**：放宽 `W` 买到训练卷收益（0.0504→0.0475），
符号闸保证这份收益不再从留出卷身上抢（0.0449→0.0433）。
单独加闸而不放宽 `W` 与㉔逐比特相同（**闸本身不带来任何收益**，它是保护性的），
这正是 ㉓ 那条方法论的第二次应验：单参数扫会把"只在联立里成立"的组合判死。

⚠ **两侧证据不对称，写死在这里**：`y_med ≥ 0.458` 的四帧里三帧来自 1299
⇒ 在两卷语料上"闸有效"与"闸 = 让 1299 永不开火"**分不开**；
而且加闸之后留出卷对 `W_min` **不再有否决权**（网格训练分和随 `W` 单调降而留出纹丝不动），
所以 `W_min=0.60` 是**钉住的工业先验**，不是本语料选出来的。
第三卷一到必须重扫"闸值 τ × `W_min`"这一对。
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

ROOT = pathlib.Path(__file__).resolve().parents[2]
MAX_DIM = 1200
ROLLS = ("1298_G200", "1299_G200")
HOLDOUT = "1299_G200"
OLD_W = 0.55            # ㉔ 出厂值（被闸换成 0.60 之前）
NO_GATE, SHIP = math.inf, None

_map = json.loads((ROOT / "fixtures/references" / "g200_manual_look.json")
                  .read_text(encoding="utf-8"))
MET = _map["method"]
REF = {e["frame_id"]: e for e in _map["frames"] if not e["variant_of"]}


def _stats(disp01):
    """验收口径：显示-sRGB，缩到 `MET["resize"]` 再取中心 70%。"""
    im = Image.fromarray((np.clip(disp01, 0, 1) * 255).round().astype(np.uint8))
    a = np.asarray(im.resize((MET["resize"],) * 2, Image.BILINEAR), np.float64) / 255.0
    c = int(a.shape[0] * (1 - MET["center_crop_frac"]) / 2)
    r = int(a.shape[1] * (1 - MET["center_crop_frac"]) / 2)
    d = a[c:a.shape[0] - c, r:a.shape[1] - r, :]
    lum = 0.2126 * d[..., 0] + 0.7152 * d[..., 1] + 0.0722 * d[..., 2]
    sat = d.max(2) - d.min(2)
    return {"mean": float(lum.mean()), "p05": float(np.percentile(lum, 5)),
            "std": float(lum.std()), "sat95": float(np.percentile(sat, 95))}


@pytest.fixture(scope="module")
def corpus():
    """**整卷两遍**跑生产（单帧跑的 `base_source` 不同），并留下按格重放所需的工件。

    出厂锚由 meta 反推：`pre = (d_lo, d_mid + δ, window_pre_hi)`；
    每一格都调**生产函数**重放（`white_window_floor` → `anchor_with_level`），
    手推公式在这条链上已经被抓到过两次（⑯、）。
    """
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
            recs.append(FrameRecord(source_path=fr["file"], source_sha256=fr["source_sha256"],
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
            a = [float(v) for v in it.print_anchor]
            d = float(ps["level_shift"])
            rows.append({"id": fr["id"], "tech": pr.result.positive_linear, "it": it,
                         "ps": ps, "anchor": a,
                         "pre": (a[0], a[1] + d, float(ps["window_pre_hi"])),
                         "ship": _stats(prod), "ref": _stats(ref_arr)})
        del results
    return {"rows": rows, "mismatches": mismatches}


def _replay(r, w_min: float, tau: float | None, flip: bool = False) -> list[float]:
    """出厂锚 → （符号闸）窗口下限 → 电平项。`tau=math.inf` = 把闸摘掉。

    `flip=True` 是**反号负控制**：选择条件取反（只在 `y_med ≥ τ` 的帧上开火），
    而开火时走"摘闸"的那条路——否则生产自己的闸会立刻把这一格变回"什么都不做"，
    负控制就成了空对照。
    """
    ps = r["ps"]
    med = float(ps["density_median"])
    area = None if ps["highlight_area"] is None else float(ps["highlight_area"])
    t = pm.PRINT_WINDOW_YMED_REFERENCE if tau is None else float(tau)
    y = pm.window_y_med(r["pre"], med)
    if flip:
        allow = (t == math.inf) or (y is None) or (y >= t)
        floored = (pm.white_window_floor(r["pre"], float(w_min), med, math.inf)[0]
                   if allow else r["pre"])
    else:
        floored = pm.white_window_floor(r["pre"], float(w_min), med, t)[0]
    out, _d, _n = pm.anchor_with_level(floored, med, area)
    return [float(v) for v in out]


def _rerender(corpus, w_min: float, tau: float | None, flip: bool = False) -> dict:
    out = {}
    for r in corpus["rows"]:
        cand = _replay(r, w_min, tau, flip)
        pm.print_curve(cand)
        it = RenderIntent.from_dict(r["it"].to_dict())
        it.print_anchor = cand
        out[r["id"]] = _stats(np.clip(
            encode_srgb(apply_render(r["tech"], it))[..., :3], 0, 1))
    return out


def _gap(s, ref):
    return {"gm": s["mean"] - ref["mean"], "gstd": s["std"] - ref["std"],
            "dist": float(np.mean([abs(s["mean"] - ref["mean"]), abs(s["p05"] - ref["p05"]),
                                   abs(s["std"] - ref["std"]),
                                   abs(s["sat95"] - ref["sat95"])]))}


def _mm(d, sel, key, f=lambda x: x):
    return float(np.mean([f(d[i][key]) for i in sel]))


def _splits(corpus):
    ids = [r["id"] for r in corpus["rows"]]
    return (ids, [i for i in ids if i.startswith(HOLDOUT)],
            [i for i in ids if not i.startswith(HOLDOUT)])


def test_gate_fields_are_recorded_and_replay_reproduces_production(corpus):
    """**前提校验**：meta 必须记下 `y_med` 与闸值；按出厂格重放必须复现产品锚
    （否则后面四格比的都不是产品）。"""
    assert corpus["mismatches"] == []
    ids, _hold, _train = _splits(corpus)
    g = _rerender(corpus, pm.PRINT_MIN_DENSITY_WINDOW, SHIP)
    worst, worst_y = 0.0, 0.0
    w = float(pm.PRINT_MIN_DENSITY_WINDOW)
    for r in corpus["rows"]:
        ps = r["ps"]
        assert ps.get("window_gate") == pytest.approx(pm.PRINT_WINDOW_YMED_REFERENCE)
        y = pm.window_y_med(r["pre"], float(ps["density_median"]))
        #: **容差为什么是 1e-4**：meta 里 `window_y_med` / `window_pre_hi` / 锚点全部按
        #: 5 位小数记录，而这里用**舍入过的锚**重放曲线，读数会漂（实测最大 4.7e-5）。
        #: 判据的余量是闸值到最近一帧的距离（0.43 对 0.404 / 0.458 ⇒ **0.026**），
        #: 1e-4 比它紧 260 倍——不是"放宽到通过"，是"贴着重建精度"。
        worst_y = max(worst_y, abs(float(ps.get("window_y_med")) - y))
        assert abs(float(ps.get("window_y_med")) - y) < 1e-4, (
            r["id"], ps.get("window_y_med"), y)
        narrow = r["pre"][2] < r["pre"][0] + w - 1e-6
        note = ps.get("window_note") or ""
        if ps.get("window_raised"):
            assert narrow and y < ps["window_gate"], f"{r['id']}：抬了不该抬的帧"
        elif narrow:
            assert y >= ps["window_gate"] or "不合法" in note, (
                f"{r['id']}：本该挨刀却被跳过，闸与曲线都给了理由吗？y_med={y:.4f} {note}")
        worst = max(worst, abs(g[r["id"]]["mean"] - r["ship"]["mean"]))
    print(f"\n  {len(ids)} 帧；出厂格重放对产品渲染的最大偏差 {worst:.6f}；"
          f"y_med 重建最大漂移 {worst_y:.2e}（闸值余量 0.026）")
    assert worst < 1e-4, "出厂格不复现产品 ⇒ 重建链写错了"


def test_gate_and_window_are_jointly_necessary(corpus):
    """**主锁**：四格联立。放宽 `W` 买训练卷、符号闸保住留出卷，**缺一不成立**。"""
    ids, hold, train = _splits(corpus)
    by = {r["id"]: r for r in corpus["rows"]}
    cells = {}
    for name, (w, tau) in (("无闸@0.55(㉔)", (OLD_W, NO_GATE)),
                           ("无闸@0.60", (pm.PRINT_MIN_DENSITY_WINDOW, NO_GATE)),
                           ("有闸@0.55", (OLD_W, SHIP)),
                           ("有闸@0.60(上线)", (pm.PRINT_MIN_DENSITY_WINDOW, SHIP))):
        g = _rerender(corpus, w, tau)
        cells[name] = {i: _gap(g[i], by[i]["ref"]) for i in ids}

    print("\n  格                    训练分和  留出|亮度|  留出距离  全部距离")
    for k, c in cells.items():
        print(f"  {k:<20}{_mm(c, train, 'gm', abs) + _mm(c, train, 'gstd', abs):>9.4f}"
              f"{_mm(c, hold, 'gm', abs):>12.4f}{_mm(c, hold, 'dist'):>11.4f}"
              f"{_mm(c, ids, 'dist'):>11.4f}", flush=True)

    old, no_gate, gate_low, ship = (cells["无闸@0.55(㉔)"], cells["无闸@0.60"],
                                    cells["有闸@0.55"], cells["有闸@0.60(上线)"])
    def tsum(c):
        return _mm(c, train, "gm", abs) + _mm(c, train, "gstd", abs)

    # ① 放宽 W 买到的是训练卷
    assert tsum(cells["有闸@0.60(上线)"]) < tsum(cells["有闸@0.55"]) - 1e-4, (
        "加闸之后放宽 W 在训练卷上没有收益 ⇒ 0.60 这个值没有依据，退回 0.55")
    # ② 闸买到的是"留出卷不再被压暗"
    assert _mm(ship, hold, "gm", abs) < _mm(no_gate, hold, "gm", abs) - 1e-4, (
        "无闸@0.60 不再比有闸@0.60 更压暗留出 ⇒ 闸没在做事，本轮前提消失")
    assert _mm(ship, hold, "gm", abs) <= _mm(old, hold, "gm", abs) + 1e-4, (
        "上线格在留出卷亮度上比 ㉔ 差 ⇒ 闸没把 ㉔ 让出去的东西换回来")
    assert _mm(ship, hold, "dist") <= _mm(old, hold, "dist") + 1e-4, "上线格让留出距离变差"
    # ③ 单独加闸（不放宽 W）必须与 ㉔ 逐比特相同——闸是**保护性**的，不带来收益
    assert abs(tsum(gate_low) - tsum(old)) < 1e-4, (
        "有闸@0.55 与无闸@0.55 不同 ⇒ 闸在 0.55 上就改变了行为，"
        "那 ㉔ 的账面就不是'无闸形状'，两格的对照全部要重述")
    # ④ 全局距离必须变好，否则这轮只是把误差搬个地方
    assert _mm(ship, ids, "dist") < _mm(old, ids, "dist"), "四统计距离没降"


def test_gate_polarity_is_load_bearing(corpus):
    """**反号负控制**：把闸反过来（只在 `y_med` 高时开火）必须明显更差。
    扫描里它把训练分和打到 0.0664——与"整个下限关掉"**逐比特相同**，
    因为反向闸恰好把所有该抬的帧都拦了、只放行不该抬的。"""
    ids, hold, train = _splits(corpus)
    by = {r["id"]: r for r in corpus["rows"]}
    ship_g = _rerender(corpus, pm.PRINT_MIN_DENSITY_WINDOW, SHIP)
    flip_g = _rerender(corpus, pm.PRINT_MIN_DENSITY_WINDOW, SHIP, flip=True)
    off_g = _rerender(corpus, 0.0, SHIP)
    ship = {i: _gap(ship_g[i], by[i]["ref"]) for i in ids}
    flip = {i: _gap(flip_g[i], by[i]["ref"]) for i in ids}
    off = {i: _gap(off_g[i], by[i]["ref"]) for i in ids}

    def tsum(c):
        return _mm(c, train, "gm", abs) + _mm(c, train, "gstd", abs)

    print(f"\n  训练分和 上线 {tsum(ship):.4f} / 反号 {tsum(flip):.4f} / 关掉下限 {tsum(off):.4f}")
    assert tsum(flip) > tsum(ship) + 1e-3, "反号闸门不比上线好 ⇒ 极性不承重，闸是摆设"
    assert abs(tsum(flip) - tsum(off)) < 1e-3, (
        f"反号应当等价于'关掉下限'（实测 {tsum(flip):.4f} 对 {tsum(off):.4f}）"
        "⇒ 反向后仍有一刀落在别处，说明闸的条件不干净")


def test_gate_blocks_at_least_one_frame_the_floor_want_to_take(corpus):
    """**反空断言**：必须存在"出厂跨度窄于 `W_min`、却被闸拦下"的帧，
    否则 ㉕ 在生产里是死代码，四格表里的差别就只是 `W` 从 0.55 到 0.60。"""
    blocked, fired = [], []
    for r in corpus["rows"]:
        lo, _mid, hi_pre = r["pre"]
        narrow = hi_pre < lo + pm.PRINT_MIN_DENSITY_WINDOW - 1e-6
        if not narrow:
            continue
        (fired if r["ps"]["window_raised"] else blocked).append(
            f"{r['id']}(y_med={r['ps'].get('window_y_med')})")
    print(f"\n  本该挨刀 {len(blocked) + len(fired)} 帧：被抬 {len(fired)}、被闸拦下 {len(blocked)}")
    print("  被拦下：" + ("、".join(blocked) if blocked else "无"))
    assert blocked, "符号闸一帧都没拦 ⇒ ㉕ 是死代码，别把它当机制留着"
    assert fired, "一帧都没抬 ⇒ 下限本身在生产里不生效，四格表没有意义"
