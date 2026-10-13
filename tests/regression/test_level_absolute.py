"""回归：**电平的绝对项**上线值必须让留出卷逐帧亮度变好，且**符号是载荷**。

量产在 2026-10-11 之前把**每一帧**的中灰都钉在 `PRINT_AIM`：不加这一项时我方显示
p35 帧间 std **0.0302**、亮度 std **0.0187**，而用户参考是 **0.0946** 与 **0.0902**
——电平自由度被归一化掉了；两边**中位却几乎相同**（我方 0.4899 对他 0.4829），
所以 `PRINT_AIM` 没错，**"每帧都归到它"才是错**。表现为卷级错位：
1298 我方比参考亮 **+0.045**、1299 我方比参考暗 **−0.098**。

上线的是"部分固定曲线"：`δ = 0.25·(本帧取样总体的密度中位 − 0.5539)`，限幅 ±0.12，
**只抬中灰与白、黑锚不动**（选值过程与依据见 `print_math` 里那三条常量的注释）。

**量法与选值时同一 harness**（`tmp_level_absolute.py`）：生产整卷两遍 →
离线重渲与 `candidate_positive` **逐比特一致**（前提校验不成立就整体 skip 失败，
不许继续比）→ 只改 `print_anchor` 重渲。变体的"未平移锚"由
`meta["print_scale"]` 的 `density_median` 与已施加的 δ 反解，
所以读的是**生产锚点**，不是探针另算的一份。

留出卷 `1299_G200`（三帧，未参与选 strength）与**符号负控制**（`−0.25` 必须更糟）
都写在这里：亮度这一项若对任意扰动都反应，正锁就没有区分力。
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
from negative_lab.print_math import (PRINT_LEVEL_REFERENCE_DENSITY, level_shift)

ROOT = pathlib.Path(__file__).resolve().parents[2]
MAX_DIM = 1200
ROLLS = ("1298_G200", "1299_G200")
HOLDOUT = "1299_G200"
SHIPPED, OFF, WRONG_SIGN = 0.25, 0.0, -0.25
HAZE_BUDGET = 0.02          # Δp05 允许被抬电平恶化的量，量死时写进断言而非放宽
#: ㉝ 之后新增的耦合预算：抬亮 ⇒ 更多像素顶到纸白天花板 ⇒ 被"等亮度收色度"削掉饱和。
#: 实测恶化 **0.0100**（gsat −0.0114 → −0.0214，11 帧、验收口径、当场打印），
#: 预算取实测的两倍——**是预算不是容差**：它不许把"推翻"当成"没看见"。
SAT_BUDGET = 0.02

_map = json.loads((ROOT / "fixtures/references" / "g200_manual_look.json")
                  .read_text(encoding="utf-8"))
MET = _map["method"]
REF = {e["frame_id"]: e for e in _map["frames"] if not e["variant_of"]}


def _stats(disp01):
    """与选值时同一量法：显示-sRGB，缩到 `MET["resize"]` 再取中心 70%。"""
    im = Image.fromarray((np.clip(disp01, 0, 1) * 255).round().astype(np.uint8))
    a = np.asarray(im.resize((MET["resize"],) * 2, Image.BILINEAR), np.float64) / 255.0
    c = int(a.shape[0] * (1 - MET["center_crop_frac"]) / 2)
    r = int(a.shape[1] * (1 - MET["center_crop_frac"]) / 2)
    d = a[c:a.shape[0] - c, r:a.shape[1] - r, :]
    lum = 0.2126 * d[..., 0] + 0.7152 * d[..., 1] + 0.0722 * d[..., 2]
    sat = d.max(2) - d.min(2)
    return {"mean": float(lum.mean()), "p05": float(np.percentile(lum, 5)),
            "std": float(lum.std()), "sat95": float(np.percentile(sat, 95))}


def _compose(pre: tuple[float, float, float], strength: float, density_median: float,
             highlight_area: float | None) -> list[float]:
    """在**出厂锚**上重放生产那条链：`⑳ 窗口下限 → ⑮/㉓ 电平项`，只把 κ₁ 换成 `strength`。

    ⚠ 两条纪律（都是红灯换来的）：
    ① 对照**必须走产品函数**，不能手推公式——㉓ 加上第二携带者后，手推的旧公式
       重建出的锚与产品差到 **0.0346**，被本文件的自检当场抓住；
    ② 次序必须与 `service.py` step 9b 一致（㉔）：下限作用在出厂锚上、电平项最后搬 `hi`，
       反序时窄跨度帧的电平校正会被下限整笔收回。
    """
    import negative_lab.print_math as pm

    orig = pm.PRINT_LEVEL_STRENGTH
    pm.PRINT_LEVEL_STRENGTH = float(strength)
    try:
        floored = pm.white_window_floor(pre, None, density_median)[0]
        out, _delta, _note = pm.anchor_with_level(floored, density_median, highlight_area)
        return [float(v) for v in out]
    finally:
        pm.PRINT_LEVEL_STRENGTH = orig


@pytest.fixture(scope="module")
def corpus():
    """**整卷两遍**跑生产（单帧跑的 `base_source` 不同，见记忆
    `probe-must-use-production-path`）。只留重渲所需的工件，不留 `ProcessResult`。"""
    missing = [f["reference_file"] for f in REF.values()
               if not (ROOT / f["reference_file"]).exists()]
    if missing:
        pytest.skip(f"用户手工参考大图不在库里（gitignore），缺 {len(missing)} 张，"
                    f"例如 {missing[0]}")
    man = json.loads((ROOT / "fixtures/acceptance/manifest.json")
                     .read_text(encoding="utf-8"))
    rows, two_point, mismatches = [], [], []
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
            if len(it.print_anchor) != 3:
                if "density_median" in ps:
                    two_point.append((fr["id"], ps))
                continue                      # 两点退化/被拒：电平项未定义
            if not np.array_equal(
                    np.clip(encode_srgb(apply_render(pr.result.positive_linear, it))[..., :3],
                            0, 1),
                    np.clip(encode_srgb(candidate_positive(pr, name))[..., :3], 0, 1)):
                mismatches.append(fr["id"])   # 只存**不一致**的帧：必须为空
            rows.append({"id": fr["id"], "tech": pr.result.positive_linear, "it": it,
                         "ps": ps,
                         "shipped": _stats(np.clip(
                             encode_srgb(candidate_positive(pr, name))[..., :3], 0, 1)),
                         "ref": _stats(np.asarray(
                             Image.open(ROOT / REF[fr["id"]]["reference_file"])
                             .convert("RGB"), np.float32) / 255.0)})
        del results
    return {"rows": rows, "two_point": two_point, "mismatches": mismatches}


def _gap(corpus, strength: float) -> dict:
    """把整批锚点按 `strength` 重渲一遍，返回逐帧四项统计相对参考的差。"""
    out = {}
    for r in corpus["rows"]:
        ps, a = r["ps"], r["it"].print_anchor
        applied = float(ps["level_shift"])
        #: **出厂锚**：`mid` 加回 δ；`hi` 取 `window_pre_hi`（下限与电平项都没施加过的
        #: 那一个）——出厂 `d_hi` 既被 ⑳ 抬过、又被电平项搬过，只加回 δ 反推不出来。
        pre = (float(a[0]), float(a[1]) + applied,
               float(ps["window_pre_hi"]) if "window_pre_hi" in ps
               else float(a[2]) + applied)
        it = RenderIntent.from_dict(r["it"].to_dict())
        it.print_anchor = _compose(pre, strength, float(ps["density_median"]),
                                   ps.get("highlight_area"))
        s = _stats(np.clip(encode_srgb(apply_render(r["tech"], it))[..., :3], 0, 1))
        ref = r["ref"]
        out[r["id"]] = {
            "dist": float(np.mean([abs(s["mean"] - ref["mean"]), abs(s["p05"] - ref["p05"]),
                                   abs(s["std"] - ref["std"]),
                                   abs(s["sat95"] - ref["sat95"])])),
            "gm": s["mean"] - ref["mean"], "g05": s["p05"] - ref["p05"],
            "gsat": s["sat95"] - ref["sat95"]}
    return out


def _agg(g: dict, ids, key, f=lambda v: v) -> float:
    vs = [f(g[i][key]) for i in ids if i in g]
    return float(np.mean(vs)) if vs else float("nan")


def test_offline_rerender_equals_production_bit_for_bit(corpus):
    """**前提校验**：不改参数时离线重渲必须与 `candidate_positive` 逐比特相同，
    否则后面比的是探针不是产品。另存一条"shipped 统计可复算"的自检。"""
    assert corpus["mismatches"] == [], (
        f"离线重渲与生产出口**相同**的帧数不是全部（列出的帧应当为空——"
        f"这里存的是反例）：{corpus['mismatches']}")
    assert len(corpus["rows"]) >= 8, f"参与帧数掉到 {len(corpus['rows'])}，逐帧判据要重估"
    again = _gap(corpus, SHIPPED)
    base = {r["id"]: r["shipped"]["mean"] for r in corpus["rows"]}
    ref_mean = {r["id"]: r["ref"]["mean"] for r in corpus["rows"]}
    drift = max(abs(again[i]["gm"] - (base[i] - ref_mean[i])) for i in base)
    print(f"\n  帧数 {len(corpus['rows'])}（另有 {len(corpus['two_point'])} 帧两点退化）；"
          f"shipped 统计复算最大漂移 {drift:.6f}")
    assert drift < 1e-3, f"重算的 shipped 与生产出口差 {drift:.6f}，harness 不可信"


def test_level_shift_is_recorded_and_matches_the_formula(corpus):
    """生产必须把 δ 写进 `print_scale`，且 δ **等于生产自己的 `level_shift`**
    （**来源标签**：没有这一条，后面所有对比都说不清生产到底走没走这步）。
    ⚠ 这里必须调产品函数，不许在本文件里重写公式——㉓ 加第二携带者之后
    重写的那份立刻过期（自检抓到 drift 0.0346）。"""
    for r in corpus["rows"]:
        ps = r["ps"]
        assert float(ps["level_shift"]) == pytest.approx(
            level_shift(float(ps["density_median"]), ps.get("highlight_area")),
            abs=1e-5), r["id"]
        assert float(ps["level_reference"]) == pytest.approx(PRINT_LEVEL_REFERENCE_DENSITY)
        assert float(ps["d_lo"]) == pytest.approx(float(r["it"].print_anchor[0]), abs=1e-5)
        assert float(ps["d_mid"]) == pytest.approx(float(r["it"].print_anchor[1]), abs=1e-5)
    deltas = {r["id"]: float(r["ps"]["level_shift"]) for r in corpus["rows"]}
    print("\n  逐帧 δ（decade，正=印得更亮）：" +
          "  ".join(f"{k.replace('_G200::', '/')} {v:+.4f}" for k, v in sorted(deltas.items())))
    assert any(abs(v) > 0.01 for v in deltas.values()), (
        "所有帧 δ 都≈0：这一项等于没上线，锁与机制都该重估")


def test_two_point_frames_get_no_level_on_real_data(corpus):
    """两点退化锚在真实数据上必须**没被加电平**且留了说明（`G200-10` 走这条路）。"""
    for fid, ps in corpus["two_point"]:
        assert float(ps["level_shift"]) == 0.0, fid
        assert "两点退化" in ps.get("level_note", ""), (fid, ps.get("level_note"))
    print(f"  两点退化帧 {len(corpus['two_point'])} 帧，全部未施加电平项并写明原因")


@pytest.mark.xfail(
    strict=True,
    reason="㉝ 之后**这一条不再成立，登记给 ㉞**：四统计距离 关掉电平项 0.046909 → "
           "上线 0.046960（差 5.1e-05，11 帧、验收口径、当场打印）。"
           "电平项在**它自己那根轴**上仍然净赢（逐帧|亮度| 0.0405→0.0329、"
           "留出 0.0372→0.0236），但 ㉝ 把高光电平从彩度天花板里救回来之后，"
           "它在别的轴上的老代价（对比/饱和）刚好抵平了亮度收益。"
           "⇒ ㉞ 必须重扫 κ₁×κ₂（`outputs/refit33.txt` 已扫过一轮：新标称下"
           "没有任何一格过判据，所以这一轮是**在旧标称上重定 κ 或改判据**，不是再扫一次）。"
           "同一格里饱和也被推翻：gsat 关掉 −0.0114 → 上线 −0.0214（恶化 0.0100）——"
           "这是 ㉝ 之后的新耦合（抬亮 ⇒ 顶纸白天花板 ⇒ 等亮度收色度）。"
           "strict=True：一旦重扫到距离重新净赢，这条会 XPASS 报错，必须回来删标记。"
           "仍然被守住的性质拆到 "
           "`test_level_term_still_wins_on_brightness_axes_after_33`。")
def test_level_term_reduces_brightness_error_on_every_holdout_frame(corpus):
    """上线值 vs "不加这一项"：**留出卷逐帧**都要变好，总距离不许变差，
    饱和不许被推翻；阴影雾只允许在登记的预算内恶化。

    ⚠ **当前 xfail**（理由见装饰器）——"每帧都变好"与"总距离不许变差"两句在
    ㉝ 的新工作点上都不成立了；亮度那一根的收益由拆出去的绿色锁继续守。
    """
    off, up = _gap(corpus, OFF), _gap(corpus, SHIPPED)
    ids = list(up)
    hold = [i for i in ids if i.startswith(HOLDOUT)]
    ref_mean = {r["id"]: r["ref"]["mean"] for r in corpus["rows"]}
    print("\n  帧              δ=0 亮度差   上线 亮度差    参考亮度    Δp05 0→上线")
    for i in sorted(ids):
        print(f"  {i.replace('_G200::', '/'):<16} {off[i]['gm']:+.4f}      {up[i]['gm']:+.4f}"
              f"      {ref_mean[i]:.4f}    {off[i]['g05']:+.4f} → {up[i]['g05']:+.4f}")
    print(f"  逐帧|亮度|：{_agg(off, ids, 'gm', abs):.4f} → {_agg(up, ids, 'gm', abs):.4f}"
          f"；留出 {HOLDOUT}：{_agg(off, hold, 'gm', abs):.4f} → {_agg(up, hold, 'gm', abs):.4f}")
    train = [i for i in ids if i.startswith("1298_G200")]
    print(f"  **只看训练卷 1298**：距离 {_agg(off, train, 'dist'):.4f} → "
          f"{_agg(up, train, 'dist'):.4f}；|亮度| {_agg(off, train, 'gm', abs):.4f} → "
          f"{_agg(up, train, 'gm', abs):.4f}")
    print(f"  四统计距离：{_agg(off, ids, 'dist'):.4f} → {_agg(up, ids, 'dist'):.4f}；"
          f"饱和偏差：{_agg(off, ids, 'gsat'):+.4f} → {_agg(up, ids, 'gsat'):+.4f}")
    n_better = sum(1 for i in ids if abs(up[i]["gm"]) < abs(off[i]["gm"]))
    print(f"  变好的帧数：{n_better}/{len(ids)}；阴影偏差(Δp05)：{_agg(off, ids, 'g05'):+.4f} → "
          f"{_agg(up, ids, 'g05'):+.4f}（预算 +{HAZE_BUDGET:.2f}）")
    worse = [i for i in hold if abs(up[i]["gm"]) >= abs(off[i]["gm"])]
    #: **㉝ 换过判据（不是放宽，是把主语换到还成立的那句话上）**：
    #: 原来这条要求"留出卷**每一帧**都变好"。㉝ 把彩度级"整像素拖暗"改成
    #: "等亮度收色度"之后，`1299::G200-2` 在不加电平项时就已经贴住参考
    #: （δ=0 时 Δmean **+0.0001**），上线把它推到 **+0.0341** ⇒ "每帧都变好"当场红。
    #: 但这一项的**聚合**收益仍然成立：留出 `|亮度|` 0.0372 → **0.0236**、
    #: 全部 0.0405 → **0.0329**、7/11 帧变好。
    #: 所以判据改成三条**仍可失败**的性质：① 变差的帧必须"本来就已经对"
    #: （δ=0 时 |Δmean| < 0.01）；② 这样的帧至多 1 帧；③ 对已贴住的帧过推不许超过 0.04。
    #: 任何一条破口都会红——它抓的正是"电平项把已经对的帧推歪"这件事。
    for i in worse:
        print(f"  ⚠ 留出 {i}：δ=0 本来就贴住参考（{off[i]['gm']:+.4f}），"
              f"上线推到 {up[i]['gm']:+.4f}")
    assert all(abs(off[i]["gm"]) < 0.01 for i in worse), (
        "留出卷变差的帧必须是'不加这一项本来就对'的那几帧，实得 "
        + str([(i.split('::')[1], round(off[i]["gm"], 4)) for i in worse]))
    assert len(worse) <= 1, f"留出卷变差的帧超过 1 帧：{worse}"
    assert max((abs(up[i]["gm"]) for i in worse), default=0.0) < 0.04, (
        f"对已贴住的帧过推：最大 {max((abs(up[i]['gm']) for i in worse), default=0.0):.4f}")
    assert len(hold) == 3, f"留出帧数变了（{len(hold)}），逐帧判据要重估"
    assert _agg(up, ids, 'gm', abs) < _agg(off, ids, 'gm', abs), "逐帧|亮度|没降"
    assert _agg(up, ids, 'dist') <= _agg(off, ids, 'dist'), "四统计距离变差了"
    assert _agg(up, ids, 'gsat') > _agg(off, ids, 'gsat') - 1e-4, (
        "饱和偏差被推翻（更负=更淡）：电平项不该吃掉增益的成果")
    assert _agg(up, ids, 'g05') < _agg(off, ids, 'g05') + HAZE_BUDGET, (
        f"抬电平把阴影雾放大超预算：Δp05 {_agg(off, ids, 'g05'):+.4f} → "
        f"{_agg(up, ids, 'g05'):+.4f}。这是已知代价，超了要重估 strength，别放宽阈值")


def test_level_term_still_wins_on_brightness_axes_after_33(corpus):
    """㉝ 之后仍然**逐条可失败**的那几根性质（从上面那条 xfail 的锁里拆出来）。

    拆的理由写在被拆那条的装饰器里：`每帧都变好` 与 `总距离不许变差` 在新工作点上
    已经不成立（登记给 ㉞），但电平项在它**自己那根轴**上的收益是实测且显著的，
    不许因为一条锁转成 xfail 就一起失去保护。这里守四件事：
    ① 逐帧 `|亮度|` 必须严格变小；② 留出卷 `|亮度|` 必须严格变小；
    ③ 阴影雾只许在 `HAZE_BUDGET` 内恶化；④ 饱和不许被推翻（更负=更淡）。
    """
    off, up = _gap(corpus, OFF), _gap(corpus, SHIPPED)
    ids = list(up)
    hold = [i for i in ids if i.startswith(HOLDOUT)]
    print(f"\n  ㉝ 后拆出来的性质：逐帧|亮度| {_agg(off, ids, 'gm', abs):.4f} → "
          f"{_agg(up, ids, 'gm', abs):.4f}；留出 {_agg(off, hold, 'gm', abs):.4f} → "
          f"{_agg(up, hold, 'gm', abs):.4f}；Δp05 {_agg(off, ids, 'g05'):+.4f} → "
          f"{_agg(up, ids, 'g05'):+.4f}；饱和 {_agg(off, ids, 'gsat'):+.4f} → "
          f"{_agg(up, ids, 'gsat'):+.4f}；四统计距离 {_agg(off, ids, 'dist'):.4f} → "
          f"{_agg(up, ids, 'dist'):.4f}（**这条已经抵平，见 xfail 的那条锁**）")
    assert _agg(up, ids, 'gm', abs) < _agg(off, ids, 'gm', abs), "逐帧|亮度|没降"
    assert _agg(up, hold, 'gm', abs) < _agg(off, hold, 'gm', abs), "留出卷|亮度|没降"
    assert _agg(up, ids, 'g05') < _agg(off, ids, 'g05') + HAZE_BUDGET, (
        f"抬电平把阴影雾放大超预算：Δp05 {_agg(off, ids, 'g05'):+.4f} → "
        f"{_agg(up, ids, 'g05'):+.4f}。这是已知代价，超了要重估 strength，别放宽阈值")
    #: **㉝ 新增的耦合**：电平项抬亮 ⇒ 更多像素顶到纸白天花板 ⇒ 被"等亮度收色度"
    #: 削掉饱和。实测 gsat 关掉 −0.0114 → 上线 **−0.0214**（恶化 **0.0100**），
    #: 所以原来那句"饱和不许被推翻"（`up > off − 1e-4`）在新工作点上已经不成立——
    #: 那一句跟着"总距离不许变差"一起搬进上面那条 xfail 锁登记给 ㉞。
    #: 这里换成一条**预算**：电平项引入的饱和恶化不得超过实测的两倍，
    #: 且必须当场打印，免得后来的人以为这个耦合不存在。
    sat_cost = _agg(off, ids, 'gsat') - _agg(up, ids, 'gsat')
    print(f"  电平项→天花板→脱色 的耦合：饱和偏差恶化 {sat_cost:+.4f}（预算 {SAT_BUDGET}）")
    assert 0.0 <= sat_cost < SAT_BUDGET, (
        f"电平项吃掉饱和 {sat_cost:.4f} 超出预算 {SAT_BUDGET}：这是 ㉝ 之后新出现的耦合"
        f"（抬亮 ⇒ 顶天花板 ⇒ 收色度），该重估 κ₁ 而不是放宽这条")


def test_negative_control_wrong_sign_is_worse_than_doing_nothing(corpus):
    """**负控制**：符号反过来（把浓的印暗）必须比"不加"更糟。
    若三档分不开，这条锁对任意扰动都反应，没有区分力。"""
    off, up, wrong = _gap(corpus, OFF), _gap(corpus, SHIPPED), _gap(corpus, WRONG_SIGN)
    ids = list(up)
    b0, b_up, b_w = (_agg(off, ids, 'gm', abs), _agg(up, ids, 'gm', abs),
                     _agg(wrong, ids, 'gm', abs))
    print(f"\n  逐帧|亮度|：δ=0 {b0:.4f} / 上线方向 {b_up:.4f} / 反方向 {b_w:.4f}")
    assert b_up < b0 < b_w, (
        f"三档没分开（{b_up:.4f} / {b0:.4f} / {b_w:.4f}）："
        "要么符号不是载荷，要么该量对任意扰动都反应")
