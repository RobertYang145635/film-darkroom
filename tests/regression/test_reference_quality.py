"""真实样片质量回归（v5 §11.1 / §12 P0-4 验收）。

对有人工区域标注的真实样片执行与批处理相同的完整管线
（加载 -> 三类 ROI -> 卷级/帧级片基 -> 技术反相 -> 质量门禁），断言：

- rotation 与标注一致；自动画面检测与标注 IoU >= 0.95（v5 §11.1）；
- 输出不保留大片片边；正常曝光样片 clamp 前剪裁 <=1%（v5 §8.2）；
- 片基参考真实可用（不是回退值 0.5）；
- 无缺色卡/缺校准类阻断；needs_review 允许导出（v5 §8.3）；
- 固定输入重复两次，参数与像素逐比特一致；
- 输出保留片边时，质量统计 mask 仍只覆盖 image_area（v5 §11.1）。

性能：图像按模块缓存，解码一次、断言多次。
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from negative_lab.calibration.annotations import load_annotation
from negative_lab.calibration.border import detect_regions, iou
from negative_lab.io.images import load_linear_rgb
from negative_lab.pipeline.geometry import roi_source_to_oriented
from negative_lab.pipeline.service import process_image

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "fixtures" / "manifest.json"


def _annotated_with_rois():
    """v5 §3.2：回归用真实标注必须真的有三类区域。"""
    if not MANIFEST.exists():
        return []
    doc = json.loads(MANIFEST.read_text(encoding="utf-8"))
    out = []
    for e in doc["images"]:
        if e["decode_status"] != "ok":
            continue
        ann = load_annotation(e["file"])
        if ann is None or not (ann.frame_roi or ann.image_area_roi):
            continue
        if not (ROOT / e["file"]).exists():
            continue
        out.append((e, ann))
    return out


CASES = _annotated_with_rois()
pytestmark = pytest.mark.skipif(
    not CASES, reason="无带三类区域的真实标注样片")


@lru_cache(maxsize=8)
def _load(rel: str) -> np.ndarray:
    return load_linear_rgb(ROOT / rel).rgb


@lru_cache(maxsize=8)
def _prior(rel: str):
    """卷级先验：**两条臂必须拿同一份**（㊴）。

    以前只有 `_run` 的裁切臂带 prior，保留片边臂不带 —— `dmax` 从此不同
    （1298::G200-4 实测 1.3630 vs 0.8062），蓝通道在一臂里削顶 1.5% 而另一臂 0%。
    那被登记成"片边泄漏进门禁统计"（B-15），实际是测试自己换了两个变量。
    """
    pf = ROOT / "fixtures" / "priors" / "KodakGold200-Z8-led.profile.json"
    if "G200" in rel and pf.exists():
        from negative_lab.profiles.profile import load_profile
        return load_profile(pf).to_initial_params()
    return None


@lru_cache(maxsize=8)
def _run(rel: str):
    ann = load_annotation(rel)
    t = _load(rel)
    prior = _prior(rel)
    step = max(1, int(np.ceil(max(t.shape[:2]) / 1600)))
    small = t[::step, ::step, :].copy()
    # 标注 ROI 是 SOURCE 全分辨率坐标：工作分辨率下必须同步缩放，
    # 否则 ROI 越界被钳制成退化区域 -> Dmin 回退（v3 实测教训的复现）。
    h, w = t.shape[:2]
    sh, sw = small.shape[:2]

    def sc(roi):
        if not roi:
            return None
        return (int(round(roi[0] * sh / h)), int(round(roi[1] * sw / w)),
                int(round(roi[2] * sh / h)), int(round(roi[3] * sw / w)))

    from negative_lab.library.models import GeometryRegions
    geo = GeometryRegions(
        rotation_degrees=ann.rotation_degrees,
        output_crop_roi=sc(ann.output_crop_roi or ann.frame_roi),
        image_area_roi=sc(ann.image_area_roi or ann.frame_roi),
        base_sample_rois=[r for r in (sc(x) for x in
                                      (ann.base_sample_rois or []
                                       or ([ann.base_roi] if ann.base_roi else [])))
                          if r])
    pr = process_image(small, geometry=geo, ann=None, prior=prior,
                       )
    return t, ann, pr, step


@pytest.mark.parametrize("e,ann", CASES, ids=[c[0]["file"].split("/")[-1]
                                              for c in CASES])
def test_rotation_matches_annotation(e, ann):
    t, _, _, _ = _run(e["file"])
    o = roi_source_to_oriented((0, 0, 10, 10), t.shape[0], t.shape[1],
                               ann.rotation_degrees)
    assert o is not None and ann.rotation_degrees in (0, 90, 180, 270)


#: **自动画面检测与用户框不一致的已知集合**（㉚b 重建：2026-10-13
#: `tmp_lock_repin.py` 在 `fixtures/manifest.json` 全部 22 张带标注样片上当场打印，
#: 纯自动路径、源图全分辨率；旧表只有 2026-10-09 量的 1299 一卷 9 条）。
#:
#: 用户把 `1298_G200` 整卷也框了之后，这张表从 9 条涨到 **19 条**——22 帧里只有
#: `1298::G200-5` 0.975、`1299::G200-4` 0.974、`1299::G200-9` 0.986 过 0.95，
#: 而且**症状不止一种**（旧表把它们统一写成"窄片边"，低估了缺陷）：
#:
#: - **模式一「窄片边被算进画面」**（IoU 0.80~0.93）：源图 8280×5520 里
#:   每侧只有约 276px（3.3% 宽），边缘对比 1.03~1.13，检测器看不见这条边。
#:   例 `1299::G200-1` auto=(935, 66, 7370, 5522) 的 x 覆盖整幅宽度，
#:   而用户框是 (1152, 342, 7110, 5160)。`1299::G200-7` 更彻底：整帧拒绝检测
#:   （"帧内边缘带与中心亮度差不足（最高对比 1.13）"）。
#: - **模式二「整段画面被截掉」**（IoU 0.32~0.72，1298 五帧 + 1299 两帧）：
#:   检测器只认出一条横带。`1298::G200-7` auto=(1771,462,3685,5434) 高只有 1914px
#:   而用户框高 4800px ⇒ **画面丢掉 60%**；`::G200-9` 丢 48%、`::G200-10` 丢 42%；
#:   `::G200-3` 从 y=3938 才开始，把上部约 2200px 整块切掉。
#:   这一类不是"边带判不准"而是**取景错到不能用**。
#:
#: **产品行为是对的**：用户框是权威（`roi_provenance.flags=manual_regions`
#: 直接跳过自动检测），所以这 19 条不挡导出、也不影响这 22 帧的成品。
#: 但它意味着**没有手框的帧拿不到可用取景**——这是 B-5 的实质加重，登记在
#: B-13 追加④。
#:
#: strict=True 意味着**检测器一旦修到 ≥0.95 就会 XPASS 报错**，届时必须把对应条目
#: 从这张表里删掉（同 `tests/unit/test_probe_roi_scaling.py` 的口径：
#: 已知缺陷要能被证伪）。
AUTO_CROP_GAP = {
    "20260709_SAMPLE-ID(1)/1298_G200/翻拍/G200-1.NEF": "模式一 IoU=0.844",
    "20260709_SAMPLE-ID(1)/1298_G200/翻拍/G200-2.NEF": "模式一 IoU=0.824",
    "20260709_SAMPLE-ID(1)/1298_G200/翻拍/G200-3.NEF": "模式二 IoU=0.441(上部 2200px 被切)",
    "20260709_SAMPLE-ID(1)/1298_G200/翻拍/G200-4.NEF": "模式一 IoU=0.812",
    "20260709_SAMPLE-ID(1)/1298_G200/翻拍/G200-6.NEF": "模式一 IoU=0.927",
    "20260709_SAMPLE-ID(1)/1298_G200/翻拍/G200-7.NEF": "模式二 IoU=0.380(画面丢 60%)",
    "20260709_SAMPLE-ID(1)/1298_G200/翻拍/G200-8.NEF": "模式一 IoU=0.655",
    "20260709_SAMPLE-ID(1)/1298_G200/翻拍/G200-9.NEF": "模式二 IoU=0.461(画面丢 48%)",
    "20260709_SAMPLE-ID(1)/1298_G200/翻拍/G200-10.NEF": "模式二 IoU=0.320(画面丢 42%)",
    "20260709_SAMPLE-ID(1)/1298_G200/翻拍/G200-11.NEF": "模式一 IoU=0.896",
    "20260709_SAMPLE-ID(1)/1299_G200/翻拍/G200-1.NEF": "模式一 IoU=0.818",
    "20260709_SAMPLE-ID(1)/1299_G200/翻拍/G200-2.NEF": "模式一 IoU=0.826",
    "20260709_SAMPLE-ID(1)/1299_G200/翻拍/G200-3.NEF": "模式一 IoU=0.918",
    "20260709_SAMPLE-ID(1)/1299_G200/翻拍/G200-5.NEF": "模式一 IoU=0.833",
    "20260709_SAMPLE-ID(1)/1299_G200/翻拍/G200-6.NEF": "模式一 IoU=0.827",
    "20260709_SAMPLE-ID(1)/1299_G200/翻拍/G200-7.NEF": "整帧拒绝检测",
    "20260709_SAMPLE-ID(1)/1299_G200/翻拍/G200-8.NEF": "模式一 IoU=0.850",
    "20260709_SAMPLE-ID(1)/1299_G200/翻拍/G200-10.NEF": "模式二 IoU=0.683",
    "20260709_SAMPLE-ID(1)/1299_G200/翻拍/G200-11.NEF": "模式二 IoU=0.715",
}


def _iou_cases():
    out = []
    for e, ann in CASES:
        f = e["file"]
        marks = ()
        gap = AUTO_CROP_GAP.get(f)
        if gap:
            marks = (pytest.mark.xfail(
                strict=True,
                reason=f"自动画面检测与用户框不一致（㉚b 实测 {gap} < 0.95）："
                       "用户框为权威，产品路径不受影响；但没有手框的帧"
                       "拿不到可用取景，见 B-13 追加①④ 与 B-5"),)
        out.append(pytest.param(e, ann, marks=marks,
                                id="/".join(f.split("/")[-2:])))
    return out


@pytest.mark.parametrize("e,ann", _iou_cases())
def test_auto_crop_iou(e, ann):
    """自动画面检测 vs 人工标注 IoU >= 0.95（纯自动路径，v5 §11.1）。"""
    t, _, _, _ = _run(e["file"])
    det = detect_regions(t)
    ref = ann.image_area_roi or ann.frame_roi
    assert det.get("image_area") is not None, det["warnings"]
    v = iou(tuple(det["image_area"]), tuple(ref))
    assert v >= 0.95, f"IoU={v:.3f} auto={det['image_area']} ann={ref}"


def test_auto_crop_gap_table_is_not_stale():
    """钉住的已知缺陷必须**仍然**在语料里：标注被删/被改归属后，
    `AUTO_CROP_GAP` 里的条目必须一起删，否则这张表会变成谎话。"""
    files = {e["file"] for e, _ in CASES}
    stale = sorted(set(AUTO_CROP_GAP) - files)
    assert not stale, f"已知缺陷表里有不在用例集里的条目（须删除）：{stale}"


@pytest.mark.parametrize("e,ann", CASES, ids=[c[0]["file"].split("/")[-1]
                                              for c in CASES])
def test_output_clipping_and_export_allowed(e, ann):
    """v5 §8.2：技术正片 clamp 前剪裁 <=1%；needs_review 允许导出。

    ⚠ **B-6（2026-10-16 用户决定）之后"超限"分两种**：**软件新引入**的那一份超线必须挡；
    **全部继承自输入端**（翻拍当场就是死数据）的降级为警告并放行。
    所以这一支判的是"该挡的有没有挡、不该挡的有没有乱挡"，两边都钉。
    """
    _, _, pr, _ = _run(e["file"])
    assert pr.gate.status != "fail" or not pr.gate.blocks_export
    pre = pr.gate.technical_preclip or {}
    attr = pr.gate.clip_attribution or {}
    over = []
    for side in ("low", "high"):
        for ch, frac in zip(("R", "G", "B"), pre.get(side, [0.0] * 3)):
            if float(frac) > 0.01:
                over.append((side, ch, float(frac)))
    for side, ch, frac in over:
        #: 与门禁同一个口径：低端没有归因（`low=None`）⇒ 按总量判；
        #: 高端只有"归因数到的贴顶像素不少于统计量"时才采信拆分。
        a = attr.get(ch) or {}
        intro = frac
        if side == "high" and a.get("high_total") is not None \
                and float(a["high_total"]) >= frac - 1e-9:
            intro = float(a["high_introduced"])
        if intro > 0.01:
            assert pr.gate.blocks_export, (
                f"{ch} {side} 软件新引入 {intro:.2%} >1% 却没挡 ⇒ 门禁被归因放过了")
        else:
            assert not pr.gate.blocks_export, (
                f"{ch} {side} 总量 {frac:.2%}、软件新引入只有 {intro:.2%}"
                f"（其余是采集端死数据）却被挡 ⇒ B-6 的降级没生效："
                f"{pr.gate.failures}")
    assert pr.gate.blocks_export is False, pr.gate.failures
    assert pr.gate.nonfinite_output == 0
    assert pr.rendered_positive_linear is not None


@pytest.mark.parametrize("e,ann", CASES, ids=[c[0]["file"].split("/")[-1]
                                              for c in CASES])
def test_base_reference_valid(e, ann):
    """片基参考必须真的可用（实测教训：ROI 越界 -> 回退 0.5 -> 全图冲淡）。"""
    _, _, pr, _ = _run(e["file"])
    assert pr.base_source != "generic_prior", \
        f"有标注的样片不应退到通用先验：{pr.frame_base.get('warnings')}"
    assert max(pr.report.params.dmin) < 0.9, "Dmin 疑似回退值"
    assert min(pr.report.params.dmin) > 0.005


@pytest.mark.parametrize("e,ann", CASES, ids=[c[0]["file"].split("/")[-1]
                                              for c in CASES])
def test_not_washed_out(e, ann):
    _, _, pr, _ = _run(e["file"])
    lum = np.nan_to_num(pr.result.positive_linear).mean(axis=2)
    p05, p50 = np.quantile(lum, [0.05, 0.5])
    assert p50 < 0.85, f"整体冲淡: p50={p50:.2f}"
    assert p05 < 0.30, f"无真实黑位: p05={p05:.2f}"


@pytest.mark.parametrize("e,ann", CASES, ids=[c[0]["file"].split("/")[-1]
                                              for c in CASES])
def test_rebate_independence(e, ann):
    """v5 §11.1：输出保留片边时，质量统计 mask 仍只覆盖 image_area。

    两次运行用**同一套**三类区域**和同一份先验**（只差 output_crop 是否保留全幅），
    因此若统计随输出裁切变化，就是污染实锤。少对齐任何一个其它入参，
    这条锁量的就不再是裁切 —— ㊴ 之前它把 `prior` 漏在第二臂外面，
    于是把"随先验变化"读成了"片边泄漏"，登记成 B-15。
    """
    t, ann, pr, step = _run(e["file"])
    h, w = t.shape[:2]
    sh, sw = t[::step, ::step, :].shape[:2]
    ann_run = load_annotation(e["file"])

    def sc(roi):
        if not roi:
            return None
        return (int(round(roi[0] * sh / h)), int(round(roi[1] * sw / w)),
                int(round(roi[2] * sh / h)), int(round(roi[3] * sw / w)))

    base_rois = [sc(x) for x in (ann_run.base_sample_rois or []
                                 or ([ann_run.base_roi] if ann_run.base_roi else []))]
    area = sc(ann_run.image_area_roi or ann_run.frame_roi)
    from negative_lab.library.models import GeometryRegions
    geo_keep = GeometryRegions(
        rotation_degrees=ann.rotation_degrees,
        image_area_roi=area,
        output_crop_roi=(0, 0, sh, sw),
        base_sample_rois=[r for r in base_rois if r])
    full = process_image(t[::step, ::step, :], geometry=geo_keep,
                         prior=_prior(e["file"]), keep_rebate=True)
    #: **容差的依据是机制，不是"调到能过为止"**。两次运行的统计 mask 都应只覆盖
    #: image_area，但 mask 是在**各自的输出坐标**里用整数减法得到的
    #: （`ay0 = area_o[0] - cy0`），保留片边与否会让 `cy0/cx0` 差一个偏移，
    #: 于是边界上可能差 1 像素。
    #: 真把片边算进统计区是多大一件事，由 `test_rebate_leak_would_be_visible`
    #: 当场量（2026-10-16，blue=0.85）：统计区 613816→1269600 像素（片边占 51.7%）
    #: 时 `preclip[high]` 动 3.803e-02 = **5836×容差**；1299::G200-1 是 37.2% / 4626×。
    #: 也就是说这条断言量的东西比它的容差大三个半数量级，不是空锁。
    #:
    #: **㊴ 更正**：2026-10-15 把这一支登记成 B-15"门禁统计在保留片边时不套 mask"，
    #: 并在当天写成"同一份代码两次运行坏的帧不一样"。两条都不成立：
    #: ① 它是确定的 —— `tmp_b15_arms.py` 两遍都是同样 3 帧（1298::G200-4/-5/-7）；
    #: ② 病根不在管线，在本测试的第二臂**漏传 `prior`** —— 裁切臂 `dmax=1.3630`、
    #: 保留片边臂 `dmax=0.8062`，蓝通道在前者削顶 0.0%、后者 1.481e-02，
    #: 于是"随输出裁切变化"其实是"随先验变化"。补齐 prior 后 22 帧的
    #: `preclip`/`output_clip` 逐通道偏差**实测 0.000e+00**（不是"小于容差"）。
    n_pr = int(np.count_nonzero(pr.valid_mask))
    n_full = int(np.count_nonzero(full.valid_mask))
    assert abs(n_pr - n_full) <= 4, (
        f"保留片边后统计区像素数变了 {n_pr} → {n_full}，mask 不再只覆盖 image_area")
    tol = 4.0 / max(min(n_pr, n_full), 1)

    def _close(x, y):
        a = np.atleast_1d(np.asarray(x, np.float64)).ravel()
        b = np.atleast_1d(np.asarray(y, np.float64)).ravel()
        return a.shape == b.shape and bool(np.all(np.abs(a - b) <= tol))

    for key in ("low", "high"):
        assert _close(full.gate.technical_preclip[key],
                      pr.gate.technical_preclip[key]), (
            f"preclip[{key}] 随输出裁切变化："
            f"{full.gate.technical_preclip[key]} vs {pr.gate.technical_preclip[key]}，"
            f"容差 {tol:.2e}（=4 像素）")
        assert _close(full.gate.output_clip[key], pr.gate.output_clip[key]), (
            f"output_clip[{key}] 随输出裁切变化："
            f"{full.gate.output_clip[key]} vs {pr.gate.output_clip[key]}，"
            f"容差 {tol:.2e}（=4 像素）")
    assert full.geometry.output_size[1] > pr.geometry.output_size[1], \
        "输出必须真的保留了片边"


#: 可见性对照（锁的负控制）：**同一张图、同一套参数**，只把统计区从
#: `valid_mask` 换成"整幅含片边"（`mask=None`），量这把尺子看不看得见泄漏。
#: 看不见的话上面那条断言就是空锁。
#:
#: 为什么不用"把 `image_area_roi` 改成全幅"来做这个对照：那会同时改掉
#: ①`_upright_rotation` 看到的画框纵横（1299 的标注框高>宽、规则补 270°，
#: 全幅框宽>高 ⇒ 不补，画面直接转歪）和 ②`frame_base_stats` 里片基条与画面的
#: 包含关系（片基条本就长在片边上）。两个混淆一起进来，量到的就不是泄漏。
LEAK_CANDIDATES = ["20260709_SAMPLE-ID(1)/1298_G200/翻拍/G200-4.NEF",
                   "20260709_SAMPLE-ID(1)/1299_G200/翻拍/G200-1.NEF"]


@pytest.mark.parametrize("rel", LEAK_CANDIDATES)
def test_rebate_leak_would_be_visible(rel):
    """把片边算进统计区，preclip 必须**大幅**变化（远大于 4 像素容差）。"""
    t, ann, _, step = _run(rel)
    h, w = t.shape[:2]
    small = t[::step, ::step, :]
    sh, sw = small.shape[:2]

    def sc(roi):
        if not roi:
            return None
        return (int(round(roi[0] * sh / h)), int(round(roi[1] * sw / w)),
                int(round(roi[2] * sh / h)), int(round(roi[3] * sw / w)))

    from negative_lab.library.models import GeometryRegions
    from negative_lab.pipeline.density import invert, preclip_ratios
    base = [r for r in (sc(x) for x in (ann.base_sample_rois or []
            or ([ann.base_roi] if ann.base_roi else []))) if r]
    leak_geo = GeometryRegions(rotation_degrees=ann.rotation_degrees,
                               image_area_roi=sc(ann.image_area_roi or ann.frame_roi),
                               output_crop_roi=(0, 0, sh, sw), base_sample_rois=base)
    good = process_image(small, geometry=leak_geo, prior=_prior(rel),
                         keep_rebate=True)
    #: `good.t` 就是这一帧保留片边时的反相输入（画面 + 一圈片边）。
    #: mask=None ⇒ 统计区覆盖包括片边在内的每一个像素 = 泄漏的样子。
    leaked = preclip_ratios(invert(good.t, good.report.params,
                                   mask=None).diagnostics)
    n_ok = int(np.count_nonzero(good.valid_mask))
    n_leak = int(good.t.shape[0] * good.t.shape[1])
    tol = 4.0 / max(n_ok, 1)

    def _gap(key):
        a = np.atleast_1d(np.asarray(leaked[key], np.float64))
        b = np.atleast_1d(np.asarray(good.gate.technical_preclip[key], np.float64))
        return float(np.abs(a - b).max()) if a.shape == b.shape else float("inf")

    hi, lo = _gap("high"), _gap("low")
    print(f"\n  {rel.split('/')[1]}/{rel.split('/')[-1]}：统计区 {n_ok} → {n_leak} 像素"
          f"（片边占 {(n_leak - n_ok) / max(n_leak, 1):.1%}），"
          f"preclip high 差 {hi:.3e} = {hi / tol:.0f}×容差，"
          f"low 差 {lo:.3e} = {lo / tol:.0f}×容差")
    assert n_leak > n_ok, "对照没有真的把片边算进统计区"
    assert max(hi, lo) > 100 * tol, (
        f"这把尺子看不见 {(n_leak - n_ok) / max(n_leak, 1):.1%} 的片边泄漏"
        f"（high {hi:.3e} / low {lo:.3e}，容差 {tol:.2e}）——"
        "test_rebate_independence 成了空锁，容差或统计口径要重定")


@pytest.mark.parametrize("e,ann", CASES, ids=[c[0]["file"].split("/")[-1]
                                              for c in CASES])
def test_idempotent(e, ann):
    _, _, a, _ = _run(e["file"])
    _, _, b, _ = _run(e["file"])
    assert a.report.params.to_parameters_dict() == b.report.params.to_parameters_dict()
    assert np.array_equal(a.result.positive_linear, b.result.positive_linear)


def test_no_calibration_blocking_anywhere():
    """v5 §8.1：任何真实样片都不允许出现缺色卡类阻断。"""
    for e, _ann in CASES:
        _, _, pr, _ = _run(e["file"])
        assert pr.gate.status != "needs_calibration"
        assert pr.gate.blocks_export is False or pr.gate.status == "fail"
        if pr.gate.status == "fail":
            assert all("剪裁" in f or "NaN" in f for f in pr.gate.failures), \
                pr.gate.failures


ROLL_GATE = "1298_G200"
MOTIVATING = "1298_G200::G200-7"


@lru_cache(maxsize=1)
def _roll_path():
    """**整卷生产路径**跑一遍 1298（与联系表 / 批处理同一个 `process_roll` 入口）。

    为什么必须有这一支：`_run` 走的是单帧 + 卷级先验 `prior`，
    而交付路径是整卷、不带 profile 先验 —— ㊴ 实测同一帧两口径
    `dmax` 1.3630 对 0.7391、蓝通道高端剪裁 **0.0% 对 1.4884%**。
    只在单帧口径上绿，等于**没测过那 10 帧被挡的帧**（本轮就是这样差点放过一个
    "归因 round 到 6 位、与统计量差 2.6e-7 就不降级"的错）。
    """
    from negative_lab.library.fingerprint import capture_fingerprint
    from negative_lab.library.models import FrameRecord, GeometryRegions
    from negative_lab.pipeline.roll_export import geometry_from_annotation
    from negative_lab.pipeline.service import process_roll

    man = json.loads((ROOT / "fixtures" / "acceptance" / "manifest.json")
                     .read_text(encoding="utf-8"))
    frs = [f for f in man["frames"]
           if f.get("required") and f["roll"] == ROLL_GATE
           and (ROOT / f["file"]).exists()]
    assert len(frs) >= 8, f"验收清单里 {ROLL_GATE} 只剩 {len(frs)} 帧，锁本身失效"
    recs, imgs, ids = [], [], []
    for fr in frs:
        loaded = load_linear_rgb(ROOT / fr["file"])
        arr = loaded.rgb
        step = max(1, int(np.ceil(max(arr.shape[:2]) / 1200.0)))
        small = arr[::step, ::step, :].copy()
        ann = load_annotation(fr["file"])
        geo = GeometryRegions()
        if ann is not None and (ann.image_area_roi or ann.base_sample_rois):
            g = geometry_from_annotation(ann)      # 与批处理/整卷导出同一个出口
            sy, sx = small.shape[0] / arr.shape[0], small.shape[1] / arr.shape[1]

            def sc(roi, _sy=sy, _sx=sx):
                if not roi:
                    return None
                o = (int(round(roi[0] * _sy)), int(round(roi[1] * _sx)),
                     int(round(roi[2] * _sy)), int(round(roi[3] * _sx)))
                return o if (o[2] - o[0] >= 4 and o[3] - o[1] >= 4) else None

            geo = GeometryRegions(
                rotation_degrees=g.rotation_degrees,
                fine_rotation_degrees=g.fine_rotation_degrees,
                mirror_horizontal=g.mirror_horizontal,
                output_crop_roi=sc(g.output_crop_roi),
                image_area_roi=sc(g.image_area_roi),
                base_sample_rois=[q for q in (sc(x) for x in g.base_sample_rois) if q])
        recs.append(FrameRecord(
            source_path=fr["file"], source_sha256=fr["source_sha256"],
            input_facts=dict(loaded.meta),
            capture_fingerprint=capture_fingerprint(
                loaded.meta, source_path=fr["file"]).value,
            geometry=geo, orientation_decided=ann is not None))
        imgs.append(small)
        ids.append(fr["id"])
        del arr, loaded
    _, results = process_roll(list(zip(recs, imgs)))
    assert len(results) == len(ids), "整卷返回的帧数与送进去的不符"
    return dict(zip(ids, results))


def test_roll_path_releases_purely_inherited_clipping():
    """B-6（2026-10-16 用户决定）在**交付路径**上的证据：超限但纯继承 ⇒ 放行并留痕。"""
    out = _roll_path()
    over = sorted(i for i, pr in out.items() if pr.gate.clip_exceeded)
    blocked = sorted(i for i, pr in out.items() if pr.gate.blocks_export)
    for i in over:
        a = (out[i].gate.clip_attribution or {}).get("B") or {}
        print(f"  {i} 超线：B 引入 {float(a.get('high_introduced', 0.0)):.4%} / "
              f"继承 {float(a.get('high_inherited', 0.0)):.4%}")
    print(f"  整卷 {len(out)} 帧，超 1% 线 {len(over)} 帧 {over}；被挡 {blocked}")
    assert over, (
        f"{ROLL_GATE} 整卷没有任何超限帧 ⇒ 这条锁空转："
        "语料或口径变了，回来重钉（别删）")
    assert not blocked, f"纯继承的超限帧仍被挡：{blocked} ⇒ B-6 的降级没生效"
    pr = out[MOTIVATING]
    assert pr.gate.clip_exceeded, "降级不等于抹掉事实：`clip_exceeded` 必须还立着"
    assert any("不挡导出" in w for w in pr.gate.warnings), pr.gate.warnings
    assert any("采集端就已丢失" in w for w in pr.gate.warnings), pr.gate.warnings
