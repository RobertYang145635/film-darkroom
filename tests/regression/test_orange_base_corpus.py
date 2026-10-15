"""语料锁：画面内片基检测在**真帧**上的两头——坏帧被救，用户画的框不许被推翻。

对应实测（`tmp_orange_base_estimator.py` / `tmp_latd_roll_aim.py`，2026-10-11）：
- `1298_G200::G200-7`：片边取样被自己判污染（contamination 0.25）而整卷又是
  `roll_partial`，于是退回通用先验；先验的红通道电平比真片基暗 0.34~0.68 decade，
  后果是**该帧 55% 的像素透射率高于 assumed base**（负片物理上不可能），
  画面平均对数密度 APD(R−G) = **−1.034**，而同卷其余十帧全在 +0.147~+0.342。
  用户点名的"水是一片死青"就是这个数。
- `1299_G200::G200-7`：用户亲手画了片基框（conf 1.00）。**他的框是绝对真值**，
  新机制只允许在"没有框、也没有可用片边"时介入。

**㉚b（2026-10-13）口径变更**：用户这天把 `1298_G200` 整卷也框了，上面那条
"没有可用片边"的语料前提不再自然成立——两条坏帧锁改成用 `_record(mode=...)`
**构造**该前提（`no_geometry` = 写锁时该帧的自然状态），断言阈值一字未放宽。
新增的另一头：有手框时 `1298::G200-7` 走 `frame+roll`（红通道 0.2404），
而画面内橙色估计在同一帧读到手框**之外**的亮区（0.4188，差 0.18 decade）——
这两个来源差这么多是 B-14㉚ 记下的新事实，也是 task #10（二维 flat-field）
存在的理由：**绝不许让橙色估计推翻手框**。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from negative_lab.calibration.annotations import load_annotation  # noqa: E402
from negative_lab.calibration.base import base_from_orange_signature  # noqa: E402
from negative_lab.io.images import load_linear_rgb  # noqa: E402
from negative_lab.library.fingerprint import capture_fingerprint  # noqa: E402
from negative_lab.library.models import FrameRecord, GeometryRegions  # noqa: E402
from negative_lab.pipeline.service import process_roll  # noqa: E402
from build_visual_contact_sheet import _downscale_geometry  # noqa: E402

MAN = json.loads((ROOT / "fixtures/acceptance/manifest.json").read_text(encoding="utf-8"))


def _record(fr: dict, max_dim: int = 1200, mode: str = "box"):
    """`mode`：**box**=手框原样（生产路径）/ **nobase**=只摘掉片基框 /
    **no_geometry**=什么几何都不给（= 这条锁写下来时该帧在语料里的自然状态）。

    ㉚b 起这三个口径都是**构造**出来的：2026-10-13 用户给 `1298::G200-7` 画了
    片基取样框，"没有片边的坏帧"这个语料前提消失了。把前提写进构造函数而不是
    去语料里找一个恰好没框的帧，这条锁才不会再随标注状态漂移。
    """
    loaded = load_linear_rgb(ROOT / fr["file"])
    arr = loaded.rgb
    step = max(1, int(np.ceil(max(arr.shape[:2]) / float(max_dim))))
    img = arr[::step, ::step, :].copy()
    ann = load_annotation(fr["file"])
    if ann is None or mode == "no_geometry":
        geo = GeometryRegions(rotation_degrees=(ann.rotation_degrees if ann else 0))
        area = (0, 0, img.shape[0], img.shape[1])
    else:
        geo = GeometryRegions(
            rotation_degrees=ann.rotation_degrees,
            output_crop_roi=ann.output_crop_roi or ann.image_area_roi,
            image_area_roi=ann.image_area_roi or ann.output_crop_roi,
            base_sample_rois=[] if mode == "nobase" else list(ann.base_sample_rois or []))
        # **必须和生产/探针同一条缩放规则**：ROI 是源图坐标，直接拿去切缩小图
        # 会整块落到画面外，于是"用户手框"在测试里悄悄消失，测出来的就不是产品。
        geo = _downscale_geometry(geo, arr.shape[:2], img.shape[:2])
        area = geo.image_area_roi or (0, 0, img.shape[0], img.shape[1])
    rec = FrameRecord(source_path=fr["file"], source_sha256=fr["source_sha256"],
                      input_facts=dict(loaded.meta),
                      capture_fingerprint=capture_fingerprint(
                          loaded.meta, source_path=fr["file"]).value,
                      geometry=geo, orientation_decided=ann is not None)
    return rec, img, area, geo


def _frame(fr_id: str) -> dict:
    hits = [f for f in MAN["frames"] if f["id"] == fr_id and f.get("required")]
    if not hits:
        pytest.skip(f"manifest 里没有 {fr_id}")
    if not (ROOT / hits[0]["file"]).exists():
        pytest.skip(f"源片不在仓库内：{hits[0]['file']}（外观真值语料不入库）")
    return hits[0]


def test_没有片边的坏帧不再吃通用先验():
    """**正锁**：`1298::G200-7` 不许退到 `generic_prior`——两条路径分别钉住。

    **㉚b 换过口径**（数字全部来自 `outputs/repin30.txt`，2026-10-13 当场打印，
    1200px、生产整卷路径）：原判据要求 `base_source == "content_orange"`，
    而用户这天给这一帧画了片基取样框，生产路径实得 **`frame+roll`**
    （conf 1.00，红通道片基 0.2404）——那是**比画面内估计更强的证据**，
    不是回归。"没有片边的坏帧"这个前提从语料里消失了，于是改成**构造**：

    - `no_geometry`（什么框都不给 = 这条锁写下来时该帧的自然状态）
      ⇒ 必须仍走画面内橙色实测 `content_orange`（conf 0.60，红通道 **0.4188**），
      且比先验的 0.216 亮出一个量级——**这一条一个字没放宽**；
    - `box`（生产现在真实走的路）⇒ 必须是实测来源，不许是 `generic_prior`。

    两种情形都还要求"不是通用先验"，因为先验的红通道电平比真片基暗 0.18~0.34 decade，
    后果就是用户点名的"水是一片死青"（APD(R−G) −1.034 对同卷 +0.147~+0.342）。
    """
    fr = _frame("1298_G200::G200-7")
    rec_ng, img, _area, _geo = _record(fr, mode="no_geometry")
    _, res = process_roll([(rec_ng, img)])
    pr = res[0]
    dmin = np.asarray(pr.report.params.dmin, np.float64)
    print(f"  构造 no_geometry 实得：base_source={pr.base_source} "
          f"conf={pr.base_confidence:.2f} dmin={np.round(dmin, 4).tolist()}")
    #: **㊲ 之后这一支换了落点，判据没有放宽**。1298 全卷按"转 180 + 水平镜像"改正手性之后，
    #: 构造"什么框都不给"时**自动检测在转正的画面上找到了可用片基条**，
    #: 于是来源从 `content_orange`（画面内橙色实测）变成 `frame+roll`（自动条 + 卷模型）
    #: ——那是**更强的证据**，不是回归；原判据写死 `content_orange` 是在钉"走哪条路"，
    #: 而这条锁真正要守的是"**不许吃通用先验、且采用的片基必须比先验亮一个量级**"。
    #: 所以这里钉的是不变量 + 物理量，不是钉路径名。
    assert pr.base_source != "generic_prior", (
        f"构造的无取样口径退回了通用先验：{pr.base_source}")
    assert pr.base_source in ("frame", "frame+roll", "content_orange", "roll"), (
        f"未知的片基来源：{pr.base_source}")
    #: 阈值锚在**实物**上而不是"调到能过"：这一帧用户亲手画的片基框实测 R=0.2404
    #: （㉚/㉛ 的"新分家"那条），所以"自动路不许比用户的手还暗"是 0.24；
    #: 旧阈值 0.26 是给 `content_orange`（画面内橙色实测 0.4188）那条路写的，
    #: 换路之后拿它当尺就是拿上一段的落点钉这一段。
    assert dmin[0] > 0.24, (
        f"采用的红通道片基不许比用户手框（0.2404）还暗，实得 {dmin.tolist()}"
        "（通用先验是 0.2164）")

    rec_box, _img, _a, geo = _record(fr, mode="box")
    assert geo.base_sample_rois, "前提：这一帧现在有用户手画的片基框"
    _, res2 = process_roll([(rec_box, _img)])
    pr2 = res2[0]
    assert pr2.base_source != "generic_prior", (
        f"有手框的帧不许吃通用先验：{pr2.base_source}")
    assert pr2.base_source in ("frame", "frame+roll", "content_orange")


def test_抬片基后高于片基的像素比例必须塌下去():
    """**对症量**：修复的直接后果是"透射率高于片基"这一物理矛盾消失。

    取样必须用 `pr.valid_mask`（生产实际看到的那块画面区）。用整幅源图会被
    黑衬底稀释，量出来的数就不是产品读到的数——这类"探针窗口 ≠ 生产窗口"
    的坑本轮已经踩过两次。

    **㉚b 换过口径，两条阈值都没放宽**：原判据的 `before > 0.30` 量在
    "该帧恰好没有框、于是统计窗 = 自动检测区"的语料状态上；用户 2026-10-13
    给这一帧画框之后，同一个先验在**他的手框窗**内只被 0.177 的红通道像素越过
    （`outputs/repin30.txt`）——因为自动检测把大片亮片边算进了窗。
    所以"先验造成大面积物理不可能"这一演示改在 `mode="no_geometry"`
    （**构造**出原来的口径）上做，`before > 0.30` / `after < 0.10` 原样保留
    （当场实得 **0.564** / **0.001**）；对手框路径只要求
    "采用的片基之下越界比例 < 0.10"（实得 **0.020**），不再要求 `before` 过线——
    那一位是"窗里有多少亮片边"的函数，不是这条机制的函数。
    """
    fr = _frame("1298_G200::G200-7")
    prior = np.array([0.21637, 0.09737, 0.02596])       # 本轮实测的通用先验取值

    def _above(pr, base):
        inner = pr.t[..., :3].reshape(-1, 3).astype(np.float64)
        m = pr.valid_mask
        if m is not None and m.shape == pr.t.shape[:2]:
            inner = pr.t[..., :3][m].reshape(-1, 3).astype(np.float64)
        return float((inner[:, 0] > base).mean())

    rec, img, area, _geo = _record(fr, mode="no_geometry")
    _, results = process_roll([(rec, img)])
    pr = results[0]
    before = _above(pr, prior[0])
    after = _above(pr, np.asarray(pr.report.params.dmin, np.float64)[0])
    assert before > 0.30, f"这条锁的前提是先验下红通道大面积越界，实得 {before:.3f}"
    assert after < 0.10, f"实测片基后红通道越界比例应塌到 10% 以下，实得 {after:.3f}"

    rec2, img2, _a, _g = _record(fr, mode="box")
    _, res2 = process_roll([(rec2, img2)])
    pr2 = res2[0]
    after_box = _above(pr2, np.asarray(pr2.report.params.dmin, np.float64)[0])
    print(f"[㉚b] 手框窗（生产真实路径）：先验位越界 {_above(pr2, prior[0]):.3f}"
          f"（窗变小，只报不判）/ 采用片基位越界 {after_box:.3f}")
    assert after_box < 0.10, (
        f"用户手框实测片基下红通道越界比例应 <0.10，实得 {after_box:.3f}")


def test_用户亲手画的片基框永远优先():
    """**纪律锁**：新机制不许推翻真值。这帧有手框，来源必须仍是实测取样。"""
    fr = _frame("1299_G200::G200-7")
    ann = load_annotation(fr["file"])
    assert ann is not None and ann.verified and ann.base_sample_rois, "前提：这帧有手框"
    rec, img, area, geo = _record(fr)
    assert geo.base_sample_rois, "前提：手框缩放后仍在图内"
    _, results = process_roll([(rec, img)])
    pr = results[0]
    assert pr.base_source in ("frame", "frame+roll"), (
        f"有用户手框时不许改用画面内估计：{pr.base_source}")
    assert pr.base_confidence > 0.9


def test_同一张真帧上手框与画面内估计的差在留出误差内():
    """把"0.02~0.03 decade"这条**留出证据**钉成锁。

    用 `1299` 里手框的帧：把框**摘掉**再跑画面内检测，估计值必须落在
    框所量到的片基附近。容差取 R/G 0.08 decade（留出实测中位 0.021/0.029，
    最差一帧 0.16 ⇒ 这里取"中位的三倍"而不是"最差值"，因为它量的是
    这条机制是否还成立，不是某一次运气）。
    """
    fr = _frame("1299_G200::G200-7")
    rec, img, area, geo = _record(fr)
    y0, x0, y1, x1 = geo.base_sample_rois[0]
    assert (y1 - y0) * (x1 - x0) >= 64, f"缩放后的框太小，量不出真值：{geo.base_sample_rois[0]}"
    truth = np.median(img[y0:y1, x0:x1, :3].reshape(-1, 3).astype(np.float64), axis=0)
    est = base_from_orange_signature(img, image_area_roi=area)
    assert est["dmin"] is not None, est["warnings"]
    err = np.log10(np.asarray(est["dmin"]) / truth)
    assert abs(err[0]) < 0.08 and abs(err[1]) < 0.08, (
        f"R/G 片基估计偏差应 <0.08 decade（留出中位 0.02~0.03），实得 {np.round(err, 3)}")
    assert abs(err[2]) < 0.25, f"B 通道允许更松（SNR 最低），实得 {np.round(err, 3)}"
