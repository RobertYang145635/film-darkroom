"""回归：**真实语料上，横躺在画布里的片带必须被转成横幅**，纵横要与他手工成品一致（B-12 追加③）。

用户点名的是 `1299_G200` 的竖片：他亲手框出的画框（`fixtures/annotations/1299_G200__*.json`，
`G200-2` 的 `verified=True`，2026-10-09）全是 **5940 高 × 4800 宽**这一类高>宽的形状，
而他一侧的手工成品是 6144×5295 / 6185×5294 / 6137×5313 三种**横幅**
（`tmp_canvas_truth_table.py` 当场列的表）。35mm 单格画框必然宽>高（齿孔在片带长边上），
所以这不是风格问题而是没回正。

锁三件事：
1. 用他的框跑生产路径 ⇒ 输出**宽>高**，来源写清是几何规则补的转，并留下复核提示；
2. 输出的纵横与**他参考图的纵横同侧**（横对横）——这是看得见的口径，不是统计量；
3. **三条负控制**，都是当场量的真值，不是编的（`tmp_canvas_truth_table.py`）：
   - `1298_G200::G200-1` 标注里是 legacy `frame_roi` = (966,376,6541,5166) =
     5575 高 × 4790 宽，**也是竖的**，但他按过 `rot=180` ⇒ 规则必须只说不转；
   - `1298_G200::G200-2` **构造**出的空几何（㉚b 起不再依赖"这帧恰好没框"）
     ⇒ 既不转也不报存疑（规则不许对没有依据的帧猜方向）；
   - `1298_G200::G200-2` 他 2026-10-13 的真实手框 = **4734×4734 正方**
     ⇒ 有框但框不提供方向证据，同样必须沉默。
   这三条放在一起是因为：规则若只是"看见竖的就转"，它会顺手改掉他亲手管着的帧；
   而它若要覆盖没有框的帧，就得依赖自动检测框——那个框在本语料上是坏的
   （`test_reference_quality::AUTO_CROP_GAP`，22 帧里 19 帧 IoU<0.95）。
"""

from __future__ import annotations

import importlib.util
import json
import pathlib

import numpy as np
import pytest
from PIL import Image

from negative_lab.calibration.annotations import load_annotation
from negative_lab.io.images import load_linear_rgb
from negative_lab.library.models import GeometryRegions
from negative_lab.pipeline.geometry import (SIDEWAYS_FRAME_ROTATION,
                                            SIDEWAYS_MIN_ASPECT,
                                            frame_output_is_portrait,
                                            sideways_correction)
from negative_lab.pipeline.service import process_image

ROOT = pathlib.Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "bvcs_frame_axis_lock", ROOT / "scripts" / "build_visual_contact_sheet.py")
kit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(kit)

SIDWAYS_ID = "1299_G200::G200-2"          # 用户框高>宽、rot=0 ⇒ 规则必须补转
DECIDED_ID = "1298_G200::G200-1"          # 他按过 rot=180，框也是竖的 ⇒ 只说不转
NOBOX_ID = "1298_G200::G200-2"            # 他按过 rot=180，但根本没有画面框 ⇒ 无从判定
MAX_DIM = 900


def _manifest_frame(fid: str) -> dict:
    man = json.loads((ROOT / "fixtures" / "acceptance" / "manifest.json")
                     .read_text(encoding="utf-8"))
    return next(f for f in man["frames"] if f["id"] == fid)


def _run(fid: str, *, strip_boxes: bool = False, override_area_source=None):
    fr = _manifest_frame(fid)
    path = ROOT / fr["file"]
    if not path.exists():
        pytest.skip(f"语料缺失：{fr['file']}")
    loaded = load_linear_rgb(path)
    arr = loaded.rgb
    step = max(1, int(np.ceil(max(arr.shape[:2]) / float(MAX_DIM))))
    img = arr[::step, ::step, :].copy()
    ann = load_annotation(fr["file"])
    assert ann is not None, f"{fid} 的标注不见了，这条控制失去意义"
    if strip_boxes:
        #: ㉚b：**构造**出来的"根本没有框"，不再靠语料恰好没框。
        geo = GeometryRegions(rotation_degrees=ann.rotation_degrees)
    else:
        geo = GeometryRegions(rotation_degrees=ann.rotation_degrees,
                              output_crop_roi=ann.output_crop_roi or ann.image_area_roi,
                              image_area_roi=ann.image_area_roi or ann.output_crop_roi,
                              base_sample_rois=list(ann.base_sample_rois or []))
        if override_area_source is not None:
            #: ㉛：把画框换成**构造**的形状（源图坐标），用来造出"真竖幅框"。
            #: 同时清掉片基取样条——构造框多半不包住它们，那会让这一帧的
            #: 失败原因从"方向规则"变成"片基取样"，混进别的账。
            geo.image_area_roi = tuple(override_area_source)
            geo.output_crop_roi = tuple(override_area_source)
            geo.base_sample_rois = []
        geo = kit._downscale_geometry(geo, arr.shape[:2], img.shape[:2])
    pr = process_image(img, geometry=geo, ann=None, make_candidates=False,
                       orientation_decided=True, frame_id=fr["file"])
    return fr, ann, geo, pr


def test_sideways_user_box_comes_out_landscape():
    """他的框是竖的 ⇒ 输出必须是横幅，而且必须说是几何规则补的转。"""
    fr, ann, geo, pr = _run(SIDWAYS_ID)
    assert ann.rotation_degrees == 0, "这帧他后来转过方向，样本要换一个"
    area = geo.image_area_roi
    assert area is not None and (area[2] - area[0]) > (area[3] - area[1]), (
        f"用户画面框应当是高>宽才构成矛盾，实测 {area}")
    h, w = pr.rendered_positive_linear.shape[:2]
    assert w > h, f"横躺的片带仍然出成竖片：输出 {h}×{w}"
    assert pr.roi_provenance["orientation"] == "geometry-rule"
    basis = pr.roi_provenance["orientation_basis"]
    assert basis["rotation_added"] == 270 and basis["area_source"] == "用户几何"
    assert basis["rotation_applied"] == 270
    assert any("35mm 画框几何补转" in r for r in pr.gate.review_reasons), \
        pr.gate.review_reasons
    assert pr.gate.blocks_export is False, "提示不许扣货（B-12 的既有纪律）"
    #: **谁决定的要能机器查到**，不能只留一句人话在 warnings 里
    sc = pr.sidecar(source_sha256=fr["source_sha256"], source_path=fr["file"])
    prov = sc["geometry"]["roi_source_space"]["provenance"]
    assert prov.get("orientation") == "geometry-rule", prov
    assert prov.get("orientation_basis", {}).get("rotation_added") == 270, prov


def test_output_aspect_is_on_his_side_of_the_wall():
    """**看得见的口径**：他这一帧的成品是横幅，我们也必须是横幅。

    统计量会骗人（B-13 追加① 的分带色相角就是），纵横比不会——他给的方向是横，
    我们出竖，并排看第一眼就是"完全不对"。参考图与我们的数组都用原始宽高，
    不做任何拉伸（上一版探针把两边都压成 24×24 方格，于是"竖的 180°"也能得高分）。
    """
    _fr, _ann, _geo, pr = _run(SIDWAYS_ID)
    look = json.loads((ROOT / "fixtures" / "references" / "g200_manual_look.json")
                      .read_text(encoding="utf-8"))
    entry = next((e for e in look["frames"]
                  if e["frame_id"] == SIDWAYS_ID and not e["variant_of"]), None)
    ref_path = ROOT / entry["reference_file"] if entry else None
    if not ref_path or not ref_path.exists():
        pytest.skip("用户手工参考大图不在库里（gitignore）")
    Image.MAX_IMAGE_PIXELS = None
    rw, rh = Image.open(ref_path).size
    h, w = pr.rendered_positive_linear.shape[:2]
    assert rw > rh, f"他这一帧的成品应当是横幅，实测 {rh}×{rw}"
    assert w > h, f"我们输出 {h}×{w} 与他 {rh}×{rw} 纵横不同侧"


#: ㉛ 构造用的"真竖幅"画框（源图坐标，画布 8280 高 × 5520 宽之内）：
#: 5600 高 × 3100 宽 ⇒ `h/w = 1.806`，远在 `SIDEWAYS_MIN_ASPECT = 1.10` 之上。
#: 35mm 单格是 36×24 = 1.50，所以"横躺的片带"无论怎么裁都不止 1.10；
#: 反过来，他给 1298 画的框 `h/w` 只有 0.981~1.0254（正方裁切），
#: 那种框**不携带方向证据**——旧判据 `h > w` 零余量，把它判成了横躺并静默转掉 90°。
PORTRAIT_PROBE_ROI = (1200, 300, 6800, 3400)


def test_negative_control_his_180_on_a_portrait_box_is_warned_not_overruled():
    """**负控制①**：框**真的**是竖幅（片带横躺）而方向他按过 `rot=180`
    ⇒ 规则看得见矛盾，也不许动像素，只许说出来。

    **㉛ 换过口径**：原先这条直接用 `1298_G200::G200-1` 的天然框
    （4776 高 × 4740 宽，`h/w=1.0076`）当"竖框"样本。纵横证据下限上线之后
    那种框**不再携带证据**（既不转也不报存疑——这正是 ㉛ 要修的东西），
    于是"看得见但不推翻"这一支在本语料里**没有自然样本**了。
    改法是把框换成**构造**的 `PORTRAIT_PROBE_ROI`（`h/w=1.806`），
    帧、旋转、`orientation_decided` 全部沿用他这一帧的真实状态。
    选它不是随手挑的：这条恰好证明规则不是"看见竖的就转"。
    """
    _fr, ann, geo, pr = _run(DECIDED_ID, override_area_source=PORTRAIT_PROBE_ROI)
    assert ann.rotation_degrees == 180, f"标注方向变了：{ann.rotation_degrees}"
    area = geo.image_area_roi
    assert area is not None and (area[2] - area[0]) > (area[3] - area[1]) * 1.5, (
        f"构造的框应当是**明显**竖幅才构成'看得见但不转'的对照，实测 {area}")
    assert pr.geometry.rotation_degrees == 180, (
        f"他定过的方向被规则改了：{pr.geometry.rotation_degrees}")
    assert pr.roi_provenance["orientation"] == "decided"
    assert "orientation_basis" not in pr.roi_provenance
    reasons = [r for r in pr.gate.review_reasons if "画面方向存疑" in r]
    assert reasons, f"看得见矛盾却没说：{pr.gate.review_reasons}"
    assert "人工定过" in reasons[0]
    assert pr.gate.blocks_export is False, "提示不许扣货"


def test_negative_control_no_box_at_all_is_neither_rotated_nor_warned():
    """**负控制②**：没有可信画框 ⇒ 纵横无从判定，既不转也不报存疑。

    这条钉的是"规则必须有依据"：没有可信画框时它的输出与改动前逐比特相同，
    不会有人某天发现软件开始对没框的帧猜方向。

    **㉚b 换过口径**：原先这条直接拿 `1298_G200::G200-2` 天然的"一个框都没有"
    当无框样本，2026-10-13 用户把这一帧框了（见下一条），前提从语料里消失。
    断言本身一个字没动，改的是**样本来源**：现在用 `strip_boxes=True` **构造**
    无框几何。这样这条锁不再依赖任何一帧的标注状态——标注被改、被删、
    被新增都不会让它悄悄变成正控制或空对照。
    """
    _fr, ann, geo, pr = _run(NOBOX_ID, strip_boxes=True)
    assert ann.rotation_degrees == 180
    assert geo.image_area_roi is None and geo.base_sample_rois == [], \
        "这条控制的前提：构造出的几何必须真的什么框都没有"
    assert pr.geometry.rotation_degrees == 180
    assert pr.roi_provenance["orientation"] == "decided"
    assert not [r for r in pr.gate.review_reasons if "画面方向存疑" in r
                or "35mm 画框几何补转" in r], pr.gate.review_reasons


def test_negative_control_his_square_box_gives_the_rule_nothing_to_do():
    """**负控制②′**（㉚b 新增）：有框但框是**正方** ⇒ 纵横不构成矛盾 ⇒ 一样沉默。

    `1298_G200::G200-2` 他 2026-10-13 画的画面框源图坐标
    `(1890, 372, 6624, 5106)` = **4734 高 × 4734 宽**，逐比特正方。
    缩放后允许差 1 像素（四舍五入），所以这里只断言"不是竖的"。
    与 ①（框是竖的、但方向他定过 ⇒ 只说不转）和 ②（没有框 ⇒ 沉默）合起来，
    三种证据状态各有锁：**竖框+已定方向 → 说；正方框 → 沉默；无框 → 沉默**。
    """
    _fr, ann, geo, pr = _run(NOBOX_ID)
    assert ann.rotation_degrees == 180
    area = geo.image_area_roi
    assert area is not None, "这帧现在有框了（用户 2026-10-13 画的），前提变了"
    h, w = area[2] - area[0], area[3] - area[1]
    print(f"\n  1298::G200-2 缩放后画面框 {h}×{w}（源图 4734×4734 正方，"
          f"允许 ±1px 取整）")
    assert h <= w + 1, f"他这一帧的框应当不是竖的（源图 4734×4734 正方），实测 {h}×{w}"
    assert pr.geometry.rotation_degrees == 180, (
        f"正方框被当成方向依据改了像素：{pr.geometry.rotation_degrees}")
    assert pr.roi_provenance["orientation"] == "decided"
    assert not [r for r in pr.gate.review_reasons if "画面方向存疑" in r
                or "35mm 画框几何补转" in r], pr.gate.review_reasons


def test_aspect_evidence_separates_the_two_rolls_on_the_real_annotations():
    """**㉛ 的语料锁（不读像素，只读标注）**：正方裁切一帧都不许转，
    真横躺一帧都不许漏。这条就是当初能抓住那个缺陷的锁。

    当场量的两簇（`fixtures/annotations/*.json`，源图坐标）：
    `1298_G200` 他裁成**正方** ⇒ `h/w` 0.981~1.0254，全部落在
    `SIDEWAYS_MIN_ASPECT = 1.10` 的证据带**之内**（⇒ `None`，规则沉默）；
    `1299_G200` 是**片带真的横躺** ⇒ `h/w` 1.2201~1.2403，全部在带**之外**
    （⇒ 必须补转 `SIDEWAYS_FRAME_ROTATION`）。
    旧判据 `h > w` 零余量 ⇒ 1298 十一帧里 8 帧被判横躺、`rot=0` 的 4 帧
    （`G200-6/-7/-10/-11`）被静默转掉 90°，看图才发现（`sat_tax_6.jpg` 第 4 行）。
    ⚠ 这条**不许**因为"某帧他重新框过就删条目"：真要是哪天两簇靠到 1.10 附近，
    说明证据带该重定，那是一次**改口径**并写进 B-12/B-14，不是把锁删掉。
    """
    def _boxes(roll: str):
        out = []
        for p in sorted((ROOT / "fixtures" / "annotations").glob(f"{roll}__*.json")):
            d = json.loads(p.read_text(encoding="utf-8"))
            a = d.get("image_area_roi") or d.get("frame_roi")
            if a:
                out.append((p.stem.split("__", 1)[1], int(d.get("rotation_degrees", 0)),
                            tuple(a)))
        return out

    sq, lying = _boxes("1298_G200"), _boxes("1299_G200")
    assert len(sq) >= 10 and len(lying) >= 10, (
        f"这条锁要两卷各自的框都还在：1298 {len(sq)} 帧 / 1299 {len(lying)} 帧")
    ratios_sq = [(a[2] - a[0]) / (a[3] - a[1]) for _n, _r, a in sq]
    ratios_ly = [(a[2] - a[0]) / (a[3] - a[1]) for _n, _r, a in lying]
    print(f"\n  证据带 {1 / SIDEWAYS_MIN_ASPECT:.3f}~{SIDEWAYS_MIN_ASPECT:.3f}；"
          f"1298 正方簇 h/w {min(ratios_sq):.4f}~{max(ratios_sq):.4f}（{len(sq)} 帧）；"
          f"1299 横躺簇 h/w {min(ratios_ly):.4f}~{max(ratios_ly):.4f}（{len(lying)} 帧）")
    for name, rot, a in sq:
        assert frame_output_is_portrait(a, rot) is None, (
            f"1298::{name} 的框 h/w={(a[2] - a[0]) / (a[3] - a[1]):.4f} 被当成方向证据"
            f"（实得 {frame_output_is_portrait(a, rot)}）⇒ 会被补转，这是 ㉛ 修的缺陷")
    for name, rot, a in lying:
        assert sideways_correction(a, rot) == SIDEWAYS_FRAME_ROTATION, (
            f"1299::{name} 的框 h/w={(a[2] - a[0]) / (a[3] - a[1]):.4f} 是真横躺，"
            f"规则漏了 ⇒ 证据带被放宽过头")
