"""回归：**真实语料上，横躺在画布里的片带必须被转成横幅**，纵横要与他手工成品一致（B-14）。

用户点名的是 `1299_G200` 的竖片：他亲手框出的画框（`fixtures/annotations/1299_G200__*.json`，
`G200-2` 的 `verified=True`，2026-10-09）全是 **5940 高 × 4800 宽**这一类高>宽的形状，
而他一侧的手工成品是 6144×5295 / 6185×5294 / 6137×5313 三种**横幅**
（`tmp_canvas_truth_table.py` 当场列的表）。35mm 单格画框必然宽>高（齿孔在片带长边上），
所以这不是风格问题而是没回正。

锁三件事：
1. 用他的框跑生产路径 ⇒ 输出**宽>高**，来源写清是几何规则补的转，并留下复核提示；
2. 输出的纵横与**他参考图的纵横同侧**（横对横）——这是看得见的口径，不是统计量；
3. **两条负控制**，都是当场量的真值，不是编的（`tmp_canvas_truth_table.py`）：
   - `1298_G200::G200-1` 标注里是 legacy `frame_roi` = (966,376,6541,5166) =
     5575 高 × 4790 宽，**也是竖的**，但他按过 `rot=180` ⇒ 规则必须只说不转；
   - `1298_G200::G200-2` 只有 `rot=180`，**一个框都没有** ⇒ 纵横无从判定，
     既不转也不报存疑（规则不许对没有依据的帧猜方向）。
   这两条放在一起是因为：规则若只是"看见竖的就转"，它会顺手改掉他亲手管着的帧；
   而它若要覆盖没有框的帧，就得依赖自动检测框——那个框在本语料上是坏的。
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


def _run(fid: str):
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
    geo = GeometryRegions(rotation_degrees=ann.rotation_degrees,
                          output_crop_roi=ann.output_crop_roi or ann.image_area_roi,
                          image_area_roi=ann.image_area_roi or ann.output_crop_roi,
                          base_sample_rois=list(ann.base_sample_rois or []))
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


def test_negative_control_his_180_on_a_portrait_box_is_warned_not_overruled():
    """**负控制①**：`1298_G200::G200-1` 的标注框 5575 高 × 4790 宽也是竖的，
    但他给这一帧按过 `rot=180` ⇒ 规则看得见矛盾，也不许动像素，只许说出来。

    选它不是随手挑的：这条恰好证明规则不是"看见竖的就转"。方向他说过自己管（B-12），
    而 1298 的参考图他裁成正方（5309×5309），纵横帮不了判断，更没有证据支持软件代转。
    """
    _fr, ann, geo, pr = _run(DECIDED_ID)
    assert ann.rotation_degrees == 180, f"标注方向变了：{ann.rotation_degrees}"
    area = geo.image_area_roi
    assert area is not None and (area[2] - area[0]) > (area[3] - area[1]), (
        f"这帧的标注框应当是竖的才构成'看得见但不转'的对照，实测 {area}")
    assert pr.geometry.rotation_degrees == 180, (
        f"他定过的方向被规则改了：{pr.geometry.rotation_degrees}")
    assert pr.roi_provenance["orientation"] == "decided"
    assert "orientation_basis" not in pr.roi_provenance
    reasons = [r for r in pr.gate.review_reasons if "画面方向存疑" in r]
    assert reasons, f"看得见矛盾却没说：{pr.gate.review_reasons}"
    assert "人工定过" in reasons[0]
    assert pr.gate.blocks_export is False, "提示不许扣货"


def test_negative_control_no_box_at_all_is_neither_rotated_nor_warned():
    """**负控制②**：`1298_G200::G200-2` 他按过 `rot=180` 而标注里根本没有画面框
    （legacy `frame_roi` 也是 None）⇒ 纵横无从判定，既不转也不报存疑。

    这条钉的是"规则必须有依据"：没有可信画框时它的输出与改动前逐比特相同，
    不会有人某天发现软件开始对没框的帧猜方向。
    """
    _fr, ann, geo, pr = _run(NOBOX_ID)
    assert ann.rotation_degrees == 180
    assert geo.image_area_roi is None, f"这帧现在有框了，样本要换一个：{geo.image_area_roi}"
    assert pr.geometry.rotation_degrees == 180
    assert pr.roi_provenance["orientation"] == "decided"
    assert not [r for r in pr.gate.review_reasons if "画面方向存疑" in r
                or "35mm 画框几何补转" in r], pr.gate.review_reasons
