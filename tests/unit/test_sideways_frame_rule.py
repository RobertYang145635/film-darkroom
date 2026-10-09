"""锁：**35mm 单格画框必然宽>高**——片带横躺时不许静默出竖片（B-14，用户点名 `G200-7` 竖片）。

规则的来历与它的两条边界，都是当场量出来的，不是推理：

- **要转**：`1299_G200` 用户亲手框出的 11 帧画框全是 5958 高 × 4818 宽（源画布
  8280×5520 是竖幅），而他一侧的手工成品是 6144×5295 横幅。高>宽的单格 35mm
  画框在物理上不存在（齿孔在片带长边上，画框 36×24），所以"输出是竖的"
  只有一个解释：片带在相机画布里横躺着，没回正。
- **往哪边转是约定不是几何**：取值 **270°（逆时针）**，来自把他那三张 1299 参考图
  当目标实测——5 个网格 × 3 个中心裁切共 15 个窗口，三帧每一格都指向逆时针
  （lead(CW−CCW) 的 min −1.210/−0.566/−0.192，中位 −1.630/−0.825/−0.254；
  探针 `tmp_orientation_sign2.py`）。
  **这个数上一版是 90（顺时针），被并排表 `all11_annotated.jpg` 推翻**：
  旧探针把负片参考一起反相了（他的参考是正片），等于拿"我方正片 vs 他的负片"
  比亮度布局，符号整体翻转。所以这条锁连同下面的 `test_rotate_step_*` 一起，
  就是那次翻案的机器可查版本。
  所以规则改过的帧**必须带复核提示且必须可一键推翻**——它是默认值，不是裁决。
- **自动检测框一律不参与方向**（既不代转也不报存疑）。第一版曾让它参与，实测被打红：
  它在 `1299_G200::G200-10` 上把用户框出的竖画框量成 4477 高 × 5445 宽的横框
  （窄片边被算进画面，`test_reference_quality.py::AUTO_CROP_GAP` 钉着 9/11 帧 IoU
  0.683~0.918），而在合成帧上它能把一幅**横向**渐变量成竖向框——
  `tests/unit/test_orientation_review.py` 的三条方向锁当场变红。
  偏差方向不唯一 ⇒ 它的纵横不能用来改方向，只配用于裁剪建议。
"""

from __future__ import annotations

import numpy as np
import pytest

from negative_lab.library.models import GeometryRegions
from negative_lab.pipeline.geometry import (SIDEWAYS_FRAME_ROTATION,
                                            frame_output_is_portrait,
                                            sideways_correction)
from negative_lab.pipeline.service import process_image

KEY_RULE = "35mm 画框几何补转"
KEY_DOUBT = "画面方向存疑"


def _portrait_strip(h: int = 128, w: int = 96) -> np.ndarray:
    """一幅**竖幅**的合成负片（通道比例固定，纵向渐变）。"""
    rows = np.linspace(0.02, 0.22, h, dtype=np.float32)[:, None]
    grid = np.repeat(rows, w, axis=1)
    return np.dstack([grid, grid * 0.88, grid * 0.74]).astype(np.float32)


def _landscape_strip(h: int = 96, w: int = 128) -> np.ndarray:
    rows = np.linspace(0.02, 0.22, w, dtype=np.float32)[None, :]
    grid = np.repeat(rows, h, axis=0)
    return np.dstack([grid, grid * 0.88, grid * 0.74]).astype(np.float32)


def _geo(deg: int = 0, area=None) -> GeometryRegions:
    return GeometryRegions(rotation_degrees=deg, image_area_roi=area,
                           output_crop_roi=area,
                           base_sample_rois=[(8, 4, 20, 12)])


# ------------------------------------------------------------------ 纯函数
@pytest.mark.parametrize("area,deg,expected", [
    ((0, 0, 100, 60), 0, True),        # 竖框不转 ⇒ 输出竖
    ((0, 0, 100, 60), 180, True),      # 180 不换纵横
    ((0, 0, 100, 60), 90, False),      # 转 90 就成横幅
    ((0, 0, 60, 100), 0, False),       # 横框不转
    ((0, 0, 60, 100), 90, True),       # 把横框转 90 反倒竖了——同样必须纠正
    ((0, 0, 100, 100), 0, None),       # 正方：无从判定
    (None, 0, None),
    ((0, 0, 50, 50), 0, None),         # 退化框（高=宽）
    ((0, 0, 50, 10), 0, True),         # 极窄也照样是竖
])
def test_portrait_predicate_is_about_the_frame_not_the_canvas(area, deg, expected):
    assert frame_output_is_portrait(area, deg) is expected


def test_correction_only_when_the_frame_would_come_out_portrait():
    assert sideways_correction((0, 0, 100, 60), 0) == SIDEWAYS_FRAME_ROTATION
    assert sideways_correction((0, 0, 100, 60), 90) == 0
    assert sideways_correction((0, 0, 60, 100), 0) == 0
    assert sideways_correction(None, 0) == 0


def test_rotation_convention_is_the_measured_one():
    """270°（逆时针）是他三张 1299 参考图上 15/15 个窗口一致的量，不是随手取的。

    改这个常数就等于改方向约定：必须重做那次窗口自检，**并且重跑并排表看图**——
    上一版就是只做了窗口自检（15/15 一致）而没看图，才把上下颠倒的 90° 当成了结论。
    """
    assert SIDEWAYS_FRAME_ROTATION == 270


# ------------------------------------------------------------------ 规则生效
def test_sideways_strip_is_rotated_and_says_so():
    """用户框是竖的 ⇒ 输出必须是横幅，且要说清是谁决定的。"""
    pr = process_image(_portrait_strip(), geometry=_geo(0, (6, 8, 122, 88)),
                       orientation_decided=True)      # 有标注≠转过方向
    out = pr.rendered_positive_linear
    assert out.shape[1] > out.shape[0], \
        f"横躺的片带仍然出成竖片 {out.shape[:2]}"
    assert pr.roi_provenance["orientation"] == "geometry-rule"
    basis = pr.roi_provenance["orientation_basis"]
    assert basis["rotation_added"] == 270 and basis["rotation_applied"] == 270
    assert basis["frame_height_width"] == [116, 80]
    assert basis["area_source"] == "用户几何"
    assert basis["requested_by_caller"] is True, "调用方说有标注，这里必须留痕"
    reasons = [r for r in pr.gate.review_reasons if KEY_RULE in r]
    assert reasons, f"补转了却没说：{pr.gate.review_reasons}"
    assert "逆时针" in reasons[0], "提示必须写清用的是哪个方向（它就是上一版搞错的东西）"
    assert "上下颠倒" in reasons[0] and "180°" in reasons[0], "提示必须可执行"


def test_rule_marking_does_not_block_export():
    """B-12 的纪律不许被 B-14 破坏：方向提示是"请看一眼"，不是扣货。"""
    pr = process_image(_portrait_strip(), geometry=_geo(0, (6, 8, 122, 88)),
                       orientation_decided=True)
    assert pr.status == "needs_review", pr.status
    assert pr.gate.blocks_export is False, pr.gate.failures
    assert not pr.gate.failures


def test_landscape_frame_is_left_alone():
    """**负控制**：本来就横着的画框一个像素都不该被转走。"""
    pr = process_image(_landscape_strip(), geometry=_geo(0, (8, 6, 88, 122)),
                       orientation_decided=True)
    assert pr.roi_provenance["orientation"] == "decided"
    assert "orientation_basis" not in pr.roi_provenance
    assert not [r for r in pr.gate.review_reasons if KEY_RULE in r]
    assert pr.geometry.rotation_degrees == 0


def test_180_of_a_portrait_frame_becomes_90_not_stays_portrait():
    """他从 0 转到 180 还是竖的：几何仍然要求补 270 ⇒ 净 (180+270)%360 = 90。"""
    pr = process_image(_portrait_strip(), geometry=_geo(180, (6, 8, 122, 88)),
                       orientation_decided=False)
    assert pr.geometry.rotation_degrees == 90
    assert pr.rendered_positive_linear.shape[1] > pr.rendered_positive_linear.shape[0]


def test_rotate_step_flips_180_only_on_sideways_strips():
    """横躺的片带上，旋转按钮一次翻 180°；正常横框的帧按键语义不变。

    因为从横幅再转 90° 得到的是竖幅，而 35mm 单格不存在竖幅——那一按要么被规则
    立刻补回去（看起来没反应），要么把他带进不可能的状态。
    """
    from negative_lab.pipeline.geometry import rotate_step
    portrait = (6, 8, 122, 88)      # 高>宽
    landscape = (8, 6, 88, 122)     # 宽>高
    assert rotate_step(portrait, 90) == 180
    assert rotate_step(portrait, 270) == 180
    assert rotate_step(landscape, 90) == 90        # **负控制**：正常帧不许改语义
    assert rotate_step(landscape, 270) == 270
    assert rotate_step(None, 90) == 90             # 没有画框 ⇒ 不擅自改
    assert rotate_step((10, 10, 110, 110), 90) == 90   # 正方画框也不是横躺


def test_human_decided_portrait_is_reported_not_overruled():
    """**边界**：他亲手定了 180 而画框仍是竖的 ⇒ 一个像素都不改，但必须报存疑。

    方向这件事他说过要自己管（B-12），几何只能指出矛盾，不能反过来推翻人的决定。
    """
    pr = process_image(_portrait_strip(), geometry=_geo(180, (6, 8, 122, 88)),
                       orientation_decided=True)
    assert pr.geometry.rotation_degrees == 180, "人工定过的方向被规则改了"
    assert pr.roi_provenance["orientation"] == "decided"
    reasons = [r for r in pr.gate.review_reasons if KEY_DOUBT in r]
    assert reasons, f"竖幅的 35mm 单格没报存疑：{pr.gate.review_reasons}"
    assert "人工定过" in reasons[0]
    assert pr.gate.blocks_export is False


def test_human_chosen_landscape_rotation_is_never_overridden():
    """**负控制**：他自己转成的横幅（rot=90）必须原样保留，不叠加第二次 90°。"""
    pr = process_image(_portrait_strip(), geometry=_geo(90, (6, 8, 122, 88)),
                       orientation_decided=True)
    assert pr.geometry.rotation_degrees == 90
    assert pr.roi_provenance["orientation"] == "decided"
    assert pr.rendered_positive_linear.shape[1] > pr.rendered_positive_linear.shape[0]


# ------------------------------------------------------------------ 边界
def test_auto_box_never_rotates():
    """只有自动检测框时**不许**用它改方向：它的纵横在语料上是坏的（见模块文档）。

    这条是"提示而不静默转"的边界：输出仍然竖着，但必须留下一句存疑。
    """
    pr = process_image(_portrait_strip(), geometry=GeometryRegions(
        rotation_degrees=0, base_sample_rois=[(8, 4, 20, 12)]),
        orientation_decided=True)
    assert pr.geometry.rotation_degrees == 0, "自动框把方向改了——规则越界"
    assert pr.roi_provenance["orientation"] != "geometry-rule"


def test_no_frame_box_at_all_never_invents_a_rotation():
    """**负控制**：拿不到画框（只有自动检测）时不许凭空造方向结论。"""
    pr = process_image(_landscape_strip(),
                       geometry=GeometryRegions(rotation_degrees=0))
    assert pr.roi_provenance["orientation"] != "geometry-rule"
    assert "orientation_basis" not in pr.roi_provenance
    assert pr.geometry.rotation_degrees == 0


def test_square_frame_is_not_called_impossible():
    """正方画框（120 的 6×6 一类）不是矛盾，既不转也不报存疑。"""
    pr = process_image(_portrait_strip(120, 120),
                       geometry=_geo(0, (10, 10, 110, 110)),
                       orientation_decided=True)
    assert pr.geometry.rotation_degrees == 0
    assert pr.roi_provenance["orientation"] == "decided"
    assert not [r for r in pr.gate.review_reasons if KEY_DOUBT in r]


# ------------------------------------------------------------------ 回写不许把自动框洗白
def test_image_area_auto_survives_serialization():
    """**信任边界靠这个键成立**，所以它必须活过 `to_dict/from_dict` 与草稿落盘。

    少了它，自动检测出的框一旦经过 `roi_source_space` 回写（`process_roll` 第二遍、
    批处理、UI 草稿都走这条路）就以"用户几何"的身份回来，规则于是开始用坏框改方向。
    """
    g = GeometryRegions(rotation_degrees=90, image_area_roi=(0, 0, 10, 4),
                        image_area_auto=True)
    back = GeometryRegions.from_dict(g.to_dict())
    assert back.image_area_auto is True and back.rotation_degrees == 90
    # 负控制：旧会话文件里没这个键 ⇒ 按"人工"处理（默认 False），不许报错也不许猜
    legacy = g.to_dict()
    del legacy["image_area_auto"]
    assert GeometryRegions.from_dict(legacy).image_area_auto is False


def test_auto_box_survives_the_round_trip_as_auto():
    """**最关键的一条负控制**：自动检测出的画框经过 `roi_source_space` 回写再喂回
    `process_image`，必须**仍然**被认成自动框，不许换个字段就冒充"用户画的框"。

    这条不是假想：`process_roll` 第二遍、批处理、UI 种草稿全都走
    `GeometryRegions.from_dict(pr.geometry.roi_source_space)` 这条路。
    """
    pr1 = process_image(_portrait_strip(),
                        geometry=GeometryRegions(rotation_degrees=0),
                        orientation_decided=True)
    rss = pr1.geometry.roi_source_space
    assert rss.get("image_area_auto") is True or rss.get("image_area_roi") is None, \
        f"自动框没被标记：{rss}"
    refeed = GeometryRegions.from_dict(rss)
    if refeed.image_area_roi:
        assert refeed.image_area_auto is True, "回写把自动框洗白成了用户框"
    pr2 = process_image(_portrait_strip(), geometry=refeed,
                        orientation_decided=True)
    assert pr2.roi_provenance["orientation"] != "geometry-rule", \
        "自动框第二次进来就驱动了方向——信任边界漏了"


# ------------------------------------------------------------------ UI 侧的按钮口径
def test_adopted_rotation_follows_the_picture_not_the_record():
    """按钮转的必须是**他看到的那张**（用户定的口径：转的是我看到的）。"""
    from types import SimpleNamespace

    from negative_lab.ui.state_store import adopted_rotation
    ruled = SimpleNamespace(roi_provenance={"orientation": "geometry-rule"},
                            geometry=SimpleNamespace(rotation_degrees=90))
    assert adopted_rotation(ruled, 0) == 90
    # 负控制：规则没动手时一个值都不改
    plain = SimpleNamespace(roi_provenance={"orientation": "decided"},
                            geometry=SimpleNamespace(rotation_degrees=0))
    assert adopted_rotation(plain, 90) == 90
    assert adopted_rotation(None, 180) == 180


def test_rotation_stale_regions_only_drops_auto_regions():
    """旋转键只该让**自动检测**的区域重来：他亲手画的框存的是 SOURCE 空间，
    画布转过来不影响胶片在画布里的位置。"""
    from negative_lab.ui.state_store import rotation_stale_regions
    assert rotation_stale_regions({"output_crop": "user", "image_area": "user",
                                   "base": "user"}) == (False, False, False)
    assert rotation_stale_regions({}) == (False, False, False)
    # **负控制**：旧实现三类一起清空，这里必须证明"有用户框时一个都不清"
    assert rotation_stale_regions({"image_area": "auto", "output_crop": "auto-image-area",
                                   "base": "auto-refined"}) == (True, True, True)
    # 画框是自动的 ⇒ 裁切跟着它一起重来（两者必须成对，否则裁切落在旧画框上）
    assert rotation_stale_regions({"image_area": "auto", "base": "annotation-verified"}) \
        == (True, True, False)


