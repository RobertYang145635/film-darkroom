"""锁：**方向未经确认的帧必须被标出来，而且不许挡住导出**（B-12 追加① 的规则落地）。

为什么需要：`rotation_degrees=0` 有两种完全不同的含义——"看过，确认就是 0°"和
"从没决定过，默认值"。二者像素一模一样，历史上软件把后者当前者，
于是用户拿到倒像/横像的成品却说不出为什么（实测 `1298_G200` 12 帧只有 4 帧
有标注文件；与用户手工参考对照，肉眼确认 G200-8 差 180°、G200-6 差 90°）。
自动判方向已被实测否定（见 B-12 追加① 的二维相关表），所以软件**只能**做一件事：
把"这一帧的方向没人决定过"说出来。

同时必须钉住**它不挡导出**：v5 §8.3 里 needs_review 是"允许导出但请人看一眼"，
如果这条理由变成硬失败，就等于用提示把用户的成品扣下了。
"""

from __future__ import annotations

import numpy as np

from negative_lab.library.models import FrameRecord, GeometryRegions
from negative_lab.pipeline.service import process_image

KEY = "画面方向未经确认"


def _frame(h: int = 96, w: int = 128) -> np.ndarray:
    """一幅够自动检测取到区域的合成"负片"（横向渐变，通道比例固定）。"""
    x = np.linspace(0.02, 0.22, w, dtype=np.float32)
    rows = np.tile(x, (h, 1))
    return np.dstack([rows, rows * 0.88, rows * 0.74]).astype(np.float32)


def _geo(deg: int = 0) -> GeometryRegions:
    return GeometryRegions(rotation_degrees=deg)


def _orientation_reasons(pr) -> list[str]:
    return [r for r in pr.gate.review_reasons if KEY in r]


def test_unconfirmed_orientation_is_marked():
    """`orientation_decided=False` ⇒ 必须出现在 review 理由里，状态 needs_review。"""
    pr = process_image(_frame(), geometry=_geo(), orientation_decided=False)
    reasons = _orientation_reasons(pr)
    assert reasons, f"没决定过方向的帧没有被标出来：{pr.gate.review_reasons}"
    assert pr.status == "needs_review", pr.status
    assert "裁切页" in reasons[0], "理由必须可执行，不能只说'未确认'"


def test_decided_orientation_is_not_marked():
    """同一帧只改一个参数：决定过 ⇒ 不得出现该理由（这条与上一条互为对照）。"""
    pr = process_image(_frame(), geometry=_geo(), orientation_decided=True)
    assert not _orientation_reasons(pr), "已决定方向的帧仍被提示，用户会忽略真正的提示"


def test_unstated_is_silent():
    """**负控制**：不表态（默认 None）必须与 False **不同**——
    否则三态只是装饰，也说明上面两条其实是被别的理由撑起来的。"""
    pr = process_image(_frame(), geometry=_geo())
    assert not _orientation_reasons(pr)
    assert pr.roi_provenance["orientation"] == "unstated"


def test_reason_names_the_angle_actually_used():
    """理由里要带上**当前真正用的角度**，否则用户不知道该不该动。"""
    pr = process_image(_frame(), geometry=_geo(90), orientation_decided=False)
    reasons = _orientation_reasons(pr)
    assert reasons and "90°" in reasons[0], reasons


def test_marking_never_blocks_export():
    """v5 §8.3：needs_review 允许导出。这条如果被破坏，提示就变成了扣货。"""
    pr = process_image(_frame(), geometry=_geo(), orientation_decided=False)
    assert pr.gate.blocks_export is False, (
        f"方向提示把导出挡住了：failures={pr.gate.failures}")
    assert not pr.gate.failures


def test_provenance_records_the_three_states():
    got = []
    for flag in (True, False, None):
        pr = process_image(_frame(), geometry=_geo(),
                           orientation_decided=flag)
        got.append(pr.roi_provenance["orientation"])
    assert got == ["decided", "unconfirmed", "unstated"], got


def test_frame_record_round_trips_all_three_states():
    for flag in (True, False, None):
        rec = FrameRecord(source_path="a.tif", orientation_decided=flag)
        back = FrameRecord.from_dict(rec.to_dict())
        assert back.orientation_decided is flag, (rec.to_dict(), back)


def test_garbage_orientation_flag_degrades_to_unstated():
    """会话文件里写进非 bool（字符串/数字）⇒ 回到"没人说过"，不许当真值。"""
    d = FrameRecord(source_path="a.tif").to_dict()
    for bad in ("yes", 1, 0, []):
        d["orientation_decided"] = bad
        assert FrameRecord.from_dict(d).orientation_decided is None, bad
