"""锁：`fixtures/acceptance/manifest.json` 里的 ROI 必须等于**当场那份标注**。

动因（㉛，2026-10-13）：清单的 `generated_utc` 是 **2026-09-02**，而用户这天在 GUI 里
重画并签字了 `1298_G200` 整卷。清单里 `1298::G200-1` 仍然存着 legacy
`frame_roi` (966,376,6541,5166)（纵横 1.164 ⇒ 会被 35mm 规则判成竖框并报"方向存疑"），
`::G200-7` 仍然"没有片基取样"⇒ 联系表把它送去走了**橙色签名回退**（红通道片基 0.4188），
而不是他的实测框（0.2404）。也就是说 **`build_visual_contact_sheet.py` 那一整批
"给他看的图"用的是六周前的几何**——回归锁没事（它们读 `load_annotation` 现值），
出事的是交付视图。

所以这条锁只读 JSON（秒级），练的是"派生数据不许过期"：
清单里每一帧的三类 ROI 与两个确认位，必须等于 `fixtures/annotations/` 的当前内容。
它红的时候**修法只有一个**：重跑 `scripts/build_acceptance_manifest.py`
（清单自己的 note 第 5 条写的就是这个流程），不许手改清单、更不许删这条锁。
"""
from __future__ import annotations

import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from negative_lab.calibration.annotations import load_annotation  # noqa: E402

MAN = ROOT / "fixtures" / "acceptance" / "manifest.json"


def _checked_frames() -> list[dict]:
    """只练"源文件在场 **且** 有标注"的帧——没标注的帧清单里本来就是 `source: none`。"""
    if not MAN.exists():
        return []
    doc = json.loads(MAN.read_text(encoding="utf-8"))
    return [f for f in doc["frames"]
            if (ROOT / f["file"]).exists() and load_annotation(f["file"]) is not None]


def test_manifest_exists_and_summary_agrees_with_its_rows():
    """清单必须在场，且 `summary` 是从明细算出来的、不是手写的。

    ⚠ **这里刻意不比日期**。第一版写成"`generated_utc` 不许早于最新标注的
    `verified_date`"，结果**当天就红**：`generated_utc` 是 UTC
    （本地 13 日凌晨 07:50 = UTC 12 日 23:50），而标注的 `verified_date` 是本地日期
    ⇒ 2026-10-12 < 2026-10-13 假红。用 mtime 更糟（新克隆里全是检出时刻，恒红）。
    "派生数据过期"这件事由下一条锁**按内容**抓，它与时区无关。
    """
    assert MAN.exists(), "先运行 scripts/build_acceptance_manifest.py"
    doc = json.loads(MAN.read_text(encoding="utf-8"))
    frames = doc["frames"]
    assert doc["summary"]["frames"] == len(frames)
    assert doc["summary"]["rois_user_confirmed"] == sum(
        1 for f in frames if f["rois"]["confirmed_by_user"])
    assert doc["summary"]["parity_required"] == sum(1 for f in frames if f["parity_required"])
    assert doc["summary"]["robustness_only"] == sum(
        1 for f in frames if f["robustness_only"])


def test_manifest_rois_equal_the_live_annotations():
    """三类 ROI + 两个确认位逐字段等于当前标注——**这条才是"派生数据过期"的探针**。"""
    cases = _checked_frames()
    assert len(cases) >= 20, f"这条锁要有对象可练（现在只有 {len(cases)} 帧）"
    bad = []
    for f in cases:
        ann = load_annotation(f["file"])
        want_base = [list(r) for r in (ann.base_sample_rois or [])]
        if not want_base and ann.base_roi:
            want_base = [list(ann.base_roi)]
        want = {
            "output_crop_roi": list(ann.output_crop_roi or ann.image_area_roi)
            if (ann.output_crop_roi or ann.image_area_roi) else None,
            "image_area_roi": list(ann.image_area_roi) if ann.image_area_roi else None,
            "base_sample_rois": want_base,
            "confirmed_by_user": bool(ann.is_user_verified()),
            "needs_reconfirm": bool(getattr(ann, "needs_reconfirm", False)),
            "source": "annotation",
        }
        got = f["rois"]
        diff = {k: (got.get(k), v) for k, v in want.items() if got.get(k) != v}
        if diff:
            bad.append((f["id"], diff))
    for fid, diff in bad[:6]:
        print(f"  {fid}: " + "; ".join(f"{k} 清单={a} 标注={b}" for k, (a, b) in diff.items()))
    assert not bad, (
        f"{len(bad)} 帧的清单 ROI 落后于标注（派生数据过期）⇒ "
        f"重跑 scripts/build_acceptance_manifest.py")


def test_negative_control_a_frame_without_annotation_stays_source_none():
    """**负控制**：没标注的帧清单里必须仍是 `source: none` + 全空 ROI。

    没有这一条，上一条锁可能被"给所有帧都塞进点什么"的做法糊过去。
    """
    doc = json.loads(MAN.read_text(encoding="utf-8"))
    hits = [f for f in doc["frames"]
            if (ROOT / f["file"]).exists() and load_annotation(f["file"]) is None]
    if not hits:
        pytest.skip("本语料里每一帧都有标注了——这条负控制失去对象，去看上一条")
    for f in hits:
        assert f["rois"]["source"] == "none", f["id"]
        assert f["rois"]["image_area_roi"] is None, f["id"]
        assert f["rois"]["base_sample_rois"] == [], f["id"]
        assert f["rois"]["confirmed_by_user"] is False, f["id"]
