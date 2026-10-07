"""锁：验收套具的**方向必须来自标注**，不许再写死 0 度。

**由来（2026-10-07 实测）**：`scripts/build_visual_contact_sheet.py::_geometry()`
此前恒为 `rotation_degrees=0`，而 `fixtures/annotations/1298_G200__G200-1.json`
明写 `rotation_degrees: 180`（备注"倒置、含片边、偏青蓝、冲淡"）。
后果：**48 帧判定条图是倒的**，而 `metrics.json` 里
`top_mean/top_std` 取的是**画面上三分之一**——对倒像来说那正是场景的**下缘**，
所以"高亮在上方"一类分层的输入是错的。
用户手工参考到手后，这个缺陷第一次变得**看得见**（并排时方向不一致）。

本文件锁四件事：
1. 有标注且归属校验通过的帧 ⇒ `_geometry()` 必须给出标注里的旋转（含微调角与镜像）；
2. 没标注的帧 ⇒ 保持 0 且**不崩**（旧行为，语料里大多数帧就是这种）；
3. 标注声明 `oriented` 空间 ⇒ **必须抛错**，不得静默按 source 空间解释
   （旋转为 0 时两种空间重合，静默=看不见的错）；
4. 钉住"**标注存在但被归属校验拒绝**"的已知集合（实测 3 帧，`1300_CN400`
   目录 `翻拍/` ↔ `翻拍2/翻拍/` 迁移所致）——它们仍按 0 度出图，
   多一帧就红，因为那说明又发生了一次没人核对的迁移。
"""

from __future__ import annotations

import importlib.util
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "build_visual_contact_sheet.py"


def _load():
    spec = importlib.util.spec_from_file_location("bvcs_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _manifest_frames() -> list[dict]:
    doc = json.loads((ROOT / "fixtures" / "acceptance" / "manifest.json")
                     .read_text(encoding="utf-8"))
    return sorted(doc["frames"], key=lambda f: str(f["id"]))


def _applied_annotation(file_rel: str):
    """**按产品的口径**取标注：归属不符会被 `load_annotation` 拒绝并返回 None。

    这里绝不绕过归属校验去读 JSON——绕过就等于把"套具看到的"和"产品看到的"
    分成两件事，那正是本文件要消灭的那类偏差。
    """
    from negative_lab.calibration.annotations import load_annotation
    return load_annotation(file_rel)


def _annotation_exists(file_rel: str) -> bool:
    from negative_lab.calibration.annotations import ann_path_for
    return ann_path_for(file_rel).is_file()


#: 2026-10-07 实测：**已知的 3 帧归属不符**（`1300_CN400` 目录迁移
#: `翻拍/` ↔ `翻拍2/翻拍/` 留下的一进一出）。它们被归属校验正确拒绝，
#: 代价是那 3 帧的判定条与分层统计仍按 0 度算。**这是记录在案的数据缺陷，
#: 不是套具的 bug**——所以钉成"必须恰好是这 3 帧"，多一帧就红。
KNOWN_REFUSED = {"1300_CN400::CN400-1", "1300_CN400::CN400-13",
                 "1300_CN400::CN400-5"}


def test_geometry_uses_the_annotation_rotation():
    mod = _load()
    checked = []
    for fr in _manifest_frames():
        ann = _applied_annotation(fr["file"])
        want = int(ann.rotation_degrees) if ann is not None else 0
        got = mod._geometry(fr).rotation_degrees
        assert got == want, f"{fr['id']}: 套具旋转 {got} ≠ 标注 {want}"
        checked.append((fr["id"], want))
    assert checked, "一帧都没查到——本锁空跑"
    assert any(r != 0 for _, r in checked), (
        "所有标注都是 0 度，本锁无法证明旋转真的传下去了")


def test_refused_annotations_stay_at_the_documented_three():
    """钉住"标注存在但被归属校验拒绝"的集合：多一帧说明又发生了目录迁移。"""
    refused = {fr["id"] for fr in _manifest_frames()
               if _annotation_exists(fr["file"])
               and _applied_annotation(fr["file"]) is None}
    assert refused == KNOWN_REFUSED, (
        f"归属不符的帧变了：新增 {sorted(refused - KNOWN_REFUSED)}，"
        f"消失 {sorted(KNOWN_REFUSED - refused)}")


def test_geometry_without_annotation_stays_zero_and_does_not_crash():
    """没标注 ⇒ 0 度（与旧行为一致）。语料里多数帧是这种，**不能因为找不到标注就崩**。"""
    mod = _load()
    missing = [fr for fr in _manifest_frames()
               if not _annotation_exists(fr["file"])]
    assert missing, "语料里每帧都有标注了，本用例的前提不再成立（请改测试，别删）"
    for fr in missing[:5]:
        assert mod._geometry(fr).rotation_degrees == 0, f"{fr['id']} 无标注却非 0 度"


def test_geometry_refuses_oriented_space_annotations(monkeypatch):
    """`oriented` 空间必须**停下**，不得按 source 空间的框去旋转。"""
    from negative_lab.calibration.annotations import Annotation

    mod = _load()
    bogus = Annotation(file="x/1298_G200/翻拍/G200-1.NEF", roll="1298_G200",
                       rotation_degrees=180, coordinate_space="oriented")
    monkeypatch.setattr(mod, "load_annotation", lambda _f: bogus)
    frame = next(f for f in _manifest_frames() if f["roll"] == "1298_G200")
    try:
        mod._geometry(frame)
    except RuntimeError as e:
        assert "oriented" in str(e), f"报错没说明是空间不匹配：{e}"
    else:
        raise AssertionError("oriented 空间的标注被静默接受了——旋转会把裁切挪到别处")


def test_fine_rotation_and_mirror_travel_with_the_rotation(monkeypatch):
    """微调角与镜像也必须来自标注：只传主旋转等于**丢了一半几何**。"""
    from negative_lab.calibration.annotations import Annotation

    mod = _load()
    ann = Annotation(file="x/1298_G200/翻拍/G200-1.NEF", roll="1298_G200",
                     rotation_degrees=90, fine_rotation_degrees=-2.5,
                     mirror_horizontal=True)
    monkeypatch.setattr(mod, "load_annotation", lambda _f: ann)
    frame = next(f for f in _manifest_frames() if f["roll"] == "1298_G200")
    geo = mod._geometry(frame)
    assert (geo.rotation_degrees, geo.fine_rotation_degrees,
            geo.mirror_horizontal) == (90, -2.5, True), (
        f"几何变换没全传：{geo.rotation_degrees}/"
        f"{geo.fine_rotation_degrees}/{geo.mirror_horizontal}")
