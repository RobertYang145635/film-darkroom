"""v4 §8 P0-F / §3.4：annotation 数据完整性契约测试。

覆盖（§3.4 + §8 验收）：
- 唯一公共保存 API（模块内不得出现第二个 save_annotation 定义）；
- 保存必经 validate()：越界/反向/空 ROI、非法旋转、verified 缺来源必须拒绝；
- 临时文件 + os.replace 原子替换，替换失败不得破坏原文件；
- load_all_verified() 可调用（不是死代码），损坏文件进 errors 不静默跳过；
- 旧 schema（frame_roi）兼容加载并迁移为 image_area_roi 草稿；
- verified_by="agent-inspected" 不构成用户签字 -> status="draft"；
- schema_version / source_sha256 落盘。

全部使用 tmp_path + monkeypatch ANNOTATIONS_DIR，不写真实 fixtures/。
"""

from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

import pytest

import negative_lab.calibration.annotations as A
from negative_lab.calibration.annotations import (Annotation, ann_path_for,
                                                  load_all_verified,
                                                  load_annotation,
                                                  save_annotation)

REL = "roll_1298_G200/shoot/G200-1.NEF"


@pytest.fixture
def ann_dir(tmp_path, monkeypatch):
    """把标注目录重定向到 tmp_path，禁止触碰真实 fixtures/annotations。"""
    monkeypatch.setattr(A, "ANNOTATIONS_DIR", tmp_path)
    return tmp_path


def _v2(**kw) -> Annotation:
    """v5 §3.2：verified 必须给出三类独立区域（片基区可以有多个）。"""
    base = dict(file=REL, roll="1298_G200", rotation_degrees=180,
                output_crop_roi=(80, 80, 520, 950),
                image_area_roi=(100, 100, 500, 900),
                base_sample_rois=[(0, 0, 50, 1000)],
                base_roi=(0, 0, 50, 1000),
                source_sha256="a" * 64)
    base.update(kw)
    return Annotation(**base)


# --------------------------------------------------------------------------
# 1. 唯一公共保存 API（v4 §1.3.1 实锤：重复定义互相覆盖）
# --------------------------------------------------------------------------

def test_only_one_public_save_annotation():
    src = Path(inspect.getfile(A)).read_text(encoding="utf-8")
    tree = ast.parse(src)
    defs = [n for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            and n.name == "save_annotation"]
    assert len(defs) == 1, [
        f"save_annotation 定义了 {len(defs)} 次（第 1 个被覆盖）: "
        f"行 {[d.lineno for d in defs]}"]
    # 赋值式别名（save_annotation = ...）也算第二定义
    assigns = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
               for t in n.targets if isinstance(t, ast.Name)
               and t.id == "save_annotation"]
    assert not assigns, "save_annotation 被二次赋值绑定"


def test_save_annotation_signature_is_rel_ann():
    params = list(inspect.signature(save_annotation).parameters)
    assert params[:2] == ["rel", "ann"], params


# --------------------------------------------------------------------------
# 2. validate() 拒绝非法输入（§3.4：越界/反向/空 ROI、非法旋转、缺来源）
# --------------------------------------------------------------------------

@pytest.mark.parametrize("field,bad", [
    ("image_area_roi", (500, 100, 100, 900)),        # y1 < y0 反向
    ("image_area_roi", (100, 900, 500, 100)),        # x1 < x0 反向
    ("image_area_roi", (100, -20, 500, 900)),        # 越界（负坐标）
    ("image_area_roi", (100, 100, 100, 900)),        # 空（高度 0）
    ("image_area_roi", (100, 100, 500, 100)),        # 空（宽度 0）
    ("output_crop_roi", (500, 100, 100, 900)),
    ("output_crop_roi", (-1, 0, 500, 900)),
    ("base_roi", (400, 0, 100, 900)),
    ("base_roi", (0, 0, 0, 0)),
])
def test_validate_rejects_bad_roi(field, bad):
    ann = _v2()
    setattr(ann, field, bad)
    assert ann.validate(), f"{field}={bad} 必须报错"


def test_validate_rejects_bad_shape_and_type():
    for bad in ((1, 2, 3), (1, 2, 3, 4, 5), "abcd"):
        ann = _v2()
        ann.image_area_roi = bad
        assert ann.validate(), f"image_area_roi={bad!r} 必须报错"


@pytest.mark.parametrize("rot", [45, -90, 360, "180", True])
def test_validate_rejects_bad_rotation(rot):
    ann = _v2()
    ann.rotation_degrees = rot
    assert ann.validate(), f"rotation_degrees={rot!r} 必须报错"


@pytest.mark.parametrize("cs", ["oriented ", "pixel", ""])
def test_validate_rejects_bad_coordinate_space(cs):
    ann = _v2()
    ann.coordinate_space = cs
    assert ann.validate(), f"coordinate_space={cs!r} 必须报错"


def test_validate_rejects_verified_without_provenance():
    ann = _v2(status="verified", verified=True, verified_by="")
    assert ann.validate(), "verified 但缺 verified_by 必须报错"


def test_validate_rejects_verified_without_image_area():
    ann = Annotation(file=REL, roll="1298_G200", status="verified",
                     verified=True, verified_by="user")
    errs = ann.validate()
    assert errs, "verified 但缺 image_area_roi 必须报错"
    assert any("image_area" in e for e in errs), errs


def test_validate_rejects_bad_sha_and_schema_version():
    ann = _v2(source_sha256="deadbeef")
    assert ann.validate(), "非法 sha256 必须报错"
    ann = _v2(schema_version=0)
    assert ann.validate(), "schema_version=0 必须报错"
    assert not _v2().validate(), f"合法标注不应报错: {_v2().validate()}"


def test_validate_rejects_bad_neutral_and_mask_refs():
    ann = _v2(neutral_refs=[{"roi": (10, 10, 5, 5), "kind": "bright"}])
    assert ann.validate(), "中性参考 ROI 反向必须报错"
    ann = _v2(neutral_refs=[{"kind": "purple"}])
    assert ann.validate(), "中性参考缺 roi/point 必须报错"
    ann = _v2(exclude_mask_rois=[(50, 50, 10, 10)])
    assert ann.validate(), "exclude mask ROI 反向必须报错"
    ok = _v2(neutral_refs=[{"roi": (10, 10, 40, 40), "kind": "bright"},
                           {"point": [12, 34], "kind": "dark"}],
             exclude_mask_rois=[(0, 0, 10, 10)])
    assert not ok.validate(), ok.validate()


# --------------------------------------------------------------------------
# 3. 保存必经 validate()：无效数据不得落盘（§1.3.3）
# --------------------------------------------------------------------------

@pytest.mark.parametrize("field,bad", [
    ("image_area_roi", (500, 100, 100, 900)),
    ("base_roi", (0, 0, 0, 0)),
])
def test_save_refuses_invalid_and_writes_nothing(ann_dir, field, bad):
    ann = _v2(status="verified", verified=True, verified_by="user")
    setattr(ann, field, bad)
    with pytest.raises(ValueError):
        save_annotation(REL, ann)
    assert list(ann_dir.glob("*.json")) == [], "非法数据不得落盘"


def test_save_refuses_verified_without_provenance(ann_dir):
    ann = _v2(status="verified", verified=True, verified_by="")
    with pytest.raises(ValueError):
        save_annotation(REL, ann)
    assert list(ann_dir.glob("*.json")) == []


def test_save_is_reachable_from_positional_api(ann_dir):
    """write_p0a_annotations.py 曾因单参数调用 TypeError（§1.3.1）。"""
    ann = Annotation(file=REL, roll="1298_G200", rotation_degrees=180,
                     frame_roi=(966, 376, 6541, 5166),
                     base_roi=(6721, 66, 7370, 5522),
                     verified=True, verified_by="agent-inspected",
                     verified_date="2026-08-31")
    p = save_annotation(REL, ann)
    assert p.exists()
    doc = json.loads(p.read_text(encoding="utf-8"))
    assert doc["status"] == "draft"          # §8.4-5：agent 检测只能 draft
    assert doc["verified"] is False


# --------------------------------------------------------------------------
# 4. 原子保存：临时文件 + os.replace；替换失败不得破坏原文件
# --------------------------------------------------------------------------

def test_save_writes_exactly_at_ann_path(ann_dir):
    ann = _v2(status="verified", verified=True, verified_by="user")
    p = save_annotation(REL, ann)
    assert p == ann_path_for(REL)
    assert p.exists()
    doc = json.loads(p.read_text(encoding="utf-8"))
    assert doc["file"] == REL
    assert tuple(doc["image_area_roi"]) == (100, 100, 500, 900)
    # 原子写不留临时文件
    leftovers = [f.name for f in ann_dir.iterdir()
                 if f.suffix != ".json" or f.name.endswith(".tmp")]
    assert leftovers == [], f"残留临时文件: {leftovers}"


def test_save_uses_atomic_replace(ann_dir, monkeypatch):
    calls = []
    real = A.os.replace

    def spy(src, dst, *a, **k):
        calls.append((Path(src), Path(dst)))
        return real(src, dst, *a, **k)

    monkeypatch.setattr(A.os, "replace", spy)
    save_annotation(REL, _v2())
    assert len(calls) == 1, "必须经由 os.replace 原子替换"
    src, dst = calls[0]
    assert dst == ann_path_for(REL)
    assert src.parent == ann_dir and src != dst


def test_replace_failure_keeps_original_intact(ann_dir, monkeypatch):
    first = _v2(notes="first")
    save_annotation(REL, first)
    target = ann_path_for(REL)
    before = target.read_text(encoding="utf-8")

    def boom(src, dst, *a, **k):
        raise OSError("replace failed")

    monkeypatch.setattr(A.os, "replace", boom)
    with pytest.raises(OSError):
        save_annotation(REL, _v2(notes="second"))
    assert target.read_text(encoding="utf-8") == before, "原文件被破坏"
    # 失败路径必须清理临时文件
    leftovers = [f.name for f in ann_dir.iterdir() if f.name != target.name]
    assert leftovers == [], f"失败后残留: {leftovers}"


# --------------------------------------------------------------------------
# 5. load_all_verified：可调用、非死代码、损坏文件不静默跳过
# --------------------------------------------------------------------------

def test_load_all_verified_empty_dir(ann_dir):
    assert load_all_verified() == []


def test_load_all_verified_returns_only_verified(ann_dir):
    save_annotation("roll/a.NEF", _v2(file="roll/a.NEF", status="verified",
                                      verified=True, verified_by="user"))
    save_annotation("roll/b.NEF", _v2(file="roll/b.NEF", status="draft",
                                      notes="draft"))
    out = load_all_verified()
    assert isinstance(out, list)
    assert [a.file for a in out] == ["roll/a.NEF"]
    assert all(isinstance(a, Annotation) for a in out)


def test_load_all_verified_reports_corrupt_files(ann_dir):
    save_annotation("roll/good.NEF", _v2(file="roll/good.NEF", status="verified",
                                         verified=True, verified_by="user"))
    (ann_dir / "broken.json").write_text("{ not json", encoding="utf-8")
    (ann_dir / "invalid.json").write_text(json.dumps({
        "file": "roll/bad.NEF", "roll": "bad", "rotation_degrees": 45}),
        encoding="utf-8")
    out = load_all_verified()
    assert [a.file for a in out] == ["roll/good.NEF"]
    anns, errors = A.scan_annotations()
    assert [a.file for a in anns] == ["roll/good.NEF"]
    assert len(errors) == 2, errors
    joined = " ".join(errors)
    assert "broken.json" in joined and "invalid.json" in joined
    assert A.LAST_LOAD_ERRORS == errors


def test_load_all_verified_ignores_other_dirs(ann_dir):
    """只扫 ANNOTATIONS_DIR 顶层 *.json（命名约定不变）。"""
    sub = ann_dir / "nested"
    sub.mkdir()
    (sub / "x.json").write_text("{ not json", encoding="utf-8")
    assert load_all_verified() == []


# --------------------------------------------------------------------------
# 6. 旧 schema 兼容：frame_roi 迁移为 image_area_roi 草稿
# --------------------------------------------------------------------------

LEGACY = {
    "file": REL, "roll": "1298_G200", "rotation_degrees": 180,
    "coordinate_space": "source",
    "frame_roi": [966, 376, 6541, 5166],
    "base_roi": [6721, 66, 7370, 5522],
    "neutral_rois": [{"roi": [100, 100, 140, 140], "kind": "bright"}],
    "mask_rois": [[0, 0, 20, 20]],
    "reference_file": "ref.jpg",
    "notes": "legacy",
    "verified": True, "verified_by": "user", "verified_date": "2026-08-31",
}


def _write_legacy(doc: dict, ann_dir) -> Path:
    p = ann_path_for(REL)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    return p


def test_legacy_frame_roi_migrates_to_draft_image_area(ann_dir):
    _write_legacy(LEGACY, ann_dir)
    got = load_annotation(REL)
    assert got is not None
    assert got.frame_roi == (966, 376, 6541, 5166)          # 原字段仍可读
    assert got.image_area_roi == (966, 376, 6541, 5166)     # 迁移
    assert got.output_crop_roi is None                      # 不猜构图
    assert got.needs_reconfirm is True
    assert got.status == "draft"
    assert got.verified is False
    assert not got.validate(), got.validate()


def test_legacy_schema_version_one_also_migrates(ann_dir):
    doc = dict(LEGACY, schema_version=1)
    _write_legacy(doc, ann_dir)
    got = load_annotation(REL)
    assert got.needs_reconfirm is True
    assert got.image_area_roi == (966, 376, 6541, 5166)


def test_legacy_neutral_and_mask_aliases_migrate(ann_dir):
    _write_legacy(LEGACY, ann_dir)
    got = load_annotation(REL)
    assert got.neutral_refs == [{"roi": (100, 100, 140, 140), "kind": "bright"}]
    assert got.exclude_mask_rois == [(0, 0, 20, 20)]


def test_v2_file_is_not_marked_needs_reconfirm(ann_dir):
    save_annotation(REL, _v2(output_crop_roi=(0, 0, 600, 1000)))
    got = load_annotation(REL)
    assert got.needs_reconfirm is False
    assert got.output_crop_roi == (0, 0, 600, 1000)
    assert got.image_area_roi == (100, 100, 500, 900)


# --------------------------------------------------------------------------
# 7. draft/verified 语义：只有 user 才是用户签字（§8.4-5）
# --------------------------------------------------------------------------

def test_agent_inspected_is_draft_on_load(ann_dir):
    doc = {"file": REL, "roll": "1298_G200", "schema_version": 2,
           "image_area_roi": [0, 0, 100, 100], "base_roi": [0, 0, 10, 10],
           "verified": True, "verified_by": "agent-inspected",
           "verified_date": "2026-08-31"}
    p = ann_path_for(REL)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc), encoding="utf-8")
    got = load_annotation(REL)
    assert got.status == "draft"
    assert got.verified is False
    assert load_all_verified() == []


def test_user_signed_is_verified_on_load(ann_dir):
    doc = {"file": REL, "roll": "1298_G200", "schema_version": A.SCHEMA_VERSION,
           "output_crop_roi": [0, 0, 110, 110],
           "image_area_roi": [0, 0, 100, 100],
           "base_sample_rois": [[0, 0, 10, 10]],
           "verified": True, "verified_by": "user",
           "verified_date": "2026-09-01"}
    p = ann_path_for(REL)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc), encoding="utf-8")
    got = load_annotation(REL)
    assert got.status == "verified"
    assert got.verified is True
    assert [a.file for a in load_all_verified()] == [REL]


def test_legacy_v2_user_signed_downgrades_to_draft(ann_dir):
    """v5 §3.2：旧 schema 只有单一 frame/base 语义，用途必须重新确认。"""
    doc = {"file": REL, "roll": "1298_G200", "schema_version": 2,
           "image_area_roi": [0, 0, 100, 100], "base_roi": [0, 0, 10, 10],
           "verified": True, "verified_by": "user",
           "verified_date": "2026-09-01"}
    p = ann_path_for(REL)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc), encoding="utf-8")
    got = load_annotation(REL)
    assert got.status == "draft" and got.needs_reconfirm is True
    assert load_all_verified() == []


def test_save_downgrades_agent_inspected_to_draft(ann_dir):
    ann = _v2(status="verified", verified=True,
              verified_by="agent-inspected", verified_date="2026-08-31")
    save_annotation(REL, ann)
    doc = json.loads(ann_path_for(REL).read_text(encoding="utf-8"))
    assert doc["status"] == "draft" and doc["verified"] is False


def test_save_fills_verified_date_for_user_signed(ann_dir):
    ann = _v2(status="verified", verified=True, verified_by="user")
    save_annotation(REL, ann)
    doc = json.loads(ann_path_for(REL).read_text(encoding="utf-8"))
    assert doc["status"] == "verified"
    assert doc["verified_date"], "user 签字必须带日期"


# --------------------------------------------------------------------------
# 8. schema_version / source_sha256 落盘（§8.10）
# --------------------------------------------------------------------------

def test_schema_version_and_sha_persisted(ann_dir):
    ann = _v2(status="verified", verified=True, verified_by="user",
              output_crop_roi=(0, 0, 600, 1000),
              neutral_refs=[{"roi": (10, 10, 40, 40), "kind": "bright"}],
              exclude_mask_rois=[(0, 0, 20, 20)])
    save_annotation(REL, ann)
    doc = json.loads(ann_path_for(REL).read_text(encoding="utf-8"))
    assert doc["schema_version"] == A.SCHEMA_VERSION == 3
    assert doc["source_sha256"] == "a" * 64
    for k in ("output_crop_roi", "image_area_roi", "base_roi",
              "base_sample_rois", "neutral_refs",
              "exclude_mask_rois", "status", "needs_reconfirm", "verified_by",
              "verified_date"):
        assert k in doc, f"缺字段 {k}"
    got = load_annotation(REL)
    assert got.source_sha256 == "a" * 64
    assert got.neutral_refs == [{"roi": (10, 10, 40, 40), "kind": "bright"}]
    assert got.exclude_mask_rois == [(0, 0, 20, 20)]
    assert [tuple(r) for r in got.base_sample_rois] == [(0, 0, 50, 1000)]


def test_frame_roi_promotes_to_image_area_on_save(ann_dir):
    """v5 §3.2：旧单一画面框只能迁移为 image_area_roi 草稿。

    试图把缺少输出裁切/片基区的记录写成 verified 必须被拒绝——
    三类区域是三种独立数据，不允许再用一个矩形冒充。
    """
    ann = Annotation(file=REL, roll="1298_G200", rotation_degrees=90,
                     frame_roi=(10, 20, 300, 400), base_roi=(0, 0, 50, 500),
                     verified=True, verified_by="user")
    with pytest.raises(ValueError, match="output_crop_roi"):
        save_annotation(REL, ann)
    assert not ann_path_for(REL).exists(), "被拒绝的写入不得落盘"
    # 补齐三类区域后才能成为用户确认标注
    ann.output_crop_roi = (10, 20, 300, 400)
    ann.base_sample_rois = [(0, 0, 50, 500)]
    save_annotation(REL, ann)
    doc = json.loads(ann_path_for(REL).read_text(encoding="utf-8"))
    assert tuple(doc["image_area_roi"]) == (10, 20, 300, 400)
    got = load_annotation(REL)
    assert got.status == "verified" and got.verified_by == "user"


# --------------------------------------------------------------------------
# 9. 命名约定与真实 fixtures 只读兼容（不写盘）
# --------------------------------------------------------------------------

def test_ann_path_for_convention(ann_dir):
    p = ann_path_for("20260709_SAMPLE-ID(1)/1298_G200/翻拍/G200-1.NEF")
    assert p == ann_dir / "1298_G200__G200-1.json"
    assert ann_path_for("only.NEF").name == "misc__only.json"


def test_load_missing_returns_none(ann_dir):
    assert load_annotation("nope/x.NEF") is None


def test_real_fixture_still_loads_as_needs_reconfirm_draft():
    """真实 1298 标注（旧 schema + agent-inspected）必须仍能加载。只读。"""
    real = Path(inspect.getfile(A)).resolve().parents[3] / "fixtures" / "annotations"
    f = real / "1298_G200__G200-1.json"
    if not f.exists():
        pytest.skip("fixtures/annotations 不存在")
    doc = json.loads(f.read_text(encoding="utf-8"))
    got = Annotation.from_dict(doc)
    assert got.status == "draft"
    assert got.needs_reconfirm is True
    assert got.image_area_roi == (966, 376, 6541, 5166)
    assert got.frame_roi == (966, 376, 6541, 5166)
    assert not got.validate(), got.validate()
