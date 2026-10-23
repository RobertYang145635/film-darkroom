"""P0-A：manifest 与真实解码探针一致性测试（整改 §P0-A 新增测试）。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from negative_lab.calibration.annotations import ANNOTATIONS_DIR, Annotation
from negative_lab.io.probe import probe_file

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "fixtures" / "manifest.json"

pytestmark = pytest.mark.skipif(not MANIFEST.exists(), reason="manifest 未生成")


@pytest.fixture(scope="module")
def doc():
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_schema_v3_fields(doc):
    assert doc["schema_version"] == 3      # v5：ECN-2 TIFF 以独立逻辑卷入库
    for e in doc["images"]:
        for k in ("file", "roll", "split", "sha256", "decode_status",
                  "decode_params", "camera", "bit_depth", "icc_profile",
                  "input_kind", "capture_type"):
            assert k in e, f"缺字段 {k}: {e['file']}"
        assert e["decode_status"] in ("ok", "unsupported", "corrupt",
                                      "missing_dependency", "unknown")


def test_p0_inputs_present(doc):
    """v5 §0.2.4/§0.2.5：C-41 与 ECN-2 必须有可解码样片（真实 fixture）。"""
    ok = [e for e in doc["images"] if e["decode_status"] == "ok"]
    by_type = {}
    for e in ok:
        by_type.setdefault(e["film_type"], []).append(e)
    assert by_type.get("c41_color"), "缺 C-41 真实样片"
    assert by_type.get("ecn2_color"), "缺 ECN-2 真实样片"
    tiffs = [e for e in ok if e["file"].lower().endswith((".tif", ".tiff"))]
    assert tiffs, "缺 16 位 TIFF 输入样片"
    # ECN-2 TIFF 如实标记来源：相机 RAW 导出，不冒充扫描仪
    assert all(e["capture_type"] == "camera_raw_export_tiff" for e in tiffs)


def test_split_isolated_by_roll(doc):
    """v5 §9.3：训练/验证/留出按卷隔离。"""
    by_roll: dict[str, set] = {}
    for e in doc["images"]:
        by_roll.setdefault(e["roll"], set()).add(e["split"])
    mixed = {r: s for r, s in by_roll.items() if len(s) > 1}
    assert not mixed, mixed


def test_probe_matches_manifest_spotcheck(doc):
    """抽查：manifest 状态必须与重新运行的真实探针一致（F-5 回归锁）。"""
    # 必查：一张 ok 的 NEF + 一张被标 unsupported 的 5219/5207 NEF
    picks = []
    ok_nef = next((e for e in doc["images"]
                   if e["decode_status"] == "ok" and e["file"].endswith(".NEF")), None)
    unsup = next((e for e in doc["images"]
                  if e["decode_status"] == "unsupported"), None)
    corrupt = {"file": "__nonexistent__.NEF", "decode_status": "corrupt"}
    if ok_nef:
        picks.append(ok_nef)
    if unsup:
        picks.append(unsup)
    assert picks, "manifest 无 NEF 样片可抽查"
    missing = [e for e in picks if not (ROOT / e["file"]).exists()]
    if missing:
        pytest.skip(f"私有语料不在库里（gitignore），抽查无对象：缺 {len(missing)} 张")
    for e in picks:
        pr = probe_file(ROOT / e["file"])
        assert pr.status == e["decode_status"], \
            f"{e['file']}: manifest={e['decode_status']} probe={pr.status} ({pr.reason})"
    assert probe_file(ROOT / corrupt["file"]).status == "corrupt"


def test_no_ok_flag_on_known_av1_rolls(doc):
    """5219/5207 为 Z8 AV1 高效 NEF：任何一张标 ok 即为矛盾（历史 F-5）。"""
    for e in doc["images"]:
        if e["roll"] in ("1171_5219", "1172_5207"):
            assert e["decode_status"] != "ok", \
                f"AV1 NEF 被标 ok（字节启发式回归）: {e['file']}"


def test_holdout_ratio_by_roll(doc):
    """按卷隔离的留出集 >= 20%（可解码卷中）。"""
    decodable_rolls = sorted({e["roll"] for e in doc["images"]
                              if e["decode_status"] == "ok"})
    assert decodable_rolls
    hold = [r for r in decodable_rolls
            if any(e["split"] == "holdout" for e in doc["images"] if e["roll"] == r)]
    assert len(hold) / len(decodable_rolls) >= 0.2, \
        f"留出卷不足: {hold} / {decodable_rolls}"
    # 同一卷不得同时出现在两个集合
    for r in decodable_rolls:
        splits = {e["split"] for e in doc["images"] if e["roll"] == r}
        assert len(splits) == 1, f"卷 {r} 跨集合: {splits}"


def test_annotations_cross_reference(doc):
    """每个标注文件必须对应 manifest 中真实存在的样片；verified 标注必须合法。"""
    files = {e["file"] for e in doc["images"]}
    for f in sorted(ANNOTATIONS_DIR.glob("*.json")):
        a = Annotation.from_dict(json.loads(f.read_text(encoding="utf-8")))
        assert not a.validate(), f"{f.name}: {a.validate()}"
        assert a.file in files, f"标注指向不存在样片: {a.file}"
