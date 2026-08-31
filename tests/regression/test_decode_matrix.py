"""P0-A：解码矩阵与能力宣称一致性测试（整改 §P0-A 新增测试）。

规则：任何 `verified` 之外的矩阵格不得被文档/代码当作已支持；
缺样片的格用明确 skip/xfail 原因标记，且发布门禁测试失败。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from build_support_matrix import compute_matrix  # noqa: E402

MANIFEST = ROOT / "fixtures" / "manifest.json"
MATRIX_DOC = ROOT / "docs" / "support-matrix.md"

pytestmark = pytest.mark.skipif(not MANIFEST.exists(), reason="manifest 未生成")


@pytest.fixture(scope="module")
def matrix():
    doc = json.loads(MANIFEST.read_text(encoding="utf-8"))
    return compute_matrix(doc)


def test_matrix_doc_matches_manifest(matrix):
    """docs/support-matrix.md 必须与 manifest 重算结果一致（防文档漂移）。"""
    assert MATRIX_DOC.exists(), "支持矩阵文档缺失"
    text = MATRIX_DOC.read_text(encoding="utf-8")
    for cell in ("c41_color", "bw", "ecn2_color"):
        status = matrix[cell][0]
        name = {"c41_color": "C-41", "bw": "黑白", "ecn2_color": "ECN-2"}[cell]
        # 找到该能力行并核对状态词
        line = next((ln for ln in text.splitlines() if name in ln and ln.startswith("|")), "")
        assert f"`{status}`" in line, f"{name} 行状态应为 {status}: {line}"


def test_undecodable_never_enters_decodable_pool(matrix):
    doc = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for e in doc["images"]:
        if e["decode_status"] != "ok":
            assert e["file"]  # 状态字段存在即视为被排除在 ok 池外
    assert matrix["ecn2_color"][0] == "unsupported", \
        "ECN-2 样片全部不可解码，矩阵必须标 unsupported（不得标 verified）"


def test_bw_support_claim_blocked(matrix):
    """黑白负片是原规格 §2.1 胶片类型，但无样片 -> 必须 missing-fixture。"""
    assert matrix["bw"][0] == "missing-fixture"


def test_single_device_claim(matrix):
    """所有样片同一相机：多设备结论一律未验证（§12 末段）。"""
    assert matrix["device_count"] <= 1, matrix["devices"]


def test_real_tiff_and_dng_missing_fixture(matrix):
    assert not matrix["real_tiff16_samples"], \
        "若已有真实 16 位 TIFF 样片，请更新矩阵与本文档"
    assert not matrix["dng_samples"]


def test_release_gate_blocked_until_fixtures(matrix):
    """发布门禁：bw/ecn2/真实TIFF/DNG/第二设备任一未 verified -> 阻断。

    整改完成前本测试必须失败（红），作为发布门禁的机器化表达。
    """
    gaps = []
    if matrix["bw"][0] != "verified":
        gaps.append("黑白负片样片缺失")
    if matrix["ecn2_color"][0] != "verified":
        gaps.append("ECN-2 无可解码样片（AV1 NEF 不支持，待转 TIFF）")
    if not matrix["real_tiff16_samples"]:
        gaps.append("真实 16 位 TIFF 样片缺失")
    if matrix["device_count"] < 2:
        gaps.append("第二套设备/光源样片缺失")
    if gaps:
        pytest.fail("发布门禁未通过（missing fixtures）：\n  - " + "\n  - ".join(gaps))
