"""v5 §4.2/§14.3：解码矩阵与能力宣称一致性测试。

规则：任何 `verified` 之外的矩阵格不得被文档/代码当作已支持；缺样片的格
用明确 skip/missing-fixture 标记，**不得伪造 PASS**。

v5 相对 v4 的变化：
- 黑白负片移出 P0（§0.2.5），不再是发布门禁项；
- ECN-2 现在有真实可解码 fixture（Z8 高效 NEF 的 16 位 TIFF 导出），
  因此"ECN-2 不可解码"的旧断言作废；
- 真实 16 位 TIFF 样片已存在，旧"缺 TIFF 样片"断言作废；
- 发布门禁改为：P0 覆盖的 C-41 与 ECN-2 必须 verified，
  且未覆盖组合必须被明确标为 missing-fixture（而不是被隐藏）。
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
    for key, name in (("c41_color", "C-41"), ("ecn2_color", "ECN-2"),
                      ("unknown_type", "未知处理类型"), ("bw", "黑白负片")):
        status = matrix[key][0]
        line = next((ln for ln in text.splitlines()
                     if name in ln and ln.startswith("|")), "")
        assert f"`{status}`" in line, f"{name} 行状态应为 {status}: {line}"


def test_p0_types_have_real_decodable_fixtures(matrix):
    """v5 §0.2.4/§0.2.5：C-41 与 ECN-2 都是 P0，必须有真实可解码样片。"""
    assert matrix["c41_color"][0] == "verified", matrix["c41_color"]
    assert matrix["ecn2_color"][0] == "verified", matrix["ecn2_color"]
    assert matrix["real_tiff16_samples"], "16 位 TIFF 输入路径必须有真实样片"


def test_uncovered_combinations_are_declared_not_hidden(matrix):
    """没有真实 fixture 的输入只能标 missing-fixture（§14.3）。"""
    assert not matrix["scanner_tiff_samples"] or matrix["dng_samples"]
    doc = MATRIX_DOC.read_text(encoding="utf-8")
    if not matrix["scanner_tiff_samples"]:
        assert "missing-fixture" in doc and "扫描仪" in doc
    if not matrix["dng_samples"]:
        assert "DNG" in doc and "missing-fixture" in doc


def test_single_device_claim(matrix):
    """所有样片同一相机：多设备结论一律未验证（v5 §9.3.4）。"""
    assert matrix["device_count"] <= 1, matrix["devices"]
    doc = MATRIX_DOC.read_text(encoding="utf-8")
    assert "跨设备泛化一律未验证" in doc


def test_bw_is_out_of_scope_not_a_gate(matrix):
    """v5 §0.2.5：黑白不在 P0，不得再作为发布门禁阻断项。"""
    assert matrix["bw"][0] == "out-of-scope"


def test_release_gate_p0_scope(matrix):
    """v5 发布门禁（机器部分）：P0 覆盖类型 verified + 未覆盖项如实标注。

    视觉门禁（逐张接受/不弱于 NLP）由 §10.2 联系表与用户签字负责，
    不在本测试范围内。
    """
    gaps = []
    if matrix["c41_color"][0] != "verified":
        gaps.append("C-41 缺少可解码标定/留出样片")
    if matrix["ecn2_color"][0] != "verified":
        gaps.append("ECN-2 缺少可解码标定/留出样片")
    doc = MATRIX_DOC.read_text(encoding="utf-8")
    if "黑白" in doc and "out-of-scope" not in doc:
        gaps.append("黑白被当作支持声明但未标注 out-of-scope")
    if gaps:
        pytest.fail("v5 P0 机器门禁未通过：\n  - " + "\n  - ".join(gaps))


def test_holdout_isolation(matrix):
    """v5 §9.3.2：训练与留出按卷隔离，同一卷不得跨集合。"""
    doc = json.loads(MANIFEST.read_text(encoding="utf-8"))
    split_by_roll: dict[str, set] = {}
    for e in doc["images"]:
        split_by_roll.setdefault(e["roll"], set()).add(e["split"])
    mixed = {r: s for r, s in split_by_roll.items() if len(s) > 1}
    assert not mixed, f"以下卷同时出现在多个集合（数据泄漏）：{mixed}"
