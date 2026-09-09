"""报告 schema 统一回归（2026-09-08 交接 §4-3 后半）。

历史缺陷：batch.report()（schema 2）与 roll_export（schema 1）键集漂移——
同一份「批处理报告」概念两套口径，消费方（batch_panel/CLI/GUI）各读各的。

统一口径（本测试锁定）：
- 共用核心键集：schema_version=2 / software_version / started_utc /
  finished_utc / outdir / kind / summary / summary_complete / items；
- items 至少含 file、status，失败项含 error；
- kind 区分批次（batch）与整卷导出（roll）；roll 另有 roll 字段，
  batch 另有 profile 字段。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import tifffile

CORE_KEYS = {"schema_version", "software_version", "started_utc",
             "finished_utc", "outdir", "kind", "summary",
             "summary_complete", "items"}


@pytest.fixture(scope="module")
def synth_tiff(tmp_path_factory):
    from negative_lab.testing.synthetic import make_synthetic_negative

    d = tmp_path_factory.mktemp("reports")
    s = make_synthetic_negative(black_border=20, dust=10, seed=47)
    src = d / "frame.tif"
    tifffile.imwrite(src, (np.clip(s["negative"], 0, 1) * 65535).astype(np.uint16),
                     photometric="rgb")
    return src


def _core_asserts(doc: dict, outdir: Path):
    missing = CORE_KEYS - set(doc)
    assert not missing, f"报告缺统一核心键：{missing}"
    assert doc["schema_version"] == 2
    assert doc["kind"] in ("batch", "roll")
    assert isinstance(doc["summary"], dict) and doc["summary"]
    assert doc["summary_complete"] is True
    assert isinstance(doc["items"], list) and doc["items"]
    for it in doc["items"]:
        assert "file" in it and "status" in it
        if it["status"] == "failed":
            assert "error" in it and it["error"]


def test_roll_report_unified_schema(synth_tiff, tmp_path):
    from negative_lab.pipeline.roll_export import export_roll_frames

    out = tmp_path / "roll"
    rep = export_roll_frames(
        [_rec(synth_tiff)], out, tiff_assumption="linear")
    assert rep["summary_complete"] is True
    doc = json.loads((out / "roll-report.json").read_text(encoding="utf-8"))
    _core_asserts(doc, out)
    assert doc["kind"] == "roll"
    assert doc["roll"] is None or isinstance(doc["roll"], str)


def test_batch_report_unified_schema(synth_tiff, tmp_path):
    from negative_lab.pipeline.batch import BatchJob

    out = tmp_path / "batch"
    job = BatchJob([synth_tiff], out, tiff_encoding="linear")
    job.run()
    doc = json.loads((out / "batch-report.json").read_text(encoding="utf-8"))
    _core_asserts(doc, out)
    assert doc["kind"] == "batch"
    assert "profile" in doc


def _rec(src: Path):
    import hashlib

    from negative_lab.library.models import FrameRecord, GeometryRegions

    sha = hashlib.sha256(src.read_bytes()).hexdigest()
    return FrameRecord(source_path=src.name, source_sha256=sha,
                       input_facts={"abs_path": str(src), "kind": "tiff"},
                       geometry=GeometryRegions())
