"""§8.1.6：批处理单个文件失败不影响其余；取消后不留下半成品。"""

import json
from pathlib import Path

import numpy as np
import pytest
import tifffile

from negative_lab.pipeline.batch import BatchJob, collect_files
from negative_lab.profiles.profile import FilmProfile
from negative_lab.testing.synthetic import make_synthetic_negative


def _write_synth(dirpath: Path, name: str, seed: int) -> Path:
    s = make_synthetic_negative(black_border=16, seed=seed)
    p = dirpath / name
    tifffile.imwrite(p, (np.clip(s["negative"], 0, 1) * 65535 + 0.5).astype(np.uint16),
                     photometric="rgb")
    return p


@pytest.fixture
def batch_inputs(tmp_path):
    d = tmp_path / "in"
    d.mkdir()
    ok1 = _write_synth(d, "roll-a.tif", 71)
    ok2 = _write_synth(d, "roll-b.tif", 72)
    bad = d / "corrupt.tif"
    bad.write_bytes(b"not a tiff at all")
    return d, [ok1, bad, ok2]


def test_single_failure_does_not_stop_batch(tmp_path, batch_inputs):
    d, files = batch_inputs
    job = BatchJob(files, tmp_path / "out", tiff_encoding="linear")
    rep = job.run()
    st = {Path(i["file"]).name: i["status"] for i in rep["items"]}
    assert st["corrupt.tif"] == "failed"
    assert st["roll-a.tif"] in ("ok", "warn", "needs_calibration")
    assert st["roll-b.tif"] in ("ok", "warn", "needs_calibration")
    # 成功项有输出，失败项没有
    for i in rep["items"]:
        if i["status"] == "failed":
            assert i["outputs"] == [] and i["error"]
        else:
            assert all(Path(o).exists() for o in i["outputs"])
    # 报告文件
    rp = job.write_report()
    doc = json.loads(rp.read_text(encoding="utf-8"))
    assert doc["summary"]["failed"] == 1
    assert (tmp_path / "out" / "batch-report.txt").exists()


def test_cancel_leaves_no_partial(tmp_path, batch_inputs):
    d, files = batch_inputs
    out = tmp_path / "out"
    job = BatchJob(files, out, tiff_encoding="linear")
    job.cancel()
    rep = job.run()
    assert all(i["status"] == "cancelled" for i in rep["items"])
    assert not list(out.glob("*.tif")) and not list(out.glob("*.jpg"))
    assert not list(out.glob("*.tmp-*"))


def test_cancel_midway_no_temp_files(tmp_path, batch_inputs):
    d, files = batch_inputs
    out = tmp_path / "out"
    job = BatchJob(files, out, tiff_encoding="linear")
    seen = []

    def cb(it):
        seen.append(it.status)
        if it.status in ("ok", "warn", "needs_calibration"):
            job.cancel()   # 第一张完成后取消

    rep = job.run(on_update=cb)
    statuses = [i["status"] for i in rep["items"]]
    assert statuses[0] in ("ok", "warn", "needs_calibration")
    assert "cancelled" in statuses
    assert not list(out.glob("*.tmp-*"))          # 无半成品
    assert not list(out.glob("corrupt*"))          # 失败项无输出


def test_retry_failed_items(tmp_path, batch_inputs, monkeypatch):
    d, files = batch_inputs
    out = tmp_path / "out"
    job = BatchJob(files, out, tiff_encoding="linear")
    job.run()
    failed = job.failed_items()
    assert [f.path.name for f in failed] == ["corrupt.tif"]
    # 模拟"修好文件"后单独重试
    _write_synth(d, "fixed.tif", 73)
    retry = BatchJob([d / "fixed.tif"], out, tiff_encoding="linear")
    rep = retry.run()
    assert rep["items"][0]["status"] in ("ok", "warn", "needs_calibration")


def test_profile_prior_is_initial_value_only(tmp_path, batch_inputs):
    """配置文件先验不得强行覆盖单张分析：Dmin 仍来自该片自己的片基检测。"""
    d, files = batch_inputs
    prof = FilmProfile(name="test", base_prior=[0.9, 0.9, 0.9])  # 故意离谱
    job = BatchJob([files[0]], tmp_path / "out", prior=prof, tiff_encoding="linear")
    rep = job.run()
    dmin = rep["items"][0]["parameters"]["dmin"]
    assert max(dmin) < 0.8, "离谱先验不应压过单张片基检测"


def test_collect_files_skips_downloading(tmp_path):
    d = tmp_path / "x"
    d.mkdir()
    (d / "a.tif").write_bytes(b"x")
    (d / "b.NEF.baiduyun.p.downloading").write_bytes(b"x")
    got = collect_files(d)
    assert [f.name for f in got] == ["a.tif"]
