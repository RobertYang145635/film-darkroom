"""v4 §9 P0-G：事务导出——共享 basename、格式开关、真部分提交回滚。"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from negative_lab.io.export import ExportError, export_triplet


@pytest.fixture()
def data():
    rng = np.random.default_rng(9)
    grad = np.linspace(0.1, 0.9, 64, dtype=np.float32)
    x = np.clip(grad[None, :, None] + rng.normal(0, 0.02, (48, 64, 1)),
                0, 1).astype(np.float32)
    return np.repeat(x, 3, axis=2)


def payload(sha="a" * 64):
    return {"schema_version": 3, "source_sha256": sha,
            "quality": {"status": "ok"}}


def test_success_writes_three(tmp_path, data):
    out = tmp_path / "o"
    files = export_triplet(out, "s", data, payload(), sources=[tmp_path / "s.tif"])
    assert len(files) == 3 and all(f.exists() for f in files)
    assert not list(out.glob(".staging-*"))


@pytest.mark.parametrize("point", ["tiff", "jpeg", "sidecar",
                                   "commit_after_0", "commit_after_1"])
def test_failure_injection_no_partial(tmp_path, data, point):
    out = tmp_path / "o"
    with pytest.raises(ExportError):
        export_triplet(out, "s", data, payload(),
                       sources=[tmp_path / "s.tif"], _inject_fail=point)
    assert not list(out.glob("s-positive*")), "失败后留下半套工件"
    assert not list(out.glob(".staging-*")), "staging 未清理"


def test_partial_commit_rollback_preserves_prior_files(tmp_path, data):
    """§9.3：事务前已有文件哈希不变；本次工件全部回滚。"""
    out = tmp_path / "o"
    out.mkdir()
    prior = out / "keep.txt"
    prior.write_text("keep", encoding="utf-8")
    prior_hash = hashlib.sha256(prior.read_bytes()).hexdigest()
    with pytest.raises(ExportError):
        export_triplet(out, "s", data, payload(),
                       sources=[tmp_path / "s.tif"], _inject_fail="commit_after_1")
    assert hashlib.sha256(prior.read_bytes()).hexdigest() == prior_hash
    assert not list(out.glob("s-positive*"))
    assert not list(out.glob(".staging-*"))


def test_rollback_delete_failure_reports_leftover(tmp_path, data, monkeypatch):
    """§9.3：回滚删除失败 -> 异常必须含残留路径，不得静默吞掉。"""
    out = tmp_path / "o"

    orig_unlink = Path.unlink

    def failing_unlink(self, *a, **k):
        if self.name.endswith(".tif"):
            raise OSError("disk full")
        return orig_unlink(self, *a, **k)

    monkeypatch.setattr(Path, "unlink", failing_unlink)
    with pytest.raises(ExportError, match="残留"):
        export_triplet(out, "s", data, payload(),
                       sources=[tmp_path / "s.tif"], _inject_fail="commit_after_1")
    monkeypatch.undo()
    for f in out.glob("s-positive*"):
        f.unlink()


def test_shared_basename_when_target_exists(tmp_path, data):
    """§9.1：预存在同名 jpg -> 整组共享同一新 suffix（不再各自 _unique）。"""
    out = tmp_path / "o"
    out.mkdir()
    (out / "s-positive.jpg").write_bytes(b"old")
    files = export_triplet(out, "s", data, payload(), sources=[tmp_path / "s.tif"])
    names = {f.name for f in files}
    assert names == {"s-positive-1.tif", "s-positive-1.jpg",
                     "s-positive-1.negative.json"}
    assert (out / "s-positive.jpg").read_bytes() == b"old"   # 原文件未动


@pytest.mark.parametrize("tiff,jpeg,expect", [
    (True, True, {"s-positive.tif", "s-positive.jpg", "s-positive.negative.json"}),
    (True, False, {"s-positive.tif", "s-positive.negative.json"}),
    (False, True, {"s-positive.jpg", "s-positive.negative.json"}),
])
def test_format_switches(tmp_path, data, tiff, jpeg, expect):
    out = tmp_path / "o"
    files = export_triplet(out, "s", data, payload(),
                           sources=[tmp_path / "s.tif"],
                           export_tiff=tiff, export_jpeg=jpeg)
    assert {f.name for f in files} == expect
    assert not list(out.glob(".staging-*"))


def test_both_false_rejected(tmp_path, data):
    out = tmp_path / "o"
    with pytest.raises(ExportError, match="不能同时为 False"):
        export_triplet(out, "s", data, payload(),
                       sources=[tmp_path / "s.tif"],
                       export_tiff=False, export_jpeg=False)


def test_unique_naming_no_overwrite(tmp_path, data):
    out = tmp_path / "o"
    f1 = export_triplet(out, "s", data, payload(), sources=[tmp_path / "s.tif"])
    f2 = export_triplet(out, "s", data, payload("b" * 64),
                        sources=[tmp_path / "s.tif"])
    assert f1[0] != f2[0] and f1[0].exists() and f2[0].exists()


def test_bad_tiff_caught_by_verification(tmp_path, data, monkeypatch):
    import negative_lab.io.export as exp

    def bad(path, arr, enc, **kw):
        path.write_bytes(b"not a tiff")
    monkeypatch.setattr(exp, "save_tiff16", bad)
    out = tmp_path / "o"
    with pytest.raises((ExportError, Exception)):
        export_triplet(out, "s", data, payload(), sources=[tmp_path / "s.tif"])
    assert not list(out.glob("s-positive*"))
