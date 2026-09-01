"""v3 P0-H：事务导出与失败注入测试（I-4 的机器证明）。"""

from __future__ import annotations

import numpy as np
import pytest

from negative_lab.io.export import ExportError, export_triplet


@pytest.fixture()
def data():
    rng = np.random.default_rng(9)
    grad = np.linspace(0.1, 0.9, 64, dtype=np.float32)
    x = np.clip(grad[None, :, None] + rng.normal(0, 0.02, (48, 64, 1)),
                0, 1).astype(np.float32)
    x = np.repeat(x, 3, axis=2)
    return x


def payload(sha="a" * 64):
    return {"schema_version": 2, "source_sha256": sha, "quality": {"status": "ok"}}


def test_success_writes_three(tmp_path, data):
    out = tmp_path / "o"
    files = export_triplet(out, "s", data, payload(), sources=[tmp_path / "s.tif"])
    assert len(files) == 3 and all(f.exists() for f in files)
    assert not list(out.glob(".staging-*"))


@pytest.mark.parametrize("point", ["tiff", "jpeg", "sidecar", "commit"])
def test_failure_injection_no_partial(tmp_path, data, point):
    out = tmp_path / "o"
    with pytest.raises(ExportError):
        export_triplet(out, "s", data, payload(),
                       sources=[tmp_path / "s.tif"], _inject_fail=point)
    assert not list(out.glob("s-positive*")), "失败后留下半套工件"
    assert not list(out.glob(".staging-*")), "staging 未清理"


def test_rollback_removes_committed(tmp_path, data):
    """commit 阶段故障：已移动的文件也必须被回滚。"""
    out = tmp_path / "o"
    with pytest.raises(ExportError):
        export_triplet(out, "s", data, payload(),
                       sources=[tmp_path / "s.tif"], _inject_fail="commit")
    assert list(out.iterdir()) == []


def test_unique_naming_no_overwrite(tmp_path, data):
    out = tmp_path / "o"
    f1 = export_triplet(out, "s", data, payload(), sources=[tmp_path / "s.tif"])
    f2 = export_triplet(out, "s", data, payload("b" * 64),
                        sources=[tmp_path / "s.tif"])
    assert f1[0] != f2[0] and f1[0].exists() and f2[0].exists()


def test_corrupt_icc_write_fails_verification(tmp_path, data, monkeypatch):
    """写坏 ICC 必须被重读校验抓住（不是静默通过）。"""
    import negative_lab.io.export as exp
    orig = exp.save_tiff16

    def bad(path, arr, enc, **kw):
        path.write_bytes(b"not a tiff")
    monkeypatch.setattr(exp, "save_tiff16", bad)
    out = tmp_path / "o"
    with pytest.raises((ExportError, Exception)):
        export_triplet(out, "s", data, payload(),
                       sources=[tmp_path / "s.tif"])
    assert not list(out.glob("s-positive*"))
