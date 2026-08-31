"""§8.1.7：输出文件不可覆盖源文件；临时文件写入失败时保持源文件不变。"""

import os

import numpy as np
import pytest

from negative_lab.io.atomic import atomic_write_bytes, ensure_not_source
from negative_lab.io.images import save_tiff16


def test_atomic_write_new_file(tmp_path):
    dst = tmp_path / "out.tif"
    atomic_write_bytes(dst, b"hello")
    assert dst.read_bytes() == b"hello"
    assert not list(tmp_path.glob("*.tmp-*"))


def test_refuses_overwrite_existing(tmp_path):
    dst = tmp_path / "out.tif"
    atomic_write_bytes(dst, b"v1")
    with pytest.raises(FileExistsError):
        atomic_write_bytes(dst, b"v2")
    assert dst.read_bytes() == b"v1"
    atomic_write_bytes(dst, b"v2", overwrite=True)
    assert dst.read_bytes() == b"v2"


def test_never_overwrite_source(tmp_path):
    src = tmp_path / "source.tif"
    src.write_bytes(b"original")
    with pytest.raises(ValueError):
        ensure_not_source(src, [src])
    # Windows 大小写不敏感
    with pytest.raises(ValueError):
        ensure_not_source(tmp_path / "SOURCE.tif", [src])
    ok = tmp_path / "out" / "source.tif"
    ensure_not_source(ok, [src])  # 不同目录同名允许
    assert src.read_bytes() == b"original"


def test_write_failure_keeps_source_and_no_partial(tmp_path, monkeypatch):
    src = tmp_path / "source.tif"
    src.write_bytes(b"original")
    dst = tmp_path / "out.tif"

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(os, "rename", boom)
    with pytest.raises(OSError):
        save_tiff16(dst, np.full((8, 8, 3), 0.5, np.float32), sources=[src])
    assert not dst.exists()
    assert not list(tmp_path.glob("*.tmp-*"))
    assert src.read_bytes() == b"original"


def test_export_same_dir_different_name_ok(tmp_path):
    src = tmp_path / "a.tif"
    src.write_bytes(b"x")
    dst = tmp_path / "a-converted.tif"
    save_tiff16(dst, np.full((8, 8, 3), 0.5, np.float32), sources=[src])
    assert dst.exists() and src.read_bytes() == b"x"
