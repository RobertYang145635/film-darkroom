"""§8.1.7：输出文件不可覆盖源文件；临时文件写入失败时保持源文件不变。"""

import os

import numpy as np
import pytest

from negative_lab.color.management import OutputEncoding
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


@pytest.mark.parametrize("target,preexist",
                         [("rename", False), ("replace", True)],
                         ids=["新建路径", "覆盖路径"])
def test_write_failure_keeps_source_and_no_partial(tmp_path, monkeypatch,
                                                   target, preexist):
    """两条落盘分支（os.rename / os.replace）都必须扛住写盘失败。

    atomic_write_bytes 里的 exists() 分支不是冗余：它是故障注入的接缝。
    只测 rename 一条时，把两分支合并成 replace 会让本测试静默失效
    （2026-09-08 实测：审查改动合并分支后此用例立刻红）。补上 replace
    参数化，两条路径都真的被钉住。
    """
    src = tmp_path / "source.tif"
    src.write_bytes(b"original")
    dst = tmp_path / "out.tif"
    if preexist:
        dst.write_bytes(b"previous-good")     # 只有目标已存在才走 os.replace

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(os, target, boom)
    with pytest.raises(OSError, match="disk full"):
        # FileExistsError 也是 OSError 子类：不显式 overwrite=True，覆盖分支
        # 会在入口就被拒掉，本用例将「假通过」而注入点根本没被执行。
        save_tiff16(dst, np.full((8, 8, 3), 0.5, np.float32), OutputEncoding.SRGB,
                    sources=[src], overwrite=True)
    if preexist:
        assert dst.read_bytes() == b"previous-good", "覆盖失败必须保住旧成品"
    else:
        assert not dst.exists()
    assert src.read_bytes() == b"original"
    assert not list(tmp_path.glob("*.tmp-*")), "失败后不得留下临时文件"
    assert not list(tmp_path.glob("*.tmp-*"))
    assert src.read_bytes() == b"original"


def test_export_same_dir_different_name_ok(tmp_path):
    src = tmp_path / "a.tif"
    src.write_bytes(b"x")
    dst = tmp_path / "a-converted.tif"
    save_tiff16(dst, np.full((8, 8, 3), 0.5, np.float32), OutputEncoding.SRGB,
                sources=[src])
    assert dst.exists() and src.read_bytes() == b"x"


def test_save_requires_encoding_enum(tmp_path):
    """整改 F-2：拒绝无枚举语义的导出请求。"""
    dst = tmp_path / "x.tif"
    with pytest.raises(TypeError):
        save_tiff16(dst, np.zeros((4, 4, 3), np.float32), "srgb")  # 字符串不行
    with pytest.raises(TypeError):
        save_tiff16(dst, np.zeros((4, 4, 3), np.float32))          # 缺省不行
