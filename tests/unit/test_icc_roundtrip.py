"""P0-B：ICC 写入-重读往返测试（整改 §P0-B：写后验证）。"""

from __future__ import annotations

import numpy as np
import pytest
import tifffile
from PIL import Image

from negative_lab.color.management import (
    OutputEncoding, icc_sha256, linear_to_srgb, srgb_icc_bytes)
from negative_lab.io.images import save_jpeg8, save_tiff16

SRGB_HASH = icc_sha256(srgb_icc_bytes())


@pytest.fixture
def linear_img():
    rng = np.random.default_rng(3)
    return rng.random((32, 24, 3)).astype(np.float32)


def test_tiff16_srgb_embeds_icc_and_matches_pixels(tmp_path, linear_img):
    p = save_tiff16(tmp_path / "o.tif", linear_img, OutputEncoding.SRGB)
    with tifffile.TiffFile(str(p)) as t:
        tag = t.pages[0].tags.get(34675)
        assert tag is not None, "TIFF 缺少 ICC"
        assert icc_sha256(bytes(tag.value)) == SRGB_HASH
        arr = t.pages[0].asarray()
    assert arr.dtype == np.uint16
    expect = (linear_to_srgb(linear_img) * 65535 + 0.5).astype(np.uint16)
    assert np.abs(arr.astype(int) - expect.astype(int)).max() <= 1


def test_jpeg8_embeds_icc(tmp_path, linear_img):
    p = save_jpeg8(tmp_path / "o.jpg", linear_img)
    with Image.open(str(p)) as im:
        icc = im.info.get("icc_profile")
        assert icc and icc_sha256(bytes(icc)) == SRGB_HASH
        assert im.mode == "RGB"


def test_tiff16_linear_has_no_icc(tmp_path, linear_img):
    p = save_tiff16(tmp_path / "lin.tif", linear_img, OutputEncoding.LINEAR)
    with tifffile.TiffFile(str(p)) as t:
        assert t.pages[0].tags.get(34675) is None
        arr = t.pages[0].asarray()
    expect = (np.clip(linear_img, 0, 1) * 65535 + 0.5).astype(np.uint16)
    assert np.abs(arr.astype(int) - expect.astype(int)).max() <= 1


def test_jpeg_rejects_linear_encoding(tmp_path, linear_img):
    with pytest.raises(ValueError):
        save_jpeg8(tmp_path / "x.jpg", linear_img, encoding=OutputEncoding.LINEAR)


def test_tiff_and_jpeg_appearance_consistent(tmp_path):
    """整改 §2：TIFF16 与 JPEG 编码/ICC 一致 -> 解码后外观一致（量化误差内）。

    用平滑渐变（真实照片主体频率）；随机噪声是 JPEG 最坏情况，不具代表性。
    """
    g, b = np.meshgrid(np.linspace(0.1, 0.9, 24), np.linspace(0.1, 0.9, 32))
    lin = np.stack([g, (g + b) / 2, b], axis=2).astype(np.float32)
    tif = save_tiff16(tmp_path / "c.tif", lin, OutputEncoding.SRGB)
    jpg = save_jpeg8(tmp_path / "c.jpg", lin)
    a = tifffile.imread(str(tif)).astype(np.float32) / 65535
    b8 = np.asarray(Image.open(str(jpg)).convert("RGB"), np.float32) / 255
    assert np.abs(a - b8).max() < 0.02   # JPEG 量化内一致
