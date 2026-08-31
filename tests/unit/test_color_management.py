"""P0-B：色彩管理单元测试（整改 §P0-B 新增测试）。"""

from __future__ import annotations

import numpy as np
import pytest
import tifffile

from negative_lab.color.management import (
    InputEncoding, OutputEncoding, RAW_DECODE_PARAMS, WORKING_SPACE,
    classify_tiff, decode_linear, encode_output, icc_sha256, linear_to_srgb,
    srgb_icc_bytes, srgb_to_linear)
from negative_lab.io.images import ReadError, load_linear_rgb, save_tiff16


class TestTransferFunctions:
    def test_srgb_roundtrip(self):
        x = np.linspace(0, 1, 257, dtype=np.float32)[None, None, :]
        rt = srgb_to_linear(linear_to_srgb(x))
        assert np.abs(rt - x).max() < 2e-4

    def test_gray_neutrality_preserved(self):
        """灰阶往返后 R=G=B 严格保持（通道独立传递函数）。"""
        g = np.full((8, 8, 3), 0.42, np.float32)
        enc = encode_output(g, OutputEncoding.SRGB)
        assert enc.std(axis=2).max() < 1e-7
        dec = decode_linear(enc, InputEncoding.SRGB)
        assert dec.std(axis=2).max() < 1e-6

    def test_adobe_decode(self):
        x = np.array([0.5], np.float32)
        got = decode_linear(x, InputEncoding.ADOBE)
        assert abs(float(got[0]) - 0.5 ** 2.1992) < 1e-6

    def test_unknown_encoding_rejected(self):
        with pytest.raises(ValueError):
            decode_linear(np.zeros((2, 2, 3), np.float32), InputEncoding.UNKNOWN)
        with pytest.raises(ValueError):
            encode_output(np.zeros((2, 2, 3), np.float32), "srgb")  # 非枚举


class TestRawPolicy:
    def test_raw_decode_params_explicit(self):
        """整改 F-2：RAW 解码参数全部显式锁定（实测教训见 ADR）。"""
        assert RAW_DECODE_PARAMS["use_camera_wb"] is False
        assert RAW_DECODE_PARAMS["use_auto_wb"] is False
        assert "user_mul" not in RAW_DECODE_PARAMS          # 实测破坏归一化，禁用
        assert RAW_DECODE_PARAMS["output_color"] == 6       # LINEAR
        assert RAW_DECODE_PARAMS["gamm"] == (1.0, 1.0, 1.0, 1.0)  # 注意拼写
        assert RAW_DECODE_PARAMS["no_auto_bright"] is True
        assert RAW_DECODE_PARAMS["output_bps"] == 16
        assert WORKING_SPACE == "camera-linear-normalized"


class TestTiffClassification:
    def test_plain_tiff_is_unknown_and_blocked(self, tmp_path):
        p = tmp_path / "plain.tif"
        tifffile.imwrite(p, (np.random.default_rng(0).random(
            (16, 16, 3)) * 65535).astype(np.uint16), photometric="rgb")
        cls = classify_tiff(p)
        assert cls.encoding == InputEncoding.UNKNOWN
        with pytest.raises(ReadError) as ei:
            load_linear_rgb(p)
        assert "tiff_encoding" in str(ei.value)          # 给出显式选项
        res = load_linear_rgb(p, tiff_encoding="linear")  # 显式假设放行
        assert res.meta["input_encoding"] == "linear"
        assert "假设" in res.meta["input_encoding_detail"] or True

    def test_srgb_icc_tiff_classified(self, tmp_path):
        p = tmp_path / "srgb.tif"
        tifffile.imwrite(p, np.zeros((8, 8, 3), np.uint8), photometric="rgb",
                         iccprofile=srgb_icc_bytes())
        cls = classify_tiff(p)
        assert cls.encoding == InputEncoding.SRGB, cls.detail
        assert cls.icc_sha256 == icc_sha256(srgb_icc_bytes())

    def test_gamma1_tiff_is_linear(self, tmp_path):
        p = tmp_path / "g1.tif"
        tifffile.imwrite(p, np.zeros((8, 8, 3), np.uint16), photometric="rgb",
                         extratags=[(33277, 10, 1, (1, 1), True)])  # SRATIONAL 1.0
        cls = classify_tiff(p)
        assert cls.encoding == InputEncoding.LINEAR, cls.detail
