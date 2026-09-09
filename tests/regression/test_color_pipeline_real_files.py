"""P0-B：真实文件色彩管线回归（整改 §P0-B 新增测试）。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import tifffile
from PIL import Image

from negative_lab.color.management import (
    OutputEncoding, icc_sha256, srgb_icc_bytes)
from negative_lab.io.images import load_linear_rgb, save_jpeg8, save_tiff16

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "fixtures" / "manifest.json"


def _entry(pred):
    if not MANIFEST.exists():
        return None
    for e in json.loads(MANIFEST.read_text(encoding="utf-8"))["images"]:
        if pred(e) and (ROOT / e["file"]).exists():
            return e
    return None


def test_raw_meta_records_explicit_policy():
    e = _entry(lambda x: x["decode_status"] == "ok" and x["file"].endswith(".NEF"))
    if e is None:
        pytest.skip("无可解码 RAW 样片（missing-fixture，见 docs/support-matrix.md）")
    from negative_lab.color.management import raw_decode_params
    res = load_linear_rgb(ROOT / e["file"])
    m = res.meta
    assert m["raw_decode_params"] == raw_decode_params()
    assert m["white_balance_policy"].startswith("libraw-default daylight")
    assert m["applied_wb_daylight"] and len(m["applied_wb_daylight"]) == 4
    assert m["orientation_policy"] == "metadata-oriented (user_flip=-1)"
    assert m["working_space"] == "srgb-linear-d65-normalized"
    assert m["input_icc_sha256"].startswith("none (RAW")
    assert m["sha256"] == e["sha256"], "样片哈希与清单不一致"


def test_nlp_reference_jpeg_is_display_encoded():
    """NLP 对照图：显示编码 + 内嵌 ICC（F-2 证据：对照图有 ICC 而旧导出没有）。"""
    e = _entry(lambda x: x.get("reference_file"))
    if e is None:
        pytest.skip("无 NLP 对照图")
    ref = ROOT / e["reference_file"]
    with Image.open(str(ref)) as im:
        assert im.info.get("icc_profile"), "NLP 对照图应含 ICC（审查证据）"
    res = load_linear_rgb(ref)
    assert res.meta["input_encoding"] == "srgb(display)"
    assert any("显示编码" in w for w in res.warnings)


def test_real_export_pipeline_has_consistent_icc(tmp_path):
    e = _entry(lambda x: x["decode_status"] == "ok" and x["file"].endswith(".NEF"))
    if e is None:
        pytest.skip("无可解码 RAW 样片")
    from negative_lab.pipeline.auto import auto_analyze
    from negative_lab.pipeline.density import invert_chunked

    t = load_linear_rgb(ROOT / e["file"]).rgb
    small = t[::8, ::8, :].copy()
    rep = auto_analyze(small)
    res = invert_chunked(small, rep.params)
    h = icc_sha256(srgb_icc_bytes())
    tif = save_tiff16(tmp_path / "o.tif", res.positive_linear, OutputEncoding.SRGB)
    jpg = save_jpeg8(tmp_path / "o.jpg", res.positive_linear)
    with tifffile.TiffFile(str(tif)) as tf:
        assert icc_sha256(bytes(tf.pages[0].tags[34675].value)) == h
    with Image.open(str(jpg)) as im:
        assert icc_sha256(bytes(im.info["icc_profile"])) == h
