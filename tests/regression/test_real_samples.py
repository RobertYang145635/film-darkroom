"""真实样片回归（P0 §2.1.8 / §8.2）。

样片不在仓库内时自动跳过（fixtures/manifest.json 记录路径与哈希）。
断言的是"不劣化"边界：可解码、参数合理、无 NaN/Inf、源文件不被改动、
自动 Dmin 与已标定基线一致。不承诺与 NLP 完全同色。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from negative_lab.io.images import load_linear_rgb, save_tiff16
from negative_lab.pipeline.auto import auto_analyze
from negative_lab.pipeline.density import invert

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "fixtures" / "manifest.json"


def _decodable_entries():
    if not MANIFEST.exists():
        return []
    doc = json.loads(MANIFEST.read_text(encoding="utf-8"))
    out = []
    for e in doc["images"]:
        p = ROOT / e["file"]
        if p.exists() and e.get("decode_status", "ok") == "ok" \
                and not e["file"].lower().endswith(".downloading"):
            out.append((e, p))
    return out


ENTRIES = _decodable_entries()
pytestmark = pytest.mark.skipif(
    not ENTRIES, reason="真实样片不在本机（fixtures/manifest.json 有记录）")


@pytest.fixture(scope="module")
def sample():
    """取 1298_G200 卷第一张可解码样片。"""
    for e, p in ENTRIES:
        if "1298_G200" in e["file"]:
            return e, p
    return ENTRIES[0]


def test_source_untouched_and_pipeline_sane(sample, tmp_path):
    entry, path = sample
    h0 = hashlib.sha256(path.read_bytes()).hexdigest()
    assert h0 == entry["sha256"], "样片哈希与清单不一致（源文件被改动？）"

    loaded = load_linear_rgb(path)
    t = loaded.rgb
    assert t.dtype == np.float32 and t.ndim == 3 and t.shape[2] == 3
    assert np.isfinite(t).all()

    small = t[::4, ::4, :].copy()          # 回归用 1/4 分辨率，控制时长
    # 色彩语义回归使用 verified 标注的 base_roi（不依赖自动检测质量——
    # 自动检测的 IoU 门禁在 P0-C test_reference_quality 单独把关）。
    from negative_lab.calibration.annotations import load_annotation
    from negative_lab.calibration.base import robust_dmin

    ann = load_annotation(entry["file"])
    assert ann is not None and ann.verified, f"缺 verified 标注: {entry['file']}"
    y0, x0, y1, x1 = ann.base_roi
    s = 4
    base = robust_dmin(small[y0 // s:y1 // s, x0 // s:x1 // s])
    dmin = np.array(base["dmin"])
    # 真实橙罩片基 B 通道透射率可低至 ~0.03（强吸收蓝），边界按物理实际
    assert (dmin > 0.01).all() and (dmin < 0.95).all(), dmin
    assert base["confidence"] > 0.5, base
    # 回归锁：与 2026-08-31 标定基线（output_color=6 LINEAR 语义）偏差 <5%
    BASELINE = np.array([0.1722, 0.1174, 0.0444])
    rel = np.abs(dmin - BASELINE) / BASELINE
    assert rel.max() < 0.05, f"G200-1 片基 Dmin 偏离基线 {rel.max():.1%}: {dmin}"
    rep = auto_analyze(small)
    assert 0.3 <= rep.params.dmax <= 4.0
    assert 0.3 <= rep.params.dmax <= 4.0
    out = invert(small, rep.params).positive_linear
    assert np.isfinite(out).all()
    lum = out.mean()
    assert 0.05 < lum < 0.95, f"输出均值异常: {lum}"

    dst = tmp_path / (path.stem + "-positive.tif")
    from negative_lab.color.management import OutputEncoding
    save_tiff16(dst, invert(small, rep.params).positive_linear,
                OutputEncoding.SRGB, sources=[path])
    assert dst.exists()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == h0, "源文件被改动"


def test_roll_consistency(sample):
    """同卷相邻两张：Dmin 应接近（同批胶片/同设备），但不强制覆盖单张。"""
    entry, path = sample
    roll = entry["roll"]
    doc = json.loads(MANIFEST.read_text(encoding="utf-8"))
    picks = [(e, ROOT / e["file"]) for e in doc["images"]
             if e["roll"] == roll and e.get("decode_status") == "ok"
             and (ROOT / e["file"]).exists()][:2]
    if len(picks) < 2:
        pytest.skip("该卷不足两张")
    dmins = []
    for e, p in picks:
        t = load_linear_rgb(p).rgb[::6, ::6, :].copy()
        rep = auto_analyze(t)
        dmins.append(np.array(rep.params.dmin))
    rel = np.abs(dmins[0] - dmins[1]) / np.maximum(dmins[0], 1e-6)
    assert rel.max() < 0.25, f"同卷 Dmin 漂移过大: {dmins}"
