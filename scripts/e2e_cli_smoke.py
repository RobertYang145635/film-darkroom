"""CLI 端到端冒烟（v5 §15：临时目录、幂等、按本次 run 断言、ASCII 安全）。

连续运行两次都必须通过；不读写 outputs/ 固定目录。
sidecar schema v4；三候选与采集指纹必须出现在 sidecar 中。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from negative_lab.testing.spectral import make_spectral_negative  # noqa: E402


def one_run(run_dir: Path) -> None:
    s = make_spectral_negative(seed=999)
    neg = s["negative"]
    src = run_dir / "synthetic-src.tif"
    tifffile.imwrite(src, (neg * 65535 + 0.5).astype(np.uint16), photometric="rgb")
    out = run_dir / "out"
    # 三类独立 ROI（v5 §3.2）：输出保留片边，分析区只含画面，片基在右侧条带
    h, w = neg.shape[:2]
    bc = s["truth"]["base_cols"]
    cmd = [sys.executable, "-m", "negative_lab.cli", "invert", str(src),
           "-o", str(out), "--tiff-encoding", "linear",
           "--output-crop", "0", "0", str(h), str(w),
           "--image-area", "8", "8", str(h - 8), str(bc[0] - 8),
           "--base-sample", "0", str(bc[0] + 4), str(h), str(w - 4),
           "--keep-rebate"]
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    r = subprocess.run(cmd, capture_output=True, cwd=ROOT, env=env)
    stdout = r.stdout.decode("utf-8", errors="strict")
    stderr = r.stderr.decode("utf-8", errors="strict")
    print(stdout)
    if stderr.strip():
        print(stderr, file=sys.stderr)
    assert r.returncode == 0, f"CLI exit {r.returncode}"
    for name, txt in (("stdout", stdout), ("stderr", stderr)):
        assert txt is not None and "Traceback" not in txt, \
            f"子进程 {name} 含 Traceback（假绿防护）"
    # 按本次 run 的三件套断言（不按目录总文件数）
    files = sorted(out.glob("synthetic-src-positive*"))
    assert len(files) == 3, [f.name for f in files]
    side = [f for f in files if f.suffix == ".json"][0]
    d = json.loads(side.read_text(encoding="utf-8"))
    assert d["schema_version"] == 4 and len(d["source_sha256"]) == 64
    rss = d["geometry"]["roi_source_space"]
    assert rss["coordinate_space"] == "source"
    assert rss["output_crop_roi"] and rss["image_area_roi"] and \
        rss["base_sample_rois"], "v5 §3.2 三类 ROI 必须全部记录"
    assert d["candidates"] is not None
    assert {c["name"] for c in (d["candidates"]["preserve"],
                                d["candidates"]["neutral"],
                                d["candidates"]["warm"])} == \
        {"preserve", "neutral", "warm"}
    assert d["capture_fingerprint"]["value"]
    assert d["base"]["base_source"] == "frame"
    assert d["color"]["output_icc_sha256"], "sidecar 必须记录输出 ICC"
    gate = d["diagnostics"]["technical"]["gate"]
    assert gate["status"] in ("ok", "warn", "needs_review"), gate["status"]
    assert gate["blocks_export"] is False, "needs_review 必须允许导出（v5 §8.3）"
    truth = s["truth"]["base"]
    err = np.max(np.abs(np.array(d["base"]["dmin"]) - np.array(truth)))
    assert err < 0.02, f"sidecar Dmin off {err}"
    # 输出保留片边 -> 输出尺寸 = 整帧（keep_rebate）
    with tifffile.TiffFile(str(out / "synthetic-src-positive.tif")) as tf:
        assert tf.pages[0].asarray().shape[:2] == (h, w), \
            "keep_rebate 时输出应保留整帧（分析区独立于输出裁切）"
    assert src.read_bytes() != files[0].read_bytes()


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="nl-e2e-") as t1, \
            tempfile.TemporaryDirectory(prefix="nl-e2e-") as t2:
        one_run(Path(t1))
        one_run(Path(t2))   # 幂等：同一脚本连续两次都必须通过
    print("E2E OK x2 (idempotent, tempdir, sidecar v4, strict UTF-8)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
