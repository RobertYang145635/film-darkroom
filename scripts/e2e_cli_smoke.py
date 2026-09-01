"""CLI 端到端冒烟（v3 P0-I：临时目录、幂等、按本次 run 断言、ASCII 控制台安全）。

连续运行两次都必须通过；不读写 outputs/ 固定目录。
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from negative_lab.testing.synthetic import make_synthetic_negative  # noqa: E402


def one_run(run_dir: Path) -> None:
    s = make_synthetic_negative(seed=999)
    neg = s["negative"]
    src = run_dir / "synthetic-src.tif"
    tifffile.imwrite(src, (neg * 65535 + 0.5).astype(np.uint16), photometric="rgb")
    out = run_dir / "out"
    bw = neg.shape[1] // 8
    r = subprocess.run(
        [sys.executable, "-m", "negative_lab.cli", "invert", str(src),
         "-o", str(out), "--tiff-encoding", "linear", "--base-roi",
         str(neg.shape[1] - bw), "0", str(bw), str(neg.shape[0])],
        capture_output=True, text=True, cwd=ROOT, encoding="utf-8")
    print(r.stdout)
    if r.stderr.strip():
        print(r.stderr, file=sys.stderr)
    assert r.returncode == 0, f"CLI exit {r.returncode}"
    # 按本次 run 的三件套断言（不按目录总文件数）
    files = sorted(out.glob("synthetic-src-positive*"))
    assert len(files) == 3, [f.name for f in files]
    side = [f for f in files if f.suffix == ".json"][0]
    d = json.loads(side.read_text(encoding="utf-8"))
    assert d["schema_version"] == 2 and len(d["source_sha256"]) == 64
    assert d["geometry"]["roi_source_space"]["coordinate_space"] == "source"
    assert d["quality"]["status"] in ("ok", "warn", "needs_calibration")
    assert d["output"]["icc_sha256"]
    truth = s["truth"]
    err = np.max(np.abs(np.array(d["parameters"]["dmin"]) - np.array(truth["dmin"])))
    assert err < 0.02, f"sidecar Dmin off {err}"
    assert src.read_bytes() != files[0].read_bytes()


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="nl-e2e-") as t1, \
            tempfile.TemporaryDirectory(prefix="nl-e2e-") as t2:
        one_run(Path(t1))
        one_run(Path(t2))   # 幂等：同一脚本连续两次都必须通过
    print("E2E OK x2 (idempotent, tempdir, sidecar v2)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
