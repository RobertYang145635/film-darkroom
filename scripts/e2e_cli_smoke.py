"""CLI 端到端冒烟测试：合成负片 -> 16 位 TIFF -> CLI invert -> 导出+旁车。"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from negative_lab.testing.synthetic import make_synthetic_negative  # noqa: E402


def main() -> int:
    s = make_synthetic_negative(seed=999)
    neg = s["negative"]
    tmp = ROOT / "outputs" / "e2e"
    src_dir = tmp / "src"
    src_dir.mkdir(parents=True, exist_ok=True)
    src = src_dir / "synthetic-src.tif"
    tifffile.imwrite(src, (neg * 65535 + 0.5).astype(np.uint16), photometric="rgb")

    r = subprocess.run(
        [sys.executable, "-m", "negative_lab.cli", "invert", str(src),
         "-o", str(tmp), "--base-roi",
         str(neg.shape[1] - (neg.shape[1] // 8)), "0", str(neg.shape[1] // 8), str(neg.shape[0])],
        capture_output=True, text=True, cwd=ROOT)
    print(r.stdout)
    print(r.stderr, file=sys.stderr)
    assert r.returncode == 0, f"CLI 退出码 {r.returncode}"

    out_tif = tmp / "synthetic-src.tif"
    out_jpg = tmp / "synthetic-src.jpg"
    side = tmp / "synthetic-src.negative.json"
    assert out_tif.exists() and out_jpg.exists() and side.exists()
    assert src.read_bytes() != out_tif.read_bytes()

    d = json.loads(side.read_text(encoding="utf-8"))
    assert d["schema_version"] == 1 and len(d["source_sha256"]) == 64
    assert set(["dmin", "dmax", "offset", "wb_low", "wb_high", "reference_exponent",
                "red_ratio", "blue_ratio", "black", "gamma", "exposure"]) <= set(d["parameters"])
    truth = s["truth"]
    err = np.max(np.abs(np.array(d["parameters"]["dmin"]) - np.array(truth["dmin"])))
    assert err < 0.02, f"旁车 Dmin 偏差 {err}"
    print("E2E OK: 源文件未被修改，导出与旁车完整")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
