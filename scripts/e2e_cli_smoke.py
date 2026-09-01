"""CLI 端到端冒烟（v3 P0-I：临时目录、幂等、按本次 run 断言、ASCII 控制台安全）。

连续运行两次都必须通过；不读写 outputs/ 固定目录。
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

from negative_lab.testing.synthetic import make_synthetic_negative  # noqa: E402


def one_run(run_dir: Path) -> None:
    s = make_synthetic_negative(seed=999)
    neg = s["negative"]
    src = run_dir / "synthetic-src.tif"
    tifffile.imwrite(src, (neg * 65535 + 0.5).astype(np.uint16), photometric="rgb")
    out = run_dir / "out"
    bw = neg.shape[1] // 8
    # v4 §10.1：bytes 捕获 + 子进程强制 UTF-8，严格解码——任何解码异常/乱码
    # 都必须失败（v4 §1.3.7 实锤：text+utf8 曾吞掉 GBK 解码错误假绿）
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    r = subprocess.run(
        [sys.executable, "-m", "negative_lab.cli", "invert", str(src),
         "-o", str(out), "--tiff-encoding", "linear", "--base-roi",
         str(neg.shape[1] - bw), "0", str(bw), str(neg.shape[0])],
        capture_output=True, cwd=ROOT, env=env)
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
    assert d["schema_version"] == 3 and len(d["source_sha256"]) == 64
    assert d["geometry"]["roi_source_space"]["coordinate_space"] == "source"
    assert d["quality"]["status"] in ("ok", "warn", "needs_calibration")
    assert d["output"]["icc_sha256"]
    truth = s["truth"]
    err = np.max(np.abs(np.array(d["technical_parameters"]["dmin"])
                        - np.array(truth["dmin"])))
    assert err < 0.02, f"sidecar Dmin off {err}"
    assert src.read_bytes() != files[0].read_bytes()


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="nl-e2e-") as t1, \
            tempfile.TemporaryDirectory(prefix="nl-e2e-") as t2:
        one_run(Path(t1))
        one_run(Path(t2))   # 幂等：同一脚本连续两次都必须通过
    print("E2E OK x2 (idempotent, tempdir, sidecar v3, strict UTF-8)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
