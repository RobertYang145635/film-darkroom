"""用户样片接收与验证（整改 P0-A.3，规格 §8.3）。

只读扫描：计算 SHA-256、真实解码探针、格式/位深检查、元数据完整性；
不复制大图、不写样片目录。输出 JSON 报告供人工核对。

运行：
    .venv\\Scripts\\python.exe scripts/sample_intake.py <目录或文件>...
        [--report outputs/intake-<时间>.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from negative_lab.io.probe import probe_file, sha256_of  # noqa: E402

MINIMAL_META_FIELDS = ("film_type", "film_stock", "process", "capture_type",
                       "device", "light_source", "contains_unexposed_border")


def scan(paths: list[Path]) -> list[dict]:
    files: list[Path] = []
    for p in paths:
        if p.is_dir():
            files += sorted(f for f in p.rglob("*")
                            if f.is_file() and not f.name.lower().endswith(".downloading"))
        elif p.is_file():
            files.append(p)
    out = []
    for f in files:
        pr = probe_file(f)
        entry = {
            "file": f.as_posix(),
            "size_bytes": f.stat().st_size if f.exists() else 0,
            "sha256": sha256_of(f) if f.exists() else "",
            **({"decode_status": pr.status, "decode_reason": pr.reason,
                "kind": pr.kind, "width": pr.width, "height": pr.height,
                "bit_depth": pr.bit_depth, "camera": pr.camera,
                "lens": pr.lens, "icc_profile": pr.icc_profile,
                "decode_params": pr.decode_params}),
            "metadata_missing": list(MINIMAL_META_FIELDS),  # 接收时全部待用户确认
        }
        print(f"[{entry['decode_status']:18s}] {f.name}  {entry['decode_reason']}",
              flush=True)
        out.append(entry)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument("--report", type=Path, default=None)
    args = ap.parse_args()
    entries = scan(args.paths)
    report = args.report or (ROOT / "outputs" /
                             f"intake-{time.strftime('%Y%m%d-%H%M%S')}.json")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": "样片接收报告：decode 状态来自真实探针；元数据字段需用户逐项确认，"
                "不能确认的保持 unknown（§8.3）。",
        "files": entries,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    ok = sum(1 for e in entries if e["decode_status"] == "ok")
    print(f"\n共 {len(entries)} 个文件，可解码 {ok}，报告：{report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
