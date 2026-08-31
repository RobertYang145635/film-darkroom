"""生成/更新 fixtures/manifest.json（规格 §8.2、§8.3）。

扫描真实样片目录，记录相对路径、字节数与 SHA-256；
元数据字段无法确认的一律 "unknown"，不猜测（§8.3）。
大图不入 Git（.gitignore 已排除样片目录）。

运行：.venv\\Scripts\\python.exe scripts/build_fixture_manifest.py
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = ROOT / "20260709_SAMPLE-ID(1)"

# 按卷的已知/推断元数据。inferred=true 的字段来自目录命名，需用户确认。
ROLL_META = {
    "1298_G200": {
        "film_stock": "Kodak Gold 200", "film_stock_inferred": True,
        "film_type": "c41_color", "film_type_inferred": True,
        "process": "C-41",
    },
    "1299_G200": {
        "film_stock": "Kodak Gold 200", "film_stock_inferred": True,
        "film_type": "c41_color", "film_type_inferred": True,
        "process": "C-41",
    },
    "1300_CN400": {
        "film_stock": "CN400", "film_stock_inferred": True,
        "film_type": "unknown", "film_type_inferred": False,
        "process": "unknown",
    },
}


def sha256_of(p: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main() -> int:
    entries = []
    for roll_dir in sorted(SAMPLE_DIR.iterdir()):
        if not roll_dir.is_dir():
            continue
        meta = ROLL_META.get(roll_dir.name, {})
        for f in sorted(roll_dir.rglob("*")):
            if not f.is_file():
                continue
            if f.name.lower().endswith(".downloading"):
                continue  # 残缺下载不入库清单
            rel = f.relative_to(ROOT).as_posix()
            print(f"hashing {rel} ...", flush=True)
            entries.append({
                "file": rel,
                "roll": roll_dir.name,
                "size_bytes": f.stat().st_size,
                "sha256": sha256_of(f),
                "film_type": meta.get("film_type", "unknown"),
                "film_stock": meta.get("film_stock", "unknown"),
                "process": meta.get("process", "unknown"),
                "capture_type": "camera",
                "device": "unknown",
                "lens": "unknown",
                "light_source": "unknown",
                "input_profile": "unknown",
                "contains_unexposed_border": "unknown",
                "reference_file": None,
                "reference_notes": None,
                "exposure_notes": None,
            })
    doc = {
        "schema_version": 1,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "notes": [
            "film_stock/film_type 由目录命名推断（*_inferred 标记），需用户确认。",
            "device/lens/light_source/边框情况未采集，保持 unknown（§8.3 不猜测）。",
            "原始图片只读；本清单不含像素数据。",
        ],
        "images": entries,
    }
    out = ROOT / "fixtures" / "manifest.json"
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"共 {len(entries)} 个文件 -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
