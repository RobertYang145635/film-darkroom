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
# 所有样片根目录（百度网盘下载，只读、不入库）
SAMPLE_DIRS = [p for p in (ROOT / "20260709_SAMPLE-ID(1)", ROOT / "20260506_SAMPLE-ID")
               if p.is_dir()]

# 按卷的已知/推断元数据。inferred=true 的字段来自目录命名，需用户确认。
# 5219/5207 是柯达电影卷号（Vision3 500T / 250D），常按 ECN-2 冲洗；仍需确认。
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
    "1171_5219": {
        "film_stock": "Kodak Vision3 500T (5219)", "film_stock_inferred": True,
        "film_type": "ecn2_color", "film_type_inferred": True,
        "process": "ECN-2",
    },
    "1172_5207": {
        "film_stock": "Kodak Vision3 250D (5207)", "film_stock_inferred": True,
        "film_type": "ecn2_color", "film_type_inferred": True,
        "process": "ECN-2",
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


def decode_status(path: Path) -> str:
    """按 RAW 条带首字节判定解码状态（不触发完整解码）。

    'CONTRACT' 标记 = Nikon Z8 高效 NEF（AV1 压缩），LibRaw 0.22.1 不支持。
    """
    try:
        import tifffile

        t = tifffile.TiffFile(path)
        sp = t.pages[0].pages[1]
        off = sp.tags[273].value[0]
        with open(path, "rb") as f:
            f.seek(off)
            head = f.read(24)
        return "av1_unsupported" if b"CONTRACT" in head else "ok"
    except Exception:  # noqa: BLE001
        return "unknown"


def camera_of(path: Path) -> str:
    try:
        import tifffile

        pg = tifffile.TiffFile(path).pages[0]
        make = pg.tags.get(271)
        model = pg.tags.get(272)
        f = lambda e: str(e.value).strip() if e else ""  # noqa: E731
        return (f(make) + " " + f(model)).strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def main() -> int:
    entries = []
    for sample_dir in SAMPLE_DIRS:
        for roll_dir in sorted(p for p in sample_dir.iterdir() if p.is_dir()):
            meta = ROLL_META.get(roll_dir.name, {})
            for f in sorted(roll_dir.rglob("*")):
                if not f.is_file():
                    continue
                if f.name.lower().endswith(".downloading"):
                    continue  # 残缺下载不入库清单
                if "NLP" in f.parts:
                    continue  # NLP 对照输出登记为 reference_file，不作为输入样片
                rel = f.relative_to(ROOT).as_posix()
                # 同目录 NLP/<stem>.jpg 作为对照参考（仅对照，非真值，§8.3）
                ref = f.parent / "NLP" / (f.stem + ".jpg")
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
                    "device": camera_of(f),
                    "lens": "unknown",
                    "light_source": "unknown",
                    "input_profile": "unknown",
                    "decode_status": decode_status(f),
                    "contains_unexposed_border": "unknown",
                    "reference_file": (ref.relative_to(ROOT).as_posix()
                                       if ref.is_file() else None),
                    "reference_notes": ("Negative Lab Pro 导出，仅作对照基线，"
                                        "不视为绝对真值（§8.4）" if ref.is_file() else None),
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
