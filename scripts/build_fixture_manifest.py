"""生成/更新 fixtures/manifest.json（规格 §8.2、§8.3；整改 P0-A）。

decode_status 来自 negative_lab.io.probe 的真实最小解码探针
（不再使用 24 字节启发式——它把 rawpy 实际失败的 Z8 AV1 NEF 标成了 ok）。
元数据无法确认的一律 "unknown"，不猜测。大图不入 Git。

运行：.venv\\Scripts\\python.exe scripts/build_fixture_manifest.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from negative_lab.io.probe import probe_file, sha256_of  # noqa: E402

# 所有样片根目录（百度网盘下载，只读、不入库）
SAMPLE_DIRS = [p for p in (ROOT / "20260709_SAMPLE-ID(1)", ROOT / "20260506_SAMPLE-ID")
               if p.is_dir()]

# 按卷的已知/推断元数据。inferred=true 的字段来自目录命名，需用户确认。
ROLL_META = {
    "1298_G200": {"film_stock": "Kodak Gold 200", "film_stock_inferred": True,
                  "film_type": "c41_color", "film_type_inferred": True,
                  "process": "C-41"},
    "1299_G200": {"film_stock": "Kodak Gold 200", "film_stock_inferred": True,
                  "film_type": "c41_color", "film_type_inferred": True,
                  "process": "C-41"},
    "1300_CN400": {"film_stock": "CN400", "film_stock_inferred": True,
                   "film_type": "unknown", "film_type_inferred": False,
                   "process": "unknown"},
    "1171_5219": {"film_stock": "Kodak Vision3 500T (5219)", "film_stock_inferred": True,
                  "film_type": "ecn2_color", "film_type_inferred": True,
                  "process": "ECN-2"},
    "1172_5207": {"film_stock": "Kodak Vision3 250D (5207)", "film_stock_inferred": True,
                  "film_type": "ecn2_color", "film_type_inferred": True,
                  "process": "ECN-2"},
    # ECN2/ 目录是上面两卷 Z8「高效 NEF」经 Lightroom 导出的 16 位 TIFF
    # （LibRaw 0.22 无法解码 AV1 压缩 RAW）。按文件名前缀拆成两个逻辑卷，
    # 保证按卷隔离；capture_type 如实记为 camera_raw_export_tiff，
    # 不冒充"扫描仪 TIFF"（v5 §14.3 缺真实扫描仪样片 -> missing-fixture）。
    "ECN2_5219": {"film_stock": "Kodak Vision3 500T (5219)",
                  "film_stock_inferred": True,
                  "film_type": "ecn2_color", "film_type_inferred": True,
                  "process": "ECN-2", "capture_type": "camera_raw_export_tiff"},
    "ECN2_5207": {"film_stock": "Kodak Vision3 250D (5207)",
                  "film_stock_inferred": True,
                  "film_type": "ecn2_color", "film_type_inferred": True,
                  "process": "ECN-2", "capture_type": "camera_raw_export_tiff"},
}

# 按卷分组的数据集划分（§9.3：按卷隔离，训练/验证/最终留出不得串卷）。
SPLIT = {
    "1298_G200": "calibration",
    "1300_CN400": "calibration",
    "1299_G200": "holdout",
    "1171_5219": "holdout",   # Z8 AV1 NEF：LibRaw 不支持，仅作不可解码记录
    "1172_5207": "holdout",   # 同上
    "ECN2_5207": "calibration",   # 可解码的 ECN-2 证据卷
    "ECN2_5219": "holdout",       # ECN-2 留出卷（未参与任何调参）
}

# ECN2/*.tif 的 NLP 对照在原始 NEF 卷目录下
NLP_FALLBACK = {
    "ECN2_5219": ROOT / "20260506_SAMPLE-ID/1171_5219/翻拍/NLP",
    "ECN2_5207": ROOT / "20260506_SAMPLE-ID/1172_5207/翻拍/NLP",
}


def _logical_roll(roll_dir_name: str, path: Path) -> str:
    """ECN2 目录按文件名前缀拆成两个逻辑卷；其余沿用顶层卷目录名。"""
    if roll_dir_name == "ECN2":
        return f"ECN2_{path.stem.split('-')[0]}"
    return roll_dir_name


def main() -> int:
    entries = []
    for sample_dir in SAMPLE_DIRS:
        for roll_dir in sorted(p for p in sample_dir.iterdir() if p.is_dir()):
            for f in sorted(roll_dir.rglob("*")):
                if not f.is_file():
                    continue
                if f.name.lower().endswith(".downloading"):
                    continue
                if "NLP" in f.parts:
                    continue  # NLP 输出登记为 reference_file，不作为输入样片
                logical = _logical_roll(roll_dir.name, f)
                meta = ROLL_META.get(logical, ROLL_META.get(roll_dir.name, {}))
                rel = f.relative_to(ROOT).as_posix()
                ref = f.parent / "NLP" / (f.stem + ".jpg")
                if not ref.is_file() and logical in NLP_FALLBACK:
                    ref = NLP_FALLBACK[logical] / (f.stem + ".jpg")
                print(f"probing {rel} ...", flush=True)
                pr = probe_file(f)
                entries.append({
                    "file": rel,
                    "roll": logical,
                    "roll_dir": roll_dir.name,
                    "split": SPLIT.get(logical, SPLIT.get(roll_dir.name, "unassigned")),
                    "size_bytes": f.stat().st_size,
                    "sha256": sha256_of(f),
                    "decode_status": pr.status,
                    "decode_reason": pr.reason,
                    "decode_params": pr.decode_params,
                    "camera": pr.camera or "unknown",
                    "lens": pr.lens or "unknown",
                    "width": pr.width, "height": pr.height,
                    "bit_depth": pr.bit_depth,
                    "icc_profile": pr.icc_profile,
                    "input_kind": pr.kind,
                    "film_type": meta.get("film_type", "unknown"),
                    "film_stock": meta.get("film_stock", "unknown"),
                    "process": meta.get("process", "unknown"),
                    "capture_type": meta.get("capture_type", "camera"),
                    "light_source": "unknown",
                    "input_profile": "unknown",
                    "contains_unexposed_border": "unknown",
                    "reference_file": (ref.relative_to(ROOT).as_posix()
                                       if ref.is_file() else None),
                    "reference_notes": ("Negative Lab Pro 导出默认/零编辑结果，"
                                        "仅作竞品对照基线，不视为绝对真值（v5 §9.2）"
                                        if ref.is_file() else None),
                    "exposure_notes": None,
                })
    n_ok = sum(1 for e in entries if e["decode_status"] == "ok")
    doc = {
        "schema_version": 3,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "notes": [
            "decode_status 来自真实最小解码探针（negative_lab.io.probe），非字节启发式。",
            "film_stock/film_type 由目录命名推断（*_inferred 标记），需用户确认。",
            "split 按卷隔离 calibration/holdout；同一卷的不同导出不得跨集。",
            "ECN2/ 是 Z8 高效 NEF 经 Lightroom 导出的 16 位 TIFF（sRGB ICC），"
            "按文件名前缀拆成 ECN2_5207 / ECN2_5219 两个逻辑卷；"
            "capture_type=camera_raw_export_tiff，不冒充扫描仪 TIFF。",
            "device/lens 来自 EXIF/TIFF 标签；light_source/边框情况未采集，保持 unknown。",
            "原始图片只读；本清单不含像素数据。",
        ],
        "summary": {
            "total": len(entries),
            "decode_ok": n_ok,
            "by_status": _count(entries, "decode_status"),
            "by_film_type_ok": _count(entries, "film_type", only_ok=True),
            "by_roll": _count(entries, "roll"),
        },
        "images": entries,
    }
    out = ROOT / "fixtures" / "manifest.json"
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"共 {len(entries)} 个文件（ok={n_ok}）-> {out}")
    print(json.dumps(doc["summary"], ensure_ascii=False, indent=2))
    return 0


def _count(entries, key, only_ok=False):
    c: dict[str, int] = {}
    for e in entries:
        if only_ok and e["decode_status"] != "ok":
            continue
        c[e[key]] = c.get(e[key], 0) + 1
    return c


if __name__ == "__main__":
    raise SystemExit(main())
