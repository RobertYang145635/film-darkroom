"""从 fixtures/manifest.json 生成 docs/support-matrix.md（v5 §4.2/§12 P0-2.4）。

矩阵格状态（稳定枚举）：
  verified          有可解码样片且 calibration+holdout 均有覆盖
  missing-fixture   宣称支持路径存在，但无任何可解码样片 -> 禁止该支持声明
  unsupported       样片存在但真实解码探针失败（格式/解码器缺口）
  out-of-scope      v5 P0 明确不覆盖（黑白/反转片），不再是发布门禁

只有真实 fixture 才标 verified；没有覆盖的输入/胶片/设备一律如实标注。

运行：.venv\\Scripts\\python.exe scripts/build_support_matrix.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

#: v5 §0.2.5：黑白与反转片不在 P0，不参与发布门禁
OUT_OF_SCOPE = ("bw",)


def compute_matrix(manifest: dict) -> dict:
    imgs = manifest["images"]
    ok = [e for e in imgs if e["decode_status"] == "ok"]

    def rolls(pool):
        return sorted({e["roll"] for e in pool})

    def cell(film_type):
        pool_ok = [e for e in ok if e["film_type"] == film_type]
        pool_all = [e for e in imgs if e["film_type"] == film_type]
        if not pool_ok:
            return ("missing-fixture" if not pool_all else "unsupported",
                    rolls(pool_all) or [])
        cal = [e for e in pool_ok if e["split"] == "calibration"]
        hold = [e for e in pool_ok if e["split"] == "holdout"]
        status = "verified" if cal and hold else "missing-fixture"
        return status, rolls(pool_all)

    def input_cell(kind):
        pool = [e for e in ok if e.get("input_kind") == kind]
        return (sorted({e["roll"] for e in pool}), len(pool))

    devices = sorted({e["camera"] for e in ok
                      if e.get("camera") and e["camera"] != "unknown"})
    formats = sorted({Path(e["file"]).suffix.lower() for e in ok})
    tiffs = [e for e in ok if Path(e["file"]).suffix.lower() in (".tif", ".tiff")]
    dngs = [e for e in imgs if Path(e["file"]).suffix.lower() == ".dng"]
    return {
        "c41_color": cell("c41_color"),
        "ecn2_color": cell("ecn2_color"),
        "unknown_type": cell("unknown"),
        "bw": ("out-of-scope", []),
        "decodable_formats": formats,
        "devices": devices,
        "device_count": len(devices),
        "real_tiff16_samples": [e["file"] for e in tiffs],
        "tiff_capture_types": sorted({e.get("capture_type", "unknown")
                                      for e in tiffs}),
        "dng_samples": [e["file"] for e in dngs],
        "input_kinds": {k: input_cell(k) for k in
                        ("raw", "tiff", "dng", "jpeg", "png")},
        "scanner_tiff_samples": [e["file"] for e in tiffs
                                 if e.get("capture_type") == "scanner"],
        "m_dng": bool(dngs),
    }


def render(m: dict, manifest: dict) -> str:
    lines = [
        "# 支持矩阵（由真实解码探针生成，v5 §4.2/§14.3）",
        "",
        "状态含义：`verified`=可解码且 calibration+holdout 均有样片；"
        "`missing-fixture`=无样片，**禁止宣称支持**；`unsupported`=样片存在但"
        "解码器不支持；`out-of-scope`=v5 P0 明确不覆盖。",
        "",
        "> v5 §0.2.5：黑白负片与反转片不在 P0，因此**不再**作为发布门禁项。",
        "",
        "| 能力格 | 状态 | 样片卷 | 说明 |",
        "|---|---|---|---|",
    ]

    def row(name, entry, note):
        status, rolls = entry
        lines.append(f"| {name} | `{status}` | {', '.join(rolls) or '—'} | {note} |")

    row("C-41 彩色负片（相机 RAW 翻拍）", m["c41_color"],
        "1298_G200 标定 / 1299_G200 留出（Nikon Z8 + 105 micro，LED 背光）")
    row("ECN-2 彩色电影负片", m["ecn2_color"],
        "ECN2_5207 标定 / ECN2_5219 留出：Z8 高效 NEF 经 Lightroom 导出的 "
        "16 位 TIFF（LibRaw 0.22 无法解码 AV1 RAW，docs/blockers.md B-1）")
    row("未知处理类型（CN400）", m["unknown_type"],
        "film_type=unknown 不阻断转换与导出（v5 §6.3），只是不参与类型宣称")
    row("黑白负片", m["bw"], "v5 P0 排除项（§2.3），不作为门禁")
    lines += [
        "",
        "## 输入路径",
        "",
        f"- 可解码输入格式：{', '.join(m['decodable_formats']) or '—'}",
        f"- 相机 RAW：{m['input_kinds']['raw'][1]} 张"
        f"（卷：{', '.join(m['input_kinds']['raw'][0]) or '—'}）",
        f"- 16 位 TIFF：{len(m['real_tiff16_samples'])} 张"
        f"（采集类型：{', '.join(m['tiff_capture_types']) or '—'}）",
        f"- 真实**扫描仪** TIFF：{len(m['scanner_tiff_samples'])} 张 -> "
        + ("missing-fixture，扫描仪路线未验证"
           if not m["scanner_tiff_samples"] else "已覆盖"),
        f"- DNG（mosaic / Linear）：{len(m['dng_samples'])} 张 -> "
        + ("missing-fixture，DNG 路线未验证（代码路径已实现，仅缺真实样片）"
           if not m["m_dng"] else "已覆盖"),
        f"- 设备覆盖：{m['device_count']} 台（{', '.join(m['devices'])}） -> "
        + "**单设备，跨设备泛化一律未验证（v5 §9.3.4）**",
        "- 光源覆盖：全部 unknown -> 多光源结论未验证",
        "",
        "## 未验证即不宣称（v5 §14.3）",
        "",
        "- 没有真实 fixture 的输入/胶片/设备组合一律标 `missing-fixture`，"
        "不得伪造 PASS；",
        "- 通用先验回退、卷级片基共享与三候选在缺样片组合上只做过合成与"
        "跨卷迁移测试，不写成设备级结论。",
        "",
        "生成命令：`.venv\\Scripts\\python.exe scripts/build_support_matrix.py`",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    manifest = json.loads((ROOT / "fixtures" / "manifest.json").read_text(encoding="utf-8"))
    m = compute_matrix(manifest)
    m["m_dng"] = bool(m["dng_samples"])
    out = ROOT / "docs" / "support-matrix.md"
    out.write_text(render(m, manifest), encoding="utf-8")
    print(f"写入 {out}")
    for k in ("c41_color", "ecn2_color", "unknown_type", "bw"):
        print(f"  {k}: {m[k][0]}")
    print(f"  devices: {m['device_count']}  formats: {m['decodable_formats']}")
    print(f"  real 16-bit TIFF: {len(m['real_tiff16_samples'])}  "
          f"scanner TIFF: {len(m['scanner_tiff_samples'])}  DNG: {len(m['dng_samples'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
