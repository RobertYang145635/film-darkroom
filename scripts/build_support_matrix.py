"""从 fixtures/manifest.json 生成 docs/support-matrix.md（整改 P0-A）。

矩阵格状态（稳定枚举）：
  verified          有可解码样片且 calibration+holdout 均有覆盖
  missing-fixture   宣称支持路径存在，但无任何可解码样片 -> 阻断该支持声明
  unsupported       样片存在但真实解码探针失败（格式/解码器缺口）
不再用总图片数掩盖类型/设备覆盖不足。

运行：.venv\\Scripts\\python.exe scripts/build_support_matrix.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


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

    devices = sorted({e["camera"] for e in ok if e["camera"] and e["camera"] != "unknown"})
    formats = sorted({Path(e["file"]).suffix.lower() for e in ok})
    m = {
        "c41_color": cell("c41_color"),
        "bw": cell("bw"),
        "ecn2_color": cell("ecn2_color"),
        "unknown_type": cell("unknown"),
        "decodable_formats": formats,
        "devices": devices,
        "device_count": len(devices),
        "real_tiff16_samples": [e["file"] for e in ok
                                if Path(e["file"]).suffix.lower() in (".tif", ".tiff")],
        "dng_samples": [e["file"] for e in imgs
                        if Path(e["file"]).suffix.lower() == ".dng"],
    }
    return m


def render(m: dict, manifest: dict) -> str:
    lines = [
        "# 支持矩阵（由真实解码探针生成，整改 P0-A）",
        "",
        "状态含义：`verified`=可解码且 calibration+holdout 均有样片；"
        "`missing-fixture`=无样片，**禁止宣称支持**；`unsupported`=样片存在但解码器不支持。",
        "",
        "| 能力格 | 状态 | 样片卷 | 说明 |",
        "|---|---|---|---|",
    ]

    def row(name, entry, note):
        status, rolls = entry
        lines.append(f"| {name} | `{status}` | {', '.join(rolls) or '—'} | {note} |")

    row("C-41 彩色负片 (camera RAW)", m["c41_color"], "Kodak Gold 200 两卷（1 标定 1 留出）")
    row("黑白负片", m["bw"], "无任何样片 -> 不得宣称支持（原规格 §2.1 胶片类型）")
    row("ECN-2 彩色电影负片", m["ecn2_color"],
       "5219/5207 共 72 张为 Z8 AV1 高效 NEF，LibRaw 0.22.1 实测不支持（docs/blockers.md B-1）；"
       "待用户转 16 位 TIFF 后重新探针")
    row("未知类型 (CN400)", m["unknown_type"], "film_type=unknown，不参与类型宣称")
    lines += [
        "",
        f"- 可解码输入格式：{', '.join(m['decodable_formats']) or '—'}"
        "（16 位 TIFF 仅合成样片，真实扫描/导出 TIFF 为 missing-fixture）",
        f"- 真实 16 位 TIFF 样片：{len(m['real_tiff16_samples'])} 张"
        + ("" if m["real_tiff16_samples"] else " -> **missing-fixture，不得宣称扫描仪路线已验证**"),
        f"- DNG 样片：{len(m['dng_samples'])} 张 -> missing-fixture" if not m["dng_samples"]
        else f"- DNG 样片：{len(m['dng_samples'])} 张",
        f"- 设备覆盖：{m['device_count']} 台（{', '.join(m['devices'])}）"
        " -> **单设备，多设备结论一律未验证（§12 末段）**",
        "- 光源覆盖：全部 unknown -> 多光源结论未验证",
        "- 灰卡/色卡样片：无 -> 中性区 ΔE2000 验收（§8.4）当前无法执行，"
        "需用户提供含色卡样片或人工取点",
        "",
        "生成命令：`.venv\\Scripts\\python.exe scripts/build_support_matrix.py`",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    manifest = json.loads((ROOT / "fixtures" / "manifest.json").read_text(encoding="utf-8"))
    m = compute_matrix(manifest)
    out = ROOT / "docs" / "support-matrix.md"
    out.write_text(render(m, manifest), encoding="utf-8")
    print(f"写入 {out}")
    for k in ("c41_color", "bw", "ecn2_color", "unknown_type"):
        print(f"  {k}: {m[k][0]}")
    print(f"  devices: {m['device_count']}  formats: {m['decodable_formats']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
