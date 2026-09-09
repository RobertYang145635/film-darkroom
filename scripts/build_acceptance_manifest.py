"""生成 v5 §10.1 的视觉验收基准清单 fixtures/acceptance/manifest.json。

每张必含：源负片 SHA-256、卷、C-41/ECN-2、输入类型与采集指纹、三类人工
确认 ROI、NLP 默认参考路径与版本设置、用户满意参考（若有）、所属集合、
以及**留给用户填写**的最终判定字段。

规则（v5 §10.1）：
- 标 `parity_required=true` 的帧必须有同源 NLP 参考，否则不能进入
  "达到或超过 NLP" 的正式对标集；
- 没有 NLP 参考的帧可以 `robustness_only=true` 收录，用于裁切/兼容/极端
  场景/数值回归，但不得用来证明相对 NLP 更好或相当；
- 执行 agent **不得**替用户填写 decision 字段。

运行：.venv\\Scripts\\python.exe scripts/build_acceptance_manifest.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from negative_lab.calibration.annotations import load_annotation  # noqa: E402
from negative_lab.library.fingerprint import capture_fingerprint  # noqa: E402

OUT = ROOT / "fixtures" / "acceptance" / "manifest.json"
SCHEMA = 1

#: NLP 对照的已知来源说明（用户提供，非本软件生成）
NLP_NOTES = ("Negative Lab Pro 默认/零编辑导出，仅作竞品对照基线；"
             "未经用户许可不作为训练目标或复制对象（v5 §9.2）")


def _fingerprint_for(entry: dict) -> dict:
    """从 manifest 已记录的解码事实生成采集指纹（不重新解码大图）。"""
    meta = {
        "kind": entry.get("input_kind") or ("tiff" if entry["file"].lower().endswith(
            (".tif", ".tiff")) else "raw"),
        "camera": entry.get("camera", ""),
        "lens": entry.get("lens", ""),
        "width": entry.get("width", 0),
        "height": entry.get("height", 0),
        "bit_depth": entry.get("bit_depth", 0),
        "raw_decode_params": entry.get("decode_params", {}),
        "input_icc_sha256": entry.get("icc_profile", "none"),
    }
    fp = capture_fingerprint(meta, source_path=entry["file"])
    return {"value": fp.value, "kind": fp.kind, "fields": fp.fields}


def _rois(entry: dict) -> dict:
    """三类 ROI：用户确认标注优先，否则自动检测草稿并标记待确认。"""
    ann = load_annotation(entry["file"])
    if ann is not None:
        base = [list(r) for r in (ann.base_sample_rois or [])]
        if not base and ann.base_roi:
            base = [list(ann.base_roi)]
        crop = ann.output_crop_roi or ann.image_area_roi
        return {
            "output_crop_roi": list(crop) if crop else None,
            "image_area_roi": list(ann.image_area_roi) if ann.image_area_roi else None,
            "base_sample_rois": base,
            "confirmed_by_user": bool(ann.is_user_verified()),
            "needs_reconfirm": bool(getattr(ann, "needs_reconfirm", False)),
            "source": "annotation",
        }
    return {"output_crop_roi": None, "image_area_roi": None,
            "base_sample_rois": [], "confirmed_by_user": False,
            "needs_reconfirm": True, "source": "none"}


def build() -> dict:
    src = json.loads((ROOT / "fixtures" / "manifest.json").read_text(encoding="utf-8"))
    frames = []
    for e in src["images"]:
        if e["decode_status"] != "ok":
            continue
        p = ROOT / e["file"]
        if not p.exists():
            continue
        ref = e.get("reference_file")
        has_nlp = bool(ref) and (ROOT / ref).exists()
        frames.append({
            "id": f"{e['roll']}::{Path(e['file']).stem}",
            "file": e["file"],
            "source_sha256": e["sha256"],
            "roll": e["roll"],
            "process": e.get("process", "unknown"),
            "film_type": e.get("film_type", "unknown"),
            "film_stock": e.get("film_stock", "unknown"),
            "input_type": e.get("input_kind") or Path(e["file"]).suffix.lstrip("."),
            "capture_type": e.get("capture_type", "unknown"),
            "capture_fingerprint": _fingerprint_for(e),
            "rois": _rois(e),
            "nlp_reference": ref if has_nlp else None,
            "nlp_reference_notes": NLP_NOTES if has_nlp else None,
            "nlp_version": "unknown（用户导出，未记录插件版本）" if has_nlp else None,
            "nlp_settings": "默认/零编辑" if has_nlp else None,
            "user_satisfied_reference": None,
            "split": e.get("split", "unassigned"),
            "required": True,
            "parity_required": bool(has_nlp),
            "robustness_only": not has_nlp,
            # ---- 以下由用户填写，执行 agent 不得代签（v5 §10.2）
            "decision": {
                "accepted": None,
                "chosen_candidate": None,
                "relative_to_nlp": None,     # better|parity|worse|no-reference
                "issue_tags": [],            # 色罩|肤色|亮度|对比度|高光|阴影|饱和度|裁切|其他
                "note": "",
                "signed_by": None,
                "signed_date": None,
            },
        })
    rolls = {}
    for f in frames:
        rolls.setdefault(f["roll"], []).append(f)
    doc = {
        "schema_version": SCHEMA,
        "spec": "负片裁切与去色罩_AI执行规格_v5.md §10.1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "notes": [
            "required=true 的帧必须全部出现在逐卷联系表中，不得只挑好看的。",
            "parity_required=true 且缺少同源 NLP 参考的帧不得进入正式对标集。",
            "robustness_only=true 的帧只能用于裁切/兼容/极端场景/数值回归。",
            "decision 字段只能由用户填写；执行 agent 不得代签（v5 §10.2）。",
            "ROI confirmed_by_user=false 表示沿用自动检测/历史草稿，"
            "用户需在裁切页确认后本清单重新生成。",
        ],
        "summary": {
            "frames": len(frames),
            "by_roll": {r: len(v) for r, v in sorted(rolls.items())},
            "parity_required": sum(1 for f in frames if f["parity_required"]),
            "robustness_only": sum(1 for f in frames if f["robustness_only"]),
            "rois_user_confirmed": sum(1 for f in frames
                                       if f["rois"]["confirmed_by_user"]),
            "processes": sorted({f["process"] for f in frames}),
        },
        "frames": frames,
    }
    return doc


def main() -> int:
    doc = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"写入 {OUT}")
    print(json.dumps(doc["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
