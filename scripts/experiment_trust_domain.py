"""实验：`scene_confidence` 到底在惩罚什么？（2026-09-09 提出的假设的实测）

**假设**：`build_candidates()` 用 `scene_confidence(ctx.technical_positive, …)`
判定 neutral 候选是否可信（`trusted = conf >= NEUTRAL_MIN_CONFIDENCE and …`），
但喂进去的是**还没去色罩的技术正片**。于是"画面高饱和""几乎没有中性面"
这两项罚项被**色罩本身**打满 —— 用症状去证明"不该治疗"，
治疗永远不会发生。全量 169 帧实测：默认候选 `preserve` 占 162/169（95.9%），
neutral 置信度中位数 0.08–0.26，而 preserve 的置信度在代码里**写死 1.0**。

本脚本把同一帧的 scene 统计在**校正前 / 校正后**各算一遍：
校正后 = 技术正片 × 本帧自己的 `correction_gain`（就是 neutral 候选要施加的那个增益）。
如果校正后 sat_frac 明显下降、neutral_frac 明显上升、conf 跨过门槛，
那么现在的门槛判的就不是"场景骗人"，而是"胶片有片基色罩"。

只读实验，不改任何生产行为、不写任何判定。

运行：
    .venv\\Scripts\\python.exe scripts/experiment_trust_domain.py [--limit 6] [--per-roll 5]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from negative_lab.io.images import ReadError, load_linear_rgb  # noqa: E402
from negative_lab.library.fingerprint import capture_fingerprint  # noqa: E402
from negative_lab.library.models import FrameRecord, GeometryRegions  # noqa: E402
from negative_lab.pipeline.candidates import (  # noqa: E402
    MAX_ESTIMATOR_SPREAD, NEUTRAL_MIN_CONFIDENCE, build_candidates,
    scene_confidence)
from negative_lab.pipeline.service import process_roll  # noqa: E402

from build_visual_contact_sheet import _downscale_geometry, _geometry  # noqa: E402

ACCEPTANCE = ROOT / "fixtures" / "acceptance" / "manifest.json"
OUT = ROOT / "outputs" / "verdicts" / "trust-domain-experiment.json"


def _stats(pos: np.ndarray, mask) -> dict:
    """scene 统计。`est={}`：同意性罚项 agree_pen 记 0，**不参与本实验**。

    实验只比较"同一批罚项在校正前/校正后"的差，agree_pen 两边都不算，
    差值有效；但别把这里的 conf 当成生产环境的 conf
    （生产环境带真实 est，全语料的真实结果看 `fixtures/verdicts/metrics.json`）。
    """
    conf, st = scene_confidence(pos, mask, {})
    return {"conf": round(float(conf), 3),
            "sat_frac": st.get("sat_frac"), "neutral_frac": st.get("neutral_frac"),
            "hue_conc": st.get("hue_concentration"), "dark_frac": st.get("dark_frac")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--per-roll", type=int, default=5)
    ap.add_argument("--max-dim", type=int, default=900)
    a = ap.parse_args()

    doc = json.loads(ACCEPTANCE.read_text(encoding="utf-8"))
    frames = sorted([f for f in doc["frames"] if f.get("required")],
                    key=lambda f: str(f["id"]))
    by_roll: dict = {}
    for f in frames:
        by_roll.setdefault(f["roll"], []).append(f)
    picked = []
    for roll in sorted(by_roll):
        picked += by_roll[roll][: a.per_roll]

    rows, n_after_pass = [], 0
    for roll in sorted({f["roll"] for f in picked}):
        rows_roll = [f for f in picked if f["roll"] == roll]
        recs, imgs, kept = [], [], []
        for fr in rows_roll:
            try:
                loaded = load_linear_rgb(ROOT / fr["file"])
            except ReadError:
                continue
            arr = loaded.rgb
            step = max(1, int(np.ceil(max(arr.shape[:2]) / float(a.max_dim))))
            small = arr[::step, ::step, :].copy()
            geo = _downscale_geometry(_geometry(fr), arr.shape[:2], small.shape[:2])
            if not geo.is_complete():
                geo = GeometryRegions(rotation_degrees=geo.rotation_degrees)
            fp = capture_fingerprint(loaded.meta, source_path=fr["file"])
            recs.append(FrameRecord(source_path=fr["file"],
                                    source_sha256=fr["source_sha256"],
                                    input_facts=dict(loaded.meta),
                                    capture_fingerprint=fp.value, geometry=geo))
            imgs.append(small)
            kept.append(fr)
        if not recs:
            continue
        _, results = process_roll(list(zip(recs, imgs)))
        for fr, pr in zip(kept, results):
            cs = build_candidates(pr)
            if cs is None or not cs.neutral:
                continue
            gain = np.asarray((cs.diagnostics or {}).get("correction_gain") or [1, 1, 1],
                              np.float32)
            if gain.shape != (3,) or not np.isfinite(gain).all():
                gain = np.ones(3, np.float32)
            pos = np.asarray(pr.technical_positive_linear, np.float32)
            corrected = np.clip(pos * gain[None, None, :], 0, 1)
            mask = getattr(pr, "valid_mask", None)
            before = _stats(pr.technical_positive_linear, mask)
            after = _stats(corrected, mask)
            spread = float((cs.diagnostics or {}).get("estimator_spread_rad") or 0.0)
            row = {"id": fr["id"], "roll": roll, "gain": [round(float(g), 3) for g in gain],
                   "before": before, "after": after,
                   "spread": round(spread, 3),
                   "min_confidence": NEUTRAL_MIN_CONFIDENCE,
                   "trusted_now": bool(before["conf"] >= NEUTRAL_MIN_CONFIDENCE
                                       and spread <= MAX_ESTIMATOR_SPREAD),
                   "trusted_if_measured_after": bool(after["conf"] >= NEUTRAL_MIN_CONFIDENCE
                                                     and spread <= MAX_ESTIMATOR_SPREAD),
                   "default_now": cs.default_name}
            n_after_pass += int(row["trusted_if_measured_after"])
            rows.append(row)
            print(f"  {fr['id'].split('::')[-1]:<12} "
                  f"conf {before['conf']:.3f} -> {after['conf']:.3f}  "
                  f"neutral_frac {before['neutral_frac']:.3f} -> {after['neutral_frac']:.3f}  "
                  f"sat {before['sat_frac']:.3f} -> {after['sat_frac']:.3f}  "
                  f"trusted {row['trusted_now']} -> {row['trusted_if_measured_after']}  "
                  f"(default {row['default_now']})", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "question": "scene_confidence 判的是场景还是色罩？",
        "frames": len(rows),
        "trusted_now": sum(1 for r in rows if r["trusted_now"]),
        "trusted_if_measured_after": n_after_pass,
        "min_confidence": NEUTRAL_MIN_CONFIDENCE,
        "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{len(rows)} 帧：现在判可信 {sum(1 for r in rows if r['trusted_now'])} 帧；"
          f"改成在**校正后**度量则 {n_after_pass} 帧可信"
          f"（门槛 conf>={NEUTRAL_MIN_CONFIDENCE}）")
    print(f"写出 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
