"""人工视觉判定基准集（verdict kit）生成器。

背景（`docs/用户痛点调研与能力差距分析_20260905.md` §6 第一优先级、
`docs/用户痛点调研与能力差距分析_20260906_复核.md` §6）：
本项目有 440 项自动化测试，但**没有任何一张结果被确认为"好看"**——
`fixtures/acceptance/manifest.json` 的 decision 字段自 2026-09-02 起全是空的。
原因不是用户懒，是工具不配合：

| 旧做法 | 后果 |
|---|---|
| 逐卷联系表把 169 帧堆成一张 300px 高的长图 | 判一张要横向滚动 7 格、纵向翻 169 行，判完一次不现实 |
| 判定写进 `outputs/quality/<run-id>/rolls/<卷>/decisions.json` | `outputs/` 被 gitignore，且 run-id 每次新生成，**签了也不保留** |
| 判定字段只有 `accepted: bool` | 表达不了"微调可救 + 花了几步"，调研的验收判据根本没法算 |

本脚本把基准集改成可执行的样子，分三步（可单跑也可 `--all`）：

1. `--measure`：整卷走**生产服务** `process_roll`（§0.3.4：不得有第二条管线），
   在**显示编码(sRGB)域**上为每帧算客观分层标记与审计数字，写
   `fixtures/verdicts/metrics.json`。分层是**代理指标**，只说明"这张踩到了
   哪类已知失效场景"，不宣称识别出了"夜景"。
2. `--select`：从 metrics 里做**确定性分层抽样**（默认 48 帧，覆盖各分层与全部卷），
   合并进 `fixtures/verdicts/decisions.json`。**已签字判定永远保留、绝不覆盖**
   （见 `diagnostics/verdicts.merge_decisions` 的三条硬规则）。
3. `--sheets`：只为选中帧各出一张判定条图（原负片 | preserve | neutral | warm |
   NLP 参考），落在 `outputs/verdicts/sheets/`（可再生的大文件，不入库）。

判定纪律：本脚本**只写空白判定**，不写任何 `verdict`/`signed_by`（v5 §10.2）。
签字之后的汇总是 `scripts/verify_verdicts.py` 的活。

运行：
    .venv\\Scripts\\python.exe scripts/build_verdict_kit.py --all
    .venv\\Scripts\\python.exe scripts/build_verdict_kit.py --select --target 50
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from PIL import Image  # noqa: E402

from negative_lab.diagnostics.verdicts import (  # noqa: E402
    ACCEPTANCE_CRITERIA, STRATA, frame_strata, merge_decisions,
    merge_metrics, select_stratified, uid_of)
from negative_lab.diagnostics.explain import (  # noqa: E402
    base_fallback_note, clip_note, clip_tag)
from negative_lab.io.atomic import atomic_write_bytes  # noqa: E402
from negative_lab.io.images import ReadError, load_linear_rgb  # noqa: E402
from negative_lab.library.fingerprint import capture_fingerprint  # noqa: E402
from negative_lab.library.models import FrameRecord, GeometryRegions  # noqa: E402
from negative_lab.pipeline.density import encode_srgb  # noqa: E402
from negative_lab.pipeline.service import candidate_positive, process_roll  # noqa: E402

# 联系表脚本已经把"读图 + 几何降采样 + 生产管线"这套动作跑通过，
# 这里复用它，坚决不开第二条管线（§0.3.4）。
from build_visual_contact_sheet import (  # noqa: E402
    _downscale_geometry, _geometry, _norm_negative, _row, _to_img)

ACCEPTANCE = ROOT / "fixtures" / "acceptance" / "manifest.json"
VDIR = ROOT / "fixtures" / "verdicts"
METRICS = VDIR / "metrics.json"
DECISIONS = VDIR / "decisions.json"
SHEETS = ROOT / "outputs" / "verdicts" / "sheets"


def _load_manifest() -> list:
    doc = json.loads(ACCEPTANCE.read_text(encoding="utf-8"))
    frames = [f for f in doc["frames"] if f.get("required")]
    frames.sort(key=lambda f: str(f["id"]))
    return frames


def _write_json(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # default=float：审计数字里混着 numpy 标量，不兜住就会在 json.dumps 里炸
    atomic_write_bytes(path, json.dumps(doc, ensure_ascii=False, indent=2,
                                        default=float).encode("utf-8"), overwrite=True)


def _density_range(transmittance: np.ndarray) -> float:
    """负片密度范围（p95−p05，decade）：薄负片/欠曝的客观数字，只记录不参与判定。

    下限取 16 位数据的最低档（1/65535）而不是 1e-6：后者让全黑区域的"密度"
    变成 6 的假数，把范围撑到 3.9 decade，量出来的就不是胶片而是量化地板。
    """
    floor = 1.0 / 65535.0
    with np.errstate(divide="ignore", invalid="ignore"):
        dens = -np.log10(np.clip(np.asarray(transmittance, dtype=np.float64)[..., :3],
                                 floor, 1.0)).mean(axis=2)
    return round(float(np.percentile(dens, 95) - np.percentile(dens, 5)), 4)


# ------------------------------------------------------------------ measure
def _plain(value):
    """把管线里的 numpy 标量/数组/嵌套结构变成可 JSON 化的普通值。

    判据测量要能随手写进文件，不能因为某个字段是 ndarray 就整个跑挂；
    转换只降表示形式，不改变数值。
    """
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, np.ndarray):
        return [_plain(v) for v in value.reshape(-1).tolist()]
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    try:
        return round(float(value), 6)
    except (TypeError, ValueError):
        return str(value)


def measure(max_dim: int, only_roll: str | None, limit: int | None) -> dict:
    frames = _load_manifest()
    if only_roll:
        frames = [f for f in frames if f["roll"] == only_roll]
    if limit:
        frames = frames[:limit]
    by_roll: dict = {}
    for f in frames:
        by_roll.setdefault(f["roll"], []).append(f)

    out_frames, t0 = [], time.perf_counter()
    for roll in sorted(by_roll):
        rows = by_roll[roll]
        recs, imgs, kept = [], [], []
        for fr in rows:
            p = ROOT / fr["file"]
            try:
                loaded = load_linear_rgb(p)
            except ReadError as e:
                out_frames.append({"id": fr["id"], "file": fr["file"], "roll": roll,
                                   "status": "decode_error", "error": str(e).splitlines()[-1]})
                continue
            arr = loaded.rgb
            step = max(1, int(np.ceil(max(arr.shape[:2]) / float(max_dim))))
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
        for fr, small, pr in zip(kept, imgs, results):
            cs = pr.candidates
            render = None
            if cs is not None:
                render = candidate_positive(pr, pr.chosen_candidate or cs.default_name)
            else:
                render = pr.rendered_positive_linear
            # 分层在**显示编码域**算：判据里"暗场/高亮"说的是人在屏幕上看到的亮度，
            # 不是线性光的数值。
            disp = encode_srgb(render) if render is not None else np.zeros((1, 1, 3), np.float32)
            scene = dict((cs.diagnostics or {}).get("scene") or {}) if cs else {}
            strata = frame_strata(
                disp, base_confidence=pr.base_confidence,
                technical_preclip=pr.gate.technical_preclip,
                scene=scene,
                has_nlp_reference=bool(fr.get("nlp_reference")))
            d = np.clip(np.asarray(disp, dtype=np.float32), 0, 1)[..., :3]
            luma = 0.2126 * d[..., 0] + 0.7152 * d[..., 1] + 0.0722 * d[..., 2]
            top = luma[: max(1, luma.shape[0] // 3)]
            out_frames.append({
                "id": fr["id"], "file": fr["file"], "roll": roll, "status": "ok",
                # 唯一键：manifest 里 169 帧只有 132 个 id，按 id 合并会静默合成一帧
                "uid": uid_of(fr), "source_sha256": fr.get("source_sha256"),
                "strata": {k: bool(v) for k, v in strata.items()},
                "audit": {
                    "median_luma": round(float(np.median(luma)), 4),
                    "p05": round(float(np.percentile(luma, 5)), 4),
                    "p95": round(float(np.percentile(luma, 95)), 4),
                    "top_mean": round(float(top.mean()), 4),
                    "top_std": round(float(top.std()), 4),
                    "negative_density_range": _density_range(small),
                    "scene": {k: v for k, v in scene.items()
                              if isinstance(v, (int, float))},
                    "technical_preclip": pr.gate.technical_preclip,
                },
                "base_source": pr.base_source,
                "base_confidence": pr.base_confidence,
                # 片基侧证据（2026-09-09 追加）。为什么必须一起量：
                # 回退 generic_prior 的 74 帧只有 1.4% 被高端剪裁挡住导出，
                # 而**真的量到片基**的 95 帧却有 21.1% 被挡 —— 一头死灰、一头死白，
                # 两头其实是同一个 Dmin 估计问题。光看 base_source 判不了案。
                "base_dmin": _plain(pr.frame_base.get("dmin")),
                "base_contamination": _plain(pr.frame_base.get("contamination")),
                "base_black_level": _plain(pr.frame_base.get("black_level")),
                "base_spatial_trend": _plain(pr.frame_base.get("spatial_trend")),
                "input_clip": _plain(pr.gate.input_clip),
                "clip_attribution": _plain(pr.gate.clip_attribution),
                "best_rejected_region": _plain(pr.frame_base.get("best_rejected_region")),
                "dmax": _plain(getattr(pr.report.params, "dmax", None)),
                "gate": pr.gate.status,
                "blocks_export": pr.gate.blocks_export,
                "chosen_candidate": pr.chosen_candidate,
                "candidate_confidence": {c.name: c.confidence for c in cs} if cs else {},
                "parity_required": bool(fr.get("parity_required")),
                "has_nlp_reference": bool(fr.get("nlp_reference")),
            })
        print(f"  measured {roll}: {len(kept)} 帧", flush=True)
    doc = {
        "schema_version": 3,   # v3：coverage（局部跑不再覆盖整份语料，见 merge_metrics）
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "max_dim": max_dim,
        "note": ("分层标记为**显示编码(sRGB)域**上的像素统计（代理指标，非场景真值）；"
                 "数字全部来自生产服务 process_roll，无第二条管线。"),
        "strata_legend": dict(STRATA),
        "seconds": round(time.perf_counter() - t0, 1),
        "frames": sorted(out_frames, key=lambda f: str(f["id"])),
    }
    # 局部跑（--roll / --limit）**并入**已有语料：本会话曾因为一次
    # `--measure --roll 1298_G200` 把 169 帧覆盖成 12 帧，而下游毫无察觉。
    partial = bool(only_roll or limit)
    existing = (json.loads(METRICS.read_text(encoding="utf-8"))
                if METRICS.exists() else None)
    doc = merge_metrics(existing, doc, partial=partial)
    _write_json(METRICS, doc)
    cov = doc["coverage"]
    print(f"写入 {METRICS}（本次 {cov['this_run_frames']} 帧 / "
          f"文件内共 {cov['frames']} 帧、{cov['rolls']} 卷，{doc['seconds']}s）")
    if cov["partial_run"]:
        print(f"  ⚠ 局部跑：已并入旧语料（覆盖同 id {cov['replaced_ids']} 帧、"
              f"保留其它 {cov['kept_from_previous']} 帧），未覆盖整份")
    elif cov["dropped_ids"]:
        print(f"  ⚠ 全量跑后语料收缩：{cov['dropped_ids']} 个 id 不再存在"
              "（manifest 变化？请确认不是取样范围写错）")
    if cov["key_scheme_changed"]:
        print("  ℹ 键方案迁移（id → id#sha8）：收缩按 id 统计，改键不算丢数据")
    if cov["unique_keys"] != cov["frames"]:
        print(f"  ⚠ 唯一键 {cov['unique_keys']} < 帧 {cov['frames']}："
              "有帧在 uid 层面互相覆盖（同 id 同 sha？检查 manifest 是否重复登记）")
    return doc


# ------------------------------------------------------------------ select
def select(target: int) -> dict:
    if not METRICS.exists():
        raise SystemExit(f"缺少 {METRICS}；先跑 --measure")
    metrics = json.loads(METRICS.read_text(encoding="utf-8"))
    cand = [f for f in metrics["frames"] if f.get("status") == "ok"]
    chosen, report = select_stratified(cand, target=target)
    existing = None
    if DECISIONS.exists():
        try:
            existing = json.loads(DECISIONS.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            raise SystemExit(f"{DECISIONS} 不是合法 JSON；不覆盖，请人工处理")
    doc = merge_decisions(existing, chosen)
    doc["selection"] = report
    doc["criteria"] = ACCEPTANCE_CRITERIA
    _write_json(DECISIONS, doc)
    signed = sum(1 for f in doc["frames"] if f.get("verdict"))
    print(f"写入 {DECISIONS}：{len(doc['frames'])} 条（已签 {signed}，本次选中 "
          f"{report['selected']}/{report['population']}）")
    print(json.dumps({"strata": {k: v for k, v in report["strata"].items()
                                 if v["available"]}, "rolls": report["rolls"]},
                     ensure_ascii=False))
    return doc


# ------------------------------------------------------------------ sheets
def render_sheets() -> dict:
    if not DECISIONS.exists():
        raise SystemExit(f"缺少 {DECISIONS}；先跑 --select")
    if not METRICS.exists():
        raise SystemExit(f"缺少 {METRICS}；先跑 --measure")
    doc = json.loads(DECISIONS.read_text(encoding="utf-8"))
    # 两处索引都按**唯一键**：2026-09-09 实测，只把 manifest 改成 uid 索引、
    # 查找仍用 id，结果是 `--sheets` **一张都不渲染却退出码 0**
    # （`src is None` 每帧都成立 -> continue -> done=0 -> "渲染 0 张"）。
    metrics = {uid_of(f): f for f in
               json.loads(METRICS.read_text(encoding="utf-8"))["frames"]}
    # 按**唯一键**索引：manifest 的 169 个 required 帧只有 132 个 id
    # （同卷 `翻拍/` 与 `翻拍2/翻拍/` 两次拍摄同名），按 id 索引会把两次拍摄
    # 合成一条，实测其中一条有 NLP 参考、另一条没有 —— 于是判定条可能显示
    # **不属于本帧**的参考成品。查不到就宁可没有参考，也不要拿错。
    manifest = {uid_of(f): f for f in _load_manifest()}   # 只读一次（325KB）
    by_roll: dict = {}
    for f in doc["frames"]:
        by_roll.setdefault(str(f.get("roll")), []).append(f)

    done, t0 = 0, time.perf_counter()
    skipped: list = []
    for roll in sorted(by_roll):
        recs, imgs, kept = [], [], []
        for entry in by_roll[roll]:
            key = str(entry.get("uid") or entry.get("id"))
            m = metrics.get(key)
            src = manifest.get(key)
            if not m or m.get("status") != "ok" or src is None:
                why = ("metrics 查不到" if not m else
                       "manifest 查不到" if src is None else "metrics status != ok")
                skipped.append(f"{key}：{why}")
                continue
            try:
                loaded = load_linear_rgb(ROOT / m["file"])
            except ReadError as e:
                print(f"  跳过 {m['id']}：{str(e).splitlines()[-1]}", flush=True)
                continue
            arr = loaded.rgb
            step = max(1, int(np.ceil(max(arr.shape[:2]) / 1200.0)))
            small = arr[::step, ::step, :].copy()
            geo = _downscale_geometry(_geometry(src), arr.shape[:2], small.shape[:2])
            if not geo.is_complete():
                geo = GeometryRegions(rotation_degrees=geo.rotation_degrees)
            fp = capture_fingerprint(loaded.meta, source_path=m["file"])
            recs.append(FrameRecord(source_path=m["file"],
                                    source_sha256=src["source_sha256"],
                                    input_facts=dict(loaded.meta),
                                    capture_fingerprint=fp.value, geometry=geo))
            imgs.append(small)
            kept.append((entry, m, small))
        if not recs:
            continue
        _, results = process_roll(list(zip(recs, imgs)))
        for (entry, m, small), pr in zip(kept, results):
            cs = pr.candidates
            renders = {c.name: _to_img(encode_srgb(candidate_positive(pr, c.name)))
                       for c in cs} if cs else {}
            nlp = None
            ref = src_of(manifest, m.get("uid") or m["id"]).get("nlp_reference")
            if ref and (ROOT / ref).exists():
                nlp = Image.open(str(ROOT / ref)).convert("RGB")
            tags = " ".join(entry.get("strata") or []) or "（无特殊分层）"
            # 归因：判定条上放短标签（长文案会挤掉图像），完整原话写进 decisions.json
            evidence = " ".join(x for x in (
                base_fallback_note(pr.frame_base, pr.base_source),
                clip_note(pr.gate)) if x)
            short = ""
            brr = (pr.frame_base or {}).get("best_rejected_region")
            if brr:
                short += f" [片基被拒:{brr.get('binding_score')}]"
            # 归因标签只看**数值**：以前用 clip_note 的文案判断，
            # 而文案在剪裁低于告警线时是空串 -> else 分支 ->
            # 明明 100% 采集端继承的帧被标成"软件引入"
            short += clip_tag(getattr(pr.gate, "clip_attribution", None))
            if evidence:
                entry["evidence"] = evidence
            row = _row([("原始负片", _to_img(_norm_negative(small))),
                        ("preserve", renders.get("preserve")),
                        ("neutral", renders.get("neutral")),
                        ("warm", renders.get("warm")),
                        (f"默认 {pr.chosen_candidate}",
                         renders.get(pr.chosen_candidate or "")),
                        ("NLP 参考", nlp)],
                       done + 1, f"{Path(m['file']).name}  [{tags}]  "
                                 f"片基={m['base_source']} conf={m['base_confidence']}"
                                 f"{short}")
            stem = Path(str(m["file"])).stem
            uid_tag = str(m.get("uid") or "")
            if "#" in uid_tag:
                # 同 id 的两次拍摄（翻拍/ 与翻拍2/）文件名相同会**互相覆盖**判定条
                stem = f"{stem}__{uid_tag.split('#', 1)[1]}"
            out = SHEETS / str(roll) / f"{stem}.jpg"
            out.parent.mkdir(parents=True, exist_ok=True)
            row.save(out, quality=88)
            entry["sheet"] = out.relative_to(ROOT).as_posix()
            done += 1
            print(f"  [{done}] {entry['sheet']}", flush=True)
    _write_json(DECISIONS, doc)
    if not done and doc["frames"]:
        # 一张都没渲染**不是成功**：以前这里打印"渲染 0 张"然后退出 0，
        # 下一步 --verify 拿着空目录照样跑（假绿的又一种形态）
        raise SystemExit(
            f"一张判定条都没渲染：{len(skipped)} 条全部跳过。样例 {skipped[:3]}。"
            "多半是判定条目与 metrics/manifest 的索引键不一致（uid/id 混用），"
            "先重跑 --measure 与 --select，不要直接开始判定。")
    if skipped:
        print(f"  ⚠ 跳过 {len(skipped)} 条：{skipped[:3]}")
    print(f"渲染 {done} 张判定条，用时 {round(time.perf_counter() - t0, 1)}s；"
          f"填写 {DECISIONS}")
    return {"sheets": done}


def src_of(manifest: dict, frame_id: str) -> dict:
    return manifest.get(str(frame_id)) or {}


def prune_stale(entries, root, *, remove: bool = False) -> dict:
    """列出/清理**本工具生成、且已无人引用**的判定条；**默认只列不删**。

    真正的安全边界都在 `negative_lab.diagnostics.verdicts.prune_orphan_sheets`
    里，并且**每条都有独立回归锁**：越界不碰、符号链接不碰、裸命名不删、
    被引用不删、`remove=False` 时一个字节都不动。这里只是薄薄一层接线。
    """
    from negative_lab.diagnostics.verdicts import prune_orphan_sheets
    rep = prune_orphan_sheets(entries, root=root, remove=remove)
    print(f"判定条目录 `{root}`：候选 {rep['candidates']} 张，"
          f"{'已删除' if rep['remove'] else '未删除（默认只列不删）'}"
          f" {rep['removed']} 张，被引用保留 {rep['referenced_kept']} 张，"
          f"裸命名交人工判断 {rep['bare_kept']} 张")
    for n in rep["samples"]:
        print(f"  - {n}")
    return rep


def build_parser() -> argparse.ArgumentParser:
    """单独抽出来：这样"**开关默认不删**"这件事**可以被测试锁住**。"""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--select", action="store_true")
    ap.add_argument("--sheets", action="store_true")
    ap.add_argument("--all", action="store_true", help="三步全跑（默认）")
    ap.add_argument("--target", type=int, default=48)
    ap.add_argument("--max-dim", type=int, default=1200)
    ap.add_argument("--roll")
    ap.add_argument("--limit", type=int, help="只测前 N 帧（调试用，会破坏整卷片基共享）")
    # B-10：判定条目录里的陈旧文件。**默认一个都不删**，要删得显式说。
    ap.add_argument("--list-stale", action="store_true",
                    help="只列出无人引用的判定条，**不删任何文件**")
    ap.add_argument("--prune-stale", action="store_true",
                    help="删除本工具生成且已无人引用的判定条；"
                         "**裸命名 / 目录外 / 被引用的一律不删**")
    return ap


def main() -> int:
    a = build_parser().parse_args()
    if not (a.measure or a.select or a.sheets or a.all
            or a.list_stale or a.prune_stale):
        a.all = True
    if a.measure or a.all:
        measure(a.max_dim, a.roll, a.limit)
    if a.select or a.all:
        select(a.target)
    if a.sheets or a.all:
        render_sheets()
    if a.list_stale or a.prune_stale:
        import json
        entries = json.loads(
            (ROOT / "fixtures" / "verdicts" / "decisions.json")
            .read_text(encoding="utf-8")).get("frames") or []
        prune_stale(entries, SHEETS, remove=bool(a.prune_stale))
    print("\n下一步：在 fixtures/verdicts/decisions.json 里逐帧填判定（只有你能签），"
          "然后跑 scripts/verify_verdicts.py 看是否达标。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
