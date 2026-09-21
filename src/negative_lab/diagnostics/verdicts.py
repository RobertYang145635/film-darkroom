"""人工视觉判定基准集的内核：客观分层、可复现抽样、判定合并与验收判据。

为什么需要这个模块（2026-09-09）
--------------------------------
`docs/用户痛点调研与能力差距分析_20260905.md` §6 的第一优先级是
「建立可复现的人眼基准：40–60 帧，覆盖夜景/单色/天空/薄负片，
三档判定 + ≥20 张满意参考成品」。此前只有两份半成品：

1. `scripts/build_visual_contact_sheet.py` 把 manifest 里 **全部 169 帧**
   堆成每张 300px 高的逐卷长图——没人能在里面判完 169 张，
   所以 `decision` 字段自 2026-09-02 起一直是空的；
2. 它写出的 `decisions.json` 落在 `outputs/quality/<run-id>/rolls/<卷>/`，
   而 `outputs/` 被 gitignore、run-id 每次新生成 ——
   **用户就算签了字，下一次跑脚本也换一个目录，判定不会自动沿用**（真缺陷）。

本模块只放**纯函数**（不读盘、不渲染、不 import 脚本），因此可单测：

- `frame_strata()`：从像素客观算出分层标记。这些是**代理指标**
  （proxy），不是场景真值——它只回答"这张踩到了哪类已知失效场景"，
  绝不宣称"这就是夜景"。阈值全部写死在常量里并有注释，改阈值=改判据。
- `select_stratified()`：按分层覆盖抽样，**同输入必得同输出**（无随机数、
  平局按 id 字典序），保证"每次改完算法复测的是同一批帧"。
- `merge_decisions()`：重建判定模板时**绝不丢弃、也绝不覆盖**已签字的判定。
- `summarize_verdicts()`：把判定汇总成验收判据；**未签满即判不通过**，
  不允许拿部分样本冒充整体绿。

判定只能由用户填写：本模块与脚本都没有任何"替用户写 verdict"的入口
（v5 §10.2 / 联系表脚本同一条纪律）。
"""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np
import pathlib

# 剪裁警告线复用质量门禁的常量：分层"踩到剪裁"和门禁"警告剪裁"必须是同一口径
from negative_lab.diagnostics.quality_gate import CLIP_WARN_FRAC

SCHEMA_VERSION = 1

#: 三档判定（用户口径来自 20260905 调研 §6.1）
VERDICT_USABLE = "usable"          # ① 可直接使用
VERDICT_TWEAKABLE = "tweakable"    # ② 微调可救（记实际步数）
VERDICT_UNUSABLE = "unusable"      # ③ 不可用
VALID_VERDICTS = (VERDICT_USABLE, VERDICT_TWEAKABLE, VERDICT_UNUSABLE)

#: 验收判据（20260905 调研 §6.1 原文三条）
ACCEPTANCE_CRITERIA = {
    "min_usable_fraction": 0.70,       # ① 直接可用 ≥ 70%
    "min_not_unusable_fraction": 0.95,  # ③ 不可用 ≤ 5%
    "min_rescued_fraction": 0.80,      # ② 里 ≥ 80% 能在 2 步内救回
    "max_rescue_steps": 2,
    "min_reference_products": 20,      # ≥ 20 张用户认可的参考成品
    "min_frames": 40,                  # 基准集规模下限
    "max_frames": 60,                  #   （同调研建议 40–60）
}

#: 分层标记（键稳定，写进判定文件供复测比对）
#:
#: 每个分层只有**一个来源**，绝大多数直接取管线自己的数字，避免"基准集另立一套
#: 场景判断"：
#: - `few_neutral_surfaces` / `hue_concentrated` / `saturated_dominant` /
#:   `night_like`：候选层 `candidates.scene_confidence()` 的现成统计，
#:   阈值取 `pipeline/service.py` 生成 scene_flags 与 `candidates.py` 罚项的同一批
#:   断点 —— 于是"基准集认为它是夜景"和"软件运行时对夜景降权"是同一件事；
#: - `*_clip` / `low_base_confidence`：`diagnostics.quality_gate` 的现成数字
#:   （剪裁复用 `CLIP_WARN_FRAC`，不另开口径）；
#: - `bright_uniform_top` / `low_result_contrast`：判定图上的像素代理
#:   （管线没有对应数字，只能自己量，故明确标注为 proxy）。
STRATA = {
    "few_neutral_surfaces": "画面里几乎没有可当中性面的像素（灰世界前提不成立，软件已降权）",
    "hue_concentrated": "色相高度集中：单色光源/舞台灯",
    "saturated_dominant": "高饱和主导：中和最容易把颜色做死",
    "night_like": "夜景/暗场（候选层 dark_frac 判据）",
    "bright_uniform_top": "画面上部大面积高亮且平滑：天空/雾，白点易被天空带跑（proxy）",
    "low_result_contrast": "成片反差低（p95-p05 窄）：薄负片/欠曝的常见表现（proxy）",
    "highlight_clip": "技术正片高端剪裁超警告线（采集或白点问题）",
    "shadow_clip": "技术正片低端剪裁超警告线（片基或 Dmax 问题）",
    "low_base_confidence": "片基 Dmin 置信度低：来源可能是回退而非实测",
    "no_nlp_reference": "没有同源 NLP 参考：只能用于稳健性，不得进对标集",
}
STRATA_KEYS = tuple(STRATA)

# 阈值。前四个抄管线断点；后两个是在本仓库真实样片上标定的像素代理。
NEUTRAL_FRAC_MIN = 0.12       # candidates.py：neutral_pen 在此以下打满
HUE_CONCENTRATION_MIN = 0.7   # service.py：scene_flags["monochrome_light"] 判据
SAT_FRAC_MIN = 0.85           # 实测标定：本语料 sat_frac 中位数≈0.75，
                              # 沿用管线罚项起点的 0.6 会在 6/6 帧全命中，
                              # 作为分层没有信息量；取头部 1/4 的 0.85
DARK_FRAC_MIN = 0.5           # service.py：scene_flags["night"] 判据
BRIGHT_TOP_LUMA = 0.55        # 实测：G200-3 天空 top=0.832/std=0.058
BRIGHT_TOP_SMOOTH_STD = 0.15  #   有云的天空 std 会到 0.15 左右，再宽就不是天空了
LOW_CONTRAST_RANGE = 0.30     # 实测：p95-p05 ≤ 0.15 的帧肉眼看已明显发灰
BASE_CONF_MIN = 0.60          # 质量门禁"片基置信度低"提示用 0.5，这里稍严一档


# ------------------------------------------------------------------ 分层
def _clip_fracs(clip: Mapping | None, side: str) -> list:
    """质量门禁的剪裁字典形状是 {"low": [r,g,b], "high": [r,g,b]}；
    也接受 {"low": {"R": ..}} 形式。取不到就当没有证据（False）。"""
    v = (clip or {}).get(side)
    if isinstance(v, Mapping):
        seq = list(v.values())
    elif isinstance(v, (list, tuple)):
        seq = list(v)
    else:
        return []
    return [float(x) for x in seq if isinstance(x, (int, float, np.floating))]


def frame_strata(pos_display: np.ndarray, *,
                 base_confidence: float | None = None,
                 technical_preclip: Mapping | None = None,
                 scene: Mapping | None = None,
                 has_nlp_reference: bool = True) -> dict:
    """算出这一帧踩到了哪些已知失效场景。

    `pos_display` 必须是**显示编码(sRGB)**的正片：判据里"高亮/平滑/反差"说的是
    人在屏幕上看到的亮度，不是线性光的数值。
    `scene` 传候选层 `scene_confidence()` 的统计字典，缺键即不判（不当真）。

    任何缺证据的维度一律 False —— 与本项目"证据不足只降置信度、不硬判"一致。

    已知适用条件（踩过才写下的）：像素代理在 **clip 到 [0,1] 之后**统计，
    所以过曝到 1.0 的纹理区 std 会被压平、更容易算作"平滑高亮"。人眼看那块
    本来也是一片死白，故行为可接受；但它不是"天空检测器"，阈值别当真相用。
    """
    out = {k: False for k in STRATA_KEYS}
    out["no_nlp_reference"] = not has_nlp_reference

    a = np.asarray(pos_display, dtype=np.float32)
    if a.ndim == 3 and a.shape[2] >= 3 and a.size:
        a = a[..., :3]
        if not np.isfinite(a).all():
            a = np.where(np.isfinite(a), a, 0.0)
        a = np.clip(a, 0.0, 1.0)
        luma = (0.2126 * a[..., 0] + 0.7152 * a[..., 1] + 0.0722 * a[..., 2])
        h3 = max(1, luma.shape[0] // 3)
        top = luma[:h3]
        out["bright_uniform_top"] = bool(
            float(top.mean()) > BRIGHT_TOP_LUMA
            and float(top.std()) < BRIGHT_TOP_SMOOTH_STD)
        p95 = float(np.percentile(luma, 95))
        p05 = float(np.percentile(luma, 5))
        out["low_result_contrast"] = bool((p95 - p05) < LOW_CONTRAST_RANGE)

    hi, lo = _clip_fracs(technical_preclip, "high"), _clip_fracs(technical_preclip, "low")
    if hi:
        out["highlight_clip"] = bool(max(hi) > CLIP_WARN_FRAC)
    if lo:
        out["shadow_clip"] = bool(max(lo) > CLIP_WARN_FRAC)
    if base_confidence is not None and float(base_confidence) < BASE_CONF_MIN:
        out["low_base_confidence"] = True

    sc = scene or {}
    if "neutral_frac" in sc:
        out["few_neutral_surfaces"] = bool(float(sc["neutral_frac"]) < NEUTRAL_FRAC_MIN)
    if "hue_concentration" in sc:
        out["hue_concentrated"] = bool(float(sc["hue_concentration"]) > HUE_CONCENTRATION_MIN)
    if "sat_frac" in sc:
        out["saturated_dominant"] = bool(float(sc["sat_frac"]) > SAT_FRAC_MIN)
    if "dark_frac" in sc:
        out["night_like"] = bool(float(sc["dark_frac"]) > DARK_FRAC_MIN)
    return out


def stratum_reasons(strata: Mapping) -> list:
    """把 True 的分层标记翻成可读依据（脚本画在判定条上、判定文件里也留一份）。"""
    return [f"{k}：{STRATA[k]}" for k in STRATA_KEYS if strata.get(k)]


# ------------------------------------------------------------------ 抽样
def select_stratified(candidates: Iterable[Mapping], *,
                      target: int = 48,
                      min_per_stratum: int = 4,
                      min_per_roll: int = 4) -> tuple:
    """从候选帧里挑出覆盖面尽量好的基准集。

    **确定性**：全程不用随机数，平局一律按 `id` 字典序 —— 同输入必得同输出，
    否则"每次改完算法复测同一批帧"这句话就是空的。

    两遍贪心：
    1. 覆盖优先：每个分层（含 `no_nlp_reference` 之外的真分层）至少选
       `min_per_stratum` 帧，选的时候按"该帧命中的稀有分层数"排序，
       同等条件下优先能补上缺额卷的帧；
    2. 补齐：剩余名额按"命中的稀有分层数多者优先"填满，凑到 `target`。

    返回 (选中帧列表[按 id 排序], 覆盖率报告)。
    """
    frames = [dict(f) for f in candidates]
    frames.sort(key=lambda f: str(f.get("id")))
    if not frames:
        return [], {"selected": 0, "requested": target, "strata": {}, "rolls": {}}

    def hits(f) -> list:
        return [k for k in STRATA_KEYS
                if k != "no_nlp_reference" and (f.get("strata") or {}).get(k)]

    rarity: dict = {}
    for f in frames:
        for k in hits(f):
            rarity[k] = rarity.get(k, 0) + 1

    chosen: list = []
    chosen_ids: set = set()
    per_stratum: dict = {k: 0 for k in STRATA_KEYS}
    per_roll: dict = {}

    def take(f):
        fid = str(f.get("id"))
        if fid in chosen_ids:
            return
        chosen_ids.add(fid)
        chosen.append(f)
        roll = str(f.get("roll") or "")
        per_roll[roll] = per_roll.get(roll, 0) + 1
        for k in hits(f):
            per_stratum[k] = per_stratum.get(k, 0) + 1

    # 1) 分层覆盖优先
    for k in STRATA_KEYS:
        if k == "no_nlp_reference":
            continue
        pool = [f for f in frames
                if (f.get("strata") or {}).get(k) and str(f.get("id")) not in chosen_ids]
        want = min(min_per_stratum, len(pool))
        for _ in range(want):
            if len(chosen) >= target:
                break
            pool.sort(key=lambda f: (
                # 先选"能一次补上最多缺额分层"的帧
                -len([h for h in hits(f) if per_stratum.get(h, 0) < min_per_stratum]),
                # 再优先能补上欠额卷的帧，卷内数量少的先补
                0 if per_roll.get(str(f.get("roll") or ""), 0) < min_per_roll else 1,
                per_roll.get(str(f.get("roll") or ""), 0),
                # 同等条件下先选命中稀有分层的帧（越稀有越要先保）
                -min((rarity.get(h, 1) for h in hits(f)), default=1),
                str(f.get("id"))))
            take(pool[0])
            pool = [f for f in pool if str(f.get("id")) not in chosen_ids]

    # 2) 补齐到 target：先补欠额的卷，再按稀有组合填
    while len(chosen) < target:
        pool = [f for f in frames if str(f.get("id")) not in chosen_ids]
        if not pool:
            break
        pool.sort(key=lambda f: (
            0 if per_roll.get(str(f.get("roll") or ""), 0) < min_per_roll else 1,
            -len(hits(f)),
            per_roll.get(str(f.get("roll") or ""), 0),
            str(f.get("id"))))
        take(pool[0])

    chosen.sort(key=lambda f: str(f.get("id")))
    # 全语料都命中的分层没有区分度（实测：橙色色罩的正片上"几乎没有中性面"
    # 是 100% 命中）。它不该假装成"覆盖到了"，直接点名告知。
    no_disc = [k for k, n in rarity.items() if n == len(frames)]
    report = {
        "requested": target,
        "selected": len(chosen),
        "population": len(frames),
        "strata": {k: {"selected": per_stratum.get(k, 0), "available": rarity.get(k, 0)}
                   for k in STRATA_KEYS if k != "no_nlp_reference"},
        "strata_without_discrimination": no_disc,
        "rolls": {r: per_roll[r] for r in sorted(per_roll)},
        "nlp_reference": sum(1 for f in chosen if not (f.get("strata") or {})
                             .get("no_nlp_reference")),
    }
    return chosen, report


# ------------------------------------------------------------------ 判定文件
def _strata_on(entry) -> set:
    """把 strata 归一成"命中的分层名集合"。

    decisions 里存的是**列表**、metrics 里存的是**字典**（两种形状都真实存在过），
    直接比会全错——本函数就是为了让比较不依赖形状。
    """
    d = entry.get("strata") if isinstance(entry, Mapping) else None
    if isinstance(d, Mapping):
        return {str(k) for k, v in d.items() if v}
    return {str(x) for x in (d or [])}


def strata_drift(entries: Iterable[Mapping], metrics_frames: Iterable[Mapping]) -> dict:
    """判定条目里的分层标记，是否与**最新一次测量**一致（只读，不改判）。

    实测（2026-09-09）：当前 48 条与最新 metrics **零漂移**——但这份一致
    纯属侥幸：以后只重跑 `--measure` 不重跑 `--select`，`decisions.json` 里的
    分层就会悄悄停在旧测量上，而验收报告照读旧分层。签收之前必须能看见这件事。
    """
    latest = {}
    for m in (metrics_frames or []):
        if isinstance(m, Mapping):
            latest[uid_of(m)] = m
    checked = drift = 0
    samples: list = []
    for f in (entries or []):
        if not isinstance(f, Mapping):
            continue
        m = latest.get(uid_of(f))
        if m is None or "strata" not in m:
            continue
        checked += 1
        old, new = _strata_on(f), _strata_on(m)
        if old != new:
            drift += 1
            if len(samples) < 3:
                samples.append(
                    f"{uid_of(f)}：新测多 {sorted(new - old) or '无'} / "
                    f"条目多 {sorted(old - new) or '无'}")
    return {"checked": checked, "drifted": drift, "samples": samples,
            "unmatched": len([f for f in (entries or []) if isinstance(f, Mapping)]) - checked}


def sheet_integrity(entries: Iterable[Mapping], *, root=None) -> dict:
    """判定条目与判定条图片的对应关系体检（**只读**，不改动任何判定）。

    实测（2026-09-09）：改键后 `--select` 生成的 48 条 `sheet` 全空，
    而当时 `--sheets` **一张都没渲染却退出 0**。那种状态下就算签了字，
    验收报告也只读 JSON 里的判定字段——**没人能复核"这条判的是哪张图"**。
    所以验收环节必须能看出"判定与证据图片脱钩"。

    只报告、**不翻判定**：`outputs/` 不入库，用户清一次缓存
    不该把已经签下的判定作废；但那必须**大声说出来**。
    """
    rows = [f for f in (entries or []) if isinstance(f, Mapping)]
    missing = signed_missing = 0
    samples: list = []
    for f in rows:
        sheet = str(f.get("sheet") or "").strip()
        ok = bool(sheet)
        if ok and root is not None:
            ok = (root / sheet).exists() if not pathlib.Path(sheet).is_absolute() \
                else pathlib.Path(sheet).exists()
        elif ok:
            ok = pathlib.Path(sheet).exists()
        if not ok:
            missing += 1
            if _is_signed(f):
                signed_missing += 1
            if len(samples) < 3:
                samples.append(f"{uid_of(f) or f.get('id')}："
                               f"{'sheet 为空' if not sheet else '图片不存在 ' + sheet}")
    return {"entries": len(rows), "missing_sheet": missing,
            "signed_missing_sheet": signed_missing, "samples": samples}


def blank_verdict(frame: Mapping) -> dict:
    """一条待用户填写的判定（所有判定字段一律为 None，脚本不得填）。"""
    return {
        "id": frame.get("id"),
        "file": frame.get("file"),
        "roll": frame.get("roll"),
        "strata": [k for k in STRATA_KEYS if (frame.get("strata") or {}).get(k)],
        "stratum_notes": stratum_reasons(frame.get("strata") or {}),
        "sheet": frame.get("sheet"),
        "has_nlp_reference": not (frame.get("strata") or {}).get("no_nlp_reference"),
        # ---- 以下五个字段只能由用户填写（v5 §10.2：执行 agent 不得代签）
        "verdict": None,          # usable | tweakable | unusable
        "tweak_steps": None,      # ② 救回来实际花了几步（判"2 步内可救"用）
        "issue_tags": [],         # 色罩|肤色|亮度|对比度|高光|阴影|饱和度|裁切|其他
        "reference_product": None,  # 是否愿意存为满意参考（bool）
        "note": "",
        "signed_by": None,
        "signed_date": None,
    }


def merge_metrics(existing: dict | None, fresh: dict, *, partial: bool) -> dict:
    """把一次测量的结果**并入** metrics；局部跑不许覆盖整份语料。

    实测教训（2026-09-09，本会话自己踩的）：为验证新字段跑了一次
    `build_verdict_kit.py --measure --roll 1298_G200`，169 帧语料被**整份覆盖成
    12 帧**。下游（分层抽样、判定条覆盖率、验收报告）读的都是这个文件，
    语料缩水不报错，只会让覆盖率看起来"变好了"——假绿的另一种形态。

    规则：
    - `partial=True`（带了 `--roll` 或 `--limit`）：按帧 id 并入，本次同 id 覆盖旧值，
      其余原样保留；
    - 全量跑：整份替换，但**必须报出消失的 id 数**（语料收缩要可见，不能静默）；
    - 两种情况都写 `coverage`，让下游一眼看出这是全量还是局部。
    """
    prev_list = list((existing or {}).get("frames") or [])
    prev = {uid_of(f): f for f in prev_list}
    fresh_frames = list(fresh.get("frames") or [])
    replaced = sum(1 for f in fresh_frames if uid_of(f) in prev)
    if partial and prev:
        pooled = dict(prev)
        pooled.update({uid_of(f): f for f in fresh_frames})
        frames = [pooled[k] for k in sorted(pooled)]
    else:
        frames = sorted(fresh_frames, key=lambda f: uid_of(f))
    prev_ids = {str(f.get("id")) for f in prev_list}
    fresh_ids = {str(f.get("id")) for f in fresh_frames}
    prev_had_uid = any(f.get("uid") or f.get("source_sha256") for f in prev_list)
    doc = dict(fresh)
    doc["frames"] = frames
    doc["coverage"] = {
        "frames": len(frames),
        "rolls": len({str(f.get("roll")) for f in frames}),
        "this_run_frames": len(fresh_frames),
        "partial_run": bool(partial and prev),
        "replaced_ids": replaced,
        "kept_from_previous": len(frames) - len(fresh_frames),
        # 收缩按 **id 集合**算：它在键方案迁移（id → id#sha8）前后都稳定。
        # 2026-09-09 实测：改键后第一次全量跑曾报"132 个唯一键不再存在"，
        # 其实是键换了个写法，差点被当成"manifest 被砍"。
        "dropped_ids": 0 if (partial and prev) else len(prev_ids - fresh_ids),
        "key_scheme_changed": bool(prev_list) and not prev_had_uid,
        # 帧数 > 唯一键数 = 有帧在 uid 层面互相覆盖（同 id 同 sha）
        "unique_keys": len({uid_of(f) for f in frames}),
    }
    return doc


def uid_of(frame: Mapping | None) -> str:
    """帧的**唯一键**：`id` 在语料里不唯一。

    2026-09-09 实测：`fixtures/acceptance/manifest.json` 的 169 个 required 帧
    只有 **132 个不同 id** —— 37 个 id 各出现两次（同一卷的 `翻拍/` 与
    `翻拍2/翻拍/` 两次拍摄，`source_sha256` 不同）。凡是按 id 合并/查找的地方
    都会把两次拍摄合成一帧：`src_of()` 会拿错 NLP 参考（实测同一 id 一帧有参考、
    另一帧没有，于是判定条上可能显示**不属于本帧**的参考成品），
    判定文件会把用户签在 A 上的签名刷新成 B，两张判定条的图片文件名还会互相覆盖。

    唯一键 = `id#sha256前8位`；没有 sha 时退回 `id`
    （宁可可见地合并，也不要静默错配到别的文件）。
    """
    if not isinstance(frame, Mapping):
        return ""
    if frame.get("uid"):
        return str(frame["uid"])
    sha = str(frame.get("source_sha256") or "")
    if sha:
        return f"{frame.get('id')}#{sha[:8]}"
    return str(frame.get("id"))


def merge_decisions(existing: Mapping | None, selected: Iterable[Mapping]) -> dict:
    """把新的抽样合并进既有判定文件，**判定内容一律以既有为准**。

    三条硬规则（都是本轮实测到的既有缺陷反推出来的）：
    - 已签字/已填判定的帧**永远保留**，即使这次没抽中它
      （旧实现把判定写在 `outputs/quality/<run-id>/` 里，重跑换目录即丢）；
    - 已填过的判定字段**绝不刷新**（哪怕重新生成分层元数据）；
    - 只有从未出现过的帧才新建空白条目。
    """
    keep: dict = {}
    # 键方案迁移（id → id#sha8）：旧条目没有 uid，新条目有，两边键对不上。
    # 2026-09-09 实测：改键后重跑 `--select`，旧 48 条**全部**被判为
    # "没抽中"而进了 retired_ids —— 现在侥幸是 0 签，若用户已签，
    # 签名就会凭空蒸发。所以旧条目在 **id 唯一可认** 时必须认亲到新键上。
    sel_list = list(selected)
    sel_by_id: dict = {}
    for frame in sel_list:
        sel_by_id.setdefault(str(frame.get("id")), []).append(frame)
    legacy_key = {i: uid_of(v[0]) for i, v in sel_by_id.items() if len(v) == 1}
    if isinstance(existing, Mapping):
        for f in existing.get("frames") or []:
            if not isinstance(f, Mapping) or f.get("id") is None:
                continue
            if _is_signed(f) or str(f.get("verdict") or "") in VALID_VERDICTS:
                key = uid_of(f)
                if not f.get("uid"):
                    moved = legacy_key.get(str(f["id"]))
                    if moved:
                        key = moved          # 签名跟着搬到新键，不丢
                # 按**唯一键**保留：按 id 会把两次拍摄合成一帧，
                # 于是"签在 A 上"和"刷新成 B"变成同一件事
                keep[key] = dict(f)   # 用户填过的，原样保留
    entries: dict = dict(keep)
    for frame in sel_list:
        key = uid_of(frame)
        if key in entries:
            base = dict(entries[key])          # 保留判定内容，只刷新元数据
            for meta in ("strata", "stratum_notes", "sheet", "has_nlp_reference"):
                if meta in frame:
                    base[meta] = frame[meta]
            base["file"] = frame.get("file", base.get("file"))
            base["roll"] = frame.get("roll", base.get("roll"))
            base["uid"] = key
            entries[key] = base
        else:
            new = blank_verdict(frame)
            new["uid"] = key
            entries[key] = new
    # 未签且这次没抽中的条目：从模板里退掉，但把键记进 retired_ids，
    # 免得"重跑一次判定悄悄少了几帧"这种事再次无声发生。
    live_ids = {str(f.get("id")) for f in entries.values()}
    retired = sorted(
        {uid_of(f) for f in ((existing or {}).get("frames") or [])
         if isinstance(f, Mapping) and f.get("id") is not None
         and uid_of(f) not in entries
         # 只是换了键（id → id#sha8）不算退休，否则一次改键会让 48 条
         # 全部"消失"在 retired_ids 里，读的人以为语料被砍了
         and str(f["id"]) not in live_ids}
        | set((existing or {}).get("retired_ids") or []))
    return {
        "schema_version": SCHEMA_VERSION,
        "criteria": ACCEPTANCE_CRITERIA,
        "strata_legend": dict(STRATA),
        "signing_rule": ("判定字段（verdict / tweak_steps / issue_tags / "
                         "reference_product / note / signed_by）只能由用户本人填写；"
                         "执行 agent 与脚本不得代签（v5 §10.2）。"),
        "frames": [entries[k] for k in sorted(entries)],
        "retired_ids": retired,
    }


def _is_signed(entry: Mapping) -> bool:
    return bool(str(entry.get("signed_by") or "").strip())


# ------------------------------------------------------------------ 汇总判据
def summarize_verdicts(doc: Mapping, *, criteria: Mapping | None = None) -> dict:
    """把判定文件汇总成验收结论。**未签满 -> 直接不通过**（不拿部分样本冒充绿）。"""
    cr = dict(ACCEPTANCE_CRITERIA)
    if criteria:
        cr.update(criteria)
    frames = [f for f in (doc or {}).get("frames") or [] if isinstance(f, Mapping)]
    total = len(frames)
    signed = [f for f in frames if str(f.get("verdict") or "") in VALID_VERDICTS
              and _is_signed(f)]
    partial = [f for f in frames
               if str(f.get("verdict") or "") in VALID_VERDICTS and not _is_signed(f)]
    unsigned = [f for f in frames if str(f.get("verdict") or "") not in VALID_VERDICTS]

    n_usable = sum(1 for f in signed if f["verdict"] == VERDICT_USABLE)
    n_tweak = sum(1 for f in signed if f["verdict"] == VERDICT_TWEAKABLE)
    n_bad = sum(1 for f in signed if f["verdict"] == VERDICT_UNUSABLE)
    n = len(signed)
    rescued = [f for f in signed
               if f["verdict"] == VERDICT_TWEAKABLE
               and isinstance(f.get("tweak_steps"), int)
               and f["tweak_steps"] <= cr["max_rescue_steps"]]
    refs = sum(1 for f in signed if f.get("reference_product") is True)

    def frac(a, b):
        return round(a / b, 4) if b else None

    usable_frac = frac(n_usable, n)
    not_bad_frac = frac(n - n_bad, n)
    rescued_frac = frac(len(rescued), n_tweak)

    checks = []

    def check(ok, label, detail):
        """`ok=None` 表示**这条判据本轮无法判定**（分母为 0）。
        它不算通过也不算失败，渲染成 ⚪：0/0 打勾是假绿的一种，
        本项目不接受（对照 `min_usable_fraction` 等硬判据自己会红）。"""
        checks.append({"ok": (True if ok is None else bool(ok)),
                       "na": ok is None, "label": label, "detail": detail})

    # 已签条目里有几个**不同帧 id**：manifest 的 169 帧只有 132 个 id
    # （同卷 `翻拍/` 与 `翻拍2/翻拍/` 两次拍摄同名），若两拍都被选中并签掉，
    # "40–60 帧"的门槛会被**同一张照片的两拍**灌水。判定不改（那是用户的），
    # 但报告必须把这个数摆出来。
    n_distinct_ids = len({str(f.get("id")) for f in signed})
    check(n >= cr["min_frames"],
          f"已签帧数 ≥ {cr['min_frames']}",
          f"{n}/{total} 已签；缺 {len(unsigned)}"
          + (f"；⚠ 去重后只有 {n_distinct_ids} 个不同帧 id —— "
             f"有 {n - n_distinct_ids} 条是同 id 的多次拍摄，帧数门槛被重复样本顶着"
             if n_distinct_ids < n else ""))
    check(not unsigned, "判定必须签满（不得用部分样本冒充整体）",
          f"未签 {len(unsigned)} 帧" + ("；另有 %d 帧填了判定但没有 signed_by" % len(partial)
                                     if partial else ""))
    check(usable_frac is not None and usable_frac >= cr["min_usable_fraction"],
          f"① 直接可用 ≥ {cr['min_usable_fraction']:.0%}",
          f"{n_usable}/{n} = " + ("-" if usable_frac is None else f"{usable_frac:.1%}"))
    check(not_bad_frac is not None and not_bad_frac >= cr["min_not_unusable_fraction"],
          f"③ 不可用 ≤ {1 - cr['min_not_unusable_fraction']:.0%}",
          f"不可用 {n_bad}/{n}")
    check(None if rescued_frac is None else rescued_frac >= cr["min_rescued_fraction"],
          f"② 里 ≥ {cr['min_rescued_fraction']:.0%} 能在 "
          f"{cr['max_rescue_steps']} 步内救回",
          (f"{len(rescued)}/{n_tweak}" + ("" if rescued_frac is None
                                        else f" = {rescued_frac:.1%}"))
          + ("（本批没有 ② 帧，无法判定，不算通过也不算失败）" if rescued_frac is None
             else "（②中未填 tweak_steps 的按未救回计）"))
    check(refs >= cr["min_reference_products"],
          f"满意参考成品 ≥ {cr['min_reference_products']}", f"{refs} 张")

    covered, missing = _coverage(frames, signed)
    check(not missing, "分层与逐卷覆盖必须齐",
          "；".join(missing) if missing else "每个分层与每卷都至少有一帧已签判定")

    return {
        "total": total, "signed": n, "unsigned": len(unsigned),
        "usable": n_usable, "tweakable": n_tweak, "unusable": n_bad,
        "rescued_in_steps": len(rescued), "reference_products": refs,
        "usable_fraction": usable_frac, "not_unusable_fraction": not_bad_frac,
        "rescued_fraction": rescued_frac,
        "signed_distinct_ids": n_distinct_ids,
        "coverage": covered,
        "checks": checks,
        "passed": all(c["ok"] for c in checks) and n > 0,
    }


def _coverage(frames: list, signed: list) -> tuple:
    signed_ids = {str(f.get("id")) for f in signed}
    per_stratum: dict = {}
    per_roll: dict = {}
    for f in frames:
        roll = str(f.get("roll") or "")
        slot = per_roll.setdefault(roll, {"frames": 0, "signed": 0})
        slot["frames"] += 1
        if str(f.get("id")) in signed_ids:
            slot["signed"] += 1
        for k in (f.get("strata") or []):
            if k == "no_nlp_reference":
                continue
            d = per_stratum.setdefault(k, {"frames": 0, "signed": 0})
            d["frames"] += 1
            if str(f.get("id")) in signed_ids:
                d["signed"] += 1
    missing = [f"分层 {k} 没有任何已签判定（共 {d['frames']} 帧）"
               for k, d in sorted(per_stratum.items()) if d["signed"] == 0]
    missing += [f"卷 {r} 没有任何已签判定（共 {s['frames']} 帧）"
                for r, s in sorted(per_roll.items()) if s["signed"] == 0]
    return {"strata": per_stratum, "rolls": per_roll}, missing
