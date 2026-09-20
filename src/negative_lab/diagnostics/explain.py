"""把新量到的证据翻译成人话（纯函数，不碰 Qt、不碰磁盘）。

为什么单独一个模块：门禁与片基估计器现在**有**证据了
（`clip_attribution`、`best_rejected_region`、分项 `scores`），
但右栏、判定条、sidecar 三个地方各自要写一遍文案的话，
三处会各自漂移，而且 UI 里的文案没法脱离 Qt 测试。
放在这里，UI 只负责贴上去，判定条也用它，测试直接调它。
"""

from __future__ import annotations

from .quality_gate import CLIP_WARN_FRAC

#: 分项分数 → 用户听得懂的说法（roll_analysis._roi_stats 的三项）
_SCORE_LABEL = {
    "spread": "颗粒/背光散布",
    "purity": "取样区污染",
    "size": "样本量不足",
}


def base_fallback_note(frame_base: dict | None, base_source: str) -> str:
    """片基为什么没用上实测值；没问题就返回空串。

    区分三种以前长得一模一样的情况（2026-09-09 实测：74/169 帧退回通用先验，
    报表统一是 `contamination=1.000 / confidence=0.000`）：
    ① 取样区量到了东西但置信度不达标（真因往往是**单个通道**的散布）；
    ② 根本没框到取样区；③ 用上了实测（不必解释）。
    """
    fb = frame_base or {}
    if base_source != "generic_prior":
        return ""
    best = fb.get("best_rejected_region")
    if best:
        mad = best.get("mad") or []
        mad_txt = ("（" + " / ".join(
            f"{ch} {v:.3f}" for ch, v in zip(("R", "G", "B"), mad)) + " decade）"
            if len(mad) == 3 else "")
        binding = best.get("binding_score") or "分项未知"
        why = _SCORE_LABEL.get(binding, binding)
        scores = best.get("scores") or {}
        score_txt = " ".join(f"{k}={v}" for k, v in scores.items()) or "无分项"
        return (f"片基未采用实测：自动取样区量到了数据"
                f"（污染 {best.get('contamination'):.0%}、保留 "
                f"{best.get('kept_frac'):.0%}），但被**{why}**否掉"
                f"（{binding}={scores.get(binding, '?')}；分项 {score_txt}）{mad_txt}；"
                f"散布取的是三通道最大值，**一个通道差就作废整帧测量**。"
                f"本帧 Dmin 用的是通用先验，不是实测。")
    if not fb.get("regions"):
        return ("片基未采用实测：没有可用的取样区"
                "（既不是脏也不是噪声，是**没框到**）；本帧 Dmin 用的是通用先验。")
    return "片基未采用实测：所有取样区均未通过置信度门槛；本帧 Dmin 用的是通用先验。"


def clip_note(gate) -> str:
    """高端剪裁的归因一句话；没有超警告线就返回空串。

    `gate` 可以是 QualityGate 实例或 `to_dict()` 出来的字典。
    """
    attr = (gate.get("clip_attribution") if isinstance(gate, dict)
            else getattr(gate, "clip_attribution", None)) or {}
    best_ch, best = "", None
    for ch, val in attr.items():
        if not isinstance(val, dict):
            continue
        total = float(val.get("high_total") or 0.0)
        if total > CLIP_WARN_FRAC and (best is None or total > float(best.get("high_total") or 0.0)):
            best_ch, best = ch, val
    if not best:
        return ""
    total = float(best["high_total"])
    inh, intro = float(best["high_inherited"]), float(best["high_introduced"])
    if total > 0 and inh / max(total, 1e-12) >= 0.8:
        return (f"高光剪裁 {best_ch} {total:.2%}：其中 {inh:.2%} 在输入端已是死数据"
                f"（软件新引入 {intro:.2%}）—— 采集端就已丢失，重扫/重拍才能根治，"
                "调算法补不回来。")
    return (f"高光剪裁 {best_ch} {total:.2%}：其中软件新引入 {intro:.2%}"
            f"（输入端死数据 {inh:.2%}）—— 这部分是本软件能改的。")
