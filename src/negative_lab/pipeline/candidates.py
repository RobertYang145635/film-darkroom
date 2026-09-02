"""三个自动颜色候选：保留氛围 / 中性 / 偏暖（v5 §6.4-§6.6）。

共享同一个物理底座：候选之间**只**在 `RenderIntent` 的色温/色调上不同，
绝不修改输出裁切、片基或技术正片（§6.5）。

本地统计/小模型只允许做四件事（§6.4）：预测低维颜色起点、融合多个经典
色恒常性估计的置信度、选择默认候选或预测有限的色温/色调偏移、按用户偏好
改进排序。禁止直接生成整张图像、禁止云端推理、禁止用当前帧参考泄漏答案。

防止错误自动中和（§6.6）：
- 只在有效画面 `image_area_roi` 内统计（排除片边与非画面区域）；
- 排除接近剪裁的高光/阴影；
- 比较 Shades-of-Gray(p=1/2/6)、Gray-Edge、白点等多个估计器的方向与幅度；
- 高饱和、单色主导、夜景、舞台灯、夕阳场景降低置信度；
- 卷内多帧稳健汇总，防止一张特殊场景污染整卷；
- 色温/色调修正设硬上限并记录触发情况；
- 分歧过大时 `neutral` 回退到 `preserve` 并显示低置信度。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..library.models import Candidate, CandidateSet
from .render import (CANDIDATE_TEMP_CAP, CANDIDATE_TINT_CAP, RenderIntent,
                     _BRADFORD, adaptation_matrix, cct_from_temperature,
                     clamp_candidate_intent, daylight_xy)

EPS = 1e-6
#: 中性候选被允许施加修正的最低置信度（低于此值回退 preserve）
NEUTRAL_MIN_CONFIDENCE = 0.50
#: 估计器方向分歧上限（弧度制夹角；超过即视为不可信）
MAX_ESTIMATOR_SPREAD = 0.30

_SRGB2XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041]], np.float64)


# ------------------------------------------------------------- 估计器
def _pixels(positive: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    x = np.asarray(positive, np.float64)
    px = x[mask] if mask is not None else x.reshape(-1, 3)
    return px[np.isfinite(px).all(axis=1)]


def _exclude_clipped(px: np.ndarray, tol: float = 0.004) -> np.ndarray:
    """排除接近剪裁的高光/阴影（v5 §6.6）。"""
    lum = px.mean(axis=1)
    keep = (px.max(axis=1) < 1.0 - tol) & (lum > tol)
    return px[keep] if keep.mean() > 0.15 else px


def shades_of_gray(px: np.ndarray, p: float = 1.0) -> np.ndarray:
    """e_c = (mean(I_c^p))^(1/p)；p=1 即灰世界，p→大趋近白点。"""
    a = np.clip(px, 0.0, None)
    if p == math.inf:
        q = np.quantile(a, 0.995, axis=0)
    else:
        q = np.power(np.mean(np.power(a + EPS, p), axis=0), 1.0 / p)
    return _normalize(q)


def gray_edge(px: np.ndarray, image: np.ndarray,
              mask: np.ndarray | None) -> np.ndarray | None:
    """Gray-Edge：一阶导数的 p 范数估计（需要二维图结构）。"""
    x = np.asarray(image, np.float64)
    if x.ndim != 3:
        return None
    if mask is not None:
        m = mask.astype(bool)
        if m.sum() < 64:
            return None
        x = np.where(m[..., None], x, np.nan)
    gy = np.diff(x, axis=0, prepend=x[:1])
    gx = np.diff(x, axis=1, prepend=x[:, :1])
    out = []
    for c in range(3):
        g = np.hypot(gy[..., c], gx[..., c])
        g = g[np.isfinite(g)]
        if g.size < 32:
            return None
        out.append(float(np.power(np.mean(np.power(g + EPS, 2.0)), 0.5)))
    return _normalize(np.asarray(out, np.float64))


def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, np.float64)
    v = np.where(np.isfinite(v) & (v > EPS), v, EPS)
    gm = float(np.exp(np.mean(np.log(v))))
    return v / max(gm, EPS)


def estimate_illuminants(positive: np.ndarray, mask: np.ndarray | None,
                         image: np.ndarray | None = None) -> dict:
    """多个经典色恒常性估计器（v5 §6.6 要求"比较方向与幅度"）。"""
    px = _exclude_clipped(_pixels(positive, mask))
    if px.shape[0] < 64:
        return {}
    est = {
        "shades_p1": shades_of_gray(px, 1.0),
        "shades_p2": shades_of_gray(px, 2.0),
        "shades_p6": shades_of_gray(px, 6.0),
        "white_patch": shades_of_gray(px, math.inf),
    }
    if image is not None:
        ge = gray_edge(px, image, mask)
        if ge is not None:
            est["gray_edge"] = ge
    return est


def estimator_agreement(est: dict) -> tuple[float, float]:
    """返回 (主族分歧角, 幅度散布)。

    主族 = Shades-of-Gray p=1/2/6（同一假设族，方向本应高度一致）；
    白点与 Gray-Edge 只作为交叉验证，单独在 `cross_spread` 中惩罚——
    它们对纹理/高光分布更敏感，与主族的天然偏差不能当成"分歧过大"。
    """
    fam = [v for k, v in est.items() if k.startswith("shades_")]
    others = [v for k, v in est.items() if not k.startswith("shades_")]
    angs: list[float] = []
    for i in range(len(fam)):
        for j in range(i + 1, len(fam)):
            angs.append(_angle(fam[i], fam[j]))
    spread = max(angs) if angs else 0.0
    if fam and others:
        ref = np.median(np.stack(fam), axis=0)
        cross = sorted(_angle(ref, o) for o in others)
        spread = max(spread, cross[len(cross) // 2] * 0.35)   # 交叉验证降权
    mags = [float(np.max(v) / max(np.min(v), EPS)) for v in est.values()]
    return (float(spread), float(np.median(mags)) if mags else 0.0)


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    c = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + EPS))
    return math.acos(float(np.clip(c, -1.0, 1.0)))


# --------------------------------------------------- 增益 -> 色温/色调
def gain_to_temp_tint(gain: np.ndarray) -> tuple[float, float]:
    """把线性 RGB 对角增益映射到 (temperature, tint) 控件值。

    adaptation_matrix 施加的是 W/D65 增益，因此这里解出的就是"要施加的
    色温/色调"，符号与 v5 §2.2 一致（>0 更暖 / >0 偏绿）。
    """
    g = _normalize(np.asarray(gain, np.float64))
    lms = _BRADFORD @ (_SRGB2XYZ @ g)
    if np.any(lms <= 0) or not np.isfinite(lms).all():
        return 0.0, 0.0
    lms = lms / float(np.exp(np.mean(np.log(lms))))
    d65_lms = _BRADFORD @ _SRGB2XYZ @ np.array([1.0, 1.0, 1.0])
    w_lms = lms * d65_lms
    xyz = np.linalg.inv(_BRADFORD) @ w_lms
    s = float(xyz.sum())
    if s <= 0 or not np.isfinite(s):
        return 0.0, 0.0
    x, y = xyz[0] / s, xyz[1] / s
    temp = _temp_from_xy(x)
    _, y_locus = daylight_xy(cct_from_temperature(temp))
    tint = float(np.clip((y - y_locus) / 0.0009, -CANDIDATE_TINT_CAP,
                         CANDIDATE_TINT_CAP))
    return float(np.clip(temp, -CANDIDATE_TEMP_CAP, CANDIDATE_TEMP_CAP)), tint


def _temp_from_xy(x: float) -> float:
    """日光轨迹 x 单调递减 -> 反解 CCT -> 色温控件值。"""
    lo, hi = 2500.0, 25000.0
    for _ in range(48):
        mid = 0.5 * (lo + hi)
        xm, _ = daylight_xy(mid)
        if xm > x:
            lo = mid
        else:
            hi = mid
    cct = 0.5 * (lo + hi)
    if cct >= 6500.0:
        t = -100.0 * math.log(cct / 6500.0) / math.log(25000.0 / 6500.0)
    else:
        t = 100.0 * math.log(cct / 6500.0) / math.log(2500.0 / 6500.0)
    return float(np.clip(t, -100.0, 100.0))


# ------------------------------------------------------- 场景可信度
def scene_confidence(positive: np.ndarray, mask: np.ndarray | None,
                     est: dict) -> tuple[float, dict]:
    """高饱和/单色主导/夜景/舞台灯/夕阳场景降低置信度（v5 §6.6）。"""
    px = _pixels(positive, mask)
    if px.shape[0] < 64:
        return 0.0, {"reason": "有效像素不足"}
    mx, mn = px.max(axis=1), px.min(axis=1)
    mean = np.maximum(px.mean(axis=1), EPS)
    sat = (mx - mn) / mean
    lum = px.mean(axis=1)
    sat_frac = float((sat > 0.35).mean())
    hi_sat_frac = float((sat > 0.75).mean())
    dark_frac = float((lum < 0.12).mean())
    bright_frac = float((lum > 0.92).mean())
    # 场景里"可当作中性面"的像素比例：这是灰世界类估计能否成立的前提。
    # 比例过低说明画面被饱和色/单色光主导，此时强行中和最危险（v5 §6.6）。
    neutral_frac = float(((sat < 0.15) & (lum > 0.15) & (lum < 0.90)).mean())
    # 色相集中度：把像素投到 a*b* 平面，看主方向占比
    hue = np.arctan2(px[:, 1] - px[:, 2], px[:, 0] - 0.5 * (px[:, 1] + px[:, 2]) + EPS)
    chroma = np.hypot(px[:, 0] - 0.5 * (px[:, 1] + px[:, 2]), px[:, 1] - px[:, 2])
    strong = chroma > np.quantile(chroma, 0.6)
    conc = 0.0
    if strong.any():
        hs = hue[strong]
        bins = np.histogram(hs, bins=12, range=(-math.pi, math.pi))[0]
        conc = float(bins.max() / max(bins.sum(), 1))
    spread, mag = estimator_agreement(est)
    agree_pen = float(np.clip(spread / MAX_ESTIMATOR_SPREAD, 0.0, 1.0))
    sat_pen = float(np.clip((sat_frac - 0.25) / 0.45, 0.0, 0.75))
    mono_pen = float(np.clip((conc - 0.35) / 0.40, 0.0, 0.85))
    night_pen = float(np.clip((dark_frac - 0.35) / 0.45, 0.0, 0.70))
    clip_pen = float(np.clip(max(bright_frac, hi_sat_frac - 0.15) / 0.30, 0.0, 0.5))
    neutral_pen = float(np.clip((0.12 - neutral_frac) / 0.12, 0.0, 0.90))
    conf = ((1.0 - 0.45 * agree_pen) * (1.0 - 0.55 * sat_pen)
            * (1.0 - 0.55 * mono_pen) * (1.0 - 0.50 * night_pen)
            * (1.0 - 0.35 * clip_pen) * (1.0 - 0.75 * neutral_pen))
    stats = {"sat_frac": round(sat_frac, 4), "dark_frac": round(dark_frac, 4),
             "neutral_frac": round(neutral_frac, 4),
             "hue_concentration": round(conc, 4),
             "estimator_spread_rad": round(spread, 4),
             "estimator_magnitude_ratio": round(mag, 4),
             "penalties": {"agreement": round(agree_pen, 3),
                           "saturation": round(sat_pen, 3),
                           "monochrome": round(mono_pen, 3),
                           "night": round(night_pen, 3),
                           "clipping": round(clip_pen, 3),
                           "neutral_pixels": round(neutral_pen, 3)}}
    return float(np.clip(conf, 0.0, 1.0)), stats


# --------------------------------------------------- 卷级稳健汇总
def roll_color_start(per_frame: list[dict]) -> dict:
    """对卷内多帧估计做稳健汇总（v5 §6.6：防止一张特殊场景污染整卷）。

    per_frame 每项 {"frame_id", "gain": (3,), "confidence": float}。
    """
    gains = [p["gain"] for p in per_frame if p.get("gain") is not None]
    confs = [float(p.get("confidence") or 0.0) for p in per_frame]
    if not gains:
        return {"gain": None, "confidence": 0.0, "n_frames": 0, "warnings":
                ["卷内没有任何可用的颜色起点估计"]}
    G = np.asarray(gains, np.float64)
    med = np.median(G, axis=0)
    dev = np.max(np.abs(G - med) / np.maximum(med, EPS), axis=1)
    keep = dev <= 0.35
    if keep.sum() >= 1:
        med = np.median(G[keep], axis=0)
    conf = float(np.median(confs)) if confs else 0.0
    warns = []
    if int((~keep).sum()):
        warns.append(f"卷内 {int((~keep).sum())} 帧的颜色起点与同卷中位偏离过大，"
                     "未参与共享起点")
    return {"gain": _normalize(med).tolist(), "confidence": round(conf, 3),
            "n_frames": int(keep.sum()), "warnings": warns}


# --------------------------------------------------------- 候选生成
@dataclass
class CandidateContext:
    """build_candidates 需要的最小上下文（全部来自同一 service 输出）。"""

    technical_positive: np.ndarray
    valid_mask: np.ndarray | None = None
    oriented_image: np.ndarray | None = None
    output_crop_roi: tuple | None = None
    roll_gain: np.ndarray | None = None          # 卷级稳健起点
    roll_confidence: float = 0.0
    base_source: str = "frame"
    model_prior: dict = field(default_factory=dict)


def build_candidates(result_or_context, *, roll_analysis=None,
                     model=None) -> CandidateSet:
    """生成 preserve / neutral / warm 三个候选（v5 §6.5）。

    入参可以是 service 的 ProcessResult，也可以是 CandidateContext。
    """
    ctx = _as_context(result_or_context)
    est = estimate_illuminants(ctx.technical_positive, ctx.valid_mask,
                               ctx.oriented_image)
    conf, stats = scene_confidence(ctx.technical_positive, ctx.valid_mask, est)
    if not est:
        return _all_preserve(ctx, conf, stats, "有效画面内没有可用统计像素")
    # 融合：以中位估计器为主，卷级起点为锚
    fused = _normalize(np.median(np.stack(list(est.values())), axis=0))
    correction_gain = _normalize(1.0 / fused)
    if ctx.roll_gain is not None:
        w = float(np.clip(ctx.roll_confidence, 0.0, 1.0))
        anchor = _normalize(np.asarray(ctx.roll_gain, np.float64))
        correction_gain = _normalize(correction_gain * (1 - 0.45 * w)
                                     + anchor * (0.45 * w))
    temp, tint = gain_to_temp_tint(correction_gain)
    spread, mag = estimator_agreement(est)
    trusted = (conf >= NEUTRAL_MIN_CONFIDENCE
               and spread <= MAX_ESTIMATOR_SPREAD)
    diag = {"estimators": {k: [round(float(v), 4) for v in val]
                           for k, val in est.items()},
            "correction_gain": [round(float(v), 4) for v in correction_gain],
            "scene": stats, "estimator_spread_rad": round(spread, 4),
            "neutral_min_confidence": NEUTRAL_MIN_CONFIDENCE,
            "max_estimator_spread": MAX_ESTIMATOR_SPREAD,
            "base_source": ctx.base_source,
            "roll_confidence": round(float(ctx.roll_confidence), 3)}
    # --- preserve：卷级颜色起点，不对单帧强行灰世界中和
    preserve = Candidate(
        name="preserve", intent=RenderIntent().to_dict(), confidence=1.0,
        rationale="使用卷级颜色起点，不对单帧做灰世界中和（保留现场氛围）",
        estimated_correction={"temperature": round(temp, 2), "tint": round(tint, 2)},
        applied_correction=None, fallback_to=None,
        estimators={k: "available" for k in est},
        output_crop_roi=ctx.output_crop_roi, base_source=ctx.base_source,
        diagnostics={"scene": stats})
    # --- neutral：仅在置信度足够时施加受限修正
    if trusted:
        raw = RenderIntent(temperature=temp, tint=tint)
        limited, capped = clamp_candidate_intent(raw)
        neutral = Candidate(
            name="neutral", intent=_p0_dict(limited), confidence=round(conf, 3),
            rationale="多估计器方向一致且场景可信，施加受限色温/色调修正",
            estimated_correction={"temperature": round(temp, 2),
                                  "tint": round(tint, 2)},
            applied_correction={"temperature": limited.temperature,
                                "tint": limited.tint},
            correction_capped=bool(capped), fallback_to=None,
            estimators={k: [round(float(v), 4) for v in val]
                        for k, val in est.items()},
            output_crop_roi=ctx.output_crop_roi, base_source=ctx.base_source,
            diagnostics={**diag, "capped": bool(capped)})
    else:
        reasons = []
        if conf < NEUTRAL_MIN_CONFIDENCE:
            reasons.append(f"中性估计置信度 {conf:.2f} < "
                           f"{NEUTRAL_MIN_CONFIDENCE}")
        if spread > MAX_ESTIMATOR_SPREAD:
            reasons.append(f"估计器方向分歧 {spread:.3f} > "
                           f"{MAX_ESTIMATOR_SPREAD}")
        neutral = Candidate(
            name="neutral", intent=RenderIntent().to_dict(),
            confidence=round(conf, 3),
            rationale="；".join(reasons) + "：已回退到保留氛围起点",
            estimated_correction={"temperature": round(temp, 2),
                                  "tint": round(tint, 2)},
            applied_correction=None, fallback_to="preserve",
            estimators={k: [round(float(v), 4) for v in val]
                        for k, val in est.items()},
            output_crop_roi=ctx.output_crop_roi, base_source=ctx.base_source,
            diagnostics=diag)
    # --- warm：在 preserve 或可信 neutral 基础上施加有上限的暖向偏移
    base_intent = neutral if trusted else preserve
    base_temp = float((base_intent.intent or {}).get("temperature", 0.0))
    warm_temp = base_temp + 18.0
    warm_intent = RenderIntent(temperature=warm_temp,
                               tint=float((base_intent.intent or {}).get("tint", 0.0)))
    warm_intent, warm_capped = clamp_candidate_intent(warm_intent)
    warm = Candidate(
        name="warm", intent=_p0_dict(warm_intent),
        confidence=round(min(1.0, base_intent.confidence + 0.05), 3),
        rationale=(f"在{'可信中性' if trusted else '保留氛围'}起点上施加受限暖向偏移"
                   "（仅色温适应，不做任意色相旋转）"),
        estimated_correction={"temperature": round(warm_temp, 2)},
        applied_correction={"temperature": warm_intent.temperature,
                            "tint": warm_intent.tint},
        correction_capped=bool(warm_capped),
        fallback_to=base_intent.name,
        output_crop_roi=ctx.output_crop_roi, base_source=ctx.base_source,
        diagnostics=diag)
    default = _choose_default(preserve, neutral, warm, conf, trusted, ctx)
    cset = CandidateSet(preserve=preserve, neutral=neutral, warm=warm,
                        default_name=default, diagnostics=diag)
    # 运行时引用：三候选共享同一技术正片对象（v5 §6.5 的机器化证明）
    for c in cset:
        c.technical_positive_linear = ctx.technical_positive
    return cset


def _p0_dict(intent: RenderIntent) -> dict:
    return {k: float(getattr(intent, k)) for k in RenderIntent.FIELD_NAMES}


def _choose_default(preserve, neutral, warm, conf, trusted, ctx) -> str:
    """默认候选由卷级统计与置信度决定（v5 §6.5）。"""
    if trusted and conf >= 0.72:
        return "neutral"
    if ctx.roll_confidence >= 0.7 and trusted:
        return "neutral"
    return "preserve"


def _all_preserve(ctx, conf, stats, why) -> CandidateSet:
    note = f"（{why}）三个候选均退化为保留氛围"
    cands = [Candidate(name=n, intent=RenderIntent().to_dict(),
                       confidence=0.0, rationale="无可用统计" + note,
                       fallback_to="preserve",
                       output_crop_roi=ctx.output_crop_roi,
                       base_source=ctx.base_source)
             for n in ("preserve", "neutral", "warm")]
    cset = CandidateSet(preserve=cands[0], neutral=cands[1], warm=cands[2],
                        default_name="preserve",
                        diagnostics={"scene": stats, "reason": why})
    for c in cset:
        c.technical_positive_linear = ctx.technical_positive
    return cset


def _as_context(obj) -> CandidateContext:
    """接受 ProcessResult 或 CandidateContext。"""
    if isinstance(obj, CandidateContext):
        return obj
    pr = obj
    geo = getattr(pr, "geometry", None)
    crop_src = None
    if geo is not None:
        crop_src = tuple(geo.roi_source_space.get("output_crop_roi") or []) or None
    return CandidateContext(
        technical_positive=pr.result.positive_linear,
        valid_mask=getattr(pr, "valid_mask", None),
        oriented_image=getattr(pr, "t", None),
        output_crop_roi=crop_src,
        roll_gain=(roll_gain_of(getattr(pr, "roll_analysis", None))),
        roll_confidence=float(getattr(getattr(pr, "roll_analysis", None),
                                      "confidence", 0.0) or 0.0),
        base_source=getattr(pr, "base_source", "frame"))


def roll_gain_of(roll_analysis) -> list[float] | None:
    """卷级共享起点（增益形式）。"""
    if roll_analysis is None:
        return None
    start = roll_analysis.shared_color_start or {}
    g = start.get("gain")
    return list(g) if g else None


def candidate_intent(candidate: Candidate) -> RenderIntent:
    """候选 -> RenderIntent（统一走 service 的渲染出口）。"""
    return RenderIntent.from_dict(candidate.intent)
