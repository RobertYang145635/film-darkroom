"""卷级片基与共享颜色起点（v5 §6.2）。

三件事，全部**不需要色卡**：

1. 每帧独立检测片基候选并计算稳健统计（中位数、MAD、有效像素比例、
   空间趋势、污染比例）；
2. 在同一 `capture_fingerprint` 内汇总为卷级片基模型，拒绝低置信度、
   齿孔、边码与异常帧；跨帧漂移超过**预先冻结**的容差时停止共享并建议拆卷；
3. 本帧没有可靠片基时自动使用同卷模型；整卷都失败时回退到版本化的
   "未知彩色负片"通用先验，并如实标记 `base_source=generic_prior`。

通用先验是明确标记的**非物理回退**，不得伪装成测得片基（§6.1.1/§6.2.6）。
"""

from __future__ import annotations

import math

import numpy as np

from ..library.models import RollAnalysis

EPS = 1e-6

#: 片基取样区污染判定的密度域容差（约 2 倍透射率）：齿孔/边码远大于此
DENSITY_TOL = 0.30

# --------------------------------------------------------------- 冻结容差
# v5 §4.4：容差必须预先冻结，不能因为某卷失败就当场放宽。
# 提升任何数值都必须同时提升 TOLERANCE_VERSION 并留下书面依据。
TOLERANCE_VERSION = "v5-p0-frozen-2026-09-01-r2"
TOLERANCE = {
    "min_base_pixels": 400,          # 密度域过滤后保留的最少像素
    "min_base_confidence": 0.45,     # 帧级片基置信度下限
    "max_base_rel_drift": 0.18,      # 卷内片基亮度漂移上限（decade）
    "max_base_chroma_drift": 0.16,   # 卷内片基色度（log10 通道比）漂移上限
    "max_black_level_drift": 0.35,   # 黑电平相对漂移
    "max_spatial_trend": 0.12,       # 归一化空间趋势斜率上限
    "max_contamination": 0.80,       # 取样区污染比例上限（齿孔带可到 ~0.8）
    "min_roll_frames": 2,            # 卷级模型最少可靠帧数
}

# ------------------------------------------------------------- 通用先验
#: 下沉到叶子 constants.py（io/sidecar.py 也要用，且不能再指向 pipeline）。
#: 此处显式自别名重导出保持旧路径可用；⚠ 2026-09-08 冷读发现此前这里是
#: 一份**独立复制的字面量**而非别名（两处会各自漂移），已改为真别名。
from ..constants import GENERIC_PRIOR as GENERIC_PRIOR  # noqa: F401,E402


# ------------------------------------------------------------- 帧级统计
def _roi_stats(seg: np.ndarray) -> dict:
    """单个片基取样区的稳健统计。

    中心用 q75（不是中位数）：未曝光片基物理上是条带里**最亮的均匀区**，
    齿孔/边码/划痕只会更暗。以 q75 为中心再做密度域过滤，可以剔除占面积
    很大的齿孔带而仍得到正确 Dmin；污染比例如实报告并压低置信度。

    过滤在**密度域**做（|log10 t - log10 center|）：片基是乘性噪声，橙罩下
    B 通道透射率可低至 0.03，用相对透射率容差会把正常颗粒误判成污染
    （真实卷实测 47-93% "污染" 全是误判，导致整卷退化到通用先验）。
    """
    pix = np.asarray(seg, np.float32).reshape(-1, 3)
    finite = np.isfinite(pix).all(axis=1)
    n_nonfinite = int((~finite).sum())
    pix = pix[finite]
    out = {"n": int(pix.shape[0]), "n_nonfinite": n_nonfinite,
           "median": None, "mad": None, "kept_frac": 0.0,
           "contamination": 1.0, "trend": (0.0, 0.0), "confidence": 0.0,
           "warnings": []}
    if pix.shape[0] < 16:
        out["warnings"].append("片基取样区有效像素过少")
        return out
    center = np.maximum(np.quantile(pix, 0.75, axis=0), EPS)
    log_t = np.log10(np.maximum(pix, EPS))
    log_c = np.log10(center)
    ddev = np.abs(log_t - log_c)                  # 密度域偏差（decade）
    keep = ddev.max(axis=1) <= DENSITY_TOL
    kept_frac = float(keep.mean())
    kept = pix[keep]
    valid2d = keep.reshape(np.asarray(seg).shape[:2])
    if kept.shape[0] < 16:
        # 兜底：放宽到最亮 25% 像素（宁可置信度低，也不退回无信息先验）
        order = np.argsort(log_t.mean(axis=1))[::-1]
        sel = order[:max(16, pix.shape[0] // 4)]
        kept = pix[sel]
        kept_frac = float(kept.shape[0] / pix.shape[0])
        valid2d = np.zeros(np.asarray(seg).shape[:2], bool)
        valid2d.reshape(-1)[sel] = True
        out["warnings"].append("片基取样区高度污染：已退回最亮 25% 像素估计")
    med = np.median(kept, axis=0)
    med_safe = np.maximum(med, EPS)
    mad = np.median(np.abs(np.log10(np.maximum(kept, EPS))
                           - np.log10(med_safe)), axis=0)
    rel_spread = float(np.nanmax(mad))
    # 空间趋势：只在密度域过滤后的片基像素上拟合（矩形含暗帧界会误报）
    trend = _spatial_trend(seg, valid2d)
    out.update({
        "median": [float(v) for v in med],
        "mad": [float(v) for v in mad],
        "kept_frac": round(kept_frac, 4),
        "contamination": round(1.0 - kept_frac, 4),
        "trend": trend,
        "rel_spread": round(rel_spread, 4),
    })
    size_score = float(min(1.0, kept.shape[0] / TOLERANCE["min_base_pixels"]))
    purity_score = float(np.clip((kept_frac - 0.20) / 0.60, 0.0, 1.0))
    spread_score = float(np.clip((0.10 - rel_spread) / 0.08, 0.0, 1.0))
    out["confidence"] = round(min(size_score, purity_score, spread_score), 3)
    # 三项分项原样写出来：置信度是 min(三项)，光看一个 0.30 分不清是
    # "样本太少"还是"太脏"还是"颗粒太散"（2026-09-09 实测：44% 帧退回通用
    # 先验，全部是 spread_score 被 B 通道噪声压死，但报表上只有一个 0.00）。
    out["scores"] = {"size": round(size_score, 3), "purity": round(purity_score, 3),
                     "spread": round(spread_score, 3)}
    if n_nonfinite:
        out["warnings"].append(f"片基取样区含 {n_nonfinite} 个非有限值，已剔除")
    if kept_frac < 1.0 - TOLERANCE["max_contamination"]:
        out["warnings"].append(
            f"片基取样区污染比例高（保留 {kept_frac:.0%}）："
            "可能框到齿孔、边码或画面内容")
    return out


def _spatial_trend(seg: np.ndarray, valid: np.ndarray | None = None) -> tuple[float, float]:
    """归一化一阶空间斜率（log 域），用于检测背光/导光不均。

    只在**通过密度域过滤的片基像素**上拟合：原始矩形常含一条暗帧界/齿孔
    边，混入后趋势必然超上限（真实卷实测 -0.216），把整卷错误地判为
    "采集条件不一致" 而拒绝共享。
    """
    a = np.asarray(seg, np.float64)
    if a.ndim != 3 or a.shape[0] < 4 or a.shape[1] < 4:
        return (0.0, 0.0)
    lum = np.log(np.maximum(a.mean(axis=2), EPS))
    if valid is not None:
        m = np.asarray(valid, bool) & np.isfinite(lum)
    else:
        m = np.isfinite(lum)
    if m.sum() < 64:
        m = np.isfinite(lum)
    h, w = lum.shape
    yy = (np.arange(h)[:, None] / max(h - 1, 1)) * np.ones((1, w))
    xx = np.ones((h, 1)) * (np.arange(w)[None, :] / max(w - 1, 1))
    A = np.column_stack([np.ones(m.sum()), yy[m], xx[m]])
    try:
        coef, *_ = np.linalg.lstsq(A, lum[m], rcond=None)
    except np.linalg.LinAlgError:
        return (0.0, 0.0)
    return (float(np.clip(coef[1] / max(abs(coef[0]), 1e-6), -2, 2)),
            float(np.clip(coef[2] / max(abs(coef[0]), 1e-6), -2, 2)))


def frame_base_stats(t_oriented: np.ndarray, base_rois: list,
                     image_area_roi=None) -> dict:
    """一帧的片基统计：多取样区独立统计 -> 稳健合并 -> 帧级结论。

    返回 {dmin, confidence, n_regions_used, regions, contamination,
          spatial_trend, black_level, warnings, rejected}。
    """
    t = np.asarray(t_oriented, np.float32)
    h, w = t.shape[:2]
    regions, used, warnings = [], [], []
    for i, roi in enumerate(base_rois or []):
        y0, x0, y1, x1 = (int(v) for v in roi)
        y0, x0 = max(0, y0), max(0, x0)
        y1, x1 = min(h, y1), min(w, x1)
        if y1 - y0 < 4 or x1 - x0 < 4:
            warnings.append(f"片基取样区 {i} 越界或过小，已忽略")
            continue
        seg = t[y0:y1, x0:x1]
        st = _roi_stats(seg)
        st["roi"] = (y0, x0, y1, x1)
        regions.append(st)
        if st["median"] is None:
            warnings.append(f"片基取样区 {i} 无有效像素")
            continue
        if st["confidence"] < TOLERANCE["min_base_confidence"]:
            warnings.append(f"片基取样区 {i} 置信度过低"
                            f"（{st['confidence']}）：污染/散布/样本量不达标，已拒绝")
            continue
        used.append(st)
    if not used:
        out = {"dmin": None, "confidence": 0.0, "n_regions_used": 0,
               "regions": regions, "contamination": 1.0,
               "spatial_trend": (0.0, 0.0), "black_level": None,
               "warnings": warnings or ["没有可用的片基取样区"],
               "rejected": [r["roi"] for r in regions]}
        # 把**被拒绝的那个区的真实数字**带出来（判定一个字不改：dmin 仍是
        # None、contamination 仍是 1.0 —— 后者是 load-bearing 的，卷级模型
        # 靠它把不可靠帧排除在外）。以前这里一律报 contamination=1.000 /
        # confidence=0.000，于是"根本没量到片基"和"片基 100% 脏"在报表、
        # sidecar、判定条上长成同一个样子，用户与后续审查都无从分辨。
        best = max(regions, key=lambda r: r["confidence"], default=None)
        if best is not None and best.get("median") is not None:
            scores = best.get("scores") or {}
            binding = min(scores, key=scores.get) if scores else None
            out["best_rejected_region"] = {
                "roi": best["roi"], "confidence": best["confidence"],
                "contamination": best["contamination"],
                "kept_frac": best["kept_frac"],
                "rel_spread": best.get("rel_spread"),
                "mad": [round(float(v), 4) for v in (best.get("mad") or [])],
                "scores": scores, "binding_score": binding,
                "median_transmittance": [round(float(v), 6)
                                         for v in best["median"]],
                "note": "该区未通过置信度门槛，**未参与**任何计算；数字仅作诊断",
            }
        return out
    meds = np.array([r["median"] for r in used], np.float64)
    confs = np.array([r["confidence"] for r in used], np.float64)
    # 多区合并：取各区中位数的中位数（对单区被齿孔/边码污染稳健）
    if len(used) == 1:
        dmin = meds[0]
    else:
        dmin = np.array([np.median(meds[:, c]) for c in range(3)], np.float64)
    dmin = np.maximum(dmin, EPS)
    # 区与区之间的一致性同样在密度域衡量（相对透射率会误判 B 通道）
    spread = float(np.max(np.ptp(np.log10(np.maximum(meds, EPS)), axis=0)))
    contamination = float(np.mean([r["contamination"] for r in used]))
    trend = (float(np.mean([r["trend"][0] for r in used])),
             float(np.mean([r["trend"][1] for r in used])))
    conf = float(np.clip(np.mean(confs) * (1.0 - min(spread / 0.20, 0.5)), 0.0, 1.0))
    if spread > 0.15:
        warnings.append(f"多个片基取样区之间密度差偏大（{spread:.2f} decade），"
                        "请检查是否框到齿孔、边码或画面内容")
    black = _black_level(t, image_area_roi)
    return {"dmin": [float(v) for v in dmin], "confidence": round(conf, 3),
            "n_regions_used": len(used), "regions": regions,
            "contamination": round(contamination, 4),
            "spatial_trend": [round(v, 4) for v in trend],
            "black_level": black, "warnings": warnings, "rejected": []}


def _black_level(t: np.ndarray, image_area_roi=None) -> float | None:
    """画面内低分位透射率（黑电平/导光底），用于跨帧漂移检查。"""
    a = np.asarray(t, np.float32)
    if image_area_roi:
        y0, x0, y1, x1 = (int(v) for v in image_area_roi)
        a = a[max(0, y0):a.shape[0], max(0, x0):a.shape[1]]
        a = a[:min(a.shape[0], max(0, y1) - max(0, y0)),
              :min(a.shape[1], max(0, x1) - max(0, x0))]
    if a.size == 0:
        return None
    lum = a.mean(axis=2).ravel()
    lum = lum[np.isfinite(lum)]
    if lum.size < 64:
        return None
    return float(np.quantile(lum, 0.002))


# ------------------------------------------------------------- 卷级汇总
def analyze_roll(frames: list[dict], *, fingerprint: str = "",
                 tolerance: dict | None = None) -> RollAnalysis:
    """把同采集指纹的多帧片基统计汇总为卷级模型（v5 §6.2.2）。

    `frames` 每项至少含：frame_id、capture_fingerprint、base_dmin、
    base_confidence，可选 black_level、spatial_trend、contamination。
    """
    tol = dict(TOLERANCE)
    if tolerance:
        unknown = set(tolerance) - set(TOLERANCE)
        if unknown:
            raise ValueError(f"未知容差键：{sorted(unknown)}")
        tol.update(tolerance)
    ra = RollAnalysis(capture_fingerprint=fingerprint,
                      analysis_version=1)
    warnings: list[str] = []
    pool, rejected = [], []
    for f in frames:
        fid = str(f.get("frame_id") or f.get("file") or "")
        fp = str(f.get("capture_fingerprint") or "")
        if fingerprint and fp and fp != fingerprint:
            rejected.append(fid)
            warnings.append(f"帧 {fid} 采集指纹与本卷不一致，不参与卷级共享")
            continue
        dmin = f.get("base_dmin")
        if not dmin:
            rejected.append(fid)
            warnings.append(f"帧 {fid} 无片基测量，不参与卷级汇总")
            continue
        if float(f.get("base_confidence") or 0.0) < tol["min_base_confidence"]:
            rejected.append(fid)
            warnings.append(f"帧 {fid} 片基置信度过低"
                            f"（{f.get('base_confidence')}），已拒绝")
            continue
        if float(f.get("contamination") or 0.0) > tol["max_contamination"]:
            rejected.append(fid)
            warnings.append(f"帧 {fid} 片基取样污染比例过高，已拒绝")
            continue
        pool.append((fid, np.maximum(np.asarray(dmin, np.float64), EPS), f))
    if not pool:
        ra.warnings = warnings + ["本卷没有任何可靠片基帧：使用通用先验回退"]
        ra.base_source = "generic_prior"
        ra.confidence = 0.0
        ra.rejected_frames = rejected
        ra.shared_with_frames = []
        return ra
    meds = np.stack([p[1] for p in pool])
    ref = np.median(meds, axis=0)
    lum_ref = float(ref.mean())
    accepted, drift_rejected = [], []
    for fid, med, f in pool:
        rel_lum = abs(math.log10(float(med.mean()) / max(lum_ref, EPS)))
        # 色度漂移同样在密度域衡量：log10(通道/通道均值) 的差
        shape_c = np.log10(np.maximum(med / max(med.mean(), EPS), EPS))
        ref_c = np.log10(np.maximum(ref / max(ref.mean(), EPS), EPS))
        rel_chroma = float(np.max(np.abs(shape_c - ref_c)))
        reasons = []
        if rel_lum > tol["max_base_rel_drift"]:
            reasons.append(f"片基亮度漂移 {rel_lum:.2f} decade")
        if rel_chroma > tol["max_base_chroma_drift"]:
            reasons.append(f"片基色度漂移 {rel_chroma:.2f} decade")
        bl = f.get("black_level")
        others = [g.get("black_level") for _, _, g in pool
                  if g.get("black_level") is not None]
        if bl is not None and others:
            ref_bl = float(np.median(others))
            if ref_bl > 0 and abs(bl - ref_bl) / ref_bl > tol["max_black_level_drift"]:
                reasons.append(f"黑电平漂移 {abs(bl - ref_bl) / ref_bl:.2f}")
        tr = f.get("spatial_trend")
        if tr and max(abs(float(v)) for v in tr) > tol["max_spatial_trend"]:
            reasons.append("空间趋势超上限")
        if reasons:
            drift_rejected.append(fid)
            rejected.append(fid)
            warnings.append(f"帧 {fid} 与同卷其他帧采集条件不一致"
                            f"（{'、'.join(reasons)}）：不共享卷级片基")
        else:
            accepted.append(fid)
    keep = [i for i, (fid, _, _) in enumerate(pool) if fid in accepted]
    if len(keep) >= 1:
        meds2 = meds[keep]
        roll_dmin = np.median(meds2, axis=0)
        mad = np.median(np.abs(meds2 - roll_dmin), axis=0) / np.maximum(roll_dmin, EPS)
        conf = float(np.clip(min(1.0, len(keep) / max(tol["min_roll_frames"], 1))
                             * (1.0 - float(np.nanmax(mad)) / 0.3), 0.0, 1.0))
        ra.base_statistics = {
            "dmin": [round(float(v), 6) for v in roll_dmin],
            "mad": [round(float(v), 4) for v in mad],
            "n_frames": int(len(keep)),
            "per_frame": {fid: [round(float(v), 6) for v in med]
                          for fid, med, _ in pool if fid in accepted},
            "tolerance_version": TOLERANCE_VERSION,
        }
        ra.confidence = round(conf, 3)
        ra.shared_with_frames = accepted
        ra.base_source = "roll" if len(keep) >= tol["min_roll_frames"] else "roll_partial"
        # 共同曝光尺度：卷级片基的亮度中位（§6.2.5，供通用先验回退使用）
        ra.generic_scale = float(np.clip(np.median(np.sum(roll_dmin)), 0.02, 0.98))
    else:
        ra.base_statistics = {"tolerance_version": TOLERANCE_VERSION}
        ra.base_source = "generic_prior"
        ra.confidence = 0.0
        scales = [float(f.get("generic_scale") or 0.0) for f in frames
                  if f.get("generic_scale")]
        ra.generic_scale = float(np.clip(np.median(scales), 0.02, 0.98)) \
            if scales else None
    ra.rejected_frames = rejected
    ra.source_frames = accepted
    if drift_rejected:
        ra.split_suggested = True
        warnings.append(
            f"检测到 {len(drift_rejected)} 帧与同卷采集条件不一致，建议一键拆成"
            "不同卷后再转换（v5 §4.4：不得静默共享错误参数）")
    ra.warnings = warnings
    return ra


# --------------------------------------------------------- 帧/卷/先验解析
def resolve_base(frame_stats: dict | None, roll: RollAnalysis | None,
                 *, t_oriented: np.ndarray | None = None,
                 image_area_roi=None, frame_id: str | None = None) -> dict:
    """按 v5 §6.2.3-6 的优先级决定本帧实际使用的片基。

    返回 {dmin, base_source, confidence, warnings, provenance}。

    `frame_id`：本帧是否参与卷级模型由 `roll.shared_with_frames` 决定；
    未参与（漂移/指纹冲突）的帧**不**使用卷级 dmin，但仍继承本卷共同
    曝光尺度（§6.2.5），避免同卷颜色起点漂移。
    """
    warnings: list[str] = []
    prov: dict = {}
    fs = frame_stats or {}
    frame_ok = (fs.get("dmin") and float(fs.get("confidence") or 0)
                >= TOLERANCE["min_base_confidence"])
    in_roll_model = (frame_id is None or not roll
                     or frame_id in roll.shared_with_frames)
    roll_ok = (roll is not None and roll.base_statistics.get("dmin")
               and float(roll.confidence) >= TOLERANCE["min_base_confidence"]
               and in_roll_model)
    if frame_ok and roll_ok:
        f = np.asarray(fs["dmin"], np.float64)
        r = np.asarray(roll.base_statistics["dmin"], np.float64)
        wf = float(fs["confidence"])
        wr = float(roll.confidence) * 0.85
        dmin = (f * wf + r * wr) / max(wf + wr, EPS)
        warnings.append("片基来源：本帧取样与卷级模型加权")
        prov = {"frame": [round(float(v), 6) for v in f],
                "roll": [round(float(v), 6) for v in r],
                "weights": [round(wf, 3), round(wr, 3)]}
        source = "frame+roll"
        conf = min(1.0, wf + 0.1)
    elif frame_ok:
        dmin = np.asarray(fs["dmin"], np.float64)
        source, conf = "frame", float(fs["confidence"])
        warnings.append("片基来源：本帧未曝光片基取样")
    elif roll_ok:
        dmin = np.asarray(roll.base_statistics["dmin"], np.float64)
        source = "roll"
        conf = float(roll.confidence)
        prov = {"source_frames": list(roll.source_frames)}
        warnings.append(
            f"本帧没有可靠片基，已使用同卷 {len(roll.source_frames)} 帧的卷级"
            f"片基模型（来源帧：{', '.join(roll.source_frames[:4])}"
            f"{'…' if len(roll.source_frames) > 4 else ''}）")
    else:
        dmin, conf = generic_base(t_oriented, image_area_roi,
                                  scale=getattr(roll, "generic_scale", None)
                                  if roll is not None else None)
        source = "generic_prior"
        scale_note = (f"（本卷共同曝光尺度 {getattr(roll, 'generic_scale', 0):.3f}）"
                      if roll is not None and getattr(roll, "generic_scale", None)
                      else "")
        warnings.append(
            "本帧与本卷都没有可用片基：使用版本化通用彩色负片先验给出保守预览"
            f"{scale_note}（{GENERIC_PRIOR['asset_version']}）。这不是实测 Dmin，"
            "结果标记为待复核。")
    return {"dmin": [float(v) for v in dmin], "base_source": source,
            "confidence": round(float(conf), 3), "warnings": warnings,
            "provenance": prov}


def generic_base(t_oriented: np.ndarray | None, image_area_roi=None,
                 scale: float | None = None) -> tuple[np.ndarray, float]:
    """通用先验回退：只从画面统计估计**共同曝光尺度**，色度形状固定。

    `scale` 由卷级分析提供（v5 §6.2.5：同一卷共用一个尺度，防止同卷
    颜色起点漂移）；没有卷级信息时才从本帧画面统计估计。
    """
    ratios = np.asarray(GENERIC_PRIOR["ratios"], np.float64)
    if scale:
        return ratios * float(np.clip(scale, 0.02, 0.98)), 0.0
    scale = 0.30
    if t_oriented is not None:
        a = np.asarray(t_oriented, np.float32)
        if image_area_roi:
            y0, x0, y1, x1 = (int(v) for v in image_area_roi)
            a = a[max(0, y0):min(a.shape[0], y1), max(0, x0):min(a.shape[1], x1)]
        if a.size:
            lum = a.mean(axis=2).ravel()
            lum = lum[np.isfinite(lum)]
            if lum.size >= 256:
                scale = float(np.quantile(lum, 0.995))
    scale = float(np.clip(scale, 0.02, 0.98))
    return ratios * scale, 0.0
