"""片基 Dmin、Dmax 与扫描曝光偏置的稳健估计（规格 §4.2.3、§4.5.3–4）。"""

from __future__ import annotations

import numpy as np

from ..pipeline.density import EPS, transmittance_to_z


def robust_dmin(base_roi: np.ndarray, trim: float = 0.10, dust_tol: float = 0.35) -> dict:
    """在未曝光片基 ROI 上计算每通道稳健基准（§4.2.3）。

    步骤：
      1. 剔除 NaN/Inf；
      2. 尘点/划痕剔除：负片片基是画面最亮区，尘点和划痕使其局部变暗，
         故剔除任一通道偏离中位数超过 dust_tol 比例的像素；
      3. 每通道去掉两端 trim 百分位后取截尾均值作为 Dmin。

    返回 dict: dmin(3,), confidence in [0,1], warnings(list), stats。
    """
    pix = np.asarray(base_roi, dtype=np.float32).reshape(-1, 3)
    warnings: list[str] = []
    finite = np.isfinite(pix).all(axis=1)
    n_nonfinite = int((~finite).sum())
    pix = pix[finite]
    n = pix.shape[0]
    if n < 16:
        return {
            # 回退透射率基准 0.5（无信息可用；置信度 0，必须人工介入）
            "dmin": np.full(3, 0.5, dtype=np.float32),
            "confidence": 0.0,
            "warnings": [f"片基 ROI 有效像素过少（{n}），Dmin 使用回退值，结果不可信"],
            "stats": {"n": n},
        }

    med = np.median(pix, axis=0)
    med_safe = np.maximum(med, EPS)
    dev = np.abs(pix - med) / med_safe
    keep = (dev <= dust_tol).all(axis=1)
    kept_frac = float(keep.mean())
    kept = pix[keep]
    if kept.shape[0] < 16:  # 容差过严时退回全部有限像素
        kept = pix
        kept_frac = 1.0
        warnings.append("片基 ROI 尘点剔除容差过严，已退回全 ROI 截尾统计")

    lo = np.quantile(kept, trim, axis=0)
    hi = np.quantile(kept, 1.0 - trim, axis=0)
    inside = ((kept >= lo) & (kept <= hi)).all(axis=1)
    core = kept[inside] if inside.sum() >= 8 else kept
    dmin = core.mean(axis=0).astype(np.float32)
    dmin = np.maximum(dmin, EPS)

    # 相对散布（MAD/median）：片基应均匀，散布大 => 污染或 ROI 放错
    mad = np.median(np.abs(core - np.median(core, axis=0)), axis=0) / med_safe
    rel_spread = float(np.nanmax(mad))

    size_score = float(min(1.0, core.shape[0] / 2000.0))
    purity_score = float(np.clip((kept_frac - 0.30) / 0.50, 0.0, 1.0))
    spread_score = float(np.clip((0.15 - rel_spread) / 0.12, 0.0, 1.0))
    confidence = round(min(size_score, purity_score, spread_score), 3)

    if n_nonfinite:
        warnings.append(f"片基 ROI 含 {n_nonfinite} 个 NaN/Inf，已剔除")
    if kept_frac < 0.7:
        warnings.append(f"片基 ROI 污染比例高（保留 {kept_frac:.0%}），Dmin 置信度降低")
    if rel_spread > 0.10:
        warnings.append(f"片基通道散布偏大（{rel_spread:.3f}），ROI 可能包含非片基内容")
    if confidence < 0.5:
        warnings.append(f"片基置信度低（{confidence}），请手动确认片基取样框")

    return {
        "dmin": dmin,
        "confidence": confidence,
        "warnings": warnings,
        "stats": {
            "n": n, "kept_frac": round(kept_frac, 3),
            "rel_spread": round(rel_spread, 4), "median": [float(v) for v in med],
        },
    }


def estimate_dmax_offset(film_roi: np.ndarray, dmin: np.ndarray,
                         black: float = 0.0) -> dict:
    """从画面区域估计 Dmax 与扫描曝光偏置（§4.5.4）。

    - Dmax：在亮度域 z_lum=mean_c(z_c) 上统计；剔除已剪裁到零的像素
      （t<=1e-5，密度信息丢失，不得进入统计），取 z_lum 的 2 分位的相反数；
      限制在 [0.3, 4.0]，防止零/负值进入模型。
    - offset：令画面最暗 0.5% 像素（z 接近 0 的一侧）恰好落到纸黑：
      offset = log10(1+black) - z_p99.5 / Dmax，截断到 [-0.5, 1.0]。
      这是启发式初值，用户可手动覆盖。
    - 输入剪裁风险：t 低于传感器黑底或达到满阱的像素比例。
    """
    t = np.asarray(film_roi, dtype=np.float32)
    t, n_bad = _finite(t)
    z = transmittance_to_z(t, dmin)
    # 亮度域统计：逐通道分位会被黑边带/片边缘标记等单通道超暗污染
    # （真实样片实测逐通道 2.96 vs 亮度域 1.89）
    z_lum = z.mean(axis=2)
    warnings: list[str] = []

    valid = (t > 1e-5).all(axis=-1)  # 剔除零剪裁像素，防止 Dmax 饱和
    if valid.mean() < 0.5:
        warnings.append("画面超过 50% 像素被剪裁到零，Dmax 估计不可靠")
    zl = z_lum[valid] if valid.any() else z_lum.ravel()
    dmax = float(np.clip(-float(np.quantile(zl, 0.02)), 0.3, 4.0))
    if n_bad:
        warnings.append(f"画面 ROI 含 {n_bad} 个 NaN/Inf，已替换为 0")

    # 接近片基（最暗场景）。钳制 <=0：齿孔/光源直射区比片基更亮（z>0），
    # 若不钳制会把 offset 拉负、抬升黑位，导致正片发灰（真实样片实测教训）。
    z_base = min(float(np.quantile(zl, 0.995)), 0.0)
    offset = float(np.clip(np.log10(1.0 + max(black, 0.0)) - z_base / dmax,
                           -0.5, 1.0))

    clip_low = float((t <= 1e-4).mean())
    clip_high = float((t >= 1.0 - 1e-4).mean())
    if clip_low > 0.01:
        warnings.append(f"输入暗部剪裁比例 {clip_low:.2%}（透射率趋零），密度信息可能丢失")
    if clip_high > 0.01:
        warnings.append(f"输入亮部剪裁比例 {clip_high:.2%}（扫描饱和），片基估计可能失真")

    # 置信度：Dmax 落在合理区间中部且无严重剪裁
    range_score = float(np.clip(1.0 - abs(dmax - 2.0) / 2.0, 0.0, 1.0))
    clip_score = float(np.clip(1.0 - 10.0 * max(clip_low, clip_high), 0.0, 1.0))
    confidence = round(min(range_score, clip_score), 3)
    if confidence < 0.5:
        warnings.append(f"Dmax/offset 置信度低（{confidence}），建议手动调整")

    return {
        "dmax": dmax,
        "offset": offset,
        "confidence": confidence,
        "warnings": warnings,
        "input_clip": {"low": clip_low, "high": clip_high},
    }


def base_roi_sanity_check(t_full: np.ndarray, base_roi: np.ndarray) -> list[str]:
    """检查所选片基 ROI 是否可能是片基：片基应是全画面最亮区域之一。

    若 ROI 中位数明显低于全图高亮分位，提示 ROI 可能落在画面内容上（§8.1.5）。
    """
    warnings: list[str] = []
    full = _finite(np.asarray(t_full, dtype=np.float32))[0].reshape(-1, 3)
    roi = _finite(np.asarray(base_roi, dtype=np.float32))[0].reshape(-1, 3)
    if full.size == 0 or roi.size == 0:
        return ["片基检查失败：ROI 或全图为空"]
    lum_full = np.quantile(full.mean(axis=1), 0.99)
    lum_roi = float(np.median(roi.mean(axis=1)))
    if lum_roi < 0.8 * max(lum_full, EPS):
        warnings.append(
            f"片基 ROI 亮度（{lum_roi:.3f}）低于全图 99 分位亮度的 80%"
            f"（{lum_full:.3f}），ROI 可能不在未曝光片基上")
    return warnings


def _finite(a: np.ndarray) -> tuple[np.ndarray, int]:
    bad = ~np.isfinite(a)
    n = int(bad.sum())
    if n:
        a = a.copy()
        a[bad] = 0.0
    return a, n
