"""片基 Dmin、Dmax 与扫描曝光偏置的稳健估计（规格 §4.2.3、§4.5.3–4）。"""

from __future__ import annotations

import numpy as np

from ..metrics import EPS, transmittance_to_z   # 叶子：消除 calibration -> pipeline 反向依赖


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


#: 画面内片基检测（橙色签名法）的参数。每一个数都是 2026-10-11 在
#: `1299_G200` 的 **11 帧用户亲手片基框**（绝对真值）上量出来的，
#: 见 `docs/blockers.md` B-15 追加 与 `tmp_orange_base_estimator.py` 的打印。
ORANGE_BASE = {
    #: 候选判据 `t_R/t_G ≥`：实测片基的 R/G 在 2.6 左右（11 帧 2.53~2.69），
    #: 取 1.8 留出余量，同时把中性白漏光（R/G≈1）挡在门外。
    "rg_min": 1.8,
    #: 候选判据 `t_R/t_B ≥`：实测片基 R/B = 8.3~9.4，取 5.0。
    #: **这条是承重的**：去掉橙色检验只按"亮"选，蓝通道误差中位从 0.073
    #: 恶化到 0.141、最差一帧从 0.160 恶化到 **0.568**（灯台漏光被当成了片基）。
    "rb_min": 5.0,
    #: 亮度门槛：低于 R 通道 p99 的这个比例就不算"片基一侧"，
    #: 防止把画面里恰好偏橙的暗部（秋叶、红车）选进来。
    "bright_frac": 0.25,
    #: 取候选集的哪一分位作为片基。
    "percentile": 99.9,
    #: 候选像素至少要占画面区的这个比例，否则视为"本帧看不到片基"。
    "min_candidate_frac": 0.002,
    #: 实测留出误差（decade）：R 0.021 / G 0.029 / B 0.073（中位）。
    #: 置信度据此给常数，不装成统计量——它表达的是"这个量级已被真值验证过"。
    "confidence": 0.6,
}


def above_base_fraction(t_oriented: np.ndarray, dmin, image_area_roi=None) -> list[float]:
    """画面区里**透射率高于 assumed 片基**的像素比例（逐通道）。

    这是一个**物理判据**，不是统计判据：负片里密度 ≥ 0，所以任何像素的透射率
    都不可能高于片基；这个比例显著大于 0 就证明 assumed 片基估低了。
    实测（`tmp_roll_inherit_contradiction.py`，23 帧）：正常帧 ≤ 0.03，
    唯一与卷模型矛盾且无自测片基的 `1298::G200-7` 是 R 0.259 / G 0.342 / B 0.396。
    """
    a = np.asarray(t_oriented, np.float64)
    if a.ndim != 3 or a.shape[-1] < 3:
        return [0.0, 0.0, 0.0]
    if image_area_roi:
        y0, x0, y1, x1 = (int(v) for v in image_area_roi)
        a = a[max(0, y0):min(a.shape[0], y1), max(0, x0):min(a.shape[1], x1)]
    pix = a.reshape(-1, 3)[:, :3]
    pix = pix[np.isfinite(pix).all(axis=1) & (pix.max(axis=1) > 1e-6)]
    if not pix.shape[0]:
        return [0.0, 0.0, 0.0]
    d = np.maximum(np.asarray(dmin, np.float64), 1e-9)
    return [float((pix[:, c] > d[c]).mean()) for c in range(3)]


def base_from_orange_signature(t_oriented: np.ndarray,
                               image_area_roi=None,
                               params: dict | None = None) -> dict:
    """**画面里没有片边时，从画面里把片基测出来**（负片的橙色色罩自己会签名）。

    为什么这不是"拿场景内容拟合片基"：负片里透射率**高于片基**的像素物理上不存在，
    所以"亮"给出的是片基的上界；而片基同时是三通道里**橙色最纯**的亮区
    （实测 R/G≈2.6、R/B≈9），翻拍台上的漏光与灯丝是**中性白**（R/G≈1），
    过不了比值闸。两条判据取交集后，候选集里剩下的就是
    "未被遮挡的片基本身"（未曝光片边、密度极低的场景高光区）。

    实测（`tmp_orange_base_estimator.py`，11 帧用户亲手框为真值，误差=
    `log10(估计/真值)` 的中位绝对值）：R **0.021** / G **0.029** / B **0.073**
    decade，最差一帧 0.16；在**整卷生产路径**上重量的同批帧是 R 0.021 / G 0.024 /
    B 0.059（`tmp_roll_inherit_contradiction.py`）。同一批帧上现行回退（通用先验）的
    误差是 **0.34~0.68** decade（B-13 追加②）。
    而 `1298::G200-7` 一片死青的原因是它的片基**被整卷平均值盖住**：
    在继承值下该帧有 R 25.9% / G 34.2% / B 39.6% 的像素比 assumed 片基还亮
    （负片物理上不可能）。⚠ 单帧跑时它是 `generic_prior`、越界比例 55%，
    **与生产不是同一个数**——诊断必须走整卷路径。

    返回 `{"dmin": ndarray|None, "confidence": float, "warnings": [...], "stats": {...}}`。
    `dmin=None` 时调用方**必须**继续走通用先验并如实标记，不得静默造一个值。
    """
    p = dict(ORANGE_BASE)
    if params:
        p.update(params)
    a = np.asarray(t_oriented, np.float64)
    warnings: list[str] = []
    if a.ndim != 3 or a.shape[-1] < 3:
        return {"dmin": None, "confidence": 0.0,
                "warnings": ["输入不是 (H,W,≥3) 数组：不做画面内片基检测"], "stats": {}}
    if image_area_roi:
        y0, x0, y1, x1 = (int(v) for v in image_area_roi)
        a = a[max(0, y0):min(a.shape[0], y1), max(0, x0):min(a.shape[1], x1)]
    pix = a.reshape(-1, 3)[:, :3]
    good = np.isfinite(pix).all(axis=1) & (pix.max(axis=1) > 1e-6)
    pix = pix[good]
    n = int(pix.shape[0])
    empty = {"dmin": None, "confidence": 0.0, "warnings": warnings, "stats": {"n": n}}
    if n < 4096:
        return {**empty, "warnings": [f"画面区可用像素不足（{n}）：不做画面内片基检测"]}

    eps = 1e-9
    rg = pix[:, 0] / np.maximum(pix[:, 1], eps)
    rb = pix[:, 0] / np.maximum(pix[:, 2], eps)
    bright = pix[:, 0] > float(np.percentile(pix[:, 0], 99)) * float(p["bright_frac"])
    sel = pix[bright & (rg >= float(p["rg_min"])) & (rb >= float(p["rb_min"]))]
    frac = float(sel.shape[0]) / max(n, 1)
    if sel.shape[0] < max(256, int(p["min_candidate_frac"] * n)):
        return {**empty, "stats": {"n": n, "candidate_frac": round(frac, 6)},
                "warnings": [f"画面内橙色亮区只占 {frac:.3%}：这帧看不到片基，"
                             "不采用画面内估计"]}

    dmin = np.percentile(sel, float(p["percentile"]), axis=0)
    viol = [(pix[:, c] > dmin[c]).mean() for c in range(3)]
    conf = float(p["confidence"])
    if max(viol) > 0.05:
        warnings.append(
            f"画面内片基估计后仍有 {max(viol):.1%} 像素透射率高于片基"
            "（漏光或片基空间不均），置信度下调")
        conf = round(conf * 0.5, 3)
    warnings.append(
        f"片基由画面内橙色亮区估计（候选 {frac:.2%}，"
        f"p{float(p['percentile']):g}）：这是实测估计，不是未曝光片边取样")
    return {"dmin": dmin.astype(np.float64), "confidence": conf,
            "warnings": warnings,
            "stats": {"n": n, "candidate_frac": round(frac, 6),
                      "percentile": float(p["percentile"]),
                      "above_base": [round(float(v), 4) for v in viol],
                      "dmin": [round(float(v), 6) for v in dmin]}}


def estimate_dmax_offset(film_roi: np.ndarray, dmin: np.ndarray,
                         black: float = 0.0) -> dict:
    """从画面区域估计 Dmax 与扫描曝光偏置（§4.5.4）。

    - Dmax：在亮度域 z_lum=mean_c(z_c) 上统计；剔除已剪裁到零的像素
      （t<=1e-5，密度信息丢失，不得进入统计），取 z_lum 的 2 分位的相反数；
      限制在 [0.3, 4.0]，防止零/负值进入模型。
    - offset：片基基准固定落在纸黑上方 ~3.4%（见下方注释；整改 F-4 修复，
      旧"压最暗0.5%到黑位"策略会压碎贴近片基的场景暗部）。
      截断到 [-0.5, 1.0]，用户可手动覆盖。
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

    # 黑位放置（整改 F-4 根因修复）：旧启发式"把最暗 0.5% 压到纸黑"会把
    # 紧贴片基的场景暗部整片压碎（G200-1 实测低端剪裁 8-9%）。
    # 新策略：片基基准 (z=0) 固定落在纸黑上方 ~6.6% 处：
    #   offset = log10(1+black) - 0.03   （paper(z=0) = (1+black)·(1-10^-0.03) ≈ 0.066）
    # 余量必须覆盖 B 通道黑位附近的读出噪声（实测 0.015 余量时 B 低端剪裁 2.4%）。
    offset = float(np.clip(np.log10(1.0 + max(black, 0.0)) - 0.03, -0.5, 1.0))
    z_base_raw = float(np.quantile(zl, 0.995))
    if z_base_raw > -0.01:
        warnings.append(
            f"画面最暗 0.5% 像素已贴近/超过片基（z_p99.5={z_base_raw:.3f}），"
            "黑位可能雾化或 Dmin 偏低，请检查片基框")

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
