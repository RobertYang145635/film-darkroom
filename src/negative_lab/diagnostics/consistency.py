"""卷内一致性度量（去色罩算法研究 R1；无真值可算的"同卷是否像同一卷"）。

回答痛点调研 #1（最强痛点）："同一卷看起来是不是同一卷"——项目此前
没有任何数字。本模块给出可复现、可回归的度量。

设计决定（研究笔记 docs/research/2026-09-consistency-rescue.md §2）：
- 度量在**渲染后正片的 sRGB 域**做 Lab 统计：这是用户眼睛看到的域，
  与技术域（InvertParams）解耦；技术契约不受影响。
- 明暗（L*）与色偏（a*/b*）分开报告：卷内"一致"主要指色偏一致；
  亮度差属于曝光语义，混在一起会把"曝光不同的两张"误判成不一致。
- 输入是同尺寸渲染图（调用方负责统一裁到有效画面/公共构图）；
  掩码可选（排除天空/留白等大面积非画面）。
- 离群判定用与卷中位色偏的偏差，不用逐对最大值（对单张污染稳健）。

已知局限（如实）：
- 不同场景内容本身的颜色差异会计入 ΔE——这是场景语义，不是处理不一致；
  因此本模块还给出色偏方向（ab_shift）的中位数与离群帧，供人工分离
  "场景本来不同"与"处理漂移"。真正的处理一致性信号见
  `neutral_pick`（R2）落地后的片基/中性区统计（后续研究）。
"""

from __future__ import annotations

import numpy as np

from .colorimetry import linear_srgb_to_lab

#: 有效像素下限：低于此值统计不可信（调用方应给出警告而非硬失败）
MIN_PIXELS = 256
#: 中位色偏偏差超过该值判离群（Lab a*/b* 单位；约 2.5 个可感知单位）
OUTLIER_AB = 4.0


def _lab_of(img: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    """sRGB 域 (H,W,3) -> 掩码内像素 Lab (N,3)。空集返回 (0,3)。"""
    x = np.asarray(img, np.float32)
    if x.ndim != 3 or x.shape[2] != 3:
        raise ValueError(f"需要 (H,W,3) 图像，得到 {x.shape}")
    px = x[mask] if mask is not None else x.reshape(-1, 3)
    px = px[np.isfinite(px).all(axis=1)]
    if px.shape[0] == 0:
        return np.zeros((0, 3), np.float64)
    return linear_srgb_to_lab(px.astype(np.float64))


def _ab_median(lab: np.ndarray) -> tuple[float, float]:
    if lab.shape[0] == 0:
        return 0.0, 0.0
    return float(np.median(lab[:, 1])), float(np.median(lab[:, 2]))


def frame_consistency(img_a: np.ndarray, img_b: np.ndarray,
                      mask: np.ndarray | None = None) -> dict:
    """两帧渲染正片的 Lab 一致性。

    返回 {"de_mean","de_p50","de_p95","l_shift","ab_shift":[da,db],"n"}：
    - de_*：掩码内逐像素 ΔE2000 统计；
    - ab_shift：B 相对 A 的中位色偏方向（[da*, db*]，蓝偏 = db* < 0）；
    - l_shift：中位 L* 差（明暗分开，不计入色偏方向）。
    """
    la = _lab_of(img_a, mask)
    lb = _lab_of(img_b, mask)
    n = min(la.shape[0], lb.shape[0])
    out = {"de_mean": 0.0, "de_p50": 0.0, "de_p95": 0.0,
           "l_shift": 0.0, "ab_shift": [0.0, 0.0], "n": int(n)}
    if la.shape[0] < MIN_PIXELS or lb.shape[0] < MIN_PIXELS:
        out["warnings"] = ["有效像素不足，一致性统计不可信"]
        return out
    from .colorimetry import delta_e_2000

    m = min(la.shape[0], lb.shape[0])
    # 两帧像素数一致（同尺寸同掩码）时天然逐点对应；不一致时取前 m 个
    # （调用方保证同尺寸；此处只防错不承诺跨尺寸语义）
    de = delta_e_2000(la[:m], lb[:m])
    out.update({
        "de_mean": round(float(de.mean()), 3),
        "de_p50": round(float(np.median(de)), 3),
        "de_p95": round(float(np.quantile(de, 0.95)), 3),
        "l_shift": round(float(np.median(lb[:m, 0]) - np.median(la[:m, 0])), 3),
        "ab_shift": [round(a, 3) for a in
                     (float(np.median(lb[:m, 1]) - np.median(la[:m, 1])),
                      float(np.median(lb[:m, 2]) - np.median(la[:m, 2])))],
        "n": int(m),
    })
    return out


def roll_consistency(frames: list[np.ndarray],
                     masks: list[np.ndarray | None] | None = None,
                     *,
                     outlier_ab: float = OUTLIER_AB) -> dict:
    """卷内一致性：两两 frame_consistency + 汇总与离群帧。

    返回 {"n_frames","n_pairs","de_mean","de_p50","de_p95","de_matrix",
          "ab_matrix","median_ab","outlier_score","outliers"}：
    - de_matrix[i,j]：帧对 ΔE 均值（对称、对角 0）；
    - median_ab：各帧中位色偏 (a*,b*)，是"卷级色偏中心"；
    - outlier_score[i]：帧 i 的色偏中心到卷中位色偏的 L2 距离；
    - outliers：score > outlier_ab 的帧索引。
    """
    n = len(frames)
    if masks is not None and len(masks) != n:
        raise ValueError("masks 与 frames 数量不一致")
    if masks is not None:
        # 审查修复（2026-09-08）：ΔE 是**逐像素配对**算的（la[i] vs lb[i]）。
        # 布尔掩码之后索引不再对应同一空间位置，各帧掩码不同就等于把不相干
        # 的两个像素相减——数字看着正常但毫无意义，而签名恰恰允许逐帧不同
        # 掩码，是个静默出错口。要么同一套，要么明确拒绝。
        base_m = masks[0]
        for k, m in enumerate(masks[1:], start=1):
            if ((m is None) != (base_m is None)
                    or (m is not None and not np.array_equal(m, base_m))):
                raise ValueError(
                    f"masks[{k}] 与 masks[0] 不一致：逐帧不同掩码会让 ΔE "
                    "配对到不同的空间位置，结果没有意义")
    out: dict = {"n_frames": n,
                 "n_pairs": n * (n - 1) // 2,
                 "de_mean": 0.0, "de_p50": 0.0, "de_p95": 0.0,
                 "de_matrix": np.zeros((n, n)).tolist(),
                 "ab_matrix": np.zeros((n, n, 2)).tolist(),
                 "median_ab": [], "outlier_score": [], "outliers": []}
    if n == 0:
        return out
    labs = [_lab_of(f, masks[i] if masks else None) for i, f in enumerate(frames)]
    ab = [(_ab_median(l) if l.shape[0] >= MIN_PIXELS else (float("nan"),
                                                           float("nan")))
          for l in labs]
    # 缺失值统一用 None（审查修复 2026-09-08）：de_matrix 已经是 None，同一份
    # 输出里两套约定逼下游写两种判空；且裸 NaN 经 json.dumps 会产出非法 JSON
    # （实测 allow_nan=False 直接抛 ValueError，严格解析器一律拒绝）。
    out["median_ab"] = [[(round(a, 3) if np.isfinite(a) else None),
                         (round(b, 3) if np.isfinite(b) else None)]
                        for a, b in ab]
    pairs: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            if labs[i].shape[0] < MIN_PIXELS or labs[j].shape[0] < MIN_PIXELS:
                de_mean = float("nan")
            else:
                from .colorimetry import delta_e_2000

                m = min(labs[i].shape[0], labs[j].shape[0])
                de_mean = float(delta_e_2000(labs[i][:m], labs[j][:m]).mean())
            out["de_matrix"][i][j] = out["de_matrix"][j][i] = (
                round(de_mean, 3) if np.isfinite(de_mean) else None)
            dv = [ab[j][0] - ab[i][0], ab[j][1] - ab[i][1]]
            out["ab_matrix"][i][j] = out["ab_matrix"][j][i] = [
                (round(v, 3) if np.isfinite(v) else None) for v in dv]
            if np.isfinite(de_mean):
                pairs.append(de_mean)
    if pairs:
        out["de_mean"] = round(float(np.mean(pairs)), 3)
        out["de_p50"] = round(float(np.median(pairs)), 3)
        out["de_p95"] = round(float(np.quantile(pairs, 0.95)), 3)
    good = [(i, a, b) for i, (a, b) in enumerate(ab) if np.isfinite(a)]
    if good:
        med_a = float(np.median([a for _, a, _ in good]))
        med_b = float(np.median([b for _, _, b in good]))
        score = [float("nan")] * n
        for i, a, b in good:
            score[i] = round(float(np.hypot(a - med_a, b - med_b)), 3)
        out["outlier_score"] = [None if not np.isfinite(s) else s
                                for s in score]
        out["outliers"] = [i for i, s in enumerate(score)
                           if s is not None and np.isfinite(s) and s > outlier_ab]
    return out
