"""密度域主模型与每通道幂指数修正（规格 §4.2–§4.4）。

============================ 符号约定（必读） ============================
与 darktable negadoctor 源码相同方向的对数坐标（规格 §4.2.4）：

    t_c   归一化线性通道透射率。扫描负片中，未曝光片基最亮（t 最大），
          曝光/密度越高的区域 t 越小。
    Dmin_c 未曝光片基的每通道透射率基准（robust median）。
    z_c = log10( max(t_c, eps) / max(Dmin_c, eps) )

因此：片基处 z_c = 0；画面内 t_c <= Dmin_c，故 z_c <= 0；
场景越亮 -> 胶片密度越高 -> t_c 越小 -> z_c 越负。
反相后输出 positive 随场景亮度单调递增（z_c 越负 -> 输出越亮）。

该方向由 tests/synthetic/test_direction.py 与 tests/unit/test_density_model.py
锁定。若把 z 的定义改写为 d_c = log10(Dmin_c / t_c)（正密度），必须同步重写
§4.3 的全部反算公式并新增等价性测试，不能只改一行（规格 §4.3 末段）。
==========================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .params import InvertParams

# 防止 log10(0) 的正数 epsilon（规格 §4.2.4）
EPS = 1e-6
# Dmax 非法时的安全回退值（规格 §4.3：报错并回退，而不是产生无穷大）
DEFAULT_DMAX = 2.0
DMAX_MIN = 1e-3


@dataclass
class InvertResult:
    """一次反相的结果与诊断（规格 §4.5.7–8）。"""

    positive_linear: np.ndarray  # (H,W,3) float32，可暂时超出 [0,1]
    preview_srgb: np.ndarray     # (H,W,3) float32 in [0,1]，显示编码后
    diagnostics: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def sanitize_finite(a: np.ndarray) -> tuple[np.ndarray, int]:
    """把 NaN/Inf 替换为可诊断的有限值 0.0 并计数（规格 §4.1.4）。

    替换值 0.0 经 z 变换后为 log10(eps/dmin)，有限且必然落在黑位，
    不会把异常值带入导出。计数写入诊断。
    """
    a = np.asarray(a, dtype=np.float32)
    bad = ~np.isfinite(a)
    n_bad = int(bad.sum())
    if n_bad:
        a = a.copy()
        a[bad] = 0.0
    else:
        a = a.astype(np.float32, copy=False)
    return a, n_bad


def sanitize_dmax(dmax: float) -> tuple[float, str | None]:
    """Dmax 必须为正；非法或接近零时回退安全默认值并给出警告（规格 §4.3）。"""
    try:
        v = float(dmax)
    except (TypeError, ValueError):
        return DEFAULT_DMAX, f"Dmax 非法值 {dmax!r}，回退默认 {DEFAULT_DMAX}"
    if not np.isfinite(v) or v < DMAX_MIN:
        return DEFAULT_DMAX, f"Dmax={dmax!r} 非正或接近零，回退默认 {DEFAULT_DMAX}"
    return v, None


def transmittance_to_z(t: np.ndarray, dmin: np.ndarray | tuple, eps: float = EPS) -> np.ndarray:
    """z_c = log10(max(t_c,eps)/max(Dmin_c,eps))，见模块头符号约定。"""
    t = np.maximum(np.asarray(t, dtype=np.float32), eps)
    d = np.maximum(np.asarray(dmin, dtype=np.float32), eps)
    return np.log10(t / d).astype(np.float32)


def channel_exponents(p: InvertParams) -> np.ndarray:
    """k_c 向量：k_G=reference_exponent, k_R=k_G*red_ratio, k_B=k_G*blue_ratio（§4.4）。"""
    ref = float(p.reference_exponent)
    return np.array([ref * p.red_ratio, ref, ref * p.blue_ratio], dtype=np.float32)


def apply_exponents(z: np.ndarray, k: np.ndarray) -> np.ndarray:
    """z'_c = k_c * z_c（§4.4）。默认 k=(1,1,1) 时 z'=z。"""
    return (z * k).astype(np.float32)


def density_to_positive(z_eff: np.ndarray, p: InvertParams,
                        warnings: list[str]) -> np.ndarray:
    """规格 §4.3 密度域主模型，逐字实现：

        wb_high_internal_c = wb_high_c / max(Dmax, eps)
        offset_c           = wb_high_c * offset * wb_low_c
        q_c                = wb_high_internal_c * z_eff_c + offset_c
        black_internal     = -exposure * (1 + black)      # 内部带符号量
        paper_linear_c     = max(0, -(exposure * 10^q_c + black_internal))
        positive_linear_c  = paper_linear_c ^ gamma

    代数上 paper_linear_c = exposure * ((1+black) - 10^q_c)：
    z=0（片基/最暗场景）且 offset=0, black=0 时 paper=0（黑位）；
    z 越负（场景越亮）10^q 越小，paper 越大 —— 方向与符号约定一致。
    """
    dmax, w = sanitize_dmax(p.dmax)
    if w:
        warnings.append(w)

    exposure = float(p.exposure)
    if not np.isfinite(exposure) or exposure <= 0:
        warnings.append(f"exposure={p.exposure!r} 非法，回退 1.0")
        exposure = 1.0
    gamma = float(p.gamma)
    if not np.isfinite(gamma) or gamma <= 0:
        warnings.append(f"gamma={p.gamma!r} 非法，回退 1.0")
        gamma = 1.0
    black = float(p.black)
    if not np.isfinite(black) or black < 0:
        warnings.append(f"black={p.black!r} 非法（纸黑须 >=0），回退 0.0")
        black = 0.0

    wb_high = np.maximum(np.asarray(p.wb_high, dtype=np.float32), EPS)
    wb_low = np.maximum(np.asarray(p.wb_low, dtype=np.float32), EPS)

    wb_high_internal = wb_high / max(dmax, EPS)
    offset_c = wb_high * np.float32(p.offset) * wb_low
    q = wb_high_internal * z_eff + offset_c  # offset_c 按最后通道维广播
    black_internal = -exposure * (1.0 + black)
    paper = np.maximum(0.0, -(exposure * np.power(10.0, q) + black_internal))
    return np.power(paper, gamma).astype(np.float32)


def soft_highlight_compress(x: np.ndarray, knee: float, strength: float) -> np.ndarray:
    """单调高光软压缩（§4.3 末）。knee 以下恒等；以上用 Reinhard 型尾巴，
    在 knee 处一阶连续且整体严格单调（strength>=0），不做不可逆自动对比度。"""
    strength = max(0.0, min(1.0, float(strength)))
    if strength == 0.0:
        return x.astype(np.float32, copy=False)
    knee = float(np.clip(knee, 1e-3, 1.0))
    out = x.copy()
    hi = x > knee
    d = x[hi] - knee
    out[hi] = knee + d / (1.0 + strength * d)
    return out.astype(np.float32)


def encode_srgb(linear: np.ndarray) -> np.ndarray:
    """sRGB 分段传递函数（仅用于预览/8 位导出；核心反相之前禁止显示曲线，§4.1.2）。"""
    x = np.clip(linear, 0.0, 1.0)
    out = np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1.0 / 2.4) - 0.055)
    return out.astype(np.float32)


def invert(t_rgb: np.ndarray, p: InvertParams) -> InvertResult:
    """完整单帧反相：线性透射率 -> z -> z' -> 密度域正片 -> 高光压缩 -> 预览编码。

    输入 t_rgb: (H,W,3) float32 线性 RGB（内部允许暂时超出 [0,1]，导出前处理，§4.1.1）。
    """
    warnings: list[str] = []
    t, n_bad = sanitize_finite(t_rgb)
    if n_bad:
        warnings.append(f"输入含 {n_bad} 个 NaN/Inf，已替换为 0.0 并记录")

    z = transmittance_to_z(t, p.dmin)
    z_eff = apply_exponents(z, channel_exponents(p))
    positive = density_to_positive(z_eff, p, warnings)
    positive = soft_highlight_compress(positive, p.highlight_knee, p.highlight_strength)

    # 诊断（§4.5.7）：输入/输出剪裁比例、非有限值
    diag: dict = {
        "input_nonfinite": n_bad,
        "input_clip": channel_clip_ratios(t),
        "output_clip": channel_clip_ratios(positive),
        "output_nonfinite": int((~np.isfinite(positive)).sum()),
    }
    preview = encode_srgb(positive)
    return InvertResult(positive_linear=positive, preview_srgb=preview,
                        diagnostics=diag, warnings=warnings)


def channel_clip_ratios(img: np.ndarray, tol: float = 1e-6) -> dict:
    """每通道 [0,1] 之外的剪裁比例（规格 §8.4 剪裁指标）。"""
    lo = (img <= tol).mean(axis=(0, 1))
    hi = (img >= 1.0 - tol).mean(axis=(0, 1))
    return {
        "low": [float(v) for v in lo],
        "high": [float(v) for v in hi],
    }
