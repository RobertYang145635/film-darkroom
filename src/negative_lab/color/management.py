"""色彩管理集中实现（整改 P0-B；决策见 docs/color-pipeline.md）。

职责：
- RAW 解码参数显式化（RAW_DECODE_PARAMS）；
- 输入分类（TIFF 线性/编码/未知 ICC）与线性化；
- 输出编码枚举与 sRGB ICC 装配、写后验证。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np


class InputEncoding(str, Enum):
    LINEAR = "linear"
    SRGB = "srgb"
    ADOBE = "adobe"
    UNKNOWN = "unknown"


class OutputEncoding(str, Enum):
    """导出像素编码枚举——save_* 拒绝无枚举语义的 ndarray。"""

    SRGB = "srgb"          # sRGB 分段传递函数（显示参照）
    LINEAR = "linear"      # 场景线性（无 ICC 或自定义，需调用方声明）


# ---------------------------------------------------------------- 传递函数

def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float32)
    return np.where(x <= 0.04045, x / 12.92,
                    np.power((x + 0.055) / 1.055, 2.4)).astype(np.float32)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float32)
    return np.where(x <= 0.0031308, 12.92 * x,
                    1.055 * np.power(np.maximum(x, 0.0), 1.0 / 2.4) - 0.055
                    ).astype(np.float32)


def decode_linear(x: np.ndarray, enc: InputEncoding) -> np.ndarray:
    """任意输入编码 -> 线性。"""
    if enc == InputEncoding.LINEAR:
        return np.asarray(x, np.float32)
    if enc == InputEncoding.SRGB:
        return srgb_to_linear(x)
    if enc == InputEncoding.ADOBE:
        return np.power(np.maximum(x, 0.0), 2.1992).astype(np.float32)
    raise ValueError(f"无法线性化未知编码 {enc}")


def encode_output(x_linear: np.ndarray, enc: OutputEncoding) -> np.ndarray:
    if not isinstance(enc, OutputEncoding):
        raise ValueError(f"输出编码必须是 OutputEncoding 枚举，得到 {enc!r}")
    if enc == OutputEncoding.SRGB:
        return linear_to_srgb(x_linear)
    if enc == OutputEncoding.LINEAR:
        return np.asarray(x_linear, np.float32)
    raise ValueError(f"不支持的输出编码 {enc!r}")


# ---------------------------------------------------------------- ICC

_SRGB_ICC_CANDIDATES = [
    r"C:\Windows\System32\spool\drivers\color\sRGB Color Space Profile.icm",
    "/Library/ColorSync/Profiles/sRGB Profile.icc",
    "/usr/share/color/icc/sRGB.icc",
    "/usr/share/color/icc/colord/sRGB.icc",
]
_srgb_cache: bytes | None = None


def srgb_icc_bytes() -> bytes:
    """sRGB IEC61966-2.1 profile 字节。

    顺序：环境变量 NEGATIVE_LAB_SRGB_ICC -> 系统标准路径 -> 抛错。
    Pillow 不提供 profile 导出 API；系统 sRGB profile 即 HP 发布的
    IEC61966-2.1 参考实现。找不到时显式失败（禁止无 ICC 静默导出）。
    """
    global _srgb_cache
    if _srgb_cache is not None:
        return _srgb_cache
    import os

    cands: list[str] = []
    env = os.environ.get("NEGATIVE_LAB_SRGB_ICC")
    if env:
        cands.append(env)
    cands += _SRGB_ICC_CANDIDATES
    for c in cands:
        p = Path(c)
        if p.is_file():
            data = p.read_bytes()
            if len(data) >= 128 and data[36:40] == b"acsp":
                _srgb_cache = data
                return data
    raise RuntimeError(
        "未找到系统 sRGB ICC profile。请设置环境变量 NEGATIVE_LAB_SRGB_ICC "
        "指向 sRGB IEC61966-2.1 .icc 文件（整改 P0-B：禁止无 ICC 静默导出）")


def icc_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------- RAW 解码

RAW_DECODE_PARAMS: dict = {
    "use_camera_wb": False,
    "use_auto_wb": False,
    # 不设 user_mul：实测 user_mul=(1,1,1,1) 会改变 LibRaw 内部通道归一化
    # （片基 R 从 0.17 塌缩到 0.11）。白平衡乘数采用 LibRaw 默认（as-shot
    # daylight）；任何逐通道增益都会在 z=log10(t/Dmin) 中被 Dmin 归一化抵消，
    # 不影响密度模型正确性——但必须如实记录策略，不得谎称"无白平衡"。
    "output_color": 6,          # LINEAR：应用相机色彩矩阵、无 gamma —— 原规格
                                # §4.1.1 的"线性 RGB"；sensor-raw(0) 通道含串扰，
                                # 且与 negadoctor/RawPedia 的线性 RGB 语义不符
    "gamm": (1.0, 1.0, 1.0, 1.0),   # rawpy 0.27 属性名就是 "gamm"（非 gamma）；
                                     # 写错会被 setattr 静默忽略 -> 输出带 2.2 gamma
                                     # （实测教训：片基 0.172 -> 0.399）
    "no_auto_bright": True,
    "highlight": 0,             # 不过滤高光（剪裁交给诊断）
    "output_bps": 16,
    "no_auto_scale": False,
    "half_size": False,
    "user_flip": -1,            # 不自动翻转：方向是显式几何参数（annotations）
}

WORKING_SPACE = "camera-linear-normalized"


def decode_raw(path: str | Path) -> tuple[np.ndarray, dict]:
    """RAW -> 归一化线性透射率 t (float32 HxWx3) + 完整解码 meta。

    所有参数显式来自 RAW_DECODE_PARAMS（整改 F-2：禁止依赖库默认值）。
    """
    import rawpy

    p = Path(path)
    with rawpy.imread(str(p)) as r:
        params = rawpy.Params()
        for k, v in RAW_DECODE_PARAMS.items():
            setattr(params, k, v)
        # 防呆：rawpy 的 C 扩展对未知属性 setattr 静默通过，
        # 必须回读校验关键参数真的生效（gamm 拼错=整条管线色彩错误）。
        if tuple(params.gamm) != tuple(RAW_DECODE_PARAMS["gamm"]):
            raise RuntimeError(
                f"rawpy gamma 参数未生效: {tuple(params.gamm)}，"
                f"LibRaw 版本接口可能已变（整改 P0-B 防呆）")
        if params.output_color != RAW_DECODE_PARAMS["output_color"]:
            raise RuntimeError("rawpy output_color 参数未生效")
        arr = r.postprocess(params)
        meta = {
            "kind": "raw",
            "raw_decode_params": dict(RAW_DECODE_PARAMS),
            "decoder": f"rawpy {rawpy.__version__} / LibRaw {rawpy.libraw_version}",
            "black_level_per_channel": [int(v) for v in r.black_level_per_channel],
            "white_level": int(r.white_level),
            "camera": f"{r.lens.make} {r.model or ''}".strip() if hasattr(r, "model") else (r.lens.make or "unknown"),
            "lens": r.lens.model or "unknown",
            "working_space": WORKING_SPACE,
            "white_balance_policy": "libraw-default as-shot multipliers (camera_wb/auto_wb off; per-channel gains cancel in z-normalization)",
            "input_encoding": "linear",
            "input_encoding_detail": "RAW 解码输出（显式参数，gamma=1，相机矩阵 LINEAR）",
            "input_icc_sha256": "none (RAW 无 ICC；线性相机 RGB)",
        }
    t = arr.astype(np.float32) / 65535.0
    t = np.clip(t, 0.0, None)
    mx = float(np.nanmax(t)) if t.size else 1.0
    if mx > 1.0:
        t = t / mx
        meta["normalized_by"] = "frame_max"
    else:
        meta["normalized_by"] = "white_level(16b)"
    meta["bit_depth"] = 16
    return t.astype(np.float32), meta


# ---------------------------------------------------------------- TIFF 分类

@dataclass
class TiffClassification:
    encoding: InputEncoding
    detail: str
    icc_sha256: str = ""
    gamma_tag: tuple | None = None


def classify_tiff(path: str | Path) -> TiffClassification:
    """读 ICC/Gamma 标签判定传递函数；无法判定返回 UNKNOWN（调用方必须拒绝或要显式假设）。"""
    import tifffile

    p = Path(path)
    with tifffile.TiffFile(str(p)) as t:
        pg = t.pages[0]
        icc_tag = pg.tags.get(34675)      # ICCProfile
        gamma_tag = pg.tags.get(33277)    # Gamma
        icc = bytes(icc_tag.value) if icc_tag is not None else None
        gam = gamma_tag.value if gamma_tag is not None else None

    g_tup = tuple(gam) if isinstance(gam, (tuple, list)) else (gam,) if gam else None
    if g_tup:
        vals = np.array(g_tup, dtype=float)
        if np.allclose(vals, 1.0, atol=1e-3):
            return TiffClassification(InputEncoding.LINEAR, "Gamma 标签=1.0",
                                       icc_sha256(icc) if icc else "", g_tup)
        if np.allclose(vals, 2.2, atol=0.05) or np.allclose(vals, 2.1992, atol=0.01):
            return TiffClassification(InputEncoding.ADOBE, f"Gamma≈{vals[0]}",
                                      icc_sha256(icc) if icc else "", g_tup)
        return TiffClassification(InputEncoding.UNKNOWN, f"Gamma 标签={vals}",
                                  icc_sha256(icc) if icc else "", g_tup)
    if icc:
        desc = _icc_description(icc).lower()
        h = icc_sha256(icc)
        if "srgb" in desc or h == icc_sha256(srgb_icc_bytes()):
            return TiffClassification(InputEncoding.SRGB, f"ICC={desc or 'sRGB'}", h)
        if "adobe" in desc:
            return TiffClassification(InputEncoding.ADOBE, f"ICC={desc}", h)
        return TiffClassification(InputEncoding.UNKNOWN,
                                  f"ICC 描述无法判定传递函数: {desc or '(无)'}", h)
    return TiffClassification(InputEncoding.UNKNOWN, "无 ICC 且无 Gamma 标签")


def _icc_description(data: bytes) -> str:
    try:
        import io as _io

        from PIL import ImageCms

        prof = ImageCms.ImageCmsProfile(_io.BytesIO(data))
        return str(ImageCms.getProfileName(prof)).strip()
    except Exception:  # noqa: BLE001
        return ""
