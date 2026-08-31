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

# 工作空间决策（v3 P0-A，2026-09-01）：
# - rawpy 0.27.1 实测 ColorSpace: raw=0, sRGB=1, Adobe=2, Wide=3, ProPhoto=4,
#   XYZ=5, ACES=6, P3D65=7, Rec2020=8。旧代码把 6 当 "LINEAR" 是错误语义
#   （v3 F-ALG-1）；"ACES 数值 + sRGB 标记" 组合已禁止。
# - 临时默认 = sRGB 线性（oc=sRGB + gamm=1）：应用相机矩阵、D65、无传递函数，
#   与导出端 sRGB TRC + sRGB ICC 自洽。最终默认由 benchmark_color_pipeline
#   三路径 bake-off 在留出集上决定（v3 P0-A.4）。
WORKING_SPACE = "srgb-linear-d65-normalized"


def raw_decode_params() -> dict:
    """显式解码参数（命名枚举运行时解析；禁止裸整数语义）。"""
    import rawpy

    return {
        "use_camera_wb": False,
        "use_auto_wb": False,
        # 不设 user_mul：实测 (1,1,1,1) 改变 LibRaw 内部通道归一化。
        # 实际生效 = LibRaw 默认 daylight 乘数，运行时读取并如实记录
        # （v3 F-ALG-8：不得笼统写 as-shot/none）。
        "output_color": int(rawpy.ColorSpace.sRGB.value),
        "output_color_name": "sRGB",
        "gamm": (1.0, 1.0, 1.0, 1.0),   # rawpy 0.27 属性名就是 "gamm"
        "no_auto_bright": True,
        "highlight": 0,                  # 不过滤高光（剪裁交给诊断）
        "output_bps": 16,
        "no_auto_scale": False,
        "half_size": False,
        # v3 F-ALG-8 如实记录：-1 = 按 RAW 元数据自动定向。
        # 本项目 SOURCE 坐标空间定义为 "metadata-oriented decode"，
        # 与主流查看器一致；不得再谎称"不自动翻转"。
        "user_flip": -1,
    }


def decode_raw(path: str | Path) -> tuple[np.ndarray, dict]:
    """RAW -> 归一化线性 sRGB-D65 RGB (float32 HxWx3) + 完整解码 meta。"""
    import rawpy

    p = Path(path)
    spec = raw_decode_params()
    with rawpy.imread(str(p)) as r:
        params = rawpy.Params()
        for k, v in spec.items():
            setattr(params, k, v)
        # 防呆：rawpy C 扩展对未知属性 setattr 静默通过，必须回读校验
        if tuple(params.gamm) != (1.0, 1.0, 1.0, 1.0):
            raise RuntimeError(f"rawpy gamma 参数未生效: {tuple(params.gamm)}")
        if int(params.output_color) != spec["output_color"]:
            raise RuntimeError("rawpy output_color 参数未生效")
        arr = r.postprocess(params)
        meta = {
            "kind": "raw",
            "raw_decode_params": {k: v for k, v in spec.items()},
            "decoder": f"rawpy {rawpy.__version__} / LibRaw {rawpy.libraw_version}",
            "black_level_per_channel": [int(v) for v in r.black_level_per_channel],
            "white_level": int(r.white_level),
            "camera": f"{r.lens.make} {r.model or ''}".strip() if hasattr(r, "model") else (r.lens.make or "unknown"),
            "lens": r.lens.model or "unknown",
            "working_space": WORKING_SPACE,
            # v3 F-ALG-8：记录实际生效的白平衡乘数，不写笼统标签
            "white_balance_policy": "libraw-default daylight multipliers (camera_wb/auto_wb off)",
            "applied_wb_daylight": [float(v) for v in r.daylight_whitebalance],
            "camera_wb_as_shot": [float(v) for v in r.camera_whitebalance],
            "rgb_xyz_matrix": np.asarray(r.rgb_xyz_matrix, dtype=float).tolist(),
            "orientation_policy": "metadata-oriented (user_flip=-1)",
            "input_encoding": "linear",
            "input_encoding_detail": "RAW 解码：相机矩阵 + sRGB 基色 + gamma=1",
            "input_icc_sha256": "none (RAW；工作空间见 working_space)",
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
