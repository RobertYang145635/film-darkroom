"""图像读取与导出（规格 §4.1、§2.1.1/6）。

Phase 1 覆盖：16 位/8 位 TIFF、PNG/JPEG（带显示曲线警告）。
RAW/DNG：依赖 rawpy/LibRaw，未安装时给出文件路径、原因和可行建议，
不静默伪造 RAW 支持（执行契约 §6、规格 §3.1）。
"""

from __future__ import annotations

import io as _io
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

RASTER_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
RAW_EXTS = {".nef", ".cr2", ".cr3", ".arw", ".raf", ".orf", ".rw2", ".dng", ".raw"}


@dataclass
class LoadResult:
    rgb: np.ndarray          # (H,W,3) float32，归一化线性透射率 [0,1]
    meta: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


class ReadError(Exception):
    """读取失败：必须携带文件路径、原因和可行建议（§2.1.1）。"""


def load_linear_rgb(path: str | Path) -> LoadResult:
    p = Path(path)
    ext = p.suffix.lower()
    if not p.exists():
        raise ReadError(f"路径：{p}\n原因：文件不存在\n建议：检查路径或重新导入")
    try:
        if ext in (".tif", ".tiff"):
            return _load_tiff(p)
        if ext in RASTER_EXTS:
            return _load_pillow(p)
        if ext in RAW_EXTS:
            return _load_raw(p)
    except ReadError:
        raise
    except Exception as e:  # noqa: BLE001 —— 统一包装为带路径/原因/建议的错误
        raise ReadError(f"路径：{p}\n原因：{type(e).__name__}: {e}\n"
                        f"建议：确认文件未损坏/下载完整（如 .downloading 后缀为残缺文件）") from e
    raise ReadError(f"路径：{p}\n原因：不支持的扩展名 {ext!r}\n"
                    f"建议：使用相机 RAW、16 位 TIFF 或 DNG 输入")


def _normalize_bitdepth(arr: np.ndarray) -> tuple[np.ndarray, int]:
    if arr.dtype == np.uint16:
        return arr.astype(np.float32) / 65535.0, 16
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0, 8
    if arr.dtype in (np.float32, np.float64):
        return arr.astype(np.float32), 32
    if arr.dtype == np.uint32:
        return (arr.astype(np.float64) / 4294967295.0).astype(np.float32), 32
    if arr.dtype == np.int32 or arr.dtype == np.int16:
        mx = np.iinfo(arr.dtype).max
        return (arr.astype(np.float32) / float(mx)).clip(0.0, None), int(np.log2(mx + 1))
    raise ValueError(f"不支持的像素类型 {arr.dtype}")


def _load_tiff(p: Path) -> LoadResult:
    import tifffile

    arr = tifffile.imread(str(p))
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))  # CHW -> HWC
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    rgb, bits = _normalize_bitdepth(arr)
    meta = {"path": str(p), "bits": bits, "shape": rgb.shape,
            "photometric": "linear-transmittance-assumed"}
    warns = ["TIFF 按线性透射率处理；若源文件带显示曲线（非相机线性数据），"
             "请在输入配置中声明（§4.1.2）"]
    return LoadResult(rgb.astype(np.float32), meta, warns)


def _load_pillow(p: Path) -> LoadResult:
    from PIL import Image

    img = Image.open(str(p))
    mode = img.mode
    arr = np.asarray(img.convert("RGB"))
    rgb, bits = _normalize_bitdepth(arr)
    warns = [f"{p.suffix} 为显示编码（gamma/压缩）图像，不是线性数据；"
             f"仅可用于预览对比，正式流程请用 RAW/16 位 TIFF（§4.1.2）"]
    return LoadResult(rgb.astype(np.float32),
                      {"path": str(p), "bits": bits, "mode": mode, "shape": rgb.shape},
                      warns)


def _load_raw(p: Path) -> LoadResult:
    try:
        import rawpy  # noqa: F401
    except ImportError as e:
        raise ReadError(
            f"路径：{p}\n原因：RAW/DNG 解码依赖 rawpy(LibRaw) 未安装"
            f"（当前 Phase 1 仅承诺 TIFF 流程）\n"
            f"建议：uv pip install rawpy，或先用相机/扫描仪导出 16 位 TIFF/DNG") from e
    import rawpy

    with rawpy.imread(str(p)) as r:
        # 线性输出：gamma=1、禁用自动亮度/相机白平衡（白平衡是显示端增益，
        # 会污染通道密度测量，§4.1.2）；16 位输出后归一化
        params = rawpy.Params(use_camera_wb=False, no_auto_bright=True,
                              half_size=False, output_bps=16)
        params.gamm = [1.0, 1.0, 1.0, 1.0]
        rgbf = r.postprocess(params).astype(np.float32) / 65535.0
        rgbf = np.clip(rgbf, 0.0, None)
        mx = float(np.nanmax(rgbf)) if rgbf.size else 1.0
        if mx > 1.0:
            rgbf = rgbf / mx
        lens = r.lens
        bits = 16
    return LoadResult(rgbf.astype(np.float32),
                      {"path": str(p), "bits": bits, "shape": rgbf.shape,
                       "camera": f"{lens.make} {lens.model}".strip() or "rawpy/LibRaw runtime"},
                      ["RAW 已解码为线性 RGB（未应用相机白平衡）；"
                       "如需设备配置请在输入侧声明（§4.1.3）"])


# ---------------------------------------------------------------- 导出

def save_tiff16(path: str | Path, rgb01: np.ndarray, *, sources: list[Path] | None = None,
                overwrite: bool = False) -> Path:
    """16 位 TIFF 导出（规格 §2.1.6）。rgb01 为 [0,1] 显示编码或线性数据，由调用方决定。"""
    import tifffile

    from .atomic import atomic_write_bytes, ensure_not_source

    path = Path(path)
    if sources:
        ensure_not_source(path, sources)
    arr = (np.clip(rgb01, 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16)
    buf = _io.BytesIO()
    tifffile.imwrite(buf, arr, photometric="rgb")
    atomic_write_bytes(path, buf.getvalue(), overwrite=overwrite)
    return path


def save_jpeg8(path: str | Path, rgb01: np.ndarray, quality: int = 95, *,
               sources: list[Path] | None = None, overwrite: bool = False) -> Path:
    """8 位 JPEG 导出。JPEG 固定 8 位，调用方界面必须明确显示这一点（§2.1.6）。"""
    from PIL import Image

    from .atomic import atomic_write_bytes, ensure_not_source

    path = Path(path)
    if sources:
        ensure_not_source(path, sources)
    arr = (np.clip(rgb01, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    img = Image.fromarray(arr, "RGB")
    buf = _io.BytesIO()
    img.save(buf, format="JPEG", quality=int(quality))
    atomic_write_bytes(path, buf.getvalue(), overwrite=overwrite)
    return path
