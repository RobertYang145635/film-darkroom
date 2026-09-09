"""图像读取与导出（整改 P0-B；决策 docs/color-pipeline.md）。

- 所有输入回答"这些数字属于哪个色域/传递函数/白平衡策略"（meta 记录）。
- TIFF 无 ICC 且无 Gamma 标签 -> 拒绝，除非调用方显式给 tiff_encoding 假设。
- 导出必须传 OutputEncoding 枚举；sRGB 输出嵌入 sRGB ICC；写后重读验证。
- RAW/DNG：rawpy/LibRaw；未安装或格式不支持时给出路径、原因、建议，
  不静默降级、不用内嵌 JPEG 冒充（整改 §1.5）。
"""

from __future__ import annotations

import io as _io
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..color.management import (
    InputEncoding, OutputEncoding, TiffClassification, classify_tiff,
    decode_linear, decode_raw, encode_output, icc_sha256, srgb_icc_bytes)
from ..color.management import linear_to_srgb   # 唯一 sRGB 编码实现

RASTER_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
RAW_EXTS = {".nef", ".cr2", ".cr3", ".arw", ".raf", ".orf", ".rw2", ".dng", ".raw"}


@dataclass
class LoadResult:
    rgb: np.ndarray          # (H,W,3) float32 归一化线性透射率
    meta: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


class ReadError(Exception):
    """读取失败：必须携带文件路径、原因和可行建议（§2.1.1）。"""


def load_linear_rgb(path: str | Path,
                    tiff_encoding: str | None = None) -> LoadResult:
    """加载为归一化线性透射率。

    tiff_encoding: "linear"|"srgb"|"adobe" —— 仅当 TIFF 自身无法判定
    （无 ICC/Gamma 标签）时作为显式假设；其余情况以文件内证据为准。
    """
    p = Path(path)
    ext = p.suffix.lower()
    if not p.exists():
        raise ReadError(f"路径：{p}\n原因：文件不存在\n建议：检查路径或重新导入")
    try:
        if ext in (".tif", ".tiff"):
            res = _load_tiff(p, tiff_encoding)
        elif ext in RASTER_EXTS:
            res = _load_pillow(p)
        elif ext in RAW_EXTS:
            res = _load_raw(p)
        else:
            raise ReadError(f"路径：{p}\n原因：不支持的扩展名 {ext!r}\n"
                            f"建议：使用相机 RAW、16 位 TIFF 或 DNG 输入")
    except ReadError:
        raise
    except Exception as e:  # noqa: BLE001
        raise ReadError(f"路径：{p}\n原因：{type(e).__name__}: {e}\n"
                        f"建议：确认文件未损坏/下载完整（.downloading 后缀为残缺文件）") from e
    from .probe import sha256_of
    res.meta.setdefault("sha256", sha256_of(p))
    return res


# ---------------------------------------------------------------- 输入

def _normalize_bitdepth(arr: np.ndarray) -> tuple[np.ndarray, int]:
    if arr.dtype == np.uint16:
        return arr.astype(np.float32) / 65535.0, 16
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0, 8
    if arr.dtype in (np.float32, np.float64):
        return arr.astype(np.float32), 32
    if arr.dtype == np.uint32:
        return (arr.astype(np.float64) / 4294967295.0).astype(np.float32), 32
    if arr.dtype in (np.int32, np.int16):
        mx = np.iinfo(arr.dtype).max
        return (arr.astype(np.float32) / float(mx)).clip(0.0, None), int(np.log2(mx + 1))
    raise ValueError(f"不支持的像素类型 {arr.dtype}")


def _load_tiff(p: Path, tiff_encoding: str | None) -> LoadResult:
    import tifffile

    cls: TiffClassification = classify_tiff(p)
    if cls.encoding == InputEncoding.UNKNOWN:
        if tiff_encoding is None:
            raise ReadError(
                f"路径：{p}\n原因：TIFF 传递函数无法从 metadata 确定"
                f"（{cls.detail}）；盲目按线性处理会产生错误密度\n"
                f"建议：确认该导入批次的来源工具编码，并作一次批次级选择 "
                f"tiff_encoding=linear|srgb|adobe（v5 §4.3.3，"
                f"不是设备配置，不要求扫描仪型号）")
        cls = TiffClassification(InputEncoding(tiff_encoding),
                                 f"用户批次级假设（文件本身：{cls.detail}）",
                                 cls.icc_sha256, cls.gamma_tag, cls.trc_sha256,
                                 cls.tag_facts)
    arr = tifffile.imread(str(p))
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    norm, bits = _normalize_bitdepth(arr)
    lin = decode_linear(norm, cls.encoding)
    warns = []
    if cls.encoding != InputEncoding.LINEAR:
        warns.append(f"TIFF 按 {cls.encoding.value} 编码线性化（{cls.detail}）")
    facts = cls.tag_facts or {}
    meta = {"path": str(p), "kind": "tiff", "bits": bits, "shape": lin.shape,
            "width": int(lin.shape[1]), "height": int(lin.shape[0]),
            "input_encoding": cls.encoding.value,
            "input_encoding_detail": cls.detail,
            "input_icc_sha256": cls.icc_sha256 or "none",
            "input_trc_sha256": cls.trc_sha256 or "none",
            "sample_format": facts.get("sample_format", "unknown"),
            "photometric": facts.get("photometric", "unknown"),
            "software": facts.get("software", ""),
            "camera": facts.get("model", "") or facts.get("make", ""),
            "tiff_tag_facts": {k: facts.get(k) for k in
                               ("photometric", "sample_format", "bits",
                                "prediction", "software")},
            "working_space": "srgb-linear-d65-normalized",
            "white_balance_policy": "n/a (file data)"}
    return LoadResult(lin.astype(np.float32), meta, warns)


def _load_pillow(p: Path) -> LoadResult:
    from PIL import Image

    img = Image.open(str(p))
    mode = img.mode
    icc = img.info.get("icc_profile")
    arr = np.asarray(img.convert("RGB"))
    norm, bits = _normalize_bitdepth(arr)
    lin = decode_linear(norm, InputEncoding.SRGB)   # JPEG/PNG 永远显示编码
    warns = [f"{p.suffix} 按 sRGB 显示编码读入并线性化；"
             f"显示编码源仅可用于预览对比，正式流程用 RAW/16 位线性 TIFF（§4.1.2）"]
    if icc and icc_sha256(bytes(icc)) != icc_sha256(srgb_icc_bytes()):
        warns.append("源文件带非 sRGB ICC，当前按 sRGB 解读——正式流程请走 TIFF 路线")
    return LoadResult(lin.astype(np.float32),
                      {"path": str(p), "kind": "raster", "bits": bits, "mode": mode,
                       "shape": lin.shape,
                       "width": int(lin.shape[1]), "height": int(lin.shape[0]),
                       "software": "",
                       "input_encoding": "srgb(display)",
                       "input_icc_sha256": icc_sha256(bytes(icc)) if icc else "none",
                       "input_trc_sha256": "none",
                       "working_space": "srgb-linear-d65-normalized",
                       "white_balance_policy": "n/a (file data)"},
                      warns)


def _load_raw(p: Path) -> LoadResult:
    try:
        import rawpy  # noqa: F401
    except ImportError as e:
        raise ReadError(
            f"路径：{p}\n原因：RAW/DNG 解码依赖 rawpy(LibRaw) 未安装\n"
            f"建议：uv pip install rawpy，或导出 16 位 TIFF/DNG") from e
    dng_facts: dict = {}
    if p.suffix.lower() == ".dng":
        from ..color.management import classify_dng
        dng_facts = classify_dng(p)
    try:
        t, meta = decode_raw(p)
    except Exception as e:  # noqa: BLE001
        name = type(e).__name__
        if "Unsupported" in name or "Unsupported" in str(e):
            raise ReadError(
                f"路径：{p}\n原因：LibRaw 不支持该 RAW 变体（如 Nikon Z8 高效 NEF/"
                f"AV1 压缩，docs/blockers.md B-1）\n"
                f"建议：用相机厂商工具导出 16 位线性 TIFF 后处理；"
                f"不会用内嵌 JPEG 冒充 RAW 解码（整改 §1.5）") from e
        raise ReadError(f"路径：{p}\n原因：{name}: {e}") from e
    if dng_facts:
        # v5 §4.2：mosaic DNG 走 RAW 解码；Linear DNG 按其矩阵/白点/位深/传递函数处理
        meta["kind"] = "dng"
        meta["dng_kind"] = dng_facts.get("dng_kind", "unknown")
        meta["dng_version"] = dng_facts.get("dng_version", "")
        meta["color_matrix"] = dng_facts.get("color_matrix")
        meta["forward_matrix"] = dng_facts.get("forward_matrix")
        meta["profile_sha256"] = dng_facts.get("profile_sha256", "")
        meta["dng_facts"] = dng_facts
        if meta["dng_kind"] == "linear":
            meta["input_encoding_detail"] = (
                "Linear DNG：已去马赛克的 RGB 数据，按 ForwardMatrix/ColorMatrix "
                "与位深处理，不走 mosaic 解马赛克路径")
    return LoadResult(t, meta,
                      ["RAW 已按显式参数解码为线性 sRGB-D65 RGB（gamma=1、"
                       "相机矩阵、LibRaw 默认 as-shot 乘数；逐通道增益被 "
                       "Dmin 归一化抵消）"])


# ---------------------------------------------------------------- 缩略图


def load_thumbnail(path: str | Path, max_dim: int = 128,
                   tiff_encoding: str | None = None):
    """低分辨率缩略图（只读；后台线程调用，失败抛异常由调用方吞掉）。

    RAW/DNG 用 half_size 快速解码；TIFF 用行步进抽样避免整帧常驻内存；
    返回 (max_dim, max_dim) 以内的 QImage（sRGB 显示编码）。
    """
    from PySide6.QtGui import QImage

    p = Path(path)
    ext = p.suffix.lower()
    if ext in RAW_EXTS:
        import rawpy

        with rawpy.imread(str(p)) as r:
            params = rawpy.Params(half_size=True, output_bps=8,
                                  no_auto_bright=True, user_flip=-1)
            arr = r.postprocess(params)
    elif ext in (".tif", ".tiff"):
        import tifffile

        with tifffile.TiffFile(str(p)) as t:
            pg = t.pages[0]
            h, w = pg.imagelength, pg.imagewidth
            step = max(1, int(np.ceil(max(h, w) / float(max_dim * 2))))
            arr = pg.asarray()[::step, ::step]
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        elif arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
            arr = np.transpose(arr, (1, 2, 0))
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        norm, _bits = _normalize_bitdepth(arr)
        cls = classify_tiff(p)
        enc = cls.encoding
        if enc == InputEncoding.UNKNOWN:
            enc = InputEncoding(tiff_encoding or "srgb")
        lin = decode_linear(np.clip(norm, 0.0, 1.0), enc)
        arr = (np.clip(linear_to_srgb(lin), 0, 1) * 255 + 0.5).astype(np.uint8)
    else:
        from PIL import Image

        with Image.open(str(p)) as im:
            im.thumbnail((max_dim, max_dim))
            arr = np.asarray(im.convert("RGB"))
    arr = np.ascontiguousarray(arr[..., :3].astype(np.uint8))
    h, w, _ = arr.shape
    img = QImage(arr.data, w, h, 3 * w, QImage.Format_RGB888)
    return img.copy()


# ---------------------------------------------------------------- 导出

def _verify_written_icc(path: Path, expect_icc: bytes | None,
                        expect_encoding: OutputEncoding) -> None:
    """写后重读：ICC 存在且与声明一致；不一致抛错（调用方事务清理）。"""
    ext = path.suffix.lower()
    if ext in (".tif", ".tiff"):
        import tifffile

        with tifffile.TiffFile(str(path)) as t:
            tag = t.pages[0].tags.get(34675)
            got = bytes(tag.value) if tag is not None else None
    else:
        from PIL import Image

        with Image.open(str(path)) as im:
            got = im.info.get("icc_profile")
    if expect_icc is None:
        if got:
            raise RuntimeError(f"导出声明无 ICC 但文件含 ICC：{path}")
        return
    if not got:
        raise RuntimeError(f"导出缺少声明的 ICC profile：{path}")
    if icc_sha256(bytes(got)) != icc_sha256(expect_icc):
        raise RuntimeError(f"导出 ICC 与声明不一致：{path}")


def save_tiff16(path: str | Path, rgb: np.ndarray, encoding: OutputEncoding, *,
                sources: list[Path] | None = None,
                overwrite: bool = False, embed_icc: bool | None = None) -> Path:
    """16 位 TIFF 导出。encoding 必须显式枚举（整改 F-2：拒绝语义不明数组）。

    SRGB：像素=sRGB 编码，嵌入 sRGB ICC；LINEAR：像素=线性，默认不嵌 ICC
    （sidecar 必须声明 linear）。
    """
    import tifffile

    from .atomic import atomic_write_bytes, ensure_not_source

    if not isinstance(encoding, OutputEncoding):
        raise TypeError(
            f"save_tiff16 需要 OutputEncoding 枚举，得到 {encoding!r}；"
            f"禁止把未声明语义的数组当 TIFF 数据（整改 P0-B）")
    path = Path(path)
    if sources:
        ensure_not_source(path, sources)
    out = encode_output(rgb, encoding)
    icc = srgb_icc_bytes() if (encoding == OutputEncoding.SRGB
                               and embed_icc is not False) else None
    arr = (np.clip(out, 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16)
    buf = _io.BytesIO()
    kw = {"photometric": "rgb"}
    if icc is not None:
        kw["iccprofile"] = icc
    tifffile.imwrite(buf, arr, **kw)
    atomic_write_bytes(path, buf.getvalue(), overwrite=overwrite)
    _verify_written_icc(path, icc, encoding)
    return path


def save_jpeg8(path: str | Path, rgb: np.ndarray, *,
               encoding: OutputEncoding = OutputEncoding.SRGB,
               quality: int = 95,
               sources: list[Path] | None = None,
               overwrite: bool = False) -> Path:
    """8 位 JPEG 导出（固定 8 位，界面必须明确显示这一点，§2.1.6）。

    JPEG 只能是显示编码：encoding 仅允许 SRGB。
    """
    from PIL import Image

    from .atomic import atomic_write_bytes, ensure_not_source

    if encoding != OutputEncoding.SRGB:
        raise ValueError(f"JPEG 只能是 sRGB 显示编码，得到 {encoding!r}")
    path = Path(path)
    if sources:
        ensure_not_source(path, sources)
    out = encode_output(rgb, OutputEncoding.SRGB)
    arr = (np.clip(out, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    img = Image.fromarray(arr, "RGB")
    icc = srgb_icc_bytes()
    buf = _io.BytesIO()
    # 4:4:4 无色度二次采样：负片色彩边缘（去色罩核心）不承受二次色度损失
    img.save(buf, format="JPEG", quality=int(quality), icc_profile=icc,
             subsampling=0)
    atomic_write_bytes(path, buf.getvalue(), overwrite=overwrite)
    _verify_written_icc(path, icc, OutputEncoding.SRGB)
    return path
