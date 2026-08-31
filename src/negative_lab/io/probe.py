"""真实最小解码探针（整改规格 P0-A）。

decode_status 不得来自未经验证的字节启发式：必须实际执行最小解码
（RAW: imread + unpack + 缩略图尺寸级 postprocess；TIFF: tifffile 读取；
JPEG/PNG: Pillow），并返回稳定枚举：

    ok | unsupported | corrupt | missing_dependency | unknown

每个结果带简短 reason 与解码参数（相机、位深、尺寸、白平衡策略、
工作色域标记、ICC 状态），供 manifest 与批处理失败分类共用。
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from pathlib import Path

RASTER_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
RAW_EXTS = {".nef", ".cr2", ".cr3", ".arw", ".raf", ".orf", ".rw2", ".dng", ".raw"}


@dataclass
class ProbeResult:
    status: str                      # ok|unsupported|corrupt|missing_dependency|unknown
    reason: str = ""
    kind: str = ""                   # raw|tiff|jpeg|png|...
    width: int = 0
    height: int = 0
    bit_depth: int = 0
    camera: str = ""
    lens: str = ""
    decode_params: dict = field(default_factory=dict)
    icc_profile: str = "none"        # none|embedded(<hash8>)|unknown
    sha256: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def sha256_of(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def probe_file(path: str | Path) -> ProbeResult:
    """按扩展名分派真实解码探针。只读，不写任何文件。"""
    p = Path(path)
    ext = p.suffix.lower()
    if not p.exists():
        return ProbeResult("corrupt", "文件不存在")
    try:
        size = p.stat().st_size
    except OSError as e:
        return ProbeResult("unknown", f"stat 失败: {e}")
    if size < 64:
        return ProbeResult("corrupt", f"文件过小({size}B)")
    if ext in RAW_EXTS:
        return _probe_raw(p)
    if ext in (".tif", ".tiff"):
        return _probe_tiff(p)
    if ext in RASTER_EXTS:
        return _probe_pillow(p)
    return ProbeResult("unknown", f"扩展名 {ext!r} 不在支持列表")


def _probe_raw(p: Path) -> ProbeResult:
    try:
        import rawpy
    except ImportError:
        return ProbeResult("missing_dependency", "rawpy/LibRaw 未安装")
    try:
        with rawpy.imread(str(p)) as r:
            # 真实最小解码：unpack 原始传感器数据（Z8 AV1 高效 NEF 在这里抛
            # LibRawFileUnsupportedError，而 24 字节启发式会漏判）
            raw = r.raw_image_visible
            if raw is None or raw.size == 0:
                return ProbeResult("corrupt", "raw_image_visible 为空")
            h, w = raw.shape
            bits = int(r.bits) if getattr(r, "bits", 0) else 16
            lens = r.lens
            other = r.other
            # 再验证一次完整后处理路径（缩略图尺寸，控制耗时）
            params = rawpy.Params(use_camera_wb=False, no_auto_bright=True,
                                  half_size=True, output_bps=8)
            params.gamm = [1.0, 1.0, 1.0, 1.0]
            img = r.postprocess(params)
            if img is None or img.size == 0:
                return ProbeResult("corrupt", "postprocess 返回空")
            return ProbeResult(
                "ok", kind="raw", width=int(w), height=int(h), bit_depth=bits,
                camera=f"{lens.make} {r.model or ''}".strip() if hasattr(r, "model") else (lens.make or "unknown"),
                lens=lens.model or "",
                decode_params={
                    "decoder": f"rawpy/raw {rawpy.__version__} LibRaw {rawpy.libraw_version}",
                    "use_camera_wb": False,
                    "use_auto_wb": False,
                    "white_balance_policy": "none (unit multipliers; 密度测量要求)",
                    "gamma": [1.0, 1.0, 1.0, 1.0],
                    "output_color": "raw-linear (no color matrix applied)",
                    "no_auto_bright": True,
                    "half_size_probe": True,
                    "black_level": [int(v) for v in r.black_level_per_channel],
                    "white_level": int(r.white_level),
                    "iso": float(other.iso_speed),
                    "shutter": float(other.shutter_speed),
                    "aperture": float(other.aperture),
                },
                icc_profile="none (RAW 无 ICC 概念，线性传感器域)",
            )
    except Exception as e:  # noqa: BLE001
        name = type(e).__name__
        msg = str(e)[:160]
        if "Unsupported" in msg or "unsupported" in name.lower():
            return ProbeResult("unsupported", f"LibRaw 不支持该 RAW 变体: {msg}")
        if name in ("LibRawFileUnsupportedError", "LibRawRawFileTypeError"):
            return ProbeResult("unsupported", f"{name}: {msg}")
        if "not a RAW" in msg or "damaged" in msg.lower() or "truncat" in msg.lower():
            return ProbeResult("corrupt", f"{name}: {msg}")
        return ProbeResult("unknown", f"{name}: {msg}")


def _probe_tiff(p: Path) -> ProbeResult:
    try:
        import tifffile
        t = tifffile.TiffFile(str(p))
        pg = t.pages[0]
        bs = pg.bitspersample
        bits = int(bs[0] if isinstance(bs, (tuple, list)) else bs) if bs else 0
        photometric = str(pg.photometric)
        icc = "none"
        try:
            if pg.tags.get(34675) is not None:  # ICCProfile tag
                blob = pg.tags[34675].value
                icc = f"embedded({hashlib.sha256(bytes(blob)).hexdigest()[:8]})"
        except Exception:  # noqa: BLE001
            icc = "present(unreadable)"
        arr = pg.asarray()
        if arr.size == 0:
            return ProbeResult("corrupt", "TIFF 页为空")
        return ProbeResult("ok", kind="tiff", width=int(pg.imagewidth),
                           height=int(pg.imagelength), bit_depth=bits,
                           decode_params={"photometric": str(photometric),
                                          "compression": str(pg.compression),
                                          "shape": list(arr.shape),
                                          "dtype": str(arr.dtype)},
                           icc_profile=icc)
    except Exception as e:  # noqa: BLE001
        name = type(e).__name__
        if "TiffFile" in name or "decode" in str(e).lower() or "not a TIFF" in str(e).lower():
            return ProbeResult("corrupt", f"{name}: {str(e)[:160]}")
        return ProbeResult("unknown", f"{name}: {str(e)[:160]}")


def _probe_pillow(p: Path) -> ProbeResult:
    try:
        from PIL import Image
        with Image.open(str(p)) as im:
            im.verify()
        with Image.open(str(p)) as im:
            im.load()
            icc = "none"
            if im.info.get("icc_profile"):
                icc = f"embedded({hashlib.sha256(im.info['icc_profile']).hexdigest()[:8]})"
            return ProbeResult("ok", kind=(im.format or "img").lower(),
                               width=im.width, height=im.height,
                               bit_depth=8 if im.mode in ("RGB", "L") else 16,
                               decode_params={"mode": im.mode},
                               icc_profile=icc)
    except Exception as e:  # noqa: BLE001
        return ProbeResult("corrupt", f"{type(e).__name__}: {str(e)[:160]}")
