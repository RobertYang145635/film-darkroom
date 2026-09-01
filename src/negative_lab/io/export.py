"""事务性三件套导出（整改 v2 I-4 / v3 P0-H）。

顺序：staging 写入（TIFF/JPEG/sidecar）-> 重读校验（ICC、形状、schema）
-> 提交（移动到最终名）-> 任一步失败：清理 staging 与本次已提交工件，
绝不留半套结果。_inject_fail 为失败注入测试钩子。
"""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

import numpy as np

from .images import save_jpeg8, save_tiff16
from ..color.management import OutputEncoding


class ExportError(RuntimeError):
    pass


def _unique(p: Path) -> Path:
    if not p.exists():
        return p
    i = 1
    while True:
        cand = p.with_name(f"{p.stem}-{i}{p.suffix}")
        if not cand.exists():
            return cand
        i += 1


def export_triplet(outdir: Path, stem: str, positive_linear: np.ndarray,
                   sidecar_payload: dict, *, sources: list[Path],
                   jpeg_quality: int = 95,
                   _inject_fail: str | None = None) -> list[Path]:
    """返回最终三件套路径。失败抛 ExportError 且不留任何本次工件。"""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    staging = outdir / f".staging-{stem}-{uuid.uuid4().hex[:8]}"
    staging.mkdir()
    committed: list[Path] = []
    try:
        tif = staging / f"{stem}-positive.tif"
        save_tiff16(tif, positive_linear, OutputEncoding.SRGB, sources=sources)
        if _inject_fail == "tiff":
            raise OSError("注入故障：tiff 写后")
        jpg = staging / f"{stem}-positive.jpg"
        if _inject_fail == "jpeg":
            raise OSError("注入故障：jpeg")
        save_jpeg8(jpg, positive_linear, quality=jpeg_quality, sources=sources)
        side = staging / f"{stem}-positive.tif.negative.json"
        if _inject_fail == "sidecar":
            raise OSError("注入故障：sidecar")
        side.write_text(json.dumps(sidecar_payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")
        # ---- 重读校验（写完不验证 = 没写）
        import tifffile
        with tifffile.TiffFile(tif) as tf:
            pg = tf.pages[0]
            if pg.tags.get(34675) is None:
                raise ExportError("staging TIFF 缺 ICC profile")
            arr = pg.asarray()
            if arr.shape[:2] != positive_linear.shape[:2]:
                raise ExportError("staging TIFF 形状不符")
        from PIL import Image
        with Image.open(jpg) as im:
            if im.size != (positive_linear.shape[1], positive_linear.shape[0]):
                raise ExportError("staging JPEG 尺寸不符")
        doc = json.loads(side.read_text(encoding="utf-8"))
        if doc.get("schema_version") != sidecar_payload.get("schema_version"):
            raise ExportError("staging sidecar 往返不一致")
        # ---- 提交
        if _inject_fail == "commit":
            raise OSError("注入故障：提交阶段")
        for f in (tif, jpg, side):
            final = _unique(outdir / f.name)
            shutil.move(str(f), final)
            committed.append(final)
        return committed
    except ExportError:
        _rollback(staging, committed)
        raise
    except Exception as e:  # noqa: BLE001
        _rollback(staging, committed)
        raise ExportError(f"事务导出失败，已回滚：{e}") from e
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _rollback(staging: Path, committed: list[Path]) -> None:
    for f in committed:
        try:
            f.unlink()
        except OSError:
            pass
