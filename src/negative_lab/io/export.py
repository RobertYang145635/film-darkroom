"""事务性三件套导出（v4 §9 P0-G：共享 basename、格式开关、真部分提交回滚）。

顺序：选定整组唯一 basename -> staging 写入（按开关 TIFF/JPEG/sidecar）
-> 重读校验（ICC、形状、schema 往返）-> 逐件提交（可注入部分提交失败）
-> 任一步失败：删除本事务已移动的全部文件 + 清空 staging；
事务前已有文件不受影响。_inject_fail 为失败注入测试钩子。
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


def _rollback(staging: Path, committed: list[Path]) -> list[Path]:
    """删除本事务已移动的文件；删除失败的残留路径返回给调用方报错。"""
    leftover: list[Path] = []
    for f in committed:
        try:
            f.unlink()
        except OSError:
            leftover.append(f)
    shutil.rmtree(staging, ignore_errors=True)
    return leftover


def export_triplet(outdir: Path, stem: str, positive_linear: np.ndarray,
                   sidecar_payload: dict, *, sources: list[Path],
                   jpeg_quality: int = 95,
                   export_tiff: bool = True, export_jpeg: bool = True,
                   _inject_fail: str | None = None) -> list[Path]:
    """按格式开关导出（sidecar 恒在）。返回最终路径列表。

    v4 §9.1：整组共享一个 basename（s-positive-1.tif/jpg/.negative.json），
    suffix 只分配一次；候选检查覆盖本次启用的全部目标文件。
    v4 §9.2：False/False 在处理前失败。
    """
    if not export_tiff and not export_jpeg:
        raise ExportError("export_tiff 与 export_jpeg 不能同时为 False"
                          "（sidecar-only 的'成功任务'无意义，v4 §9.2）")
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    staging = outdir / f".staging-{stem}-{uuid.uuid4().hex[:8]}"
    staging.mkdir()
    committed: list[Path] = []
    try:
        # ---- 共享 basename：一次分配（§9.1）
        base = f"{stem}-positive"
        i = 0
        while True:
            cand = [outdir / f"{base}{'' if i == 0 else f'-{i}'}.{ext}"
                    for ext in (("tif",) if export_tiff else ())
                    + (("jpg",) if export_jpeg else ()) + ("negative.json",)]
            if not any(c.exists() for c in cand):
                break
            i += 1
        suffix = "" if i == 0 else f"-{i}"
        tif = staging / f"{base}{suffix}.tif" if export_tiff else None
        jpg = staging / f"{base}{suffix}.jpg" if export_jpeg else None
        side = staging / f"{base}{suffix}.negative.json"

        if tif is not None:
            save_tiff16(tif, positive_linear, OutputEncoding.SRGB,
                        sources=sources)
            if _inject_fail == "tiff":
                raise OSError("注入故障：tiff 写后")
        if jpg is not None:
            if _inject_fail == "jpeg":
                raise OSError("注入故障：jpeg")
            save_jpeg8(jpg, positive_linear, quality=jpeg_quality,
                       sources=sources)
        if _inject_fail == "sidecar":
            raise OSError("注入故障：sidecar")
        from .atomic import atomic_write_bytes    # 原子写：报告/配方与成品同等对待
        atomic_write_bytes(side, json.dumps(sidecar_payload, ensure_ascii=False,
                                           indent=2).encode("utf-8"),
                           overwrite=True)

        # ---- 重读校验（写完不验证 = 没写）
        if tif is not None:
            import tifffile
            with tifffile.TiffFile(tif) as tf:
                pg = tf.pages[0]
                if pg.tags.get(34675) is None:
                    raise ExportError("staging TIFF 缺 ICC profile")
                if pg.asarray().shape[:2] != positive_linear.shape[:2]:
                    raise ExportError("staging TIFF 形状不符")
        if jpg is not None:
            from PIL import Image
            with Image.open(jpg) as im:
                if im.size != (positive_linear.shape[1],
                               positive_linear.shape[0]):
                    raise ExportError("staging JPEG 尺寸不符")
        doc = json.loads(side.read_text(encoding="utf-8"))
        if doc.get("schema_version") != sidecar_payload.get("schema_version"):
            raise ExportError("staging sidecar 往返不一致")

        # ---- 提交（v4 §9.3：真正可测的部分提交回滚）
        finals = [outdir / f.name for f in
                  ([x for x in (tif, jpg) if x is not None] + [side])]
        for k, sf in enumerate([x for x in (tif, jpg) if x is not None] + [side]):
            if _inject_fail == f"commit_after_{k}":
                raise OSError(f"注入故障：提交第 {k} 件后")
            shutil.move(str(sf), finals[k])
            committed.append(finals[k])
        return committed
    except ExportError as e:
        leftover = _rollback(staging, committed)
        if leftover:
            raise ExportError(f"{e}；回滚删除失败，残留："
                              f"{[str(p) for p in leftover]}") from e
        raise
    except Exception as e:  # noqa: BLE001
        leftover = _rollback(staging, committed)
        msg = f"事务导出失败，已回滚：{e}"
        if leftover:
            msg += f"；残留：{[str(p) for p in leftover]}"
        raise ExportError(msg) from e
    finally:
        shutil.rmtree(staging, ignore_errors=True)
