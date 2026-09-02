"""导入会话构建（v5 §2.1.1/§2.1.2）。

导入只做**元数据探针 + 哈希**，不解码全图：一个 272 MB 的 16 位 TIFF
在浏览阶段不需要像素。像素在裁切/转换页按需加载。
"""

from __future__ import annotations

import time
from pathlib import Path

from ..io.probe import probe_file, sha256_of
from .models import (FrameRecord, FrameStatus, GeometryRegions, ImportSession,
                     RollGroup)

SUPPORTED_EXTS = (".tif", ".tiff", ".nef", ".dng", ".cr2", ".cr3", ".arw",
                  ".raf", ".orf", ".rw2", ".raw")


def iter_supported_files(root: Path) -> list[Path]:
    """递归收集受支持输入；跳过 .downloading 残缺文件。"""
    root = Path(root)
    if root.is_file():
        return [root]
    return sorted(p for p in root.rglob("*")
                  if p.suffix.lower() in SUPPORTED_EXTS
                  and not p.name.lower().endswith(".downloading"))


def make_frame_record(path: Path, *, root: Path | None = None,
                      sha256: str | None = None,
                      probe: bool = True) -> FrameRecord:
    """单文件 -> FrameRecord（imported / decode_error）。

    `source_path` 语义与标注/manifest 一致：**仓库相对路径**；
    文件在仓库之外时保留绝对路径。两种情况都另外记录 `abs_path`，
    供 UI/导出在没有会话根信息时精确定位。
    """
    path = Path(path)
    repo_root = Path(__file__).resolve().parents[3]
    rel = path.as_posix()
    base_root = Path(root).resolve() if root else repo_root
    try:
        rel = path.resolve().relative_to(base_root).as_posix()
    except ValueError:
        pass
    try:
        rel_repo = path.resolve().relative_to(repo_root).as_posix()
    except ValueError:
        rel_repo = None
    if rel_repo:
        rel = rel_repo
    digest = sha256 or sha256_of(path)
    facts: dict = {"path": path.as_posix(), "abs_path": path.resolve().as_posix(),
                   "size_bytes": path.stat().st_size}
    status = FrameStatus.IMPORTED.value
    warnings: list[str] = []
    if probe:
        pr = probe_file(path)
        facts.update({"probe_status": pr.status, "probe_reason": pr.reason,
                      "kind": pr.kind, "width": pr.width, "height": pr.height,
                      "bit_depth": pr.bit_depth, "camera": pr.camera,
                      "lens": pr.lens, "icc_profile": pr.icc_profile,
                      "decode_params": pr.decode_params})
        if pr.status not in ("ok",):
            status = FrameStatus.DECODE_ERROR.value
            warnings.append(f"探针状态 {pr.status}：{pr.reason}")
    return FrameRecord(source_path=rel, source_sha256=digest,
                       input_facts=facts, status=status, warnings=warnings,
                       geometry=GeometryRegions())


def import_session(root_or_files, *, root: Path | None = None,
                   probe: bool = True, sha256_cache: dict | None = None
                   ) -> ImportSession:
    """混合文件夹导入 -> ImportSession（全部 needs_group，等用户分卷）。"""
    p = Path(root_or_files)
    files = [p] if p.is_file() else iter_supported_files(p)
    root = root or (p.parent if p.is_file() else p)
    cache = sha256_cache or {}
    sess = ImportSession(created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                   time.gmtime()))
    for f in files:
        digest = cache.get(str(f)) or cache.get(f.as_posix())
        rec = make_frame_record(f, root=root, sha256=digest, probe=probe)
        rec.status = (FrameStatus.NEEDS_GROUP.value
                      if rec.status == FrameStatus.IMPORTED.value
                      else rec.status)
        sess.files.append(rec)
    return sess


def suggest_groups_by_directory(session: ImportSession, *,
                                apply: bool = False) -> list[tuple[str, list[str]]]:
    """按直接父目录给出**建议**分组；是否采纳由用户决定（v5 §0.2.6）。"""
    buckets: dict[str, list[str]] = {}
    for f in session.files:
        key = Path(f.source_path).parent.as_posix()
        buckets.setdefault(key, []).append(f.source_path)
    suggested = [(Path(k).name or k, v) for k, v in sorted(buckets.items())]
    if apply:
        for name, paths in suggested:
            existing = next((r for r in session.roll_groups
                             if r.display_name == name), None)
            if existing is None:
                session.create_roll(name, paths)
            else:
                session.move_frames(paths, existing.id)
    return suggested


def roll_of(session: ImportSession, frame_path: str) -> RollGroup | None:
    f = session.frame(frame_path)
    return session.roll(f.roll_id) if f and f.roll_id else None
