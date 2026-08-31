"""原子导出与源文件保护（规格 §9、§8.1.7）。

规则：
- 原始文件永远只读；导出目标解析后与任何源文件相同则拒绝。
- 先写临时文件，成功后 os.replace 原子替换目标新文件。
- 临时文件写入/替换失败时，目标与源文件保持原状。
"""

from __future__ import annotations

import os
import secrets
from pathlib import Path


def ensure_not_source(dst: Path, sources: list[Path]) -> None:
    """导出目标不得覆盖源文件（大小写不敏感比较，Windows 语义）。"""
    d = _norm(dst)
    for s in sources:
        if _norm(s) == d:
            raise ValueError(f"导出目标与源文件相同，拒绝覆盖：{dst} == {s}")


def _norm(p: Path) -> str:
    try:
        return os.path.normcase(os.path.realpath(p))
    except OSError:
        return os.path.normcase(str(p.resolve()))


def atomic_write_bytes(dst: Path, data: bytes, overwrite: bool = False) -> None:
    """临时文件 + 原子替换。目标已存在且 overwrite=False 时拒绝。"""
    dst = Path(dst)
    if dst.exists() and not overwrite:
        raise FileExistsError(f"目标已存在，默认不覆盖：{dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f"{dst.name}.tmp-{secrets.token_hex(6)}")
    try:
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        if dst.exists():  # overwrite=True 时替换既有目标（仍非源文件，已由调用方校验）
            os.replace(tmp, dst)
        else:
            os.rename(tmp, dst)
    except BaseException:
        # 失败清理：删除半成品临时文件，目标保持不存在/原状
        try:
            if tmp.exists():
                os.unlink(tmp)
        except OSError:
            pass
        raise
