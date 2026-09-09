"""可写存储位置的单一入口（v6.2 §3 阶段 A）。

问题
----
标注目录与 GUI 草稿目录此前是各自模块里的硬编码 Path，测试/冒烟脚本
要么只重定向其中一个（导致 `outputs/drafts/` 被测试写入真实样片的工作稿），
要么完全无法在子进程中复现隔离（冷启动双进程测试必需）。

规则
----
- 默认仍落在仓库内（`fixtures/annotations`、`outputs/drafts`），行为不变；
- 任何可写目录都可以被环境变量整体改道；环境变量由子进程继承，因此
  "起一个新进程重新打开窗口" 的冷启动测试可以与父进程共用同一个临时根，
  却不共享任何内存对象；
- 解析优先级：**显式改道（模块全局）> 环境变量 > 仓库默认**（与 resolve()
  一致；旧版本此处的两行注释写反过，见下方 resolve 的说明）。
  保留模块全局这一层是因为既有测试与审计脚本用
  `patch.object(annotations, "ANNOTATIONS_DIR", ...)` 改路径，不能失效。

绝不静默创建：只有 `ensure=True` 时才 mkdir，避免只读检查留下空目录。
"""

from __future__ import annotations

import contextlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

#: 仓库/安装根。此前 11 处各自用 Path(__file__).parents[3] 反推，深度写死在各自
#: 层级上：模块挪目录、装进 site-packages 或被打包后各处的「根」会互相不一致
#: （storage 自己是 parents[2]）。统一从这里取，改解析规则只改一处。
REPO_ROOT = Path(__file__).resolve().parents[2]


def repo_path(*parts: str) -> Path:
    """仓库根下的路径（唯一入口，替代散落的 parents[3] 反推）。"""
    return REPO_ROOT.joinpath(*parts)

ENV_ANNOTATIONS = "NEGATIVE_LAB_ANNOTATIONS_DIR"
ENV_DRAFTS = "NEGATIVE_LAB_DRAFTS_DIR"

#: 所有可改道的写目录（用于清单与"前后哈希一致"检查）
DEFAULTS = {
    "annotations": REPO_ROOT / "fixtures" / "annotations",
    "drafts": REPO_ROOT / "outputs" / "drafts",
}


def env_override(var: str) -> Path | None:
    """返回环境变量指定的目录；未设置/为空返回 None。"""
    raw = os.environ.get(var)
    if raw is None or not str(raw).strip():
        return None
    p = Path(str(raw)).expanduser()
    return p if p.is_absolute() else (REPO_ROOT / p)


def resolve(var: str, default_key: str, *, module_global: Path | None = None,
            ensure: bool = False) -> Path:
    """解析优先级：**显式改道 > 环境变量 > 仓库默认**。

    - 显式改道：调用方把模块全局（如 annotations.ANNOTATIONS_DIR）改成不是
      仓库默认的值 —— 这是测试/脚本的一次性定向重定向，必须赢，否则会话级
      环境变量会把它无声吞掉（实测让 10 个标注测试写进错误的目录）。
    - 环境变量：整进程隔离（子进程继承），用于测试会话与冒烟脚本。
    - 仓库默认：正常用户使用位置。
    """
    p: Path | None = None
    if module_global is not None:
        mg = Path(module_global)
        if mg != DEFAULTS[default_key]:
            p = mg                      # 显式改道，最高优先
    if p is None:
        p = env_override(var)
    if p is None:
        p = DEFAULTS[default_key]
    if ensure:
        p.mkdir(parents=True, exist_ok=True)
    return p


def annotations_dir(*, module_global: Path | None = None, ensure: bool = False) -> Path:
    return resolve(ENV_ANNOTATIONS, "annotations",
                   module_global=module_global, ensure=ensure)


def drafts_dir(*, module_global: Path | None = None, ensure: bool = False) -> Path:
    return resolve(ENV_DRAFTS, "drafts", module_global=module_global, ensure=ensure)


@dataclass
class Storage:
    root: Path
    annotations: Path
    drafts: Path

    def describe(self) -> dict:
        return {"root": str(self.root),
                "annotations": str(self.annotations),
                "drafts": str(self.drafts)}


@contextlib.contextmanager
def isolated_storage(root: str | Path | None = None, *,
                     copy_annotations: bool = False):
    """把全部可写目录改道到 root（默认新建临时目录），退出时恢复。

    用途：测试/冒烟脚本运行时绝不允许写真实的用户工作稿与确认标注。
    嵌套安全（保存并恢复原环境变量值）。

    copy_annotations=True 时把仓库内既有标注**复制**进临时根：读得到用户
    标定（冒烟才有代表性），但任何写入仍落在临时目录。只读与可写必须能分开：
    只有开关、没有粒度时，会误把"被测行为本来就这样"当成"隔离改坏了它"。

    实测教训（2026-09-08）：隔离后 gui_smoke 的高光裁切门禁失败，我一度推断
    是"丢了真实标注导致几何变化"。用 copy_annotations=True 重跑，数字逐位相同
    （high = [0.0343, 0.0449, 0.0]）——假设被自己的实验推翻，门禁失败与标注
    在不在无关。详见 outputs/v6_2/20260908-state-pick-fixes/v6.2_handover.md 3.1。
    """
    if root is None:
        tmp = Path(tempfile.mkdtemp(prefix="nl-storage-"))
    else:
        tmp = Path(root).expanduser().resolve()
    ann = tmp / "annotations"
    dr = tmp / "drafts"
    ann.mkdir(parents=True, exist_ok=True)
    dr.mkdir(parents=True, exist_ok=True)
    if copy_annotations and DEFAULTS["annotations"].exists():
        for f in sorted(DEFAULTS["annotations"].glob("*.json")):
            try:
                (ann / f.name).write_bytes(f.read_bytes())
            except OSError:      # 复制失败退化为"无标注"，不静默改道一半
                pass
    saved = {k: os.environ.get(k) for k in (ENV_ANNOTATIONS, ENV_DRAFTS)}
    os.environ[ENV_ANNOTATIONS] = str(ann)
    os.environ[ENV_DRAFTS] = str(dr)
    try:
        yield Storage(root=tmp, annotations=ann, drafts=dr)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def inventory(paths: list[Path]) -> dict:
    """目录内文件的 (相对路径 -> 大小, mtime_ns) 清单，用于证明未被写入。"""
    import hashlib
    out: dict[str, tuple[int, int, str]] = {}
    for base in paths:
        base = Path(base)
        if not base.exists():
            continue
        for f in sorted(base.rglob("*")):
            if f.is_dir():
                continue
            st = f.stat()
            out[str(f.relative_to(base))] = (
                st.st_size, st.st_mtime_ns,
                hashlib.sha256(f.read_bytes()).hexdigest()[:16])
    return out
