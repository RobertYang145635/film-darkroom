"""持久状态选择：确认基线 vs 工作草稿（v6.2 §5.2）。

为什么要单独一个模块
--------------------
v6.1 审计 F4 的根因是"确认标注无条件优先于草稿"：`_annotation_for()` 先看
`is_user_verified()`，命中就直接 return，**草稿永远轮不到**；反过来若草稿
无条件优先，用户签过字的基线又会被一次误操作的工作稿无声覆盖。两个方向都
错在同一个地方——没有"草稿派生自哪一版确认基线"这个身份信息。

规则（可测、可复述）
--------------------
| 现场 | 结果 |
| --- | --- |
| 无确认、无草稿 | none：用自动检测结果，需要用户确认 |
| 只有确认 | confirmed：确认基线生效 |
| 只有草稿 | work_draft：草稿生效，但状态仍是 draft（不算签字） |
| 两者都有，草稿指纹 == 确认内容指纹 | work_draft：草稿是在当前基线上做的后续工作，生效 |
| 两者都有，草稿没有指纹（旧版草稿）或指纹不符 | conflict：**确认基线生效**，草稿原地保留，标记需要重新确认 |
| 源图哈希变了 | stale_source：该帧的标定全部失效，重新检测 |

冲突时**绝不**把草稿几何并进确认基线，也**绝不**删除草稿：只做"哪个生效"
的决定，并把原因写进 UI 与 sidecar。
"""

from __future__ import annotations

import hashlib
import json
import numbers
from dataclasses import dataclass
from typing import Any

import numpy as np                       # _canonical 需识别 numpy 标量

from .annotations import (STATUS_VERIFIED, Annotation)

SOURCE_NONE = "none"
SOURCE_CONFIRMED = "confirmed"
SOURCE_WORK_DRAFT = "work_draft"
SOURCE_CONFLICT = "conflict"
SOURCE_STALE = "stale_source"

#: 基线内容指纹只覆盖"会影响成像与几何"的字段：备注、签字日期、
#: 草稿自身的指纹/修订都不进指纹，否则每次写盘都自认冲突。
#:
#: 审查修复（2026-09-08）：`status` 原先**在**指纹字段里，但它是签字标签
#: 而非内容，且 `Annotation.from_dict()` 会调 `reconcile_status()` 就地改写
#: 它 —— 于是同一份几何的「内存对象」与「落盘读回对象」指纹必然不同
#: （实测 稳定=False）。谁哪天用内存里的 Annotation 盖一次 `base_annotation_sha`，
#: 该帧此后每次重启都会判 conflict、工作草稿永久失效（正是本模块要根治的
#: F4 类故障换个触发器复活）。签字与否由 `is_confirmed()` 单独判，不进指纹。
_FINGERPRINT_FIELDS = (
    "file", "roll", "rotation_degrees", "coordinate_space",
    "output_crop_roi", "image_area_roi", "base_sample_rois", "neutral_refs",
    "exclude_mask_rois", "source_sha256", "fine_rotation_degrees",
    "mirror_horizontal", "keep_rebate",
)


def _canonical(value: Any) -> Any:
    """归一化为「内容」表示：tuple→list、numpy/整数标量→float、bool 保留。

    审查修复（2026-09-08）：`json.dumps(..., default=str)` 下 numpy 整数被写成
    字符串 `"100"`、python int 写成数字 `100`、float 写成 `100.0` —— 同一份
    ROI 因此算出**三个**不同指纹（实测）。指纹只能对内容敏感，不能对
    落盘/取数路径决定的表示形式敏感，否则 conflict 判定形同抛硬币。
    """
    if isinstance(value, (tuple, list)):
        return [_canonical(v) for v in value]
    if isinstance(value, dict):
        return {k: _canonical(value[k]) for k in sorted(value)}
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, numbers.Integral):       # 含 np.integer
        return float(int(value))
    if isinstance(value, numbers.Real):           # 含 np.float64（本就是 float 子类）
        return float(value)
    return value


def content_fingerprint(ann: Annotation) -> str:
    """确认基线的内容指纹（sha256, 16 hex 足够人工核对且抗碰撞够用）。"""
    proj = {k: _canonical(getattr(ann, k, None)) for k in _FINGERPRINT_FIELDS}
    blob = json.dumps(proj, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


@dataclass
class ResolvedState:
    """一帧的持久状态解析结果（生效标注 + 为什么 + 需要用户做什么）。"""

    source: str = SOURCE_NONE
    annotation: Annotation | None = None      #: 生效标注（几何/方向来源）
    draft: Annotation | None = None           #: 工作草稿（可能未生效）
    confirmed: Annotation | None = None       #: 确认基线（可能未生效）
    needs_reconfirm: bool = False
    reason: str = ""
    confirmed_fingerprint: str = ""
    draft_fingerprint_match: bool | None = None
    draft_revision: int = 0

    @property
    def has_persistent_state(self) -> bool:
        return self.source not in (SOURCE_NONE,)

    def to_dict(self) -> dict:
        """可直接进 sidecar / UI 的纯数据（不含标注对象本体）。"""
        return {
            "source": self.source,
            "needs_reconfirm": self.needs_reconfirm,
            "reason": self.reason,
            "confirmed_fingerprint": self.confirmed_fingerprint,
            "draft_fingerprint_match": self.draft_fingerprint_match,
            "draft_revision": self.draft_revision,
            "has_confirmed": self.confirmed is not None,
            "has_draft": self.draft is not None,
        }


def resolve_state(*, confirmed: Annotation | None, draft: Annotation | None,
                  source_sha256: str = "") -> ResolvedState:
    """按上表决定哪一份持久状态生效。纯函数，不读盘、不写盘。"""
    # 源图变了：两份标定都不再对应当前像素
    if source_sha256 and (
            (confirmed is not None and confirmed.source_sha256
             and confirmed.source_sha256 != source_sha256)
            or (draft is not None and draft.source_sha256
                and draft.source_sha256 != source_sha256)):
        return ResolvedState(
            source=SOURCE_STALE, annotation=None, draft=draft,
            confirmed=confirmed, needs_reconfirm=True,
            reason="源图哈希与标定记录不一致（文件被替换或重新扫描），"
                   "已按自动检测重新分析，需重新确认三类区域")

    if confirmed is None and draft is None:
        return ResolvedState(source=SOURCE_NONE, needs_reconfirm=True,
                             reason="尚无确认标注与工作草稿，按自动检测给出初值")
    if confirmed is None:
        return ResolvedState(
            source=SOURCE_WORK_DRAFT, annotation=draft, draft=draft,
            needs_reconfirm=True, reason="只有工作草稿（未签字），可继续调整",
            draft_revision=int(getattr(draft, "draft_revision", 0) or 0))
    if draft is None:
        fp = content_fingerprint(confirmed)
        return ResolvedState(
            source=SOURCE_CONFIRMED, annotation=confirmed, confirmed=confirmed,
            needs_reconfirm=bool(confirmed.needs_reconfirm),
            reason="确认基线生效", confirmed_fingerprint=fp)

    fp = content_fingerprint(confirmed)
    draft_base = str(getattr(draft, "base_annotation_sha", "") or "")
    match = bool(draft_base) and draft_base == fp
    rev = int(getattr(draft, "draft_revision", 0) or 0)
    if match:
        return ResolvedState(
            source=SOURCE_WORK_DRAFT, annotation=draft, draft=draft,
            confirmed=confirmed, needs_reconfirm=True,
            reason=f"工作草稿在当前确认基线（{fp}）上，草稿生效",
            confirmed_fingerprint=fp, draft_fingerprint_match=True,
            draft_revision=rev)
    return ResolvedState(
        source=SOURCE_CONFLICT, annotation=confirmed, draft=draft,
        confirmed=confirmed, needs_reconfirm=True,
        reason=("工作草稿缺少当前确认基线的指纹"
                if not draft_base else
                f"工作草稿基于旧确认基线 {draft_base}，当前基线 {fp}"),
        confirmed_fingerprint=fp,
        draft_fingerprint_match=None if not draft_base else False,
        draft_revision=rev)


def is_confirmed(ann: Annotation | None) -> bool:
    """只有 status=verified 且 verified_by=user 才算用户签字。"""
    return bool(ann is not None and ann.status == STATUS_VERIFIED
                and getattr(ann, "verified_by", "") == "user")
