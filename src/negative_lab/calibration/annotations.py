"""人工标注读写（v5 §3.2/§5.2 重建：单一保存 API + schema v3 + 数据完整性）。

fixtures/annotations/<roll>__<stem>.json，只存小型 JSON，不复制大图。

坐标语义
--------
  coordinate_space:
    - "source": 坐标基于原始未旋转文件的像素空间 (row0,col0,row1,col1)。
    - "oriented": 坐标基于按 rotation_degrees 顺时针旋转后的空间。
  本模块的 ROI 一律写在 source 空间；rotation_degrees 为应用到 source 得到
  正确观看方向的**顺时针**角度 (0|90|180|270)。

schema v3（v5 §3.2）：三类互不相同的数据，片基取样区可以有多个
----------------------------------------------------------------
  output_crop_roi    最终导出构图（用户构图决定，可含片边/齿孔）
  image_area_roi     算法与统计区域（必须排除片边、齿孔、明显非画面区域）
  base_sample_rois   一个或多个未曝光片基取样区（不得含齿孔/边码/画面内容）
  neutral_refs       中性/颜色参考 [{roi|point, kind: bright|dark}]（可选）
  exclude_mask_rois  排除区域（灰尘、反光、非画面杂物）
  source_sha256      源图哈希；源图变化时标注失效
  schema_version     落盘 schema 版本（v3 = 3）

draft / verified
---------------
  status 是唯一真相源："draft" | "verified"；布尔 verified 由 status 派生。
  只有 verified_by == "user" 才算用户签字。agent 自动检测结果（含历史
  verified_by="agent-inspected"）**永远不构成签字**：
    - 加载：任何声明 verified 但来源不是 user 的记录，一律降级 draft
      （加载不得抛错：批处理会为每个样片读标注）。
    - 保存：已知 agent 来源降级为 draft；来源缺失/不明的 verified 声明
      不静默改写，直接由 validate() 拒绝并抛 ValueError。

保存契约
--------
  唯一公共保存 API：``save_annotation(rel, ann)``。保存前无条件
  ``ann.validate()``，无效数据抛 ValueError（绝不允许写成 verified），
  再通过"同目录临时文件 + os.replace"原子替换；替换失败保留原文件。

旧 schema 兼容
--------------
  schema_version 缺失或 < 3 的旧文件可以加载：
    - v1/v2 的单值 base_roi -> base_sample_rois=[base_roi]；
    - 旧 frame_roi -> image_area_roi 草稿、output_crop_roi 保持 None、
      needs_reconfirm=True、status 降级 draft（用途必须由用户重新确认）。
"""

from __future__ import annotations

import json
import math
import numbers
import os
import re
import sys
import tempfile
from dataclasses import dataclass, field, fields
from datetime import date
from pathlib import Path

ANNOTATIONS_DIR = Path(__file__).resolve().parents[3] / "fixtures" / "annotations"

VALID_ROT = (0, 90, 180, 270)
VALID_COORDINATE_SPACES = ("source", "oriented")

SCHEMA_VERSION = 3
LEGACY_SCHEMA_VERSIONS = (1, 2)
STATUS_DRAFT = "draft"
STATUS_VERIFIED = "verified"
VALID_STATUS = (STATUS_DRAFT, STATUS_VERIFIED)
USER_VERIFIER = "user"
#: 不构成用户签字的已知自动/代签来源（§8.4-5）
AGENT_VERIFIERS = ("agent-inspected", "agent", "auto", "script")

#: 最近一次 scan_annotations()/load_all_verified() 的诊断（损坏或非法文件）
LAST_LOAD_ERRORS: list[str] = []

_ROI_FIELDS = ("output_crop_roi", "image_area_roi", "base_roi", "frame_roi")
_ROI_LIST_FIELDS = ("base_sample_rois", "exclude_mask_rois")


def _is_number(v) -> bool:
    if isinstance(v, bool) or not isinstance(v, numbers.Real):
        return False
    return math.isfinite(float(v))


def _check_roi(name: str, roi) -> str | None:
    """单个 (y0,x0,y1,x1) ROI 的合法性；返回 ASCII 错误信息或 None。"""
    if roi is None:
        return None
    if isinstance(roi, (str, bytes)):
        return f"{name} must be a (y0,x0,y1,x1) sequence, got {roi!r}"
    try:
        vals = list(roi)
    except TypeError:
        return f"{name} must be a (y0,x0,y1,x1) sequence, got {roi!r}"
    if len(vals) != 4:
        return f"{name} must have 4 values, got {len(vals)}: {roi!r}"
    if not all(_is_number(v) for v in roi):
        return f"{name} values must be finite numbers, got {roi!r}"
    y0, x0, y1, x1 = (int(v) for v in roi)
    if min(y0, x0, y1, x1) < 0:
        return f"{name} has negative coordinates (out of bounds): {roi!r}"
    if y1 <= y0 or x1 <= x0:
        return f"{name} is reversed or empty: {roi!r}"
    return None


@dataclass
class Annotation:
    file: str                                   # 相对仓库根的样片路径
    roll: str
    rotation_degrees: int = 0                   # 顺时针
    coordinate_space: str = "source"            # source|oriented
    schema_version: int = SCHEMA_VERSION
    status: str = STATUS_DRAFT                  # draft|verified（唯一真相源）
    needs_reconfirm: bool = False               # 旧 frame_roi 迁移待用户确认
    output_crop_roi: tuple | None = None        # 最终导出构图
    image_area_roi: tuple | None = None         # 算法/统计区域
    base_sample_rois: list = field(default_factory=list)   # 未曝光片基取样区（多个）
    base_roi: tuple | None = None               # 已废弃：单值片基区（v2）
    neutral_refs: list = field(default_factory=list)     # [{roi|point, kind}]
    exclude_mask_rois: list = field(default_factory=list)
    source_sha256: str = ""
    reference_file: str | None = None
    notes: str = ""
    verified: bool = False                      # 由 status 派生（兼容字段）
    verified_by: str = ""                       # user | agent-inspected | ...
    verified_date: str = ""
    # v6 §6.1：几何变换与 ROI 一起持久化（附加可选字段，旧文件可读）
    fine_rotation_degrees: float = 0.0
    mirror_horizontal: bool = False
    frame_roi: tuple | None = None              # 已废弃：语义含混（v4 §8.6）
    neutral_rois: list = field(default_factory=list)    # 已废弃别名
    mask_rois: list = field(default_factory=list)       # 已废弃别名

    # ------------------------------------------------------------------ 迁移
    def migrate(self, *, legacy: bool) -> "Annotation":
        """就地迁移：旧别名 -> v3 字段；旧 schema 强制 draft + 需重新确认。"""
        if self.image_area_roi is None and self.frame_roi is not None:
            # frame_roi 用途含混，只能作为"画面分析区"草稿（§8.6）
            self.image_area_roi = tuple(self.frame_roi)
            self.needs_reconfirm = True
        if not self.base_sample_rois and self.base_roi is not None:
            self.base_sample_rois = [tuple(self.base_roi)]
        if not self.base_roi and self.base_sample_rois:
            self.base_roi = tuple(self.base_sample_rois[0])   # 兼容读侧
        if not self.neutral_refs and self.neutral_rois:
            self.neutral_refs = list(self.neutral_rois)
        if not self.exclude_mask_rois and self.mask_rois:
            self.exclude_mask_rois = [tuple(r) for r in self.mask_rois]
        self.neutral_refs = [_neutral_ref_dict(r) for r in self.neutral_refs]
        self.base_sample_rois = [tuple(r) for r in self.base_sample_rois
                                 if r is not None]
        self.exclude_mask_rois = [tuple(r) for r in self.exclude_mask_rois
                                  if r is not None]
        if legacy:
            # 旧文件里 frame_roi 的用途、以及是否用户签字都无从判断
            self.needs_reconfirm = True
            self.status = STATUS_DRAFT
            self.verified = False
        else:
            self.schema_version = SCHEMA_VERSION
        return self

    def reconcile_status(self, *, downgrade_unknown_provenance: bool = False) -> "Annotation":
        """verified 声明与 status 对齐；verified 布尔由 status 派生。"""
        declared = bool(self.verified) or self.status == STATUS_VERIFIED
        if declared:
            if self.verified_by == USER_VERIFIER:
                self.status = STATUS_VERIFIED
            elif downgrade_unknown_provenance or self.verified_by in AGENT_VERIFIERS:
                self.status = STATUS_DRAFT
            # 其余来源：不猜、不改写，交给 validate() 拒绝
        else:
            self.status = STATUS_DRAFT
        self.verified = self.status == STATUS_VERIFIED
        return self

    # ------------------------------------------------------------------ 序列化
    def to_dict(self) -> dict:
        d: dict = {}
        for f in fields(self):
            d[f.name] = getattr(self, f.name)
        for k in _ROI_FIELDS:
            if d[k] is not None:
                d[k] = list(d[k])
        for k in _ROI_LIST_FIELDS:
            d[k] = [list(r) for r in (d.get(k) or [])]
        d["neutral_refs"] = [_neutral_ref_dict(r, as_json=True)
                             for r in d["neutral_refs"]]
        d.pop("neutral_rois", None)     # 旧别名不再落盘（已迁移进 neutral_refs）
        d.pop("mask_rois", None)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Annotation":
        """加载并迁移：不抛业务错误（结构不合法才抛，由调用方收集诊断）。"""
        if not isinstance(d, dict):
            raise ValueError(f"annotation must be a JSON object, got {type(d).__name__}")
        if not d.get("file"):
            raise ValueError("annotation is missing the required 'file' field")
        raw_version = d.get("schema_version")
        legacy = raw_version is None or (
            isinstance(raw_version, int) and not isinstance(raw_version, bool)
            and raw_version in LEGACY_SCHEMA_VERSIONS)
        known = set(cls.__dataclass_fields__)
        ann = cls(**{k: v for k, v in d.items() if k in known})
        for k in _ROI_FIELDS:
            if getattr(ann, k) is not None:
                setattr(ann, k, tuple(getattr(ann, k)))
        for k in _ROI_LIST_FIELDS:
            setattr(ann, k, [tuple(r) for r in (getattr(ann, k) or [])])
        ann.migrate(legacy=legacy)
        ann.reconcile_status(downgrade_unknown_provenance=True)
        return ann

    # ------------------------------------------------------------------ 校验
    def validate(self) -> list[str]:
        """返回 ASCII 错误列表（空 = 合法）。不修改自身。"""
        errs: list[str] = []
        if not self.file:
            errs.append("'file' is required")
        if isinstance(self.rotation_degrees, bool) \
                or self.rotation_degrees not in VALID_ROT:
            errs.append(f"rotation_degrees must be one of {VALID_ROT}, "
                        f"got {self.rotation_degrees!r}")
        if self.coordinate_space not in VALID_COORDINATE_SPACES:
            errs.append(f"coordinate_space must be one of "
                        f"{VALID_COORDINATE_SPACES}, got {self.coordinate_space!r}")
        if isinstance(self.schema_version, bool) \
                or self.schema_version not in (1, 2, SCHEMA_VERSION):
            errs.append(f"schema_version must be 1, 2 or {SCHEMA_VERSION}, "
                        f"got {self.schema_version!r}")
        if self.status not in VALID_STATUS:
            errs.append(f"status must be one of {VALID_STATUS}, "
                        f"got {self.status!r}")
        for name in _ROI_FIELDS:
            e = _check_roi(name, getattr(self, name))
            if e:
                errs.append(e)
        if not isinstance(self.base_sample_rois, (list, tuple)):
            errs.append(f"base_sample_rois must be a list, got "
                        f"{self.base_sample_rois!r}")
        else:
            for i, r in enumerate(self.base_sample_rois):
                e = _check_roi(f"base_sample_rois[{i}]", r)
                if e:
                    errs.append(e)
        for i, r in enumerate(self.exclude_mask_rois):
            e = _check_roi(f"exclude_mask_rois[{i}]", r)
            if e:
                errs.append(e)
        for i, r in enumerate(self.neutral_refs):
            e = _check_neutral_ref(f"neutral_refs[{i}]", r)
            if e:
                errs.append(e)
        if self.source_sha256:
            s = self.source_sha256
            if not isinstance(s, str) or len(s) != 64 \
                    or any(c not in "0123456789abcdefABCDEF" for c in s):
                errs.append(f"source_sha256 must be 64 hex chars, got {s!r}")
        if self.status == STATUS_VERIFIED:
            if not self.verified_by:
                errs.append("verified status requires verified_by (provenance)")
            elif self.verified_by != USER_VERIFIER:
                errs.append(f"only verified_by='{USER_VERIFIER}' is a user sign-off; "
                            f"{self.verified_by!r} stays draft "
                            f"(v4 8.4-5: agent output is not user approval)")
            # v5 §5.2：verified = 用户确认过三类独立 ROI
            if self.image_area_roi is None:
                errs.append("verified status requires image_area_roi "
                            "(the algorithm/statistics region)")
            if self.output_crop_roi is None:
                errs.append("verified status requires output_crop_roi "
                            "(the final composition region, v5 3.2)")
            if not self.base_sample_rois:
                errs.append("verified status requires at least one "
                            "base_sample_rois entry (v5 3.2)")
            if not self.verified_date:
                errs.append("verified status requires verified_date")
        elif self.verified:
            errs.append("verified=True conflicts with status="
                        f"{self.status!r} (status is the source of truth)")
        return errs

    # ------------------------------------------------------------------ 便捷
    def is_user_verified(self) -> bool:
        return self.status == STATUS_VERIFIED and self.verified_by == USER_VERIFIER


def _neutral_ref_dict(r, as_json: bool = False) -> dict:
    """中性参考条目归一：roi/point 转元组（as_json 时转 list）。"""
    if not isinstance(r, dict):
        return r
    out = dict(r)
    for k in ("roi", "point"):
        if out.get(k) is not None and not isinstance(out[k], (str, bytes)):
            seq = tuple(out[k])
            out[k] = list(seq) if as_json else seq
    return out


def _check_neutral_ref(name: str, r) -> str | None:
    if not isinstance(r, dict):
        return f"{name} must be an object, got {r!r}"
    roi, point = r.get("roi"), r.get("point")
    if roi is None and point is None:
        return f"{name} needs either 'roi' or 'point'"
    if roi is not None:
        e = _check_roi(f"{name}.roi", roi)
        if e:
            return e
    if point is not None:
        if isinstance(point, (str, bytes)) or not isinstance(point, (list, tuple)) \
                or len(point) != 2 or not all(_is_number(v) for v in point):
            return f"{name}.point must be (y, x), got {point!r}"
        if min(int(v) for v in point) < 0:
            return f"{name}.point has negative coordinates: {point!r}"
    kind = r.get("kind", "bright")
    if kind not in ("bright", "dark"):
        return f"{name}.kind must be 'bright' or 'dark', got {kind!r}"
    return None


def ann_path_for(file_rel: str) -> Path:
    p = Path(file_rel)
    roll = p.parts[1] if len(p.parts) > 2 else "misc"
    safe = re.sub(r"[^\w.-]+", "_", roll)
    return ANNOTATIONS_DIR / f"{safe}__{p.stem}.json"


def load_annotation(file_rel: str) -> Annotation | None:
    """按命名约定加载单个标注（自动迁移旧 schema）。文件不存在返回 None。

    结构不合法（非对象 / 缺 file）或 JSON 损坏会抛异常——不得静默吞。
    """
    f = ann_path_for(file_rel)
    if not f.exists():
        return None
    return Annotation.from_dict(json.loads(f.read_text(encoding="utf-8")))


def save_annotation(rel: str, ann: Annotation) -> Path:
    """唯一公共保存入口（v4 §8.1-2）。

    rel 是权威样片路径；写入前无条件 ``validate()``，无效数据抛
    ValueError（绝不落盘，更不会被写成 verified）。落盘方式：同目录临时
    文件 + ``os.replace`` 原子替换；替换失败时保留原文件并清理临时文件。

    会就地规范化 ann：frame_roi 迁移为 image_area_roi 草稿、已知 agent
    来源降级 draft、user 签字补 verified_date、verified 布尔对齐 status。
    """
    if not isinstance(ann, Annotation):
        raise TypeError(f"save_annotation(rel, ann) needs an Annotation, "
                        f"got {type(ann).__name__}")
    ann.file = rel
    ann.migrate(legacy=False)
    ann.reconcile_status(downgrade_unknown_provenance=False)
    if ann.status == STATUS_VERIFIED and not ann.verified_date:
        ann.verified_date = date.today().isoformat()
    ann.schema_version = SCHEMA_VERSION
    errs = ann.validate()
    if errs:
        raise ValueError("annotation validation failed: " + "; ".join(errs))

    path = ann_path_for(rel)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(ann.to_dict(), ensure_ascii=False, indent=2)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp",
                                    dir=str(path.parent), text=True)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)     # 失败不得留下临时文件
        raise
    return path


def scan_annotations(*, only_verified: bool = True,
                     verbose: bool = True) -> tuple[list[Annotation], list[str]]:
    """扫描 ANNOTATIONS_DIR 顶层 *.json。

    返回 (标注列表, 诊断列表)。损坏 JSON、非法标注**不会静默跳过**：进入
    errors（并写进 LAST_LOAD_ERRORS，stderr 打印 ASCII 诊断）。
    """
    global LAST_LOAD_ERRORS
    anns: list[Annotation] = []
    errors: list[str] = []
    if ANNOTATIONS_DIR.exists():
        for f in sorted(ANNOTATIONS_DIR.glob("*.json")):
            try:
                a = Annotation.from_dict(json.loads(f.read_text(encoding="utf-8")))
            except Exception as exc:  # noqa: BLE001 —— 损坏标注必须可见
                errors.append(f"{f.name}: load failed {type(exc).__name__}: {exc}")
                continue
            errs = a.validate()
            if errs:
                errors.append(f"{f.name}: invalid: " + "; ".join(errs))
                continue
            if only_verified and a.status != STATUS_VERIFIED:
                continue
            anns.append(a)
    LAST_LOAD_ERRORS = errors
    if verbose:
        for line in errors:
            print(f"[annotations] WARNING {line}", file=sys.stderr)
        print(f"[annotations] {len(anns)} usable / {len(errors)} rejected "
              f"in {ANNOTATIONS_DIR}", file=sys.stderr)
    return anns, errors


def load_all_verified() -> list[Annotation]:
    """status=="verified"（用户签字）的标注列表；诊断见 LAST_LOAD_ERRORS。"""
    anns, _ = scan_annotations(only_verified=True)
    return anns


def load_all_annotations() -> tuple[list[Annotation], list[str]]:
    """全部合法标注（含 draft）与诊断列表。"""
    return scan_annotations(only_verified=False)
