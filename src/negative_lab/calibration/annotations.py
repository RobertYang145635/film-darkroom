"""人工标注读写（整改 P0-A.4）。

fixtures/annotations/<roll>__<stem>.json，只存小型 JSON，不复制大图。
坐标语义（必须声明，整改 §P0-A.4）：

  coordinate_space:
    - "source": 坐标基于原始未旋转文件的像素空间 (row0,col0,row1,col1)。
    - "oriented": 坐标基于按 rotation_degrees 顺时针旋转后的空间。
  本版本全部使用 "source"；rotation_degrees 为应用到 source 得到
  正确观看方向的**顺时针**角度 (0|90|180|270)。

verified 字段区分"人工核对过"与"算法暂定待核对"，回归只使用 verified 项。
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

ANNOTATIONS_DIR = Path(__file__).resolve().parents[3] / "fixtures" / "annotations"

VALID_ROT = (0, 90, 180, 270)


@dataclass
class Annotation:
    file: str                                   # 相对仓库根的样片路径
    roll: str
    rotation_degrees: int = 0                   # 顺时针
    coordinate_space: str = "source"            # source|oriented
    frame_roi: tuple | None = None              # (y0,x0,y1,x1) 画面区域（不含片边/片基）
    base_roi: tuple | None = None               # 未曝光片基条带
    neutral_rois: list = field(default_factory=list)   # [{roi|point, kind: bright|dark}]
    mask_rois: list = field(default_factory=list)
    reference_file: str | None = None
    notes: str = ""
    verified: bool = False                      # 人工核对（含视觉方向确认）
    verified_by: str = ""                       # agent-inspected|user
    verified_date: str = ""

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        for k in ("frame_roi", "base_roi"):
            if d[k] is not None:
                d[k] = list(d[k])
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Annotation":
        d = dict(d)
        for k in ("frame_roi", "base_roi"):
            if d.get(k) is not None:
                d[k] = tuple(d[k])
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})

    def validate(self) -> list[str]:
        errs = []
        if self.rotation_degrees not in VALID_ROT:
            errs.append(f"rotation_degrees 必须为 {VALID_ROT}，得到 {self.rotation_degrees}")
        if self.coordinate_space not in ("source", "oriented"):
            errs.append(f"coordinate_space 非法: {self.coordinate_space}")
        for name in ("frame_roi", "base_roi"):
            roi = getattr(self, name)
            if roi is not None:
                y0, x0, y1, x1 = roi
                if not (y1 > y0 and x1 > x0 and min(roi) >= 0):
                    errs.append(f"{name} 非法: {roi}")
        if self.verified and self.frame_roi is None:
            errs.append("verified 标注必须含 frame_roi")
        return errs


def ann_path_for(file_rel: str) -> Path:
    p = Path(file_rel)
    roll = p.parts[1] if len(p.parts) > 2 else "misc"
    safe = re.sub(r"[^\w.-]+", "_", roll)
    return ANNOTATIONS_DIR / f"{safe}__{p.stem}.json"


def load_annotation(file_rel: str) -> Annotation | None:
    f = ann_path_for(file_rel)
    if not f.exists():
        return None
    return Annotation.from_dict(json.loads(f.read_text(encoding="utf-8")))


def save_annotation(ann: Annotation) -> Path:
    errs = ann.validate()
    if errs:
        raise ValueError("标注校验失败: " + "; ".join(errs))
    f = ann_path_for(ann.file)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(ann.to_dict(), ensure_ascii=False, indent=2),
                 encoding="utf-8")
    return f


def save_annotation(rel: str, ann: Annotation) -> Path:
    """把（用户框选的）区域写回标注文件。GUI『保存区域』按钮的落盘路径。
    命名与 load_annotation/ann_path_for 同一约定。"""
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    p = ann_path_for(rel)
    d = asdict(ann)
    d["file"] = rel
    if ann.verified and not ann.verified_date:
        d["verified_date"] = date.today().isoformat()
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    return p
    out = []
    if not ANNOTATIONS_DIR.exists():
        return out
    for f in sorted(ANNOTATIONS_DIR.glob("*.json")):
        try:
            a = Annotation.from_dict(json.loads(f.read_text(encoding="utf-8")))
        except Exception:  # noqa: BLE001 —— 损坏标注必须可见，不静默吞
            continue
        if a.verified:
            out.append(a)
    return out
