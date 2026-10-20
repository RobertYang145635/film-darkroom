"""v5 §3.1 核心数据模型（可序列化、可脱离 GUI 测试）。

对象语义互不合并（§0.2.9）：

- `ImportSession`   一次导入：文件清单 + 卷分组
- `RollGroup`       用户手动整理出的一卷
- `FrameRecord`     单帧：源事实、采集指纹、几何、分析、选定候选、微调覆盖
- `GeometryRegions` 三类独立 ROI + 全部几何变换
- `RollAnalysis`    卷级片基与共享颜色起点（**不需要色卡**）
- `CandidateSet`    保留氛围 / 中性 / 偏暖 三个自动候选

坐标约定（唯一权威，见 pipeline/geometry.py）：所有 ROI 以 **SOURCE** 空间
半开区间 (y0, x0, y1, x1) 保存；预览与输出只做映射，不改写存储值。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field   # os 随原子写改造后不再需要
from enum import Enum
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 4          # sidecar / 会话持久化 schema（v5 §3.4）


# --------------------------------------------------------------------- 状态机
#: 真正的硬阻断状态（v5 §3.3/§8.2）：只有解码/处理错误阻断导出。
BLOCKING_STATUSES = ("decode_error", "processing_error")


class FrameStatus(str, Enum):
    """v5 §3.3 允许的用户可见状态。

    禁止把"没有色卡 / 没有 verified 校准 / 未知胶片型号 / 没有手工中性点"
    设为阻断状态——它们只影响来源说明与置信度。
    """

    IMPORTED = "imported"
    NEEDS_GROUP = "needs_group"
    NEEDS_CROP_REVIEW = "needs_crop_review"
    READY_TO_CONVERT = "ready_to_convert"
    CONVERTED = "converted"
    NEEDS_REVIEW = "needs_review"
    EXPORTED = "exported"
    DECODE_ERROR = "decode_error"
    PROCESSING_ERROR = "processing_error"
    #: 被质量门禁**故意挡下**（B-9）：**记录性状态**，故**不在** `BLOCKING_STATUSES` 里——
    #: 阻断与否只由既有那一条路决定（metrics `blocks_export` + `GateBlocked` + 退出码），
    #: 见 `docs/blockers.md` B-9"单一事实源"决定；`is_blocking("gate_blocked")` 必须为 False。
    GATE_BLOCKED = "gate_blocked"

    @classmethod
    def is_blocking(cls, value: "str | FrameStatus") -> bool:
        v = value.value if isinstance(value, FrameStatus) else str(value)
        return v in BLOCKING_STATUSES


# --------------------------------------------------------------------- 工具
def _roi(value):
    """ROI 归一：None 或 (y0,x0,y1,x1) 整数元组。"""
    if value is None:
        return None
    seq = [int(v) for v in value]
    if len(seq) != 4:
        raise ValueError(f"ROI 必须是 (y0,x0,y1,x1)，得到 {value!r}")
    return tuple(seq)


#: ROI 结构判定的**唯一实现**。此前 library/models._roi_error 与
#: calibration.annotations._check_roi 各写一份，且已经漂移：models 版不查
#: str/bytes 与非有限数，纯数字字符串 "1234" 会被当成合法 ROI（实测漏洞）。
#: 两处现在共用判据，但各自**保留原有错误文案**，不改用户可见信息。
ROI_OK = "ok"
ROI_BAD_TYPE = "bad_type"
ROI_NEGATIVE = "negative"
ROI_REVERSED = "reversed"


def roi_facts(roi):
    """返回 (判定, (y0,x0,y1,x1) 或 None, 元素个数或 -1)。不生成文案。"""
    if roi is None:
        return ROI_OK, None, -1
    if isinstance(roi, (str, bytes)):
        return ROI_BAD_TYPE, None, -1
    try:
        vals = list(roi)
    except TypeError:
        return ROI_BAD_TYPE, None, -1
    n = len(vals)
    if n != 4:
        return ROI_BAD_TYPE, None, n
    try:
        nums = [float(v) for v in vals]          # 容忍 numpy 标量（含 np.int64）
    except (TypeError, ValueError):
        return ROI_BAD_TYPE, None, n
    if any(v != v or v in (float("inf"), float("-inf")) for v in nums):
        return ROI_BAD_TYPE, None, n
    y0, x0, y1, x1 = (int(v) for v in vals)
    if min(y0, x0, y1, x1) < 0:
        return ROI_NEGATIVE, (y0, x0, y1, x1), n
    if y1 <= y0 or x1 <= x0:
        return ROI_REVERSED, (y0, x0, y1, x1), n
    return ROI_OK, (y0, x0, y1, x1), n


def _roi_error(name: str, roi) -> str | None:
    """单个 ROI 的结构校验（ASCII 错误信息，供 validate 汇总）。文案不变。"""
    kind, _coords, _n = roi_facts(roi)
    if kind is ROI_OK:
        return None
    if kind is ROI_NEGATIVE:
        return f"{name} has negative coordinates: {roi!r}"
    if kind is ROI_REVERSED:
        return f"{name} is empty or reversed: {roi!r}"
    return f"{name} must be (y0,x0,y1,x1) integers, got {roi!r}"


def _jsonable(obj: Any) -> Any:
    """numpy 数组/元组 -> JSON 可序列化结构（诊断字段常用）。"""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if hasattr(obj, "tolist"):          # numpy scalar/array
        return _jsonable(obj.tolist())
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float) and obj != obj:      # NaN
        return None
    return obj


# ----------------------------------------------------------------- 几何区域
@dataclass
class GeometryRegions:
    """三类独立 ROI + 全部几何变换（v5 §3.1/§3.2/§5.4）。"""

    rotation_degrees: int = 0                    # 0/90/180/270 顺时针
    fine_rotation_degrees: float = 0.0           # 小范围拉直
    mirror_horizontal: bool = False
    output_crop_roi: tuple | None = None         # 最终构图（可含片边）
    image_area_roi: tuple | None = None          # 有效画面（算法/统计）
    base_sample_rois: list = field(default_factory=list)   # 未曝光片基取样区
    #: `image_area_roi` 是不是自动检测给的。**B-12 追加③ 的画框几何规则只认人工框**：
    #: 自动框在本语料上把左右窄片边算进画面（9/11 帧 IoU 0.683~0.918），
    #: 它的**纵横**因此不可信，不能拿来决定方向。这个标记必须跟着回写走——
    #: UI/整卷处理会把检测出的框种回 `rec.geometry` 再传回 service，
    #: 没有它，一个自动框换个字段就冒充成"用户画的框"。
    image_area_auto: bool = False

    def __post_init__(self):
        self.output_crop_roi = _roi(self.output_crop_roi)
        self.image_area_roi = _roi(self.image_area_roi)
        self.base_sample_rois = [_roi(r) for r in (self.base_sample_rois or [])]

    # ---- 校验
    def validate(self, source_size: tuple[int, int] | None = None) -> list[str]:
        """结构合法性（不要求三类区域都已填齐——那是完整性问题）。"""
        errs: list[str] = []
        if self.rotation_degrees not in (0, 90, 180, 270):
            errs.append(f"rotation_degrees must be 0|90|180|270, "
                        f"got {self.rotation_degrees!r}")
        if not -10.0 <= float(self.fine_rotation_degrees) <= 10.0:
            errs.append(f"fine_rotation_degrees out of P0 range [-10,10]: "
                        f"{self.fine_rotation_degrees!r}")
        for name in ("output_crop_roi", "image_area_roi"):
            e = _roi_error(name, getattr(self, name))
            if e:
                errs.append(e)
        for i, r in enumerate(self.base_sample_rois):
            e = _roi_error(f"base_sample_rois[{i}]", r)
            if e:
                errs.append(e)
        # 语义约束：分析区必须与输出裁切相交（v5 §5.2）
        if self.output_crop_roi and self.image_area_roi:
            if not _intersects(self.output_crop_roi, self.image_area_roi):
                errs.append("image_area_roi does not intersect output_crop_roi")
        if source_size:
            h, w = source_size
            for name in ("output_crop_roi", "image_area_roi"):
                roi = getattr(self, name)
                if roi and (roi[2] > h or roi[3] > w):
                    errs.append(f"{name} exceeds source bounds {h}x{w}: {roi!r}")
            for i, r in enumerate(self.base_sample_rois):
                if r[2] > h or r[3] > w:
                    errs.append(f"base_sample_rois[{i}] exceeds source bounds "
                                f"{h}x{w}: {r!r}")
        return errs

    def completeness_errors(self) -> list[str]:
        """三类区域是否都已给出（缺失 -> needs_crop_review，不是硬失败）。"""
        errs: list[str] = []
        if self.output_crop_roi is None:
            errs.append("output_crop_roi not set")
        if self.image_area_roi is None:
            errs.append("image_area_roi not set")
        if not self.base_sample_rois:
            errs.append("base_sample_rois is empty")
        return errs

    def is_complete(self) -> bool:
        return not self.completeness_errors()

    # ---- 归一化相对位置（整卷同步用，v5 §5.5）
    def normalized(self, source_size: tuple[int, int]) -> dict:
        h, w = source_size

        def norm(roi):
            if roi is None:
                return None
            return [roi[0] / h, roi[1] / w, roi[2] / h, roi[3] / w]

        return {
            "rotation_degrees": self.rotation_degrees,
            "fine_rotation_degrees": float(self.fine_rotation_degrees),
            "mirror_horizontal": bool(self.mirror_horizontal),
            "output_crop_roi": norm(self.output_crop_roi),
            "image_area_roi": norm(self.image_area_roi),
            "base_sample_rois": [norm(r) for r in self.base_sample_rois],
        }

    @classmethod
    def from_normalized(cls, doc: dict, source_size: tuple[int, int],
                        *, clamp: bool = True) -> "GeometryRegions":
        """由归一化相对位置重建，并对每帧重新钳制与验证（v5 §5.5）。"""
        h, w = source_size

        def den(n):
            if not n:
                return None
            y0, x0, y1, x1 = (float(n[0]) * h, float(n[1]) * w,
                              float(n[2]) * h, float(n[3]) * w)
            if clamp:
                y0, x0 = max(0, int(round(y0))), max(0, int(round(x0)))
                y1, x1 = min(h, int(round(y1))), min(w, int(round(x1)))
            else:
                y0, x0, y1, x1 = (int(round(v)) for v in (y0, x0, y1, x1))
            return (y0, x0, y1, x1)

        return cls(
            rotation_degrees=int(doc.get("rotation_degrees", 0)),
            fine_rotation_degrees=float(doc.get("fine_rotation_degrees", 0.0)),
            mirror_horizontal=bool(doc.get("mirror_horizontal", False)),
            output_crop_roi=den(doc.get("output_crop_roi")),
            image_area_roi=den(doc.get("image_area_roi")),
            base_sample_rois=[r for r in (den(n) for n in
                                          doc.get("base_sample_rois") or []) if r],
        )

    def to_dict(self) -> dict:
        return {
            "rotation_degrees": int(self.rotation_degrees),
            "fine_rotation_degrees": float(self.fine_rotation_degrees),
            "mirror_horizontal": bool(self.mirror_horizontal),
            "output_crop_roi": list(self.output_crop_roi) if self.output_crop_roi
            else None,
            "image_area_roi": list(self.image_area_roi) if self.image_area_roi
            else None,
            "base_sample_rois": [list(r) for r in self.base_sample_rois],
            "image_area_auto": bool(self.image_area_auto),
            "coordinate_space": "source",
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GeometryRegions":
        return cls(
            rotation_degrees=int(d.get("rotation_degrees", 0)),
            fine_rotation_degrees=float(d.get("fine_rotation_degrees", 0.0)),
            mirror_horizontal=bool(d.get("mirror_horizontal", False)),
            output_crop_roi=d.get("output_crop_roi"),
            image_area_roi=d.get("image_area_roi"),
            base_sample_rois=list(d.get("base_sample_rois") or []),
            image_area_auto=bool(d.get("image_area_auto", False)),
        )


def _intersects(a, b) -> bool:
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


# ----------------------------------------------------------------- 卷级分析
@dataclass
class RollAnalysis:
    """按卷片基汇总与共享颜色起点（v5 §6.2）。**不要求色卡。**"""

    roll_id: str = ""
    capture_fingerprint: str = ""
    base_statistics: dict = field(default_factory=dict)   # dmin/mad/n/...
    base_surface: dict = field(default_factory=dict)      # coverage/warnings
    shared_color_start: dict = field(default_factory=dict)
    source_frames: list = field(default_factory=list)     # 参与汇总的帧
    confidence: float = 0.0
    warnings: list = field(default_factory=list)
    rejected_frames: list = field(default_factory=list)   # 低置信/漂移/污染
    #: 被**明确排除**在卷级共享之外的帧：自己测到了片基，但与同卷其他帧漂移不一致，
    #: 或采集指纹与本卷不符。v5 §6.2.3 的继承服务对象恰恰是"自己没测到"的帧，
    #: 它们不该出现在这个列表里（见 `resolve_base` 的 `in_roll_model`）。
    excluded_from_share: list = field(default_factory=list)
    shared_with_frames: list = field(default_factory=list)
    split_suggested: bool = False
    base_source: str = "none"     # frame|roll|roll_partial|frame+roll|content_orange|generic_prior
    #: 本卷共同曝光尺度（v5 §6.2.5）：通用先验回退时全卷共用一个尺度，
    #: 不允许每帧各猜一个（否则同卷颜色起点漂移）。
    generic_scale: float | None = None
    analysis_version: int = 1
    model_asset_hash: str = ""

    def to_dict(self) -> dict:
        return _jsonable(asdict(self))

    @classmethod
    def from_dict(cls, d: dict) -> "RollAnalysis":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})

    @property
    def is_generic_fallback(self) -> bool:
        return self.base_source == "generic_prior"


# ------------------------------------------------------------------- 三候选
@dataclass
class Candidate:
    """一个自动颜色候选。

    候选之间**只**在 RenderIntent 的色温/色调（与受限的影调起点）上不同；
    输出裁切、片基与技术正片必须完全一致（v5 §6.5）。
    numpy 数组等运行时产物不进入 to_dict（保持可序列化）。
    """

    name: str = "preserve"                       # preserve|neutral|warm
    intent: dict = field(default_factory=dict)   # RenderIntent 参数
    confidence: float = 0.0
    rationale: str = ""
    estimated_correction: dict | None = None     # 估计出的色温/色调（始终记录）
    applied_correction: dict | None = None       # 实际施加的色温/色调
    correction_capped: bool = False              # 是否触发硬上限
    fallback_to: str | None = None               # 分歧过大时回退到哪个候选
    estimators: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)
    # ---- 运行时（不序列化）
    technical_positive_linear: Any = None
    rendered_positive_linear: Any = None
    output_crop_roi: tuple | None = None
    base_source: str = ""

    def to_dict(self) -> dict:
        return _jsonable({
            "name": self.name, "intent": self.intent,
            "confidence": round(float(self.confidence), 4),
            "rationale": self.rationale,
            "estimated_correction": self.estimated_correction,
            "applied_correction": self.applied_correction,
            "correction_capped": bool(self.correction_capped),
            "fallback_to": self.fallback_to,
            "estimators": self.estimators,
            "diagnostics": self.diagnostics,
            "output_crop_roi": list(self.output_crop_roi)
            if self.output_crop_roi else None,
            "base_source": self.base_source,
        })

    @classmethod
    def from_dict(cls, d: dict) -> "Candidate":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class CandidateSet:
    preserve: Candidate = field(default_factory=lambda: Candidate("preserve"))
    neutral: Candidate = field(default_factory=lambda: Candidate("neutral"))
    warm: Candidate = field(default_factory=lambda: Candidate("warm"))
    default_name: str = "preserve"
    diagnostics: dict = field(default_factory=dict)

    def get(self, name: str) -> Candidate:
        for c in (self.preserve, self.neutral, self.warm):
            if c.name == name:
                return c
        raise KeyError(f"未知候选 {name!r}")

    def __iter__(self):
        return iter((self.preserve, self.neutral, self.warm))

    def to_dict(self) -> dict:
        return _jsonable({
            "preserve": self.preserve.to_dict(),
            "neutral": self.neutral.to_dict(),
            "warm": self.warm.to_dict(),
            "default_name": self.default_name,
            "diagnostics": self.diagnostics,
        })

    @classmethod
    def from_dict(cls, d: dict) -> "CandidateSet":
        return cls(
            preserve=Candidate.from_dict(d.get("preserve") or {"name": "preserve"}),
            neutral=Candidate.from_dict(d.get("neutral") or {"name": "neutral"}),
            warm=Candidate.from_dict(d.get("warm") or {"name": "warm"}),
            default_name=d.get("default_name", "preserve"),
            diagnostics=d.get("diagnostics") or {})


# ------------------------------------------------------------------- 帧记录
@dataclass
class FrameRecord:
    """单帧事实（v5 §3.1）。source_path + source_sha256 是唯一键。"""

    source_path: str = ""
    source_sha256: str = ""
    input_facts: dict = field(default_factory=dict)       # 解码/ICC/位深事实
    capture_fingerprint: str = ""
    geometry: GeometryRegions = field(default_factory=GeometryRegions)
    frame_analysis: dict = field(default_factory=dict)    # 帧级片基/统计
    chosen_candidate: str | None = None
    render_intent_override: dict | None = None
    status: str = FrameStatus.IMPORTED.value
    roll_id: str | None = None
    warnings: list = field(default_factory=list)
    #: v6.1 §4：当前帧「保留片边」的用户选择（导出链路的唯一数据源）。
    keep_rebate: bool = False
    #: B-12：这一帧的**方向是不是有人决定过**。
    #: `rotation_degrees=0` 既可能是"看过确认是 0°"，也可能是"从没决定过的默认值"，
    #: 二者像素完全一样，只有这个字段能分开它们。实测 `1298_G200` 12 帧里
    #: 只有 4 帧有标注文件，其余多数帧的方向是默认值；与用户手工参考对照后
    #: 确认"方向本来就对"的帧内容相关领先 0.35~0.46，而默认帧肉眼可见地差
    #: 90°/180°（G200-6、G200-8）。**软件不能猜方向，但必须说出没决定过。**
    #: 三态：`None`=没人说过（不判定，单元测试与旧会话文件就是这个状态）；
    #: `False`=明确"没决定过"⇒ 门禁给 needs_review；`True`=决定过。
    #: 界面与验收套具**必须显式给值**，这条由 `test_orientation_flag_call_sites` 钉住。
    orientation_decided: bool | None = None
    #: ㊶ 追加⑨/⑩：用户**声明**的片种键（`pipeline/film_types.py` 的 `key`）。
    #: 空 = 未声明 ⇒ 技术级通道斜率走现产默认（逐比特不变）。
    #: 为什么放在帧级而不是卷级：软件**不许猜**片种（三卷 G200/CN400 的 capture
    #: 指纹完全相同，`film-corpus-traps` 第 7 条），而声明要能落到可核对的单位。
    film_type: str = ""

    def key(self) -> str:
        return f"{self.source_path}#{self.source_sha256[:16]}"

    def validate(self) -> list[str]:
        errs: list[str] = []
        if not self.source_path:
            errs.append("frame requires source_path")
        if self.source_sha256 and len(self.source_sha256) != 64:
            errs.append(f"source_sha256 must be 64 hex chars, got "
                        f"{self.source_sha256!r}")
        if self.status not in {s.value for s in FrameStatus}:
            errs.append(f"unknown status {self.status!r}")
        errs.extend(self.geometry.validate())
        return errs

    def to_dict(self) -> dict:
        return _jsonable({
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "input_facts": self.input_facts,
            "capture_fingerprint": self.capture_fingerprint,
            "geometry": self.geometry.to_dict(),
            "frame_analysis": self.frame_analysis,
            "chosen_candidate": self.chosen_candidate,
            "render_intent_override": self.render_intent_override,
            "status": self.status,
            "roll_id": self.roll_id,
            "warnings": list(self.warnings),
            "keep_rebate": bool(self.keep_rebate),      # v6.1 §4 显式布尔
            "orientation_decided": (None if self.orientation_decided is None
                                    else bool(self.orientation_decided)),   # B-12
            "film_type": str(self.film_type),                    # ㊶ 追加⑨
        })

    @classmethod
    def from_dict(cls, d: dict) -> "FrameRecord":
        known = set(cls.__dataclass_fields__) - {"geometry", "keep_rebate",
                                                 "orientation_decided"}
        kw = {k: v for k, v in d.items() if k in known}
        kr = d.get("keep_rebate", False)
        # v6.1 §4：非 bool 值不能静默当真；旧文件默认 False
        kw["keep_rebate"] = kr if isinstance(kr, bool) else False
        od = d.get("orientation_decided", None)     # B-12：旧文件=没人说过
        kw["orientation_decided"] = od if isinstance(od, bool) else None
        ft = d.get("film_type", "")                # ㊶ 追加⑨：旧文件=未声明
        kw["film_type"] = ft.strip() if isinstance(ft, str) else ""
        kw["geometry"] = GeometryRegions.from_dict(d.get("geometry") or {})
        return cls(**kw)


# ------------------------------------------------------------------- 卷分组
@dataclass
class RollGroup:
    """用户手动整理的一卷（v5 §0.2.6/§2.1.2）。"""

    id: str = ""
    display_name: str = ""
    frame_ids: list = field(default_factory=list)
    capture_fingerprint: str = ""
    roll_analysis: RollAnalysis | None = None
    shared_candidate: CandidateSet | None = None
    shared_render_intent: dict | None = None
    conflicting_frames: list = field(default_factory=list)

    def validate(self) -> list[str]:
        errs: list[str] = []
        if not self.id:
            errs.append("roll requires id")
        if not self.display_name:
            errs.append("roll requires display_name")
        dup = {f for f in self.frame_ids if self.frame_ids.count(f) > 1}
        if dup:
            errs.append(f"duplicate frame ids: {sorted(dup)}")
        return errs

    def to_dict(self) -> dict:
        return _jsonable({
            "id": self.id,
            "display_name": self.display_name,
            "frame_ids": list(self.frame_ids),
            "capture_fingerprint": self.capture_fingerprint,
            "roll_analysis": self.roll_analysis.to_dict()
            if self.roll_analysis else None,
            "shared_candidate": self.shared_candidate.to_dict()
            if self.shared_candidate else None,
            "shared_render_intent": self.shared_render_intent,
            "conflicting_frames": list(self.conflicting_frames),
        })

    @classmethod
    def from_dict(cls, d: dict) -> "RollGroup":
        ra = d.get("roll_analysis")
        sc = d.get("shared_candidate")
        return cls(
            id=d.get("id", ""), display_name=d.get("display_name", ""),
            frame_ids=list(d.get("frame_ids") or []),
            capture_fingerprint=d.get("capture_fingerprint", ""),
            roll_analysis=RollAnalysis.from_dict(ra) if ra else None,
            shared_candidate=CandidateSet.from_dict(sc) if sc else None,
            shared_render_intent=d.get("shared_render_intent"),
            conflicting_frames=list(d.get("conflicting_frames") or []))


# ----------------------------------------------------------------- 导入会话
@dataclass
class ImportSession:
    """一次导入的完整可恢复状态（v5 §3.1）。UI 不持有唯一业务状态。"""

    files: list = field(default_factory=list)          # [FrameRecord]
    roll_groups: list = field(default_factory=list)    # [RollGroup]
    schema_version: int = SCHEMA_VERSION
    batch_encoding_assumption: dict | None = None      # v5 §4.3.3 批次级选择
    created_utc: str = ""

    # ---- 索引
    def frame(self, path_or_id: str) -> FrameRecord | None:
        for f in self.files:
            if f.source_path == path_or_id or f.key() == path_or_id:
                return f
        return None

    def roll(self, roll_id: str) -> RollGroup | None:
        for r in self.roll_groups:
            if r.id == roll_id:
                return r
        return None

    def ungrouped(self) -> list[FrameRecord]:
        return [f for f in self.files if not f.roll_id]

    # ---- 分组操作（v5 §2.1.2：建立、重命名、拆分、移动）
    def create_roll(self, display_name: str, frame_paths: list[str],
                    roll_id: str | None = None) -> RollGroup:
        rid = roll_id or f"roll-{len(self.roll_groups) + 1:03d}"
        if self.roll(rid):
            raise ValueError(f"roll id 已存在：{rid}")
        roll = RollGroup(id=rid, display_name=display_name,
                         frame_ids=list(frame_paths))
        for p in frame_paths:
            f = self.frame(p)
            if f is None:
                raise ValueError(f"未知帧：{p}")
            f.roll_id = rid
            f.status = FrameStatus.NEEDS_CROP_REVIEW.value
        self.roll_groups.append(roll)
        return roll

    def rename_roll(self, roll_id: str, display_name: str) -> None:
        r = self.roll(roll_id)
        if r is None:
            raise KeyError(roll_id)
        r.display_name = display_name

    def move_frames(self, frame_paths: list[str], target_roll_id: str) -> None:
        target = self.roll(target_roll_id)
        if target is None:
            raise KeyError(target_roll_id)
        for p in frame_paths:
            f = self.frame(p)
            if f is None:
                raise ValueError(f"未知帧：{p}")
            src = self.roll(f.roll_id) if f.roll_id else None
            if src is not None and p in src.frame_ids:
                src.frame_ids.remove(p)
            if p not in target.frame_ids:
                target.frame_ids.append(p)
            f.roll_id = target.id
            f.status = FrameStatus.NEEDS_CROP_REVIEW.value

    def split_roll(self, roll_id: str, frame_paths: list[str],
                   new_name: str) -> RollGroup:
        """一键拆成不同卷（v5 §4.4 指纹冲突时使用）。"""
        src = self.roll(roll_id)
        if src is None:
            raise KeyError(roll_id)
        for p in frame_paths:
            if p in src.frame_ids:
                src.frame_ids.remove(p)
        new = self.create_roll(new_name, frame_paths)
        return new

    # ---- 校验与持久化
    def validate(self) -> list[str]:
        errs: list[str] = []
        if self.schema_version > SCHEMA_VERSION:
            errs.append(f"session schema_version {self.schema_version} newer "
                        f"than supported {SCHEMA_VERSION}")
        seen: set[str] = set()
        for f in self.files:
            if f.source_path in seen:
                errs.append(f"duplicate frame {f.source_path}")
            seen.add(f.source_path)
            errs.extend(f"{f.source_path}: {e}" for e in f.validate())
        for r in self.roll_groups:
            errs.extend(r.validate())
            for p in r.frame_ids:
                f = self.frame(p)
                if f is None:
                    errs.append(f"roll {r.id} references unknown frame {p}")
                elif f.roll_id != r.id:
                    errs.append(f"frame {p} roll_id={f.roll_id} inconsistent "
                                f"with roll {r.id}")
        return errs

    def to_dict(self) -> dict:
        return _jsonable({
            "schema_version": self.schema_version,
            "created_utc": self.created_utc,
            "batch_encoding_assumption": self.batch_encoding_assumption,
            "files": [f.to_dict() for f in self.files],
            "roll_groups": [r.to_dict() for r in self.roll_groups],
        })

    @classmethod
    def from_dict(cls, d: dict) -> "ImportSession":
        return cls(
            files=[FrameRecord.from_dict(x) for x in d.get("files") or []],
            roll_groups=[RollGroup.from_dict(x)
                         for x in d.get("roll_groups") or []],
            schema_version=int(d.get("schema_version", SCHEMA_VERSION)),
            batch_encoding_assumption=d.get("batch_encoding_assumption"),
            created_utc=d.get("created_utc", ""))

    def save(self, path: str | Path) -> Path:
        errs = self.validate()
        if errs:
            raise ValueError("session 不可保存：" + "; ".join(errs[:8]))
        path = Path(path)
        # 审查修复（2026-09-08）：此前是「固定名 .tmp + write_text + os.replace」——
        # 既无 fsync（断电后会话文件可能 0 字节/半截 JSON），临时名又固定
        # （重入/并发相撞），失败还不清理残留。统一走 io.atomic 的正解。
        from ..io.atomic import atomic_write_bytes
        atomic_write_bytes(path, json.dumps(self.to_dict(), ensure_ascii=False,
                                            indent=2).encode("utf-8"),
                           overwrite=True)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "ImportSession":
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(doc)
