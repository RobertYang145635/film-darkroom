"""自动采集指纹（v5 §4.4）。

程序自己从 EXIF / DNG 标签 / ICC / TIFF 标签 / 解码事实生成**不可编辑**的
`capture_fingerprint`；用户不管理"设备配置文件"，也不需要输入扫描仪型号。

指纹**不包含**文件名或卷名：同一卷改名、换目录都必须得到同一指纹。
它只能识别 metadata 中可见的设备事实，不能保证背光与曝光真的一致——
跨帧漂移由 pipeline.roll_analysis 用像素统计另行检查。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

SCHEMA = 1

#: 参与指纹的字段（按输入类型）。缺失字段以 "<none>" 占位，不静默丢弃。
MISSING = "<none>"


@dataclass(frozen=True)
class CaptureFingerprint:
    """不可编辑的采集指纹。value 是规范化字段的 SHA-256。"""

    value: str
    kind: str                                  # raw|dng_mosaic|dng_linear|tiff|raster|unknown
    fields: dict = field(default_factory=dict)
    editable: bool = False                     # 恒为 False：用户不维护设备配置

    def to_dict(self) -> dict:
        return {"schema_version": SCHEMA, "value": self.value, "kind": self.kind,
                "fields": json.loads(json.dumps(self.fields, default=str)),
                "editable": self.editable}

    @classmethod
    def from_dict(cls, d: dict) -> "CaptureFingerprint":
        return cls(value=d["value"], kind=d.get("kind", "unknown"),
                   fields=d.get("fields") or {}, editable=False)

    @property
    def completeness(self) -> float:
        """参与字段的可用比例（metadata 不足时用于 needs_review 提示）。"""
        vals = [v for k, v in self.fields.items() if k != "input_kind"]
        if not vals:
            return 0.0

        def filled(v):
            if v is None or v == MISSING or v == "" or v == [] or v == [0, 0]:
                return False
            if isinstance(v, str) and v.startswith("n/a"):
                return False
            return True

        return round(sum(1 for v in vals if filled(v)) / len(vals), 3)

    def short(self) -> str:
        return self.value[:12]


def _h(parts: dict) -> str:
    blob = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _matrix_id(mat) -> str:
    """相机矩阵 -> 稳定标识（量化到 1e-4，避免浮点噪声造成假冲突）。"""
    if not mat:
        return MISSING
    try:
        flat = [round(float(v), 4) for v in
                (x for row in mat for x in (row if isinstance(row, (list, tuple))
                                            else [row]))]
    except (TypeError, ValueError):
        return MISSING
    return ",".join(f"{v:.4f}" for v in flat) if flat else MISSING


def capture_fingerprint(meta: dict, *, source_path: str | None = None,
                        batch_assumption: dict | None = None
                        ) -> CaptureFingerprint:
    """从加载 meta 生成采集指纹。

    meta 由 io.images.load_linear_rgb 产生（RAW/DNG/TIFF/raster 各有字段）。
    source_path **只**用于判定输入类型，绝不进入哈希。
    """
    meta = meta or {}
    kind = _input_kind(meta, source_path)
    parts: dict = {"input_kind": kind}

    if kind in ("raw", "dng_mosaic", "dng_linear"):
        parts["camera"] = meta.get("camera") or MISSING
        parts["lens"] = meta.get("lens") or MISSING
        parts["pixel_size"] = [int(meta.get("width") or 0), int(meta.get("height") or 0)]
        parts["raw_decode_policy"] = _decode_policy(meta)
        parts["camera_matrix"] = _matrix_id(meta.get("rgb_xyz_matrix"))
        parts["color_matrix"] = _matrix_id(meta.get("color_matrix"))
        parts["forward_matrix"] = _matrix_id(meta.get("forward_matrix"))
        parts["profile_hash"] = meta.get("profile_sha256") or MISSING
        parts["bit_depth"] = int(meta.get("bit_depth") or 0)
        parts["orientation_policy"] = meta.get("orientation_policy") or MISSING
    if kind in ("dng_mosaic", "dng_linear"):
        parts["unique_camera_model"] = meta.get("camera") or MISSING
        parts["dng_version"] = meta.get("dng_version") or MISSING

    if kind in ("tiff", "raster"):
        parts["icc_sha256"] = meta.get("input_icc_sha256") or MISSING
        parts["trc_sha256"] = meta.get("input_trc_sha256") or MISSING
        parts["bits"] = int(meta.get("bits") or meta.get("bit_depth") or 0)
        parts["sample_format"] = meta.get("sample_format") or MISSING
        parts["photometric"] = meta.get("photometric") or MISSING
        parts["software"] = meta.get("software") or MISSING
        parts["tiff_tags"] = _tiff_tag_id(meta.get("tiff_tag_facts"))
        parts["pixel_size"] = [int(meta.get("width") or 0),
                               int(meta.get("height") or 0)]

    # 所有输入：用户已确认的批次编码假设（v5 §4.3.3）
    if batch_assumption:
        parts["batch_encoding"] = batch_assumption.get("encoding") or MISSING
        parts["batch_assumption_id"] = batch_assumption.get("id") or MISSING

    return CaptureFingerprint(value=_h(parts), kind=kind, fields=parts)


def _input_kind(meta: dict, source_path: str | None) -> str:
    kind = (meta.get("kind") or "").lower()
    ext = (source_path or meta.get("path") or "").lower()
    if kind == "dng" or ext.endswith(".dng"):
        dng_kind = (meta.get("dng_kind") or "").lower()
        if dng_kind in ("mosaic", "linear"):
            return f"dng_{dng_kind}"
        return "dng_unknown"
    if kind == "raw" or ext.endswith((".nef", ".cr2", ".cr3", ".arw", ".raf",
                                      ".orf", ".rw2", ".raw")):
        return "raw"
    if kind == "tiff" or ext.endswith((".tif", ".tiff")):
        return "tiff"
    if ext.endswith((".png", ".jpg", ".jpeg", ".bmp")):
        return "raster"
    return kind or "unknown"


def _decode_policy(meta: dict) -> str:
    rp = meta.get("raw_decode_params") or {}
    if not rp:
        return MISSING
    keys = ("use_camera_wb", "use_auto_wb", "output_color_name", "gamm",
            "no_auto_bright", "highlight", "output_bps", "user_flip",
            "half_size")
    return "|".join(f"{k}={rp.get(k)}" for k in keys if k in rp)


def _tiff_tag_id(facts) -> str:
    if not facts:
        return MISSING
    if isinstance(facts, dict):
        return "|".join(f"{k}={facts[k]}" for k in sorted(facts))
    return str(facts)


def fingerprints_compatible(a, b) -> bool:
    """两个指纹是否可共享卷分析（v5 §4.4）。

    只有完全相同的采集事实才兼容；任一方缺失指纹（value 为空）视为不兼容，
    不得静默共享错误参数。
    """
    av = a.value if isinstance(a, CaptureFingerprint) else str(a or "")
    bv = b.value if isinstance(b, CaptureFingerprint) else str(b or "")
    if not av or not bv:
        return False
    return av == bv


def group_by_fingerprint(items) -> dict[str, list]:
    """{fingerprint_value: [items]}，供卷内冲突检测使用。"""
    out: dict[str, list] = {}
    for key, fp in items:
        out.setdefault(str(fp or ""), []).append(key)
    return out
