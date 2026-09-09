"""StateStore：主窗口的落盘状态与几何换算辅助（2026-09-08 MainWindow 拆分）。

只搬不改（交接 §12 未完成清单 (6)）：本模块收纳 ui/main_window 的模块级
状态辅助——确认基线/工作草稿解析、标注->几何、ROI 上下采样、尺寸/状态文案。
全部为纯函数或显式传参，不持有 Qt 控件。

依赖注入说明：
- 草稿目录经 `drafts_dir` 参数传入（main_window 保留 `_DRAFTS_DIR` 模块
  全局与 `_drafts_dir()`，测试 monkeypatch 该全局后仍经 main_window 的
  包装函数生效——既有测试契约不变）；
- `_LAST_STATE_LOAD_ERRORS` 仍为模块级诊断列表（scan 语义：每次
  resolve_persistent_state 前清空，不跨帧串诊断，冷读 R3 已核实）。
"""

from __future__ import annotations

import json
from pathlib import Path

from ..calibration.annotations import Annotation, ann_path_for, load_annotation
from ..calibration.state import ResolvedState, is_confirmed, resolve_state
from .. import storage
from ..library.models import FrameRecord, GeometryRegions

MAX_PREVIEW_DIM = 1400          # 交互预览（fit 视图在 150% DPI 下约需 1400 物理px；再大走按需高清分片）
IMPORT_PROBE = False            # 导入只取元数据+哈希（完整探针太慢，用户实测）
#: v6.1 §3：后台高清替换默认关闭——当前实现把全分辨率整帧塞进使用
#: small ROI 坐标的场景，且把 worker 返回的 {"ok","value"} 当数组用，
#: 会显示错误内容；在实现同场景高清 tile 之前必须保持 False。
ENABLE_HD_PREVIEW = False

# 画布归属哨兵：self.current 已切到新帧、但画布还挂着上一帧像素的那段窗口。
# 用它（而不是 None）表达「确定不同源」，这样从未走过 _frame_done 的手工装配
# 调用方（测试）不会被误判。
_CANVAS_STALE = object()


def _shape_hw(value):
    """统一取 (H, W)：ndarray 用 shape[:2]，尺寸元组原样通过。

    v6.1 §2：任何 source 入口都可能是全分辨率 RGB 数组；绝不能把数组
    本身传给只接受 (H,W) 的 normalized()/from_normalized()，也不能对
    ndarray 做真假判断（ambiguous truth value）。
    """
    if value is None:
        return None
    shape = getattr(value, "shape", None)
    if shape is not None and len(shape) >= 2:
        return int(shape[0]), int(shape[1])
    try:
        if len(value) >= 2:
            return int(value[0]), int(value[1])
    except TypeError:
        pass
    raise ValueError(f"invalid image/size value: {value!r}")


def _abs(path: str, rec: FrameRecord | None = None) -> Path:
    """记录的 abs_path 优先（导入根不一定是仓库根）。

    回退分支此前用 Path(__file__).parents[3]——2026-09-08「仓库根解析统一」
    漏掉的最后一处（交接 §2），本次拆分顺手统一到 storage.repo_path()：
    仓库内行为不变；打包版下与全仓 11 处一样走环境变量改道，不再被吞。
    """
    if rec is not None:
        abs_path = (rec.input_facts or {}).get("abs_path")
        if abs_path and Path(abs_path).exists():
            return Path(abs_path)
    p = Path(path)
    if p.is_absolute():
        return p
    return storage.repo_path(*Path(path).parts)


def _geometry_from_annotation(ann: Annotation) -> GeometryRegions:
    return GeometryRegions(
        rotation_degrees=ann.rotation_degrees,
        fine_rotation_degrees=float(getattr(ann, "fine_rotation_degrees", 0.0) or 0.0),
        mirror_horizontal=bool(getattr(ann, "mirror_horizontal", False)),
        output_crop_roi=ann.output_crop_roi or ann.image_area_roi,
        image_area_roi=ann.image_area_roi,
        base_sample_rois=list(ann.base_sample_rois or []))


def _geometry_for_job(ann, small, source, geo: GeometryRegions | None = None,
                      ann_source=None) -> GeometryRegions:
    """把草稿（SOURCE 全分辨率）映射到降采样工作副本坐标。

    - geo 给定：直接按 source 形状归一化映射（编辑路径）；
    - geo 为 None：从旧标注迁移（加载路径），annotation 坐标即 SOURCE。
    source / ann_source 可为 (H,W,3) 数组或 (H,W) 尺寸元组。
    """
    small_hw = _shape_hw(small)
    source_hw = _shape_hw(source)
    if geo is not None and source_hw is not None:
        return GeometryRegions.from_normalized(
            geo.normalized(source_hw), small_hw)
    g = _geometry_from_annotation(ann) if ann is not None else GeometryRegions()
    ann_hw = _shape_hw(ann_source) if ann_source is not None else small_hw
    return GeometryRegions.from_normalized(
        g.normalized(ann_hw), small_hw)


def _as_roi(v):
    if not v:
        return None
    try:
        return tuple(int(x) for x in v)
    except (TypeError, ValueError):
        return None


def _unscale_geometry(rss: dict, src_shape, small_shape) -> dict:
    """service 的 roi_source_space（small 空间坐标）-> SOURCE 全分辨率。

    service 在降采样副本上检测 ROI；`roi_source_space` 里的值是 small
    空间的 SOURCE 语义。真正的换算用归一化相对位置：small 上的 ROI /
    small 尺寸 == SOURCE 上的 ROI / SOURCE 尺寸。
    """
    sy = src_shape[0] / max(small_shape[0], 1)
    sx = src_shape[1] / max(small_shape[1], 1)

    def up(roi):
        if not roi:
            return None
        return (int(round(roi[0] * sy)), int(round(roi[1] * sx)),
                int(round(roi[2] * sy)), int(round(roi[3] * sx)))

    return {
        "output_crop_roi": up(rss.get("output_crop_roi")),
        "image_area_roi": up(rss.get("image_area_roi")),
        "base_sample_rois": [r for r in (up(x) for x in
                                         (rss.get("base_sample_rois") or []))
                             if r],
    }


def _size_of(rec: FrameRecord) -> tuple[int, int]:
    w = int((rec.input_facts or {}).get("width") or 0)
    h = int((rec.input_facts or {}).get("height") or 0)
    return (h, w) if h and w else (0, 0)


def _frame_status_text(rec: FrameRecord, pr) -> str:
    sh, sw = "", ""
    if pr is not None:
        h, w = pr.geometry.source_size
        sh, sw = str(h), str(w)
    saved = "已保存" if rec.geometry.is_complete() else "草稿(不完整)"
    gate = pr.gate.status if pr is not None else "-"
    return f"{Path(rec.source_path).name} | {sw}x{sh} | 草稿: {saved} | 门禁: {gate}"


def _qrect(x, y, w, h):
    from PySide6.QtCore import QRectF

    return QRectF(x, y, w, h)


# 审查修复（2026-09-08，我的 P0）：此前两个 loader 把所有异常吞成 None，
# 「文件损坏」与「用户从没标定过」在下游不可区分 -> resolve_state 判 none ->
# 界面显示自动检测、用户在其上再签字 -> 可恢复的旧基线被覆盖。这就是
# state.py 想根治的「标定无声消失」，只是换了个入口发生。现在记录诊断，
# 由 resolve_persistent_state 并进 reason，UI 明示并**拒绝覆盖式保存**。
_LAST_STATE_LOAD_ERRORS: list[str] = []


def _load_confirmed_annotation(rec: FrameRecord) -> Annotation | None:
    """只取用户签字的确认基线（非 user 来源在加载时已被降级为 draft）。"""
    try:
        a = load_annotation(rec.source_path)
    except Exception as e:  # noqa: BLE001 —— 不得阻断打开，但必须留痕
        _LAST_STATE_LOAD_ERRORS.append(
            f"确认标注读取失败（{type(e).__name__}: {e}）")
        return None
    return a if is_confirmed(a) else None


def _load_draft_annotation(rec: FrameRecord, drafts_dir: Path) -> Annotation | None:
    """工作草稿（经 storage 改道的 drafts/，重启可续；损坏**必须可见**）。"""
    f = drafts_dir / (ann_path_for(rec.source_path).name)
    try:
        if f.exists():
            draft = Annotation.from_dict(
                json.loads(f.read_text(encoding="utf-8")))
            from ..calibration.ownership import same_source
            recorded = str(getattr(draft, "file", "") or "")
            if recorded and not same_source(recorded, rec.source_path):
                # 冷读审查 L2-④：不同帧可能映射到同一个草稿文件名
                _LAST_STATE_LOAD_ERRORS.append(
                    f"工作草稿归属不符（{f.name} 记的是 {recorded!r}），"
                    "已忽略，未套用别的帧的几何")
                return None
            return draft
    except Exception as e:  # noqa: BLE001 —— 不得阻断打开，但必须留痕
        _LAST_STATE_LOAD_ERRORS.append(
            f"工作草稿读取失败（{f.name}：{type(e).__name__}: {e}）")
        return None
    return None


def resolve_persistent_state(rec: FrameRecord,
                             drafts_dir: Path) -> ResolvedState:
    """确认基线 + 工作草稿 -> 生效状态（v6.2 §5.2；纯读盘，不写盘）。

    规则表见 calibration/state.py：草稿只有带当前确认基线指纹时才生效；
    来源不可证明或基于旧基线 -> conflict，确认基线生效、草稿原地保留。
    """
    _LAST_STATE_LOAD_ERRORS.clear()          # 逐帧重置：诊断不得串到下一帧
    resolved = resolve_state(confirmed=_load_confirmed_annotation(rec),
                             draft=_load_draft_annotation(rec, drafts_dir),
                             source_sha256=rec.source_sha256 or "")
    if _LAST_STATE_LOAD_ERRORS:
        # 损坏 = 需要重新确认，且绝不能顺手覆盖：并进 reason 让 UI 说清楚。
        resolved.needs_reconfirm = True
        resolved.reason = ((resolved.reason + "；") if resolved.reason else "") + \
            "；".join(_LAST_STATE_LOAD_ERRORS) + \
            "。已停止自动覆盖，请先备份该文件再继续确认"
    return resolved


def annotation_for(rec: FrameRecord, drafts_dir: Path) -> Annotation | None:
    """生效标注：确认基线与工作草稿按 §5.2 规则决定谁生效。

    v6.1 此前是"确认标注无条件优先"，命中后草稿永远轮不到（审计 F4）。
    """
    return resolve_persistent_state(rec, drafts_dir).annotation
