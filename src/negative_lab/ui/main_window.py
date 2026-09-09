"""主窗口（v6 §5）：只有 裁切 与 转换 两个一级页面。

数据流（v6 §0.7/§5.2）：画布编辑 -> 草稿 `GeometryRegions`（FrameRecord）
-> 注释/会话持久化 -> 同一 `pipeline.service.process_image()` 刷新预览。
UI 不是唯一业务状态；service warning/error 不阻断画布编辑与保存（§5.2），
只给出轻量 inline 提示（§5.6.7）。

预览在降采样工作副本上调用同一 service（坐标经归一化相对位置映射），
导出走全分辨率（后续导出门禁）；UI 内不复制第二套图像算法。
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np
from PySide6.QtCore import QThread, QTimer, Qt
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QHBoxLayout, QLabel,
    QMainWindow, QMessageBox, QPushButton, QVBoxLayout, QWidget)

from .. import __version__
from ..calibration.annotations import (Annotation, ann_path_for,
                                       load_annotation, save_annotation)
from ..calibration.state import ResolvedState, is_confirmed, resolve_state
from ..io.images import ReadError, load_linear_rgb
from ..library.fingerprint import capture_fingerprint, fingerprints_compatible
from ..library.models import (FrameRecord, FrameStatus, GeometryRegions,
                              ImportSession, RollAnalysis)
from ..library.session import import_session, make_frame_record, suggest_groups_by_directory
from ..pipeline.geometry import OrientTransform
from ..pipeline.render import RenderIntent
from ..pipeline.roll_analysis import analyze_roll
from ..pipeline.service import process_image

from .. import storage   # 模块级：目录常量在 import 期就需要它（v6.2 审查修复）
_DRAFTS_DIR = storage.repo_path("outputs", "drafts")
from . import theme
from .library_panel import FilmStrip, LibraryPanel
from .preview_view import PreviewView, RoiKind
from .right_panel import RightPanel
from .workers import BusyState as _Busy

# 画布归属哨兵：self.current 已切到新帧、但画布还挂着上一帧像素的那段窗口。
# 用它（而不是 None）表达「确定不同源」，这样从未走过 _frame_done 的手工装配
# 调用方（测试）不会被误判。必须放在 import 之后，否则整块 import 变 E402。
_CANVAS_STALE = object()


def _drafts_dir() -> Path:
    """生效草稿目录：显式改道（模块全局）> 环境变量 > 仓库默认。

    此前测试/冒烟只改标注目录、不改草稿目录，导致真实样片的 GUI 工作稿
    被测试画进去（`outputs/drafts/1298_G200__G200-1.json` 实锤）。
    现在所有写入都经此函数，测试用 `storage.isolated_storage()` 整体改道。
    """
    from .. import storage
    return storage.drafts_dir(module_global=_DRAFTS_DIR)

MAX_PREVIEW_DIM = 1400          # 交互预览（fit 视图在 150% DPI 下约需 1400 物理px；再大走按需高清分片）
IMPORT_PROBE = False            # 导入只取元数据+哈希（完整探针太慢，用户实测）
#: v6.1 §3：后台高清替换默认关闭——当前实现把全分辨率整帧塞进使用
#: small ROI 坐标的场景，且把 worker 返回的 {"ok","value"} 当数组用，
#: 会显示错误内容；在实现同场景高清 tile 之前必须保持 False。
ENABLE_HD_PREVIEW = False


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

KIND_LABELS = {RoiKind.OUTPUT_CROP: "输出裁切",
               RoiKind.IMAGE_AREA: "有效画面",
               RoiKind.BASE: "片基取样"}


class MainWindow(QMainWindow):
    #: 画布显示图 -> 整幅技术正片的原点偏移（v6.2 F5）。
    #: 显示图为输出裁切子矩形时非零；任何画布坐标采样都必须先加它。
    _view_origin: tuple = (0, 0)
    _canvas_rec: object = None   # 画布像素当前归属的 FrameRecord
    _view_space: str = "positive_full"
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"负片裁切与去色罩 negative-lab v{__version__}")
        self.setMinimumSize(1024, 640)
        # 启动即最大化：窗口尺寸不得超过屏幕（150% DPI 下 1440 逻辑像素
        # = 2160 物理，会把右侧检查器整个裁出屏幕外——2026-09-02 实测教训）
        self.session: ImportSession | None = None
        self.current: FrameRecord | None = None
        self.t_source: np.ndarray | None = None
        self._small: np.ndarray | None = None
        self._small_scale = 1
        self._draft_touched = False
        self.pr = None
        self.roll_analysis: RollAnalysis | None = None
        self._intent = RenderIntent()
        self._candidate = "preserve"
        self._ann: Annotation | None = None
        self._load_meta: dict = {}
        self._fingerprint = None
        self._banner_text = ""
        self._viewer_background = None
        self._busy = _Busy()
        self._preview_pending = False
        self._frame_data: dict | None = None
        self._positive_full: np.ndarray | None = None

        self.setStyleSheet(theme.qss())
        central = QWidget()
        central.setObjectName("Root")
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(theme.SPACING["sm"], theme.SPACING["sm"],
                                theme.SPACING["sm"], 0)
        root.setSpacing(theme.SPACING["sm"])

        # ---- 顶部栏 44px：全局动作 | 帧状态摘要
        head = QHBoxLayout()
        head.setSpacing(theme.SPACING['sm'])
        self.btn_import = QPushButton('导入文件夹…')
        self.btn_import_file = QPushButton('导入文件…')
        self.btn_suggest = QPushButton('按目录建议分组')
        self.btn_export = QPushButton('整卷导出…')
        self.btn_export.setProperty('accent', True)
        for b in (self.btn_import, self.btn_import_file, self.btn_suggest,
                  self.btn_export):
            head.addWidget(b)
        head.addSpacing(theme.SPACING['md'])
        self.btn_confirm_rois = QPushButton('确认并保存标注')
        self.btn_confirm_rois.setToolTip(
            '把当前三类区域保存为用户签字标注（verified_by=user）')
        head.addWidget(self.btn_confirm_rois)
        head.addStretch(1)
        self.btn_hold_orig = QPushButton('按住看原片')
        self.btn_hold_orig.setProperty('flat', True)
        self.btn_hold_orig.setToolTip(
            '按住临时显示原负片（before），松开回到处理后的正片（after）')
        head.addWidget(self.btn_hold_orig)
        root.addLayout(head)

        # ---- 主体三列：左栏(卷) | 中央正片 | 右栏(节面板)  + 底部胶片条
        from PySide6.QtWidgets import QSplitter

        self.library = LibraryPanel()
        self.library.setMinimumWidth(theme.SIZES['left_w'] - 50)
        self.library.setMaximumWidth(theme.SIZES['left_w'] + 80)

        center = QWidget()
        center_lay = QVBoxLayout(center)
        center_lay.setContentsMargins(0, 0, 0, 0)
        center_lay.setSpacing(theme.SPACING['xs'])

        # 引导 pill（竞品式）：永远告诉用户下一步做什么（非模态）
        self.pill = QLabel('')
        self.pill.setProperty('pill', True)
        self.pill.setAlignment(Qt.AlignCenter)
        self.pill.setVisible(False)
        center_lay.addWidget(self.pill)

        # 查看区（相对定位容器：toast/缩放提示悬浮其上）
        from PySide6.QtWidgets import QGridLayout
        self.view_host = QWidget()
        view_grid = QGridLayout(self.view_host)
        view_grid.setContentsMargins(0, 0, 0, 0)
        self.view = PreviewView()
        self.view.setStyleSheet(
            f"background: {theme.COLORS['viewer']}; border: none;")
        view_grid.addWidget(self.view, 0, 0)
        # 处理中 toast（居中悬浮，非模态，竞品式）
        self.toast = QLabel('处理中…')
        self.toast.setProperty('toast', True)
        self.toast.setAlignment(Qt.AlignCenter)
        self.toast.setVisible(False)
        view_grid.addWidget(self.toast, 0, 0, Qt.AlignCenter)
        # 缩放提示（左下角 pill）
        self.lbl_zoom = QLabel('100%')
        self.lbl_zoom.setProperty('pill', True)
        self.lbl_zoom.setVisible(False)
        view_grid.addWidget(self.lbl_zoom, 0, 0,
                            Qt.AlignLeft | Qt.AlignBottom)
        view_grid.setContentsMargins(0, 0, 0, 0)
        center_lay.addWidget(self.view_host, 1)

        self.filmstrip = FilmStrip()
        self.filmstrip.frame_selected.connect(self.load_frame)
        center_lay.addWidget(self.filmstrip)

        self.right = RightPanel()

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self.library)
        splitter.addWidget(center)
        splitter.addWidget(self.right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setHandleWidth(1)
        splitter.setSizes([theme.SIZES['left_w'], 10000,
                           theme.SIZES['right_w']])
        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.addWidget(splitter)
        root.addLayout(body, 1)

        # ---- 状态行 24px
        row = QHBoxLayout()
        row.setSpacing(theme.SPACING['sm'])
        self.status = QLabel('就绪：导入包含 RAW / 16 位 TIFF / DNG 的文件夹')
        self.status.setProperty('caption', True)
        row.addWidget(self.status, 0)
        self.lbl_banner = QLabel('')
        self.lbl_banner.setProperty('caption', True)
        self.lbl_banner.setVisible(False)
        row.addWidget(self.lbl_banner, 1)
        root.addLayout(row)
        self._viewer_background = theme.COLORS['viewer']
        self._orig_held = False
        self._stage = 0          # 0=分析（负片） 1=调整（正片）
        self.lbl_stage_tag = None   # 阶段标签 pill（首帧时创建）
        # 草稿落盘防抖：拖动中只写内存，停顿 600ms 后才写盘（修卡顿）
        self._draft_flush_timer = QTimer(self)
        self._draft_flush_timer.setSingleShot(True)
        self._draft_flush_timer.setInterval(600)
        self._draft_flush_timer.timeout.connect(self._flush_draft)
        self._hd_tile_pending = False
        self._hd_generation = 0         # 高清任务代数：旧任务结果过期丢弃
        self.view.zoom_changed.connect(self._on_zoom_changed)
        self._tag_stage('原负片')
        self.pill.setText('导入包含 RAW / 16 位 TIFF / DNG 的文件夹后开始')
        self.pill.setVisible(True)

        # ---- 顶部动作
        self.btn_import.clicked.connect(self._pick_folder)
        self.btn_import_file.clicked.connect(self._pick_files)
        self.btn_suggest.clicked.connect(self._suggest_groups)
        self.btn_confirm_rois.clicked.connect(self.save_regions_as_annotation)
        self.btn_export.clicked.connect(self._on_export_button)
        self.library.frame_selected.connect(self.load_frame)
        self.library.grouping_changed.connect(self._on_grouping_changed)
        self.library.roll_selected.connect(self._load_first_frame_of)
        self.library.request_split.connect(self._on_split)

        # ---- 右栏（Stage 模型）
        rp = self.right
        rp.stage_changed.connect(self._set_stage)
        rp.area_action.connect(self._on_area_action)
        rp.area_clear.connect(self._on_area_clear)
        rp.auto_detect_requested.connect(self._auto_crop)
        rp.keep_rebate_toggled.connect(self._on_keep_rebate)
        rp.pick_neutral_toggled.connect(self._on_pick_neutral)
        rp.undo_pick_requested.connect(self._undo_pick_rescue)
        rp.rotate_requested.connect(self._rotate)
        rp.mirror_requested.connect(self._mirror)
        rp.fine_changed.connect(self._set_fine_rotation)
        rp.go_edit_requested.connect(lambda: self._set_stage(1))
        rp.reset_geometry.connect(self._reset_all_geometry)
        rp.candidate_chosen.connect(self._choose_candidate)
        rp.tone_changed.connect(self._intent_changed)
        rp.tone_committed.connect(self._intent_committed)
        rp.sync_tone_requested.connect(self._sync_intent)
        rp.intent_reset_requested.connect(self._reset_all_intent)
        self.btn_hold_orig.pressed.connect(lambda: self._hold_original(True))
        self.btn_hold_orig.released.connect(lambda: self._hold_original(False))

        # ---- 画布
        self.view.roi_created.connect(self._on_roi_edited)
        self.view.roi_changed.connect(self._on_roi_edited)
        self.view.roi_deleted.connect(self._on_roi_deleted)
        self.view.roi_live_changed.connect(self._on_roi_live_changed)
        self.view.roi_selected.connect(self._on_roi_selected)
        self.view.replace_hint.connect(lambda t: self._set_banner(t, 'info'))
        self.view.draw_rejected.connect(self._on_draw_rejected)
        self.view.point_picked.connect(self._on_point_picked)
        self.view.pick_mode_changed.connect(self.right.set_pick_mode_active)
        self._stage = 0          # 0=分析（负片） 1=调整（正片）
        # R5b 拾点救援状态：撤销栈（每帧独立，换帧清空）
        self._pick_undo_stack: list = []

    # ------------------------------------------------------- Stage 切换
    def current_page(self) -> int:
        return self._stage

    def page_count(self) -> int:
        return 1

    def page_title(self, index: int = 0) -> str:
        return '工作区'

    def _set_stage(self, i: int):
        """Stage 切换（竞品 Convert/Edit 模型）：

        - 分析：中央显示**原负片**（分析要在负片上看片基/文字），
          三类 ROI 叠加可见；
        - 调整：中央显示**转换结果**（缓存合成正片，毫秒级），
          ROI 隐藏（构图在分析阶段已定）。
        """
        self._stage = 0 if i == 0 else 1
        self.right.set_stage(self._stage)
        # v6.2 §7.6 / 审查 F9：中性点拾点只允许存在于调整阶段。
        # set_draw_mode(None) 只在 kind 非空时清 _pick_mode，靠它清不掉拾点态，
        # 曾出现"看着负片拾点、实际按正片像素求解"。
        picking = bool(self._stage == 1
                       and getattr(self.view, "_pick_mode", None) == "neutral")
        if not picking:
            self.view.set_pick_mode(None)       # 幂等退出
        self.right.set_pick_mode_active(picking)
        self.right.btn_pick_neutral.blockSignals(True)
        self.right.btn_pick_neutral.setChecked(picking)
        self.right.btn_pick_neutral.blockSignals(False)
        if self._stage == 0:
            self.view.set_rois_visible(True)
            self.view.set_draw_mode(None)
            self.view._crop_dim = 0.0
            if self._small is not None and not self._orig_held:
                self._show_analysis_image(fit=True)
            self._tag_stage('原负片')
        else:
            self.view.set_draw_mode(None)
            self.view.set_rois_visible(False)
            self._compose_view(fit=True)
            self._tag_stage('转换结果')
        self._update_guidance()
        self._update_area_states()

    def _tag_stage(self, text: str):
        """图像左上角的阶段标签 pill（竞品 Original Negative/Converted）。"""
        if self.lbl_stage_tag is None:

            self.lbl_stage_tag = QLabel(text)
            self.lbl_stage_tag.setProperty('pill', True)
            lay = self.view_host.layout()
            lay.addWidget(self.lbl_stage_tag, 0, 0,
                          Qt.AlignTop | Qt.AlignLeft)
        else:
            self.lbl_stage_tag.setText(text)
        self.lbl_stage_tag.setVisible(True)

    def _update_guidance(self):
        """引导 pill：按当前状态告诉用户下一步（竞品核心 UX）。"""
        if self.current is None:
            self.pill.setVisible(False)
            return
        g = self.current.geometry
        if self._stage == 0:
            if not g.image_area_roi:
                text = '拖动绿框圈出单帧画面区域（右栏「有效画面 · 框选」）'
            elif not g.base_sample_rois:
                text = '片基未取样：点「片基 Dmin · 取样」，在未曝光片基上拖出一小块'
            else:
                text = '分析完成：切到「调整」查看去色罩结果'
        else:
            text = '拖动滑杆微调，直方图实时更新；按住「看原片」对比'
        self.pill.setText(text)
        self.pill.setVisible(True)

    def _update_area_states(self):
        """右栏分析区状态行（彩点 + 文本 + 清除按钮）。"""
        if self.current is None:
            for key in ('image_area', 'base', 'output_crop'):
                self.right.set_area_state(key, has=False, text='未载入帧')
            return
        g = self.current.geometry
        self.right.set_area_state(
            'image_area', has=bool(g.image_area_roi),
            text='已框选' if g.image_area_roi else '未框选',
            allow_clear=bool(g.image_area_roi))
        self.right.set_area_state(
            'base', has=bool(g.base_sample_rois),
            text=(f'已取样 {len(g.base_sample_rois)} 块'
                  if g.base_sample_rois else '未取样'),
            allow_clear=bool(g.base_sample_rois))
        self.right.set_area_state(
            'output_crop', has=bool(g.output_crop_roi),
            text='已裁切' if g.output_crop_roi else '未裁切（导出用画面区）',
            allow_clear=bool(g.output_crop_roi))

    AREA_KIND = {'image_area': RoiKind.IMAGE_AREA,
                 'base': RoiKind.BASE,
                 'output_crop': RoiKind.OUTPUT_CROP}

    def _on_area_action(self, key: str):
        """分析区动作按钮：进入对应类型的绘制模式（分析阶段在负片上）。"""
        if self._stage != 0 or self.current is None:
            self._set_stage(0)
        kind = self.AREA_KIND[key]
        self.view.set_rois_visible(True)
        self.view.set_draw_mode(kind)
        label = {'image_area': '有效画面', 'base': '片基 Dmin',
                 'output_crop': '输出裁切'}[key]
        self._set_banner(f'在画布上拖出{label}区域（Esc 取消）', 'info')

    def _on_area_clear(self, key: str):
        kind = self.AREA_KIND[key]
        self.view.clear_rois(kind)
        self._update_draft_from_view()
        self._update_area_states()
        self._update_guidance()

    def _on_keep_rebate(self, on: bool):
        """v6.1 §4：显式检查当前帧后写入 FrameRecord（不得写 None/串帧）。"""
        if self.current is None:
            return
        self.current.keep_rebate = bool(on)
        self._save_draft_annotation(self.current, quiet=True)
        self._set_banner("保留片边已开启：导出将包含整幅（含胶片边框）"
                         if on else "保留片边已关闭：导出使用输出裁切", "info")

    def _sync_keep_rebate_ui(self, rec: FrameRecord | None):
        """帧加载/切换后：右栏勾选状态反映 rec.keep_rebate（不串上一帧）。"""
        self.right.set_keep_rebate(bool(rec.keep_rebate)
                                   if rec is not None else False)

    def _set_fine_rotation(self, deg: float):
        if self.current is None:
            return
        self.current.geometry.fine_rotation_degrees = float(deg)
        self.right.sec_areas.set_fine_rotation(float(deg))
        self._update_draft_from_view()
        self._compose_view(fit=True) if self._stage == 1 else \
            self._refresh_preview(seed_if_untouched=False)

    def _hold_original(self, on: bool):
        """按住看原片（负片）；松开回到当前阶段视图。"""
        self._orig_held = bool(on)
        if self._small is None:
            return
        if on or self._stage == 0:
            self._show_analysis_image()
        else:
            self._compose_view()

    def _update_input_info(self, rec: FrameRecord, meta: dict):
        """右栏「输入」节：只读元信息（竞品 Film Type/Decoder 式）。"""
        self.right.set_input_info(
            file=Path(rec.source_path).name,
            camera=str(meta.get('camera_model') or meta.get('make') or '—'),
            decoder=str(meta.get('kind') or '—'),
            bits=f"{meta.get('bits', '')}".strip() or '16bit?',
            fingerprint=rec.capture_fingerprint or '')

    def _consistency_text(self) -> str:
        """R5c：右栏「技术信息」的卷级片基一致性行（研究 #1 的数字）。

        - 实测片基帧的色度极差（log10 通道/均值，decade）＝同卷处理
          一致性的物理锚点；generic_prior 帧的形状是资产常量，不参与
          统计（混入会伪造漂移，研究 #1 §2.4 教训）；
        - prior 帧计数本身是风险信号（片基取样薄弱）。
        """
        roll = self.roll_analysis
        if roll is None:
            return ""
        stats = (roll.base_statistics or {})
        per_frame = stats.get("per_frame") or {}
        if len(per_frame) >= 2:
            import math as _math
            C = []
            for v in per_frame.values():
                d = np.asarray(v, np.float64)
                m = float(d.mean())
                if m > 1e-9:
                    C.append([_math.log10(max(x, 1e-9) / m) for x in d])
            if len(C) >= 2:
                Ca = np.asarray(C)
                drift = float(np.max(Ca.max(axis=0) - Ca.min(axis=0)))
                qual = ("很好" if drift <= 0.03 else
                        "可接受" if drift <= 0.06 else "有漂移")
                line = (f"本卷片基一致性：{qual}"
                        f"（{len(per_frame)} 帧，色度极差 {drift:.3f} decade）")
                if roll.base_source == "generic_prior":
                    line += "；注意：本帧使用通用先验回退"
                return line
        if roll.base_source == "generic_prior":
            return ("本卷没有可靠片基帧：全部使用通用先验回退"
                    "（建议重画片基取样区）")
        return ""

    def _update_tech_info(self):
        """右栏「技术信息」节：片基来源/门禁/一致性（R5c）。"""
        if self.current is None or self.pr is None:
            return
        self.right.set_tech_info(
            base_source=self.pr.base_source,
            gate=self.pr.gate.status,
            review=list(self.pr.gate.review_reasons),
            warnings=list(self.pr.report.warnings)[:4],
            consistency=self._consistency_text())

    def _update_histograms(self):
        """两处直方图：分析=负片；调整=当前合成正片。"""
        try:
            if self._small is not None:
                x = np.clip(self._small, 0, None)
                self.right.hist_convert.set_image(
                    x / max(float(x.max()), 1e-6))
            pos = self._rendered_full() if self._positive_full is not None \
                else None
            self.right.hist_edit.set_image(pos)
        except Exception:  # noqa: BLE001 直方图失败不影响编辑
            pass

    def _pick_folder(self):
        d = QFileDialog.getExistingDirectory(self, "导入包含负片的文件夹")
        if d:
            self.load_folder(Path(d))

    def _pick_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "导入负片", "",
            "支持 (*.NEF *.nef *.TIF *.tif *.TIFF *.tiff *.DNG *.dng);;所有文件 (*)")
        if files:
            self.load_files([Path(f) for f in files])

    def load_folder(self, folder: Path) -> ImportSession:
        """后台导入（§15.7 响应性）：只取元数据+哈希，不做完整解码探针
        （12 张 NEF 完整探针 20s——用户实测过久；解码失败在打开帧时
        给出完整错误即可）。"""
        self._set_banner(f"正在导入 {folder.name} …（读取元数据与哈希，"
                         "界面保持可用）", "info")
        from .workers import run_in_thread
        self._busy_enter("import")
        return run_in_thread(
            lambda cb: import_session(folder, probe=IMPORT_PROBE),
            on_done=lambda r: self._import_done(r, folder),
            on_progress=lambda m: self._set_banner(f"导入：{m}", "info"))

    def _import_done(self, result: dict, folder: Path):
        self._busy_leave("import")
        if not result.get("ok"):
            self._set_banner(f"导入失败：{result.get('error')}", "danger")
            return
        self._adopt_session(result["value"])
        # 导入即预览（兜底）：若自动选卷链路没触发加载，直接载第一帧
        if self.current is None and self.session is not None \
                and self.session.files:
            self._load_first_frame_of(self.library.current_roll)

    def load_files(self, files: list[Path]) -> ImportSession:
        sess = ImportSession()
        root = Path(files[0]).parent if files else Path(".")
        for f in files:
            sess.files.append(make_frame_record(f, root=root, probe=False))
        return self._adopt_session(sess)

    def _adopt_session(self, sess: ImportSession) -> ImportSession:
        self.session = sess
        self.library.set_session(sess)
        self.filmstrip.set_frames([], pixmap_of=self._strip_pixmap)
        if sess.roll_groups:
            # 导入即预览：选卷 -> 库面板选中信号会自动加载第一帧
            self.library.select_roll(sess.roll_groups[0].id)
        else:
            # 无分组：选"未分卷"（select_roll(None)=第一项），触发
            # _roll_changed -> _populate_grid -> roll_selected -> 自动预览
            self.library.select_roll(None)
        self._set_banner(f"已导入 {len(sess.files)} 个文件", "info")
        return sess

    def _load_first_frame_of(self, rid: str | None):
        """自动预览：载入该卷（或未分组）的第一帧。"""
        frames = self._frames_of(rid or "__ungrouped__")
        if frames:
            self.load_frame(frames[0].source_path)

    def _strip_pixmap(self, rec: FrameRecord):
        """胶片条取图：与左栏缩略图同源（后台生成后由 library 缓存）。"""
        return self.library._pixmaps.get(rec.source_path)

    def _sync_filmstrip(self, rec: FrameRecord | None = None):
        rid = self.library.current_roll
        frames = self._frames_of(rid or "")
        self.filmstrip.set_frames(frames, pixmap_of=self._strip_pixmap)
        cur = rec or self.current
        if cur is not None:
            self.filmstrip.select(cur.source_path)

    def _suggest_groups(self):
        if self.session is None:
            return
        suggest_groups_by_directory(self.session, apply=True)
        self.library.set_session(self.session)
        self._set_banner("已按目录建立卷分组（可继续改名、拆分、移动）", "success")

    def _on_grouping_changed(self):
        if self.session is None:
            return
        rid = self.library.current_roll
        if rid and rid != "__ungrouped__":
            self._analyze_current_roll()
        frames = self.library.current_frames()
        if frames:
            self.load_frame(frames[0])

    def _on_split(self, _rid: str, _frames: list[str]):
        self._analyze_current_roll()

    # ---------------------------------------------------------------- 帧
    def load_frame(self, path: str):
        """选帧：解码+卷分析+首预览全部在后台线程（主线程零阻塞）。

        同一时刻只允许一个帧加载任务（BusyState 防抖）；查看区立即进入
        “处理中”状态并保留上一帧的 ROI（v6 §5.7 加载状态）。
        """
        if self.session is None:
            return
        rec = self.session.frame(path)
        if rec is None:
            return
        if not self._busy_enter("frame"):
            # 冷读审查 L1-②：这里原本写"稍候自动更新"，但代码里**不存在**任何
            # 待处理帧队列，用户必须自己再点一次。承诺不成立的文案比没有更糟。
            self._set_banner("上一帧仍在处理中，本次切换已忽略：请等它完成后重新点这一帧",
                             "warning")
            return
        # 审查修复（GUI 审查 重要-3）：600ms 防抖窗口内切帧，定时器随后会在
        # **新** self.current 上落盘，把旧帧的血缘（_state_fingerprint/
        # _draft_revision）写进新帧草稿 -> 下次加载必然 conflict。切帧前先
        # 替旧帧 flush 并停表。
        if self._draft_flush_timer.isActive():
            self._draft_flush_timer.stop()
            self._flush_draft()
        self._canvas_rec = _CANVAS_STALE   # 画布仍是上一帧数据：期间拒绝拾点（重要-6）
        self.current = rec
        self.library.set_current_frame(path)
        self._set_status(f"{Path(path).name} | 处理中…")
        from .workers import run_in_thread
        run_in_thread(lambda cb: self._load_frame_job(rec, cb),
                      on_done=lambda r: self._frame_done(r, rec),
                      on_progress=lambda m: self._set_banner(m, "info"))

    def _load_frame_job(self, rec: FrameRecord, cb):
        """后台执行：解码 -> 指纹 -> 卷分析 -> 预览。返回所有 UI 需要的数据。"""
        cb("解码原图…")
        loaded = self._load(rec)
        fp = capture_fingerprint(
            loaded.meta, source_path=rec.source_path,
            batch_assumption=(self.session.batch_encoding_assumption
                              if self.session else None))
        cb("卷级片基分析…")
        # v6.2 F1：先解析持久状态（确认基线 / 工作草稿），再决定用什么几何跑算法。
        # 此前这里传的是 ann=None + rec.geometry（新会话必为空），算法实际跑的是
        # 自动检测，"重启后恢复用户标定"全靠 _frame_done 事后回填 —— 界面显示的
        # 是用户标定，缓存的技术正片却是自动检测的结果，两者不一致。
        state = resolve_persistent_state(rec)
        ann = state.annotation
        roll = self._analyze_roll_frames()
        cb("生成预览…")
        k = max(1, int(np.ceil(max(loaded.rgb.shape[:2]) / MAX_PREVIEW_DIM)))
        small = loaded.rgb[::k, ::k, :].copy()
        # 只有**完整**的用户草稿/标注才传给 service；空标注传空几何会经
        # 归一化映射钳制出荒谬 ROI（实测：60px 横带被当 image_area，
        # 预览被裁成一条线）。不完整时交给 service 自动检测再种回草稿。
        eff_geo = rec.geometry
        if state.annotation is not None and not eff_geo.is_complete():
            # 标注/草稿里的三类区域按 SOURCE 归一化，_geometry_for_job 再映到工作副本
            eff_geo = _geometry_from_annotation(state.annotation)
        seed_geo = eff_geo if eff_geo.is_complete() else None
        # keep_rebate=True：技术正片按**整幅**计算并缓存——用户的裁切/
        # 微调之后都是缓存上的即时合成（毫秒级），只有片基变化才重跑
        pr = process_image(
            small,
            geometry=(_geometry_for_job(None, small, loaded.rgb, seed_geo)
                      if seed_geo is not None else None),
            ann=None, roll_analysis=roll,
            render_intent=self._intent, candidate=self._candidate,
            keep_rebate=True, capture_fingerprint=fp, film_type="unknown")
        return {"rec": rec, "rgb": loaded.rgb, "meta": loaded.meta,
                "fingerprint": fp, "annotation": ann, "roll": roll,
                "preview": pr, "small": small, "scale": k,
                "state": state.to_dict(), "geometry_seeded": seed_geo is not None}

    def _frame_done(self, result: dict, rec: FrameRecord):
        self._busy_leave("frame")
        if not result.get("ok"):
            self._set_banner(f"帧处理失败：{result.get('error')}", "danger")
            self._set_status(f"{Path(rec.source_path).name} | 失败")
            return
        d = result["value"]
        if d["rec"] is not self.current:
            return          # 用户已切到别的帧：丢弃过期结果
        self._frame_data = d
        self.t_source = d["rgb"]
        self._small = d["small"]
        self._small_scale = d["scale"]
        self._load_meta = d["meta"]
        self._fingerprint = d["fingerprint"]
        self._ann = d["annotation"]
        # v6.2 §5.2 / F4：记住本帧持久状态的来源，之后写草稿要带上确认基线指纹；
        # 冲突或源图失效必须当场说清，不能让用户以为草稿还接着上次确认的标注。
        self._state_resolution = dict(d.get("state") or {})
        self._state_fingerprint = str(self._state_resolution.get(
            "confirmed_fingerprint") or "")
        self._draft_revision = int(self._state_resolution.get(
            "draft_revision") or 0)
        self._state_notice = (
            str(self._state_resolution.get("reason") or "")
            if self._state_resolution.get("source") in
            ("conflict", "stale_source") else "")
        self.roll_analysis = d["roll"]
        rec = d["rec"]
        rec.capture_fingerprint = d["fingerprint"].value
        # 合并而非覆盖：导入期写入的 abs_path 不在 load 的 meta 里，
        # 整体覆盖会让仓库外文件在后续 _abs()/导出时回退到仓库根拼接。
        rec.input_facts = {**(rec.input_facts or {}), **dict(d["meta"]),
                           "kind": d["meta"].get("kind", "")}
        if self._ann is not None:
            # v6.1 §4：标注/草稿里的「保留片边」状态先于几何恢复（无几何时
            # 也要恢复，避免旧会话勾选状态丢失）
            rec.keep_rebate = bool(getattr(self._ann, "keep_rebate", False))
        if self._ann is not None and not rec.geometry.is_complete():
            rec.geometry = _geometry_from_annotation(self._ann)
        self.pr = d["preview"]
        # 缓存整幅技术正片：裁切/微调的即时合成数据源
        self._positive_full = self.pr.result.positive_linear
        if not rec.geometry.is_complete() and self.pr is not None:
            # 自动种子只作初始草稿；service 在 small 副本上检测，必须换算
            # 回 SOURCE 全分辨率（与 _refresh_preview 同一规则）
            src_geo = _unscale_geometry(self.pr.geometry.roi_source_space,
                                        self.t_source.shape[:2],
                                        self._small.shape[:2])
            rec.geometry = GeometryRegions(
                rotation_degrees=rec.geometry.rotation_degrees,
                fine_rotation_degrees=rec.geometry.fine_rotation_degrees,
                mirror_horizontal=rec.geometry.mirror_horizontal,
                output_crop_roi=src_geo.get("output_crop_roi"),
                image_area_roi=src_geo.get("image_area_roi"),
                base_sample_rois=src_geo.get("base_sample_rois") or [])
            self._save_draft_annotation(rec, quiet=True)
        self._draft_touched = rec.geometry.is_complete()
        self._sync_filmstrip(rec)
        # 先把草稿 ROI 放上画布（_show_frame），再合成显示（_compose_view）
        # ——顺序反了或缺了都会让"框消失"（2026-09-02 实测教训）
        self._show_frame()
        # 竞品式：分析阶段看原负片；调整阶段看转换结果
        if self._stage == 0:
            self._show_analysis_image(fit=True)
            self._tag_stage('原负片')
        else:
            self._compose_view(fit=True)
            self._tag_stage('转换结果')
        self._update_input_info(rec, d['meta'])
        self._update_tech_info()
        self._pick_undo_stack.clear()       # 换帧：撤销栈/拾点提示清空
        self.right.set_undo_pick_enabled(False)
        self.right.set_pick_hint("")
        self.view.clear_points()           # 上一帧的中性点圆点不得留在新帧画面上
        self._sync_keep_rebate_ui(rec)      # v6.1 §4：勾选状态随帧恢复
        self._update_area_states()
        self._update_guidance()
        self._update_histograms()
        # 卷指纹冲突：worker 线程只交出数据（见 _analyze_roll_frames），
        # 落控件的动作必须在主线程、且在算最终 banner 之前。
        conflicts = getattr(self, "_pending_conflicts", None) or []
        self._pending_conflicts = None
        if conflicts:
            self.library.mark_conflicts(conflicts)
            self._state_notice = (
                (self._state_notice + "；") if self._state_notice else "") + (
                    f"本卷有 {len(conflicts)} 帧采集指纹不一致：已停止共享卷分析，"
                    "可用“拆分”一键分成不同卷")
        sh, sw = self.t_source.shape[:2]
        info_line = (f"{Path(rec.source_path).name} | {sw}x{sh} | "
                     f"片基来源 {self.pr.base_source}")
        # 审查修复（GUI 审查 重要-1）：这里原本无条件用 info 覆盖同一条
        # QLabel，conflict/stale_source 的警告在同一函数里闪一下就没了 ——
        # F4 要求「当场说清」实际做不到。现在把警告并入最终 banner。
        if self._state_notice:
            self._set_banner(f"⚠ {self._state_notice}｜{info_line}", "warning")
        else:
            self._set_banner(info_line, "info")
        self._canvas_rec = rec       # 画布与 current 同源，拾点/换算方可放行

    def _load(self, rec: FrameRecord):
        assumption = None
        if self.session and self.session.batch_encoding_assumption:
            assumption = self.session.batch_encoding_assumption.get("encoding")
        path = _abs(rec.source_path, rec)
        try:
            return load_linear_rgb(path, tiff_encoding=assumption)
        except ReadError as e:
            if "tiff_encoding" not in str(e):
                raise
            choice = self._ask_batch_encoding(rec, e)
            if not choice:
                raise
            self.session.batch_encoding_assumption = {
                "encoding": choice, "id": f"batch-{choice}",
                "confirmed_by": "user", "applied_to_roll": rec.roll_id}
            return load_linear_rgb(path, tiff_encoding=choice)

    def _ask_batch_encoding(self, rec: FrameRecord, err: ReadError) -> str | None:
        """v5 §4.3.3：编码事实矛盾时一次**导入批次级**选择（非设备配置）。"""
        app = QApplication.instance()
        if app is not None and QThread.currentThread() is not app.thread():
            # 审查修复（GUI 审查 致命-1）：本函数会被 _load() 在**后台解码线程**
            # 里调到。Qt 规定 QWidget 只能在 GUI 线程创建与 exec()，在 worker
            # 里弹窗是未定义行为（轻则提示丢失，重则 access violation）。
            # 宁可干净地失败让用户显式指定编码，也不能崩。
            # 下一轮的正解：把「请求选择」做成主线程信号 + worker 等待事件。
            raise ReadError(
                "传递函数歧义，且当前在后台线程无法弹窗：请先在导入批次设置中"
                "显式指定编码假设（线性 / sRGB）后重试。\n原始诊断：" + str(err))
        box = QMessageBox(self)
        box.setWindowTitle("输入批次编码假设")
        box.setText(
            "这一批文件的传递函数无法从 metadata 确定。\n"
            "请选择该批次的编码假设（同一批次复用，不要求扫描仪型号）：\n\n"
            f"{err}")
        b_lin = box.addButton("线性扫描", QMessageBox.AcceptRole)
        b_srgb = box.addButton("sRGB/显示编码扫描", QMessageBox.AcceptRole)
        box.addButton("取消导入", QMessageBox.RejectRole)
        box.exec()
        return {b_lin: "linear", b_srgb: "srgb"}.get(box.clickedButton())

    def _analyze_current_roll(self):
        """卷级片基汇总（同采集指纹内共享；冲突帧不共享，v5 §4.4）。"""
        self.roll_analysis = self._analyze_roll_frames()

    def _analyze_roll_frames(self) -> RollAnalysis | None:
        if self.session is None or not self.library.current_roll:
            return None
        frames = [f for f in self._frames_of(self.library.current_roll)
                  if f.capture_fingerprint]
        if not frames:
            return None
        main_fp = max({f.capture_fingerprint for f in frames},
                      key=lambda fp: sum(1 for f in frames
                                         if f.capture_fingerprint == fp))
        stats = []
        for rec in frames:
            st = rec.frame_analysis or {}
            if not st.get("base_dmin"):
                continue
            stats.append({
                "frame_id": rec.source_path,
                "capture_fingerprint": rec.capture_fingerprint,
                "base_dmin": st["base_dmin"],
                "base_confidence": st.get("base_confidence", 0.0),
                "contamination": st.get("contamination"),
                "black_level": st.get("black_level"),
                "spatial_trend": st.get("spatial_trend")})
        ra = analyze_roll(stats, fingerprint=main_fp)
        conflicts = [f.source_path for f in frames
                     if not fingerprints_compatible(f.capture_fingerprint, main_fp)]
        # 审查修复（GUI 审查 致命-1）：本函数在**后台解码线程**里被调用，
        # 而 mark_conflicts() 改 QListWidget item、_set_banner() 改 QLabel ——
        # Qt 规定控件只能在 GUI 线程动，workers.py 的契约也写明「worker 不
        # 直接触碰控件」。这里改为只交出数据，由 _frame_done 回主线程后落地。
        self._pending_conflicts = list(conflicts)
        return ra

    def _frames_of(self, rid: str) -> list[FrameRecord]:
        if self.session is None:
            return []
        if rid == "__ungrouped__":
            return self.session.ungrouped()
        r = self.session.roll(rid)
        return [f for f in (self.session.frame(p) for p in (r.frame_ids if r else []))
                if f is not None]

    # --------------------------------------------------- 草稿 <-> 预览
    def _prepare_frame_views(self):
        """（同步路径，测试/兜底用）：管线同步跑一次，状态与图一次到位。"""
        rec = self.current
        k = max(1, int(np.ceil(max(self.t_source.shape[:2]) / MAX_PREVIEW_DIM)))
        self._small = self.t_source[::k, ::k, :].copy()
        self._small_scale = k
        self._draft_touched = rec.geometry.is_complete()
        geo_snapshot = rec.geometry
        seed = not self._draft_touched
        self.pr = process_image(
            self._small,
            geometry=(_geometry_for_job(None, self._small, self.t_source,
                                        geo_snapshot)
                      if rec.geometry.is_complete() else None),
            ann=None, roll_analysis=self.roll_analysis,
            render_intent=self._intent, candidate=self._candidate,
            keep_rebate=True, capture_fingerprint=self._fingerprint,
            film_type="unknown")
        self._positive_full = self.pr.result.positive_linear
        if seed:
            src_geo = _unscale_geometry(self.pr.geometry.roi_source_space,
                                        self.t_source.shape[:2],
                                        self._small.shape[:2])
            rec.geometry = GeometryRegions(
                rotation_degrees=rec.geometry.rotation_degrees,
                fine_rotation_degrees=rec.geometry.fine_rotation_degrees,
                mirror_horizontal=rec.geometry.mirror_horizontal,
                output_crop_roi=src_geo.get("output_crop_roi"),
                image_area_roi=src_geo.get("image_area_roi"),
                base_sample_rois=src_geo.get("base_sample_rois") or [])
            self._save_draft_annotation(rec, quiet=True)
        self._show_frame()
        self._sync_keep_rebate_ui(rec)      # v6.1 §4：同步路径同样恢复勾选
        self._compose_view(fit=True)

    def _geometry_for_service(self) -> GeometryRegions:
        """把视图（ORIENTED 预览坐标）的三类 ROI 换算回 SOURCE 草稿。"""
        rec = self.current
        tr = OrientTransform(self.t_source.shape[:2],
                             rec.geometry.rotation_degrees,
                             rec.geometry.mirror_horizontal,
                             rec.geometry.fine_rotation_degrees)
        s = self._small_scale
        h, w = tr.oriented_size

        def to_src(rect):
            if rect is None:
                return None
            y0 = int(rect.y()) * s
            x0 = int(rect.x()) * s
            y1 = min(int(rect.bottom()) * s, h)
            x1 = min(int(rect.right()) * s, w)
            if y1 - y0 < 8 or x1 - x0 < 8:
                return None
            return tr.bbox_oriented_to_source((y0, x0, y1, x1))

        return GeometryRegions(
            rotation_degrees=rec.geometry.rotation_degrees,
            fine_rotation_degrees=rec.geometry.fine_rotation_degrees,
            mirror_horizontal=rec.geometry.mirror_horizontal,
            output_crop_roi=to_src(self.view.roi(RoiKind.OUTPUT_CROP)),
            image_area_roi=to_src(self.view.roi(RoiKind.IMAGE_AREA)),
            base_sample_rois=[r for r in (
                to_src(self.view.roi(RoiKind.BASE, i))
                for i in range(self.view.roi_count(RoiKind.BASE))) if r])

    def _small_geometry(self, geo: GeometryRegions) -> GeometryRegions:
        """草稿 -> 降采样工作副本坐标（归一化相对位置映射）。"""
        return _geometry_for_job(None, self._small, self.t_source, geo)

    def _busy_enter(self, key: str) -> bool:
        if not self._busy.try_enter(key):
            return False
        if key == "preview":
            self.view.set_busy(True)
            self._show_toast('正在分析片基与密度范围…')
        return True

    def _busy_leave(self, key: str):
        self._busy.leave(key)
        if key == "preview":
            self.view.set_busy(self._busy._locks.__contains__("preview"))
            if not self._busy._locks.__contains__("preview"):
                self._hide_toast()

    def _refresh_preview(self, *, seed_if_untouched: bool = False,
                         reason: str = "crop"):
        """管线刷新：只处理**影响技术底座**的变化。

        v6 交互模型（对齐 Lightroom，用户 2026-09-02 指示）：
        - 每帧只算一次**整幅**技术正片并缓存（keep_rebate=True）；
        - 裁切/微调 = 缓存正片上的即时合成（_compose_view，毫秒级），
          不进本函数；
        - 片基取样区或有效画面区变化才重跑管线（v6.2 F2：两者都决定
          Dmin/Dmax 与色罩统计范围；这行此前与 _changes_statistics 判据不一致）；
        - 后台线程执行，主线程零阻塞。
        """
        rec = self.current
        if rec is None or self._small is None:
            return
        if not self._busy_enter("preview"):
            self._preview_pending = True
            return
        geo_snapshot = rec.geometry
        touched = self._draft_touched
        seed = bool(seed_if_untouched and not touched)
        small = self._small
        fp = self._fingerprint
        roll = self.roll_analysis
        intent_snapshot = self._intent.copy()
        cand_snapshot = self._candidate

        def job(cb):
            return process_image(
                small,
                geometry=_geometry_for_job(None, small, self.t_source,
                                           geo_snapshot),
                ann=None, roll_analysis=roll,
                render_intent=intent_snapshot, candidate=cand_snapshot,
                keep_rebate=True,          # 整幅技术正片（裁切走即时合成）
                capture_fingerprint=fp, film_type="unknown")

        def done(result: dict):
            self._busy_leave("preview")
            if result.get("ok") is False:
                self._set_banner(f"预览刷新失败：{result.get('error')}"
                                 "（区域编辑与草稿保存不受影响）", "danger")
            elif result.get("ok"):
                pr = result["value"]
                if self.current is not rec:
                    return                      # 已切帧：过期结果丢弃
                self.pr = pr
                rec.frame_analysis = {
                    "base_dmin": pr.frame_base.get("dmin"),
                    "base_confidence": pr.frame_base.get("confidence"),
                    "contamination": pr.frame_base.get("contamination"),
                    "black_level": pr.frame_base.get("black_level"),
                    "spatial_trend": pr.frame_base.get("spatial_trend"),
                    "regions": [list(r) for r in geo_snapshot.base_sample_rois]}
                if seed and not rec.geometry.is_complete():
                    # service 在降采样副本上检测：ROI 必须换算回 SOURCE
                    # 全分辨率（service 的 roi_source_space 是 small 空间坐标，
                    # 直接写草稿会产生越界框——2026-09-02 实测教训）
                    src_geo = _unscale_geometry(
                        pr.geometry.roi_source_space,
                        self.t_source.shape[:2], self._small.shape[:2])
                    rec.geometry = GeometryRegions(
                        rotation_degrees=geo_snapshot.rotation_degrees,
                        fine_rotation_degrees=geo_snapshot.fine_rotation_degrees,
                        mirror_horizontal=geo_snapshot.mirror_horizontal,
                        output_crop_roi=src_geo.get("output_crop_roi"),
                        image_area_roi=src_geo.get("image_area_roi"),
                        base_sample_rois=src_geo.get("base_sample_rois") or [])
                    self._save_draft_annotation(rec, quiet=True)
                self._set_status(_frame_status_text(rec, pr))
                # 技术底座已更新：刷新缓存正片并按当前阶段上屏
                self._positive_full = pr.result.positive_linear
                if self._stage == 1:
                    self._compose_view(fit=True)
                else:
                    self._show_frame()          # 分析阶段：负片+ROI
                self._update_area_states()
                self._update_guidance()
                self._update_tech_info()
                self._set_banner('片基分析完成', 'success')
            if getattr(self, "_preview_pending", False):
                self._preview_pending = False
                self._refresh_preview()

        from .workers import run_in_thread
        run_in_thread(job, on_done=done)

    # ------------------------------------------------ 即时合成（零管线）
    def _rendered_full(self) -> np.ndarray:
        """缓存整幅技术正片 + 当前 RenderIntent -> 显示用渲染图（ms 级）。"""
        from ..pipeline.render import apply_render

        return apply_render(self._positive_full, self._intent)

    def _compose_view(self, *, fit: bool = False):
        """把当前状态合成到中央常驻 viewer（毫秒级，零管线调用）。

        - 裁切模式：显示**整幅渲染正片**，输出裁切框外由遮罩压暗
          （Lightroom 式"框内=成品"），三类 ROI 手柄可继续调；
        - 转换模式：只显示输出裁切内的渲染正片（构图即所见）；
        - "按住看原片"：显示原负片。
        """
        rec = self.current
        if rec is None or self._small is None or self._positive_full is None:
            return
        if self._orig_held or self.btn_hold_orig.isDown():
            self._show_analysis_image()
            if fit:
                self.view.fit()
            return
        rendered = self._rendered_full()
        tr = OrientTransform(self.t_source.shape[:2],
                             rec.geometry.rotation_degrees,
                             rec.geometry.mirror_horizontal,
                             rec.geometry.fine_rotation_degrees)
        s = self._small_scale
        out = rec.geometry.output_crop_roi
        origin = (0, 0)
        # v6.1 §4：显示层按当前帧状态决定是否裁切——勾选「保留片边」时
        # 导出是整幅，显示也必须是整幅（不得只显示输出裁切内）
        if out is not None and self.current_page() == 1 \
                and not rec.keep_rebate:
            o = tr.bbox_source_to_oriented(tuple(out))
            x0, y0 = max(0, int(round(o[1] / s))), max(0, int(round(o[0] / s)))
            x1 = min(self._positive_full.shape[1], int(round(o[3] / s)))
            y1 = min(self._positive_full.shape[0], int(round(o[2] / s)))
            if x1 - x0 > 8 and y1 - y0 > 8:
                rendered = rendered[y0:y1, x0:x1]
                origin = (x0, y0)
        # v6.2 F5：显示图可能是整幅技术正片的子矩形，原点偏移必须显式记录。
        # 一切画布采样（中性点拾点等）先加偏移再取 _positive_full；
        # 旧注释"场景坐标 == 正片坐标，零换算"只在裁切为整幅/保留片边时成立。
        self._view_origin = origin
        self._view_space = ("positive_full_cropped" if origin != (0, 0)
                            else "positive_full")
        self.view.set_image(rendered)
        if fit:
            self.view.fit()

    def _update_draft_from_view_silent(self):
        """拖动中的草稿更新：**只写内存**，不落盘。

        2026-09-02 实测教训：此前每次（30fps）都原子写盘，磁盘 fsync
        拖垮主线程——裁切拖动卡顿的根因。落盘交给 _flush_draft（停顿
        600ms 后）或松手时立即执行。
        """
        rec = self.current
        if rec is None or self.t_source is None:
            return
        if getattr(self, "_canvas_rec", None) is _CANVAS_STALE:
            # 冷读审查（2026-09-08）：静默 return 会把用户刚拖的框无声吃掉，
            # 正是本次整改一直在反制的 A 类缺陷。拒绝可以，必须说。
            self._set_banner("本帧仍在加载（或加载失败），画布还是上一帧的像素，"
                             "此刻框选不会写入本帧：请等加载完成或重选该帧", "warning")
            return
        rec.geometry = self._geometry_for_service()
        self._draft_touched = True
        self._draft_flush_timer.start()

    def _flush_draft(self):
        """把内存中的草稿写盘（防抖落地）。"""
        rec = self.current
        if rec is None or not self._draft_touched:
            return
        self._save_draft_annotation(rec, quiet=True)

    def _analysis_image(self) -> np.ndarray | None:
        """唯一的「分析阶段显示图」：small 预览 + 当前 FrameRecord 的
        OrientTransform（v6.1 §2.1）。

        rotation/mirror 非 0 时底图与 ROI 必须在同一坐标系（ORIENTED），
        否则会出现底图与框错位；90/270 度宽高交换由 OrientTransform 的
        图像变换与 oriented_size 共同保证，不硬编码修补。
        """
        if self._small is None:
            return None
        rec = self.current
        if rec is None:
            return self._small
        g = rec.geometry
        if (not g.rotation_degrees and not g.mirror_horizontal
                and abs(float(g.fine_rotation_degrees)) < 1e-9):
            return self._small
        src_hw = (_shape_hw(self.t_source) if self.t_source is not None
                  else _shape_hw(self._small))
        tr = OrientTransform(src_hw, g.rotation_degrees,
                             g.mirror_horizontal, g.fine_rotation_degrees)
        return tr.apply(self._small)

    def _show_analysis_image(self, *, fit: bool = False):
        """分析阶段/看原片统一上屏入口：底图与 ROI 坐标系一致。"""
        img = self._analysis_image()
        if img is None:
            return
        self.view.set_image(img)
        if fit:
            self.view.fit()

    def _show_frame(self):
        """把草稿 ROI 上屏（正片显示由 _compose_view 负责）。"""
        if self.current is None or self._small is None:
            return
        rec = self.current
        self._sync_filmstrip(rec)
        s = self._small_scale
        tr = OrientTransform(self.t_source.shape[:2],
                             rec.geometry.rotation_degrees,
                             rec.geometry.mirror_horizontal,
                             rec.geometry.fine_rotation_degrees)
        view = self.view
        draw_kind = view.draw_mode()
        # 必须先把图像放进场景（场景尺寸就绪），再放 ROI——
        # 顺序反了会让 add_roi 在 1x1 场景里钳制出负坐标框（实测教训）
        self._show_analysis_image()
        view.clear_rois()

        def put(kind, roi_src, multiple=False):
            if not roi_src:
                return
            for r in (roi_src if multiple else [roi_src]):
                # SOURCE 全分辨率 -> oriented -> small 预览坐标
                o = tr.bbox_source_to_oriented(tuple(r))
                view.add_roi(kind, _qrect(round(o[1] / s), round(o[0] / s),
                                          max(1, round((o[3] - o[1]) / s)),
                                          max(1, round((o[2] - o[0]) / s))))

        rss = rec.geometry
        put(RoiKind.OUTPUT_CROP, rss.output_crop_roi)
        put(RoiKind.IMAGE_AREA, rss.image_area_roi)
        put(RoiKind.BASE, rss.base_sample_rois, multiple=True)
        view.set_draw_mode(draw_kind)
        self._set_status(_frame_status_text(rec, self.pr))

    # ------------------------------------------------- 高清放大（按需）
    def _on_zoom_changed(self, zoom: float):
        """缩放变化：只更新缩放状态。

        v6.1 §3：后台高清替换默认关闭（ENABLE_HD_PREVIEW=False）——
        在实现「与 small 坐标同尺寸的高清 tile + ROI/滚动同步映射」之前，
        全分辨率整帧替换会显示错误内容；缩放、平移与 100% 检查保留。
        """
        if self.current is None or self.t_source is None:
            return
        self.lbl_zoom.setText(f'{zoom * 100:.0f}%')
        self.lbl_zoom.setVisible(True)
        if not ENABLE_HD_PREVIEW:
            return
        # ---- 以下为高清替换路径（默认不启用；重新启用前必须满足 §3
        # 全部要求：同尺寸 tile、generation 防竞态、worker 解包契约）----
        if zoom > 1.01 and not self._hd_tile_pending:
            self._hd_tile_pending = True
            self._hd_generation += 1
            gen = self._hd_generation
            from .workers import run_in_thread
            rec = self.current
            tr = OrientTransform(self.t_source.shape[:2],
                                 rec.geometry.rotation_degrees,
                                 rec.geometry.mirror_horizontal,
                                 rec.geometry.fine_rotation_degrees)
            run_in_thread(
                lambda cb: self._hd_tile_job(rec, tr),
                on_done=lambda r: self._hd_tile_done(r, rec, zoom, gen))
        elif zoom <= 1.01:
            self._hd_tile_pending = False
            self._hd_generation += 1        # 使在途任务过期
            self._show_frame()

    def _hd_tile_job(self, rec: FrameRecord, tr):
        """后台高清帧（仅在 ENABLE_HD_PREVIEW=True 时使用）。

        v6.1 §3 调整阶段硬要求：必须调用同一 process_image(loaded.rgb,
        geometry=..., keep_rebate=True) 取**正确正片**后再编码；
        不得用 tr.apply(loaded.rgb) 把原始负片当正片上屏。
        """
        loaded = self._load(rec)
        if self._stage == 1:
            pos = self._positive_full
            if pos is not None and pos.shape[:2] == loaded.rgb.shape[:2]:
                arr = pos
            else:
                pr = process_image(
                    loaded.rgb, geometry=rec.geometry,
                    render_intent=self._intent, candidate=self._candidate,
                    keep_rebate=True, film_type="unknown")
                arr = pr.result.positive_linear
        else:
            arr = tr.apply(loaded.rgb)
        from ..pipeline.density import encode_srgb
        return encode_srgb(np.clip(arr, 0, None))

    def _hd_tile_done(self, result, rec, zoom, generation: int = 0):
        """worker 契约统一解包（v6.1 §3）：失败显示可读错误并保留小图；
        成功只把 result["value"] 交给 set_image()；过期任务（帧已切换、
        已缩回、或更新一代任务已发出）一律丢弃，不覆盖新状态。"""
        self._hd_tile_pending = False
        if generation and generation != self._hd_generation:
            return                          # 旧任务：不得覆盖新状态
        if not result.get("ok"):
            self._set_banner(f"高清预览失败：{result.get('error')}", "warning")
            return
        arr = result.get("value")
        if rec is not self.current or arr is None:
            return
        # 只有仍处于放大态才替换（避免缩回后覆盖小图）
        if self.view.current_zoom() > 1.01:
            self.view.set_image(arr)

    # ------------------------------------------------------- 编辑事件
    def _on_roi_edited(self, kind, index=0, _rect=None):
        """画布释放鼠标（Stage 模型，对齐竞品）：

        - 分析阶段（负片）：更新草稿 + 分析区状态 + 引导文案；
          片基变化 -> 后台重跑管线（唯一慢路径，toast 提示）；
        - 调整阶段：即时合成（毫秒级）。
        """
        # v6.2 F2：凡改变**统计范围**的编辑都必须让分析失效。
        # 片基取样区和有效画面区都决定 Dmin/Dmax 与色罩估计；
        # 旧实现只重跑片基，改完有效画面还挂着按旧范围算出的分析结果。
        # 输出裁切只改构图、不改统计 -> 保持零管线调用的合成快路径。
        needs_rerun = self._changes_statistics(kind)
        self._update_draft_from_view(rerun=needs_rerun)
        self._update_area_states()
        self._update_guidance()
        if needs_rerun:
            self._show_toast('正在按新的统计范围重新分析…')
        elif self._stage == 1:
            self._compose_view()

    def _show_toast(self, text: str):
        """居中非模态处理提示（竞品式）。"""
        self.toast.setText(text)
        self.toast.setVisible(True)

    def _hide_toast(self):
        self.toast.setVisible(False)

    def _on_roi_live_changed(self):
        """拖动中（~30fps）：草稿静默更新 + 遮罩即时跟随，零管线调用。"""
        self._update_draft_from_view_silent()
        self.view._overlay.update()

    def _on_draw_rejected(self, text: str):
        """绘制被拒：用引导 pill 突出显示（状态行太隐蔽，用户看不到）。"""
        self._set_banner(text, 'warning')
        self.pill.setText(text)
        self.pill.setStyleSheet(f'color: {theme.COLORS['danger']};')
        QTimer.singleShot(2500, lambda: (
            self.pill.setStyleSheet(''),
            self._update_guidance()))

    @staticmethod
    def _changes_statistics(kind) -> bool:
        """该 ROI 种类的增删改是否会改变统计范围（Dmin/Dmax/色罩来源）。"""
        return kind in (RoiKind.BASE, RoiKind.IMAGE_AREA)

    def _on_roi_selected(self, kind, index=0):
        pass          # 状态引导由 _update_area_states 负责

    def _on_roi_deleted(self, kind, index=0):
        label = KIND_LABELS.get(kind, str(kind))
        # 审查修复（GUI 审查 重要-4）：F2 的规则是「凡改变统计范围都要重算」，
        # 但删除/清空路径仍按旧判据 -> 删掉有效画面区后分析结果不作废，
        # 恰好复刻 F2 要修的病。
        self._update_draft_from_view(rerun=self._changes_statistics(kind))
        self._update_area_states()
        self._update_guidance()
        if kind is not RoiKind.BASE:
            self._set_banner(f"已删除{label}：状态 = 需要裁切复核"
                             "（不会自动用其他框代替）", "warning")
        else:
            self._set_banner(f"已删除{label} {index + 1}", "info")

    def _update_draft_from_view(self, *, rerun: bool = False):
        """草稿 <- 画布（松手路径：立即落盘）。rerun=True 才重跑管线。"""
        rec = self.current
        if rec is None or self.t_source is None:
            return
        if getattr(self, "_canvas_rec", None) is _CANVAS_STALE:
            # 加载期间画布仍是上一帧像素，反算出的几何属于旧帧；但**不能静默**
            # 丢掉用户刚松手的这次编辑（冷读审查 L1-①）。
            self._set_banner("本帧仍在加载（或加载失败），刚完成的框选未写入："
                             "请稍后重试这一步", "warning")
            return
        rec.geometry = self._geometry_for_service()
        self._draft_touched = True
        if not rec.geometry.is_complete():
            rec.status = FrameStatus.NEEDS_CROP_REVIEW.value
        self._draft_flush_timer.stop()      # 松手即写，取消防抖定时
        self._save_draft_annotation(rec, quiet=True)
        self._set_status(_frame_status_text(rec, self.pr))
        if rerun:
            self._refresh_preview()

    def _save_draft_annotation(self, rec: FrameRecord, *, quiet: bool = False):
        """编辑后把草稿写入 **outputs/drafts/**（绝不写真实标注目录）。

        教训（2026-09-02）：自动种子草稿曾覆盖 fixtures/annotations 里的
        历史标注——真实标注目录只接受用户显式“确认并保存标注”
        （verified_by=user）。草稿目录在 outputs/ 下，不入 Git。
        """
        try:
            geo = rec.geometry
            from ..calibration.annotations import ann_path_for
            from ..io.atomic import atomic_write_bytes
            ann = Annotation(
                file=rec.source_path, roll=rec.roll_id or "misc",
                rotation_degrees=geo.rotation_degrees,
                fine_rotation_degrees=geo.fine_rotation_degrees,
                mirror_horizontal=geo.mirror_horizontal,
                keep_rebate=bool(rec.keep_rebate),   # v6.1 §4 状态随草稿持久化
                output_crop_roi=geo.output_crop_roi,
                image_area_roi=geo.image_area_roi,
                base_sample_rois=list(geo.base_sample_rois),
                source_sha256=rec.source_sha256,
                status="draft", verified=False,
                # v6.2 §5.2：草稿必须带来源身份——派生自哪一版确认基线 + 第几次修订。
                # 没有它，草稿与确认标注并存时无法判断先后，只能判为冲突。
                base_annotation_sha=str(getattr(self, "_state_fingerprint", "") or ""),
                draft_revision=int(getattr(self, "_draft_revision", 0)) + 1,
                notes="GUI 草稿（v6 §6.2：每次编辑保存）")
            payload = json.dumps(ann.to_dict(), ensure_ascii=False, indent=2)
            target = _drafts_dir() / ann_path_for(rec.source_path).name
            atomic_write_bytes(target, payload.encode("utf-8"), overwrite=True)
            self._draft_file = target
            self._draft_revision = int(getattr(self, "_draft_revision", 0)) + 1
        except Exception as e:  # noqa: BLE001 —— 保存失败可见但不阻断
            self._set_banner(f"草稿保存失败：{type(e).__name__}: {e}", "danger")

    def _reset_all_geometry(self):
        self.view.clear_rois()
        self._update_draft_from_view()
        self._set_banner("已清空全部几何：状态 = 需要裁切复核", "warning")

    def _auto_crop(self):
        """重新检测：清空用户草稿标记后按自动结果重种（用户显式动作）。"""
        if self.current is None:
            return
        self.current.geometry = GeometryRegions(
            rotation_degrees=self.current.geometry.rotation_degrees,
            fine_rotation_degrees=self.current.geometry.fine_rotation_degrees,
            mirror_horizontal=self.current.geometry.mirror_horizontal)
        self._draft_touched = False
        self._refresh_preview(seed_if_untouched=True)
        self._show_frame()
        # v6.2 F3：此前调用不存在的方法，"重新检测"按钮一点就 AttributeError
        self._set_banner("已按自动检测重新生成三类区域（可继续调整）", "info")

    def _rotate(self, delta: int):
        if self.current is None:
            return
        g = self.current.geometry
        g.rotation_degrees = (g.rotation_degrees + delta) % 360
        g.output_crop_roi = g.image_area_roi = None
        g.base_sample_rois = []
        self._draft_touched = False
        self._refresh_preview(seed_if_untouched=True)
        self._show_frame()
        self._set_banner("方向已改变：三类区域重新检测，请确认", "info")

    def _mirror(self):
        if self.current is None:
            return
        self.current.geometry.mirror_horizontal = \
            not self.current.geometry.mirror_horizontal
        self._refresh_preview(seed_if_untouched=True)
        self._show_frame()
        self._update_draft_from_view()   # 审查修复（GUI 重要-5）：镜像此前只活在内存，
        self._set_banner("镜像已切换并写入草稿（重启可续）", "info")   #   几何完整时不落盘

    def _step(self, direction: int):
        paths = self.library.current_frames()
        if not paths:
            return
        try:
            i = paths.index(self.current.source_path) if self.current else 0
        except ValueError:
            i = 0
        self.load_frame(paths[max(0, min(len(paths) - 1, i + direction))])

    def _choose_candidate(self, name: str):
        self._candidate = name
        if self.current is not None:
            self.current.chosen_candidate = name
        if self.pr is not None and self.pr.candidates is not None:
            cand = self.pr.candidates.get(name)
            self._intent = RenderIntent.from_dict(cand.intent)
            # v6.2 F3：v6 单工作区已没有 convert_panel，滑杆在右栏
            self.right.panel.set_intent(self._intent, silent=True)
        self._compose_view()

    def _intent_changed(self, field: str, value: float):
        """微调滑杆：即时合成（RenderIntent 应用在缓存的正片上，毫秒级），
        不重跑管线；技术正片按契约不变。"""
        setattr(self._intent, field, float(value))
        self._compose_view()

    def _intent_committed(self, field: str, value: float):
        setattr(self._intent, field, float(value))
        if self.current is not None:
            self.current.render_intent_override = self._intent.to_dict()
        self._compose_view()

    def _reset_all_intent(self):
        """「全部重置」：只清 RenderIntent（风格契约所在层）。

        候选与技术正片一概不动：reset_all() 清的是十个微调控件。
        此前这条链断了 —— 按钮 emit 出去没有任何接收者。
        """
        self._intent.reset_all()
        if self.current is not None:
            self.current.render_intent_override = self._intent.to_dict()
        self.right.panel.set_intent(self._intent, silent=True)
        self._compose_view()
        self._set_banner("已重置全部微调项（技术正片与候选未变）", "info")

    # ------------------------------------------------ R5b 中性点救援
    def _on_pick_neutral(self, on: bool):
        """拾点救援开关（调整阶段）：进入/退出 view 的 neutral 拾点模式。"""
        # 先判数据再切阶段：此前顺序反了，空文档也会被打到「调整」页
        if on and self.current is None:
            self.right.btn_pick_neutral.setChecked(False)
            self.right.set_pick_hint("还没有帧可拾点：请先导入并选择一帧")
            return
        if on and self._stage != 1:
            self._set_stage(1)
        self.view.set_pick_mode("neutral" if on else None)
        if on:
            self._set_banner("中性点救援：在画面上点击应该呈现中性灰的位置"
                             "（Esc 退出）", "info")

    def _on_point_picked(self, kind: str, x: float, y: float):
        """画布拾点：中性点 -> 精确救援 -> 即时合成 + 撤销栈。"""
        if kind != "neutral":
            return
        rec = self.current
        if rec is None or self._positive_full is None:
            return
        # 审查修复（GUI 审查 重要-6）：帧加载期间画布仍显示上一帧，而
        # self.current 已指向新帧 —— 此时拾点会把「上一帧像素解出的 intent」
        # 写进新帧记录。校验画布数据归属，不一致直接拒绝。
        if getattr(self, "_canvas_rec", None) is _CANVAS_STALE:
            self.right.set_pick_hint("帧数据尚未就绪（正在加载/刚切换），请稍候再拾点")
            return
        # v6.2 §7.6 / 审查 F9：拾点求解取的是**技术正片**像素，只能发生在
        # 调整阶段；分析阶段中央显示的是负片，此时点击必须拒绝而不是按正片求解。
        if self._stage != 1:
            self.right.set_pick_hint("中性点救援只在「调整」阶段可用")
            return
        if self._orig_held or self.btn_hold_orig.isDown():
            # 冷读审查 L4-①：「按住看原片」时 _compose_view 提前 return，既不刷新
            # _view_origin 也不换采样源：此刻底图是负片，采样却按正片 + 上一次
            # 裁切偏移取像素，解出来的调色与所见画面毫无关系。
            self.right.set_pick_hint("请先松开「按住看原片」再拾点")
            return
        from ..pipeline.rescue import neutral_pick_intent_exact

        # v6.2 F5（修正 v6.1 的"零换算"断言）：调整阶段且未勾选保留片边时，
        # 画布显示的是整幅技术正片的整体裁切子矩形，_compose_view 已把该子矩形
        # 原点记入 _view_origin；画布坐标必须先加偏移才是 _positive_full 索引。
        # 仅当显示整幅（裁切为空 / 勾选保留片边）时偏移为 (0, 0)。
        ox, oy = self._view_origin
        x_full, y_full = float(x) + ox, float(y) + oy
        H, W = self._positive_full.shape[:2]
        if not (np.isfinite(x_full) and np.isfinite(y_full)) \
                or not (0 <= int(round(x_full)) < W
                        and 0 <= int(round(y_full)) < H):
            self.right.set_pick_hint("拾点失败：请点在画面内")
            self._set_banner("拾点位置在画面外，未应用救援", "warning")
            return
        px, py = int(round(x_full)), int(round(y_full))
        try:
            intent, warns = neutral_pick_intent_exact(
                self._positive_full, py, px, base_intent=self._intent,
                return_warnings=True)
        except ValueError as e:
            self.right.set_pick_hint(f"拾点失败：{e}")
            return
        self._pick_undo_stack.append(self._intent.copy())
        self._intent = intent
        if rec is not None:
            rec.render_intent_override = intent.to_dict()
        self.right.panel.set_intent(intent, silent=True)
        self._compose_view()
        self.right.set_undo_pick_enabled(True)
        gain_note = (f"temp {intent.temperature:+.1f} / tint {intent.tint:+.1f}")
        if warns:
            self.right.set_pick_hint("⚠ " + "；".join(warns[:2]) +
                                     f"（{gain_note}）")
            self._set_banner("中性点救援已应用，但拾点可信度存疑（见右栏提示）",
                             "warning")
        else:
            self.right.set_pick_hint(f"已把该点拉向中性灰（{gain_note}）。"
                                     "不满意可再点别处或「撤销拾点」。")
            self._set_banner("中性点救援已应用", "success")

    def _undo_pick_rescue(self):
        """撤销上一次拾点救援（恢复栈里的上一个 intent）。

        栈只在 _frame_done 成功路径清空；加载失败时会留着上一帧的 intent，
        此时撤销会把 A 帧的调色写进 B 帧（冷读审查 L4-②）。
        """
        if not self._pick_undo_stack:
            return
        if getattr(self, "_canvas_rec", None) is _CANVAS_STALE:
            self._set_banner("本帧尚未加载完成，撤销拾点暂不可用（栈里是上一帧的结果）",
                             "warning")
            return
        prev = self._pick_undo_stack.pop()
        self._intent = prev
        if self.current is not None:
            self.current.render_intent_override = prev.to_dict()
        self.right.panel.set_intent(prev, silent=True)
        self._compose_view()
        if not self._pick_undo_stack:
            self.right.set_undo_pick_enabled(False)
            self.right.set_pick_hint("")
        self._set_banner("已撤销上一次拾点救援", "info")

    # ---------------------------------------------------------------- 提示
    def closeEvent(self, ev):
        """关窗前把防抖窗口里的最后一次几何编辑落盘（冷读审查 L2-③）。

        拖动 ROI 只改内存并起 600ms 单次定时器；在那段窗口内直接关窗，
        那次编辑就永久丢失。落盘失败只提示，不阻塞退出。
        """
        try:
            if self._draft_flush_timer.isActive():
                self._draft_flush_timer.stop()
                self._flush_draft()
        except Exception as e:  # noqa: BLE001 —— 退出不得被保存卡住
            self._set_banner(f"退出前草稿落盘失败：{type(e).__name__}: {e}",
                             "danger")
        super().closeEvent(ev)

    def _set_status(self, text: str):
        self.status.setText(text)

    def clear_banner(self):
        self._set_banner("", "info")

    def _set_banner(self, text: str, level: str = "info"):
        """底部状态行右侧的轻量 inline 提示（§5.6.7），非模态、单行。"""
        self._banner_text = text if level != "info" else ""
        color = {"info": theme.COLORS["text_dim"],
                 "warning": theme.COLORS["warning"],
                 "danger": theme.COLORS["danger"],
                 "success": theme.COLORS["success"]}.get(level,
                                                         theme.COLORS["text_dim"])
        self.lbl_banner.setText(text)
        self.lbl_banner.setStyleSheet(f"color: {color};")
        self.lbl_banner.setVisible(bool(text))

    # ---------------------------------------------------------------- 同步

    def _sync_intent(self, fields: list[str]):
        # v6.2 F3：作用域下拉在右栏微调面板里（convert_panel 已随 v6 移除）
        scope = self.right.panel.combo_sync_scope.currentText()
        targets = self._targets_for(scope)
        for rec in targets:
            base = RenderIntent.from_dict(rec.render_intent_override or {})
            base.merge(self._intent, fields)
            rec.render_intent_override = base.to_dict()
        roll = (self.session.roll(self.current.roll_id)
                if self.current and self.current.roll_id else None)
        if roll is not None and scope == "整卷":
            roll.shared_render_intent = self._intent.to_dict()
        self._set_banner(f"微调项 {', '.join(fields)} 已同步到 {len(targets)} 帧",
                         "success")

    def _targets_for(self, scope: str) -> list[FrameRecord]:
        if self.session is None:
            return []
        if scope == "当前帧":
            return [self.current] if self.current else []
        if scope == "选中帧":
            return [f for f in (self.session.frame(p)
                                for p in self.library.selected_frames()) if f]
        return self._frames_of(self.library.current_roll or "")

    # ---------------------------------------------------------------- 导出
    def _on_export_button(self):
        """同一按钮承担「开始 / 取消」两态。

        导出改到后台线程后必须给出取消入口，否则用户只能干等；
        不加新控件是因为右栏/工具条布局刚定稿，不值得为它改版式。
        """
        if getattr(self, "_export_running", False):
            self.cancel_export()
            return
        self.export_roll()

    def export_roll(self):
        if self.session is None:
            return
        rid = self.library.current_roll
        frames = self._frames_of(rid or "")
        if not frames:
            QMessageBox.information(self, "整卷导出", "当前卷没有帧")
            return
        if getattr(self, "_export_running", False):
            self._set_banner("上一次整卷导出仍在进行，请先取消或等待其完成",
                             "warning")
            return
        out = QFileDialog.getExistingDirectory(self, "选择导出目录")
        if not out:
            return
        # 审查修复（2026-09-08）：整卷导出 = 每帧全管线 + 16bit TIFF + JPEG +
        # 写后重读校验，是分钟级重活。此前直接跑在 GUI 线程上，违反
        # ui/workers.py 自己立的契约「一切 >200ms 的工作都必须在 QThread 中
        # 执行」：窗口「未响应」数分钟；progress 回调因事件循环被阻塞而根本
        # 不重绘；也没有取消入口，导致 roll_export 新增的 cancelled 计数在
        # 主流程里永远不可达。现在走 run_in_thread + cancel 事件。
        from .workers import run_in_thread
        self._export_cancel = threading.Event()
        self._export_running = True
        self.btn_export.setText("取消导出")
        self._set_banner(f"整卷导出开始（{len(frames)} 帧）…", "info")
        run_in_thread(
            lambda cb: self._export_job(frames, Path(out), rid, cb),
            on_done=self._on_export_done,
            on_progress=lambda m: self._set_banner(str(m), "info"))

    def _export_job(self, frames, outdir, roll_id, progress):
        """后台线程执行整卷导出：只算不碰控件（结果经信号回主线程）。"""
        from ..pipeline.roll_export import export_roll_frames
        return export_roll_frames(
            frames, outdir, session=self.session, roll_id=roll_id,
            intent_resolver=lambda rec: RenderIntent.from_dict(
                rec.render_intent_override or self._intent.to_dict()),
            tiff_assumption=(self.session.batch_encoding_assumption or {}).get(
                "encoding"),
            cancel=self._export_cancel, progress=progress)

    def cancel_export(self):
        """取消整卷导出：当前帧完成后停止（已写出的成品不回滚）。"""
        if getattr(self, "_export_running", False):
            self._export_cancel.set()
            self._set_banner("取消中：当前帧完成后停止…", "warning")

    def _on_export_done(self, res):
        """导出结束（主线程）：如实报告四类计数。

        审查修复：此前标题恒为「整卷导出完成」且不报 cancelled —— 取消掉
        半卷仍被说成成功，正是 v6.2 §6.6 想消灭的静默丢帧，只是修在数据层。
        """
        self._export_running = False
        self.btn_export.setText("整卷导出…")
        ok = isinstance(res, dict) and bool(res.get("ok"))
        if not ok:
            err = res.get("error") if isinstance(res, dict) else str(res)
            QMessageBox.critical(self, "导出失败（已回滚）",
                                 f"{err}\n源文件未改动")
            self._set_banner("整卷导出失败（已回滚，源文件未改动）", "danger")
            return
        rep = res.get("value") or {}
        cancelled = int(rep.get("cancelled") or 0)
        failed = int(rep.get("failed") or 0)
        line = (f"成功 {int(rep.get('ok') or 0)} / "
                f"待复核 {int(rep.get('needs_review') or 0)} / "
                f"失败 {failed}")
        if cancelled:
            line += f" / 已取消 {cancelled}"
        detail = f"报告：{rep.get('report_path', '')}"
        if rep.get("summary_complete") is False:
            detail += "（计数不完整，请查看报告核对逐帧结果）"
        title = ("整卷导出已取消" if cancelled
                 else "整卷导出结束（有失败帧）" if failed
                 else "整卷导出完成")
        if cancelled or failed:
            QMessageBox.warning(self, title, f"{line}\n{detail}")
        else:
            QMessageBox.information(self, title, f"{line}\n{detail}")
        self._set_banner(f"{title}：{line}",
                         "warning" if (cancelled or failed) else "success")

    # ------------------------------------------------------- 保存区域
    def save_regions_as_annotation(self):
        """用户显式确认：三类区域保存为**用户签字**标注（§6.2）。"""
        if self.current is None or not self.current.geometry.is_complete():
            self._set_banner("三类区域不完整，无法保存为确认标注", "warning")
            return
        rec = self.current
        # 审查修复（我的 P0）：该帧原标注读取失败时，保存会**覆盖掉可恢复的**
        # 损坏文件（截断 JSON 常能人工救回）。显式拒绝，指引人工备份。
        if "读取失败" in str(self._state_resolution.get("reason") or ""):
            self._set_banner("该帧原有标注/草稿读取失败（已损坏），为避免覆盖可恢复"
                             "数据已拒绝保存：请先备份 fixtures/annotations 下对应"
                             "文件，再重新标定", "danger")
            return
        geo = self._geometry_for_service()
        ann = Annotation(
            file=rec.source_path, roll=rec.roll_id or "misc",
            rotation_degrees=geo.rotation_degrees,
            fine_rotation_degrees=geo.fine_rotation_degrees,
            mirror_horizontal=geo.mirror_horizontal,
            keep_rebate=bool(rec.keep_rebate),       # v6.1 §4 随确认标注持久化
            output_crop_roi=geo.output_crop_roi,
            image_area_roi=geo.image_area_roi,
            base_sample_rois=list(geo.base_sample_rois),
            source_sha256=rec.source_sha256,
            verified=True, verified_by="user", notes="GUI 用户框选确认")
        try:
            p = save_annotation(rec.source_path, ann)
        except ValueError as e:
            self._set_banner(f"标注保存失败：{e}", "danger")
            return
        self._ann = ann
        # 审查修复（GUI 审查 重要-2）：确认动作**改变血缘**，必须立刻刷新本帧
        # 的基线指纹与草稿修订号。否则其后每次编辑写盘带的都是旧/空的
        # base_annotation_sha -> 重启必判 conflict，确认之后的微调被搁置，
        # §5.2 的指纹机制等于自废。
        from ..calibration.state import content_fingerprint
        self._state_fingerprint = content_fingerprint(ann)
        self._draft_revision = 0
        self._state_resolution = {
            "source": "confirmed",
            "reason": "确认基线生效（本会话内刚签字）",
            "confirmed_fingerprint": self._state_fingerprint,
            "draft_revision": 0}
        self._state_notice = ""
        # 冷读审查 L3：刚签字就把草稿血缘同步到新基线。否则磁盘上的草稿仍写着
        # 旧 base_annotation_sha，重启必判 conflict、黄条说"草稿基于旧确认基线"，
        # 而它的内容其实已经等于新基线 —— 假冲突会诱导用户重做标定。
        # 只刷新血缘、不删草稿（"绝不删用户工作稿"是本模块的既定原则）。
        self._save_draft_annotation(rec, quiet=True)
        self._set_banner(f"三类区域已保存为用户确认标注：{p.name}", "success")


# ------------------------------------------------------------------ 工具
def _abs(path: str, rec: FrameRecord | None = None) -> Path:
    """记录的 abs_path 优先（导入根不一定是仓库根）。"""
    if rec is not None:
        abs_path = (rec.input_facts or {}).get("abs_path")
        if abs_path and Path(abs_path).exists():
            return Path(abs_path)
    p = Path(path)
    if p.is_absolute():
        return p
    return Path(__file__).resolve().parents[3] / path


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


def _load_draft_annotation(rec: FrameRecord) -> Annotation | None:
    """工作草稿（经 storage 改道的 drafts/，重启可续；损坏**必须可见**）。"""
    f = _drafts_dir() / (ann_path_for(rec.source_path).name)
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


def resolve_persistent_state(rec: FrameRecord) -> ResolvedState:
    """确认基线 + 工作草稿 -> 生效状态（v6.2 §5.2；纯读盘，不写盘）。

    规则表见 calibration/state.py：草稿只有带当前确认基线指纹时才生效；
    来源不可证明或基于旧基线 -> conflict，确认基线生效、草稿原地保留。
    """
    _LAST_STATE_LOAD_ERRORS.clear()          # 逐帧重置：诊断不得串到下一帧
    resolved = resolve_state(confirmed=_load_confirmed_annotation(rec),
                             draft=_load_draft_annotation(rec),
                             source_sha256=rec.source_sha256 or "")
    if _LAST_STATE_LOAD_ERRORS:
        # 损坏 = 需要重新确认，且绝不能顺手覆盖：并进 reason 让 UI 说清楚。
        resolved.needs_reconfirm = True
        resolved.reason = ((resolved.reason + "；") if resolved.reason else "") + \
            "；".join(_LAST_STATE_LOAD_ERRORS) + \
            "。已停止自动覆盖，请先备份该文件再继续确认"
    return resolved


def _annotation_for(rec: FrameRecord) -> Annotation | None:
    """生效标注（兼容入口）：确认基线与工作草稿按 §5.2 规则决定谁生效。

    v6.1 此前是"确认标注无条件优先"，命中后草稿永远轮不到（审计 F4）。
    """
    return resolve_persistent_state(rec).annotation


def _geometry_from_annotation(ann: Annotation) -> GeometryRegions:
    return GeometryRegions(
        rotation_degrees=ann.rotation_degrees,
        fine_rotation_degrees=float(getattr(ann, "fine_rotation_degrees", 0.0) or 0.0),
        mirror_horizontal=bool(getattr(ann, "mirror_horizontal", False)),
        output_crop_roi=ann.output_crop_roi or ann.image_area_roi,
        image_area_roi=ann.image_area_roi,
        base_sample_rois=list(ann.base_sample_rois or []))


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


def main() -> int:
    import sys

    app = QApplication(sys.argv)
    app.setApplicationName("negative-lab")
    # 高分屏：跟随系统 DPI 缩放（否则 150% 下控件按 96dpi 渲染后又被
    # 系统拉伸，字大糊、布局溢出）
    app.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    theme.apply(app)
    win = MainWindow()
    win.showMaximized()          # 永不超出屏幕（§15 布局混乱根因修复）
    if len(sys.argv) > 1:
        p = Path(sys.argv[1])
        if p.is_dir():
            win.load_folder(p)
        else:
            win.load_files([p])
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
