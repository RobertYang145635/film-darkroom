"""主窗口（v6 §5）：只有 裁切 与 转换 两个一级页面。

数据流（v6 §0.7/§5.2）：画布编辑 -> 草稿 `GeometryRegions`（FrameRecord）
-> 注释/会话持久化 -> 同一 `pipeline.service.process_image()` 刷新预览。
UI 不是唯一业务状态；service warning/error 不阻断画布编辑与保存（§5.2），
只给出轻量 inline 提示（§5.6.7）。

预览在降采样工作副本上调用同一 service（坐标经归一化相对位置映射），
导出走全分辨率（后续导出门禁）；UI 内不复制第二套图像算法。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import (
    QApplication, QHBoxLayout, QLabel,
    QMainWindow, QMessageBox, QPushButton, QVBoxLayout, QWidget)

from .. import __version__
from ..calibration.annotations import (Annotation, ann_path_for,  # noqa: F401 重导出（测试引用 mw.ann_path_for）
                                       load_annotation,  # noqa: F401
                                       save_annotation)
from ..calibration.state import (ResolvedState, is_confirmed,  # noqa: F401
                                 resolve_state)  # noqa: F401
from ..library.models import (FrameRecord, ImportSession, RollAnalysis)
from ..pipeline.render import RenderIntent
from ..pipeline.service import process_image  # noqa: F401 —— 测试契约：patch mw.process_image 拦截 UI 管线

from .. import storage   # 模块级：目录常量在 import 期就需要它（v6.2 审查修复）
_DRAFTS_DIR = storage.repo_path("outputs", "drafts")
from . import theme
from .library_panel import FilmStrip, LibraryPanel
from .preview_view import PreviewView, RoiKind
from .right_panel import RightPanel
from .workers import BusyState as _Busy

from .export_coordinator import ExportCoordinatorMixin
from .frame_loader import FrameLoaderMixin
from .geometry_bridge import GeometryBridgeMixin
from .intent_controller import IntentControllerMixin

# MainWindow 拆分（2026-09-08 交接 §12-(6)，只搬不改行为）：模块级状态辅助
# 与常量移到 ui/state_store.py；四个方法族移到 mixin（frame_loader /
# geometry_bridge / intent_controller / export_coordinator）。
# 这里显式重导出全部既有公开名，既有测试与调用方零改动（noqa: F401）。
from .state_store import (  # noqa: F401 —— 显式重导出（兼容旧路径）
    ENABLE_HD_PREVIEW, IMPORT_PROBE, MAX_PREVIEW_DIM, _CANVAS_STALE,
    _as_roi, _frame_status_text, _geometry_for_job,
    _geometry_from_annotation, _qrect, _shape_hw, _size_of, _unscale_geometry,
    annotation_for as _annotation_for_impl,
    resolve_persistent_state as _resolve_state_impl,
)
from .state_store import _abs as _abs  # noqa: F401

# 画布归属哨兵等已随 state_store 下沉；_DRAFTS_DIR 留在本模块：
# 测试 monkeypatch 的是 main_window._DRAFTS_DIR，_drafts_dir() 必须就地读它。


def _drafts_dir() -> Path:
    """生效草稿目录：显式改道（模块全局）> 环境变量 > 仓库默认。

    此前测试/冒烟只改标注目录、不改草稿目录，导致真实样片的 GUI 工作稿
    被测试画进去（`outputs/drafts/1298_G200__G200-1.json` 实锤）。
    现在所有写入都经此函数，测试用 `storage.isolated_storage()` 整体改道。
    """
    from .. import storage
    return storage.drafts_dir(module_global=_DRAFTS_DIR)


def resolve_persistent_state(rec: FrameRecord) -> ResolvedState:
    """（兼容包装）确认基线 + 工作草稿 -> 生效状态；实现见 ui/state_store.py。"""
    return _resolve_state_impl(rec, drafts_dir=_drafts_dir())


def _annotation_for(rec: FrameRecord) -> Annotation | None:
    """（兼容包装）生效标注；实现见 ui/state_store.annotation_for。"""
    return _annotation_for_impl(rec, drafts_dir=_drafts_dir())

class MainWindow(ExportCoordinatorMixin, IntentControllerMixin,
                 GeometryBridgeMixin, FrameLoaderMixin, QMainWindow):
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


    # --------------------------------------------------- 草稿 <-> 预览






    # ------------------------------------------------ 即时合成（零管线）







    # ------------------------------------------------- 高清放大（按需）



    # ------------------------------------------------------- 编辑事件



















    # ------------------------------------------------ R5b 中性点救援



    # ---------------------------------------------------------------- 提示
    def closeEvent(self, ev):
        """关窗前：草稿防抖落盘（冷读审查 L2-③）+ 导出进行中必须显式确认
        （冷读 B 轴，2026-09-08）。

        拖动 ROI 只改内存并起 600ms 单次定时器；在那段窗口内直接关窗，
        那次编辑就永久丢失。落盘失败只提示，不阻塞退出。
        整卷导出在 daemon 线程里跑，进程退出会直接丢弃它——此前关窗
        **静默**杀掉未完成的帧；现在给出显式选择，并在用户坚持退出时
        置 cancel 让当前帧完成后尽快收尾（事务机制保证不出错误成品）。
        """
        try:
            if self._draft_flush_timer.isActive():
                self._draft_flush_timer.stop()
                self._flush_draft()
        except Exception as e:  # noqa: BLE001 —— 退出不得被保存卡住
            self._set_banner(f"退出前草稿落盘失败：{type(e).__name__}: {e}",
                             "danger")
        if getattr(self, "_export_running", False):
            if not self._confirm_close_during_export():
                ev.ignore()
                self._set_banner("整卷导出仍在进行：先取消导出或等待完成",
                                 "warning")
                return
            self._export_cancel.set()
        super().closeEvent(ev)

    def _confirm_close_during_export(self) -> bool:
        """导出进行中关窗的确认框。返回 True = 用户坚持立即退出。"""
        box = QMessageBox(self)
        box.setWindowTitle("导出进行中")
        box.setText("整卷导出仍在后台进行。\n"
                    "现在退出将放弃未完成的帧（已写出的成品不受影响）。")
        box.addButton("继续等待导出完成", QMessageBox.AcceptRole)
        quit_btn = box.addButton("取消剩余帧并退出", QMessageBox.DestructiveRole)
        box.exec()
        return box.clickedButton() is quit_btn

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
