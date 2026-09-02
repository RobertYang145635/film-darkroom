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
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication, QButtonGroup, QFileDialog, QHBoxLayout, QLabel,
    QMainWindow, QMessageBox, QPushButton, QStackedWidget, QToolButton,
    QVBoxLayout, QWidget)

from .. import __version__
from ..calibration.annotations import (Annotation, ann_path_for,
                                       load_annotation, save_annotation)
from ..io.images import ReadError, load_linear_rgb
from ..library.fingerprint import capture_fingerprint, fingerprints_compatible
from ..library.models import (FrameRecord, FrameStatus, GeometryRegions,
                              ImportSession, RollAnalysis)
from ..library.session import import_session, make_frame_record, suggest_groups_by_directory
from ..pipeline.geometry import OrientTransform
from ..pipeline.render import RenderIntent
from ..pipeline.roll_analysis import analyze_roll
from ..pipeline.service import candidate_positive, process_image

_DRAFTS_DIR = Path(__file__).resolve().parents[3] / "outputs" / "drafts"
from . import icons, theme
from .library_panel import FilmStrip, LibraryPanel
from .preview_view import PreviewView, RoiKind
from .right_panel import RightPanel
from .workers import BusyState as _Busy

MAX_PREVIEW_DIM = 1100          # 交互预览（快）；导出走全分辨率
IMPORT_PROBE = False            # 导入只取元数据+哈希（完整探针太慢，用户实测）


def _geometry_for_job(ann, small, source, geo: GeometryRegions | None = None,
                      ann_source=None) -> GeometryRegions:
    """把草稿（SOURCE 全分辨率）映射到降采样工作副本坐标。

    - geo 给定：直接按 source 形状归一化映射（编辑路径）；
    - geo 为 None：从旧标注迁移（加载路径），annotation 坐标即 SOURCE。
    """
    if geo is not None and source is not None:
        return GeometryRegions.from_normalized(
            geo.normalized(source), small.shape[:2])
    g = _geometry_from_annotation(ann) if ann is not None else GeometryRegions()
    return GeometryRegions.from_normalized(
        g.normalized(ann_source) if ann_source else g.normalized(small.shape[:2]),
        small.shape[:2])

KIND_LABELS = {RoiKind.OUTPUT_CROP: "输出裁切",
               RoiKind.IMAGE_AREA: "有效画面",
               RoiKind.BASE: "片基取样"}


class MainWindow(QMainWindow):
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
        self.view = PreviewView()
        self.view.setStyleSheet(
            f"background: {theme.COLORS['viewer']}; border: none;")
        center_lay.addWidget(self.view, 1)
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
        self._compare_on = False

        # ---- 顶部动作
        self.btn_import.clicked.connect(self._pick_folder)
        self.btn_import_file.clicked.connect(self._pick_files)
        self.btn_suggest.clicked.connect(self._suggest_groups)
        self.btn_confirm_rois.clicked.connect(self.save_regions_as_annotation)
        self.btn_export.clicked.connect(self.export_roll)
        self.library.frame_selected.connect(self.load_frame)
        self.library.grouping_changed.connect(self._on_grouping_changed)
        self.library.request_split.connect(self._on_split)

        # ---- 右栏
        rp = self.right
        rp.crop_activate.connect(self._set_crop_active)
        rp.rotate_requested.connect(self._rotate)
        rp.mirror_requested.connect(self._mirror)
        rp.auto_crop_requested.connect(self._auto_crop)
        rp.fine_changed.connect(self._set_fine_rotation)
        rp.reset_geometry.connect(self._reset_all_geometry)
        rp.show_base_toggled.connect(
            lambda on: self.view.set_kind_visible(RoiKind.BASE, on))
        rp.show_image_area_toggled.connect(
            lambda on: self.view.set_kind_visible(RoiKind.IMAGE_AREA, on))
        rp.candidate_chosen.connect(self._choose_candidate)
        rp.tone_changed.connect(self._intent_changed)
        rp.tone_committed.connect(self._intent_committed)
        rp.sync_tone_requested.connect(self._sync_intent)
        self.btn_hold_orig.pressed.connect(lambda: self._hold_original(True))
        self.btn_hold_orig.released.connect(lambda: self._hold_original(False))

        # ---- 画布
        self.view.roi_created.connect(self._on_roi_edited)
        self.view.roi_changed.connect(self._on_roi_edited)
        self.view.roi_deleted.connect(self._on_roi_deleted)
        self.view.roi_live_changed.connect(self._on_roi_live_changed)
        self.view.roi_selected.connect(self._on_roi_selected)
        self.view.replace_hint.connect(lambda t: self._set_banner(t, 'info'))
        self.view.draw_rejected.connect(
            lambda t: self._set_banner(t, 'warning'))

    # ------------------------------------------------------- 裁切叠加工具
    def page_count(self) -> int:
        return 1

    def page_title(self, index: int = 0) -> str:
        return '工作区'

    def current_page(self) -> int:
        return 0          # 单工作区：裁切叠加/转换共用同一视图

    def _set_crop_active(self, on: bool):
        """激活/退出裁切叠加（右栏小工具节按钮；Esc/完成退出）。

        激活：输出裁切框出现 + 框外压暗（即时遮罩）；输出裁切不存在时
        自动进入"画新框"状态。退出：恢复整幅正片（框外不再压暗）。
        """
        self.right.set_crop_active(on)
        if on:
            self.view.set_rois_visible(True)
            if self.view.roi(RoiKind.OUTPUT_CROP) is None:
                self._toggle_draw(RoiKind.OUTPUT_CROP)
            self.view._crop_dim = 0.55
        else:
            self.view.set_draw_mode(None)
            self.view._crop_dim = 0.0
        self.view._overlay.update()

    def _toggle_draw(self, kind: RoiKind):
        self.view.set_draw_mode(
            None if self.view.draw_mode() is kind else kind)

    def _set_fine_rotation(self, deg: float):
        if self.current is None:
            return
        self.current.geometry.fine_rotation_degrees = float(deg)
        self._update_draft_from_view()
        self._compose_view(fit=True)

    def _hold_original(self, on: bool):
        """按住看原片（负片）；松开回到合成正片。"""
        self._orig_held = bool(on)
        if self._small is None:
            return
        if on:
            self.view.set_image(self._small)
        else:
            self._compose_view()

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
        from .workers import BusyState
        self._busy_leave("import")
        if not result.get("ok"):
            self._set_banner(f"导入失败：{result.get('error')}", "danger")
            return
        self._adopt_session(result["value"])

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
            self.library.select_roll(sess.roll_groups[0].id)
        self._set_banner(f"导入 {len(sess.files)} 个文件；请整理成卷后标定区域",
                         "info")
        return sess

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
            self._set_banner("上一帧仍在处理中…（可继续查看/框选，稍候自动更新）",
                             "warning")
            return
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
        ann = _annotation_for(rec)
        roll = self._analyze_roll_frames()
        cb("生成预览…")
        k = max(1, int(np.ceil(max(loaded.rgb.shape[:2]) / MAX_PREVIEW_DIM)))
        small = loaded.rgb[::k, ::k, :].copy()
        # 只有**完整**的用户草稿/标注才传给 service；空标注传空几何会经
        # 归一化映射钳制出荒谬 ROI（实测：60px 横带被当 image_area，
        # 预览被裁成一条线）。不完整时交给 service 自动检测再种回草稿。
        seed_geo = rec.geometry if rec.geometry.is_complete() else None
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
                "preview": pr, "small": small, "scale": k}

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
        self.roll_analysis = d["roll"]
        rec = d["rec"]
        rec.capture_fingerprint = d["fingerprint"].value
        rec.input_facts = {**dict(d["meta"]), "kind": d["meta"].get("kind", "")}
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
        # Lightroom 语义：进来看到的就是**处理后的正片**（原负片用
        # "按住看原片"查看），不是负片+框
        self._compose_view(fit=True)
        sh, sw = self.t_source.shape[:2]
        self._set_banner(f"{Path(rec.source_path).name} | {sw}x{sh} | "
                         f"片基来源 {self.pr.base_source}", "info")

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
        if conflicts:
            self.library.mark_conflicts(conflicts)
            self._set_banner(
                f"本卷有 {len(conflicts)} 帧采集指纹不一致：已停止共享卷分析，"
                "可用“拆分”一键分成不同卷", "warning")
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
        return True

    def _busy_leave(self, key: str):
        self._busy.leave(key)
        if key == "preview":
            self.view.set_busy(self._busy._locks.__contains__("preview"))

    def _refresh_preview(self, *, seed_if_untouched: bool = False,
                         reason: str = "crop"):
        """管线刷新：只处理**影响技术底座**的变化。

        v6 交互模型（对齐 Lightroom，用户 2026-09-02 指示）：
        - 每帧只算一次**整幅**技术正片并缓存（keep_rebate=True）；
        - 裁切/微调 = 缓存正片上的即时合成（_compose_view，毫秒级），
          不进本函数；
        - 只有片基取样区变化（影响 Dmin）才重跑管线；
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
        keep = self.right.keep_rebate()
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
                # 技术底座已更新：刷新缓存正片并按当前模式即时合成
                self._positive_full = pr.result.positive_linear
                self._compose_view(fit=True)
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
            self.view.set_image(self._small)
            if fit:
                self.view.fit()
            return
        rendered = self._rendered_full()
        if False and \
                self.current_page() == 1:
            # 并排对比：左半原负片、右半渲染正片（同一构图）
            neg = np.clip(self._small, 0, None)
            neg = neg / max(float(neg.max()), 1e-6)
            split = rendered.shape[1] // 2
            rendered = neg.copy()
            rendered[:, :split] = self._rendered_full()[:, :split]
        tr = OrientTransform(self.t_source.shape[:2],
                             rec.geometry.rotation_degrees,
                             rec.geometry.mirror_horizontal,
                             rec.geometry.fine_rotation_degrees)
        s = self._small_scale
        out = rec.geometry.output_crop_roi
        if out is not None and self.current_page() == 1:
            o = tr.bbox_source_to_oriented(tuple(out))
            x0, y0 = max(0, int(round(o[1] / s))), max(0, int(round(o[0] / s)))
            x1 = min(self._positive_full.shape[1], int(round(o[3] / s)))
            y1 = min(self._positive_full.shape[0], int(round(o[2] / s)))
            if x1 - x0 > 8 and y1 - y0 > 8:
                rendered = rendered[y0:y1, x0:x1]
        self.view.set_image(rendered)
        if fit:
            self.view.fit()

    def _update_draft_from_view_silent(self):
        rec = self.current
        if rec is None or self.t_source is None:
            return
        rec.geometry = self._geometry_for_service()
        self._draft_touched = True
        self._save_draft_annotation(rec, quiet=True)

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
        view.set_image(self._small)
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
        self.right.set_fine_rotation(rec.geometry.fine_rotation_degrees)
        self.right.set_keep_rebate(self.right.keep_rebate())
        self._set_status(_frame_status_text(rec, self.pr))

    # ------------------------------------------------------- 编辑事件
    def _on_roi_edited(self, kind, index=0, _rect=None):
        """画布释放鼠标（v6 交互模型，对齐 Lightroom）：

        - 输出裁切 / 有效画面：即时合成——正片缓存按新构图即时呈现
          （转换模式重推渲染图；裁切模式靠框外压暗遮罩），**不重跑管线**；
        - 片基取样区：影响 Dmin 技术底座 -> 后台重跑管线（唯一慢路径）。
        """
        self._update_draft_from_view(rerun=(kind is RoiKind.BASE))
        if kind is RoiKind.BASE:
            self._set_banner("片基取样已更新：正在按新片基重新计算去色罩…",
                             "info")
        else:
            self._compose_view()
        # roi_edited 对外信号已无订阅者（快/慢路径在 MainWindow 内闭环）

    def _on_roi_live_changed(self):
        """拖动中（~30fps）：草稿静默更新 + 遮罩即时跟随，零管线调用。"""
        self._update_draft_from_view_silent()
        self.view._overlay.update()

    def _on_roi_selected(self, kind, index=0):
        self.right.sec_crop.set_active(True)

    def _on_roi_deleted(self, kind, index=0):
        label = KIND_LABELS.get(kind, str(kind))
        self._update_draft_from_view()
        if kind is not RoiKind.BASE:
            self._set_banner(f"已删除{label}：状态 = 需要裁切复核"
                             "（不会自动用其他框代替）", "warning")
        else:
            self._set_banner(f"已删除{label} {index + 1}", "info")

    def _update_draft_from_view(self, *, rerun: bool = False):
        """草稿 <- 画布。rerun=True 才重跑管线（只有片基变化需要）。"""
        rec = self.current
        if rec is None or self.t_source is None:
            return
        rec.geometry = self._geometry_for_service()
        self._draft_touched = True
        if not rec.geometry.is_complete():
            rec.status = FrameStatus.NEEDS_CROP_REVIEW.value
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
                output_crop_roi=geo.output_crop_roi,
                image_area_roi=geo.image_area_roi,
                base_sample_rois=list(geo.base_sample_rois),
                source_sha256=rec.source_sha256,
                status="draft", verified=False,
                notes="GUI 草稿（v6 §6.2：每次编辑保存）")
            payload = json.dumps(ann.to_dict(), ensure_ascii=False, indent=2)
            target = _DRAFTS_DIR / ann_path_for(rec.source_path).name
            atomic_write_bytes(target, payload.encode("utf-8"), overwrite=True)
            self._draft_file = target
        except Exception as e:  # noqa: BLE001 —— 保存失败可见但不阻断
            self._set_banner(f"草稿保存失败：{type(e).__name__}: {e}", "danger")

    def _reset_kind(self, kind: RoiKind):
        label = KIND_LABELS.get(kind, str(kind))
        self.view.clear_rois(kind)
        self._update_draft_from_view()
        if kind is not RoiKind.BASE:
            self._set_banner(f"已清空{label}：状态 = 需要裁切复核", "warning")
        else:
            self._set_banner(f"已清空全部{label}区", "info")

    def _reset_all_geometry(self):
        self.view.clear_rois()
        self._update_draft_from_view()
        self._set_banner("已清空全部几何：状态 = 需要裁切复核", "warning")

    def _delete_active_roi(self):
        if self.view.delete_active():
            self._update_draft_from_view()

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
        self._set_banner_status("已按自动检测重新生成三类区域（可继续调整）",
                                  "info")

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
            self.convert_panel.panel.set_intent(self._intent, silent=True)
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

    # ---------------------------------------------------------------- 提示
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
    def _sync_geometry(self, scope: str):
        if self.current is None or self.session is None:
            return
        targets = self._targets_for(scope)
        if not targets:
            return
        src = self._geometry_for_service()
        size = self.t_source.shape[:2]
        norm = src.normalized(size)
        for rec in targets:
            own = GeometryRegions.from_normalized(norm, _size_of(rec))
            errs = own.validate()
            if errs:
                self._set_banner(f"同步被拒绝：{errs[0]}", "danger")
                continue
            own.rotation_degrees = src.rotation_degrees
            own.mirror_horizontal = src.mirror_horizontal
            rec.geometry = own
            rec.status = FrameStatus.NEEDS_CROP_REVIEW.value
            self._save_draft_annotation(rec, quiet=True)
        self._set_banner(f"方向与三类 ROI 已同步到 {len(targets)} 帧"
                         "（按归一化相对位置，逐帧重新钳制与验证）", "success")

    def _sync_intent(self, fields: list[str]):
        scope = self.convert_panel.panel.combo_sync_scope.currentText()
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
    def export_roll(self):
        if self.session is None:
            return
        rid = self.library.current_roll
        frames = self._frames_of(rid or "")
        if not frames:
            QMessageBox.information(self, "整卷导出", "当前卷没有帧")
            return
        out = QFileDialog.getExistingDirectory(self, "选择导出目录")
        if not out:
            return
        from ..pipeline.roll_export import export_roll_frames

        try:
            report = export_roll_frames(
                frames, Path(out), session=self.session,
                roll_id=rid, intent_resolver=lambda rec: RenderIntent.from_dict(
                    rec.render_intent_override or self._intent.to_dict()),
                tiff_assumption=(self.session.batch_encoding_assumption or {}).get(
                    "encoding"),
                progress=lambda msg: self._set_banner(msg, "info"))
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "导出失败（已回滚）",
                                 f"{type(e).__name__}: {e}\n源文件未改动")
            return
        QMessageBox.information(
            self, "整卷导出完成",
            f"成功 {report['ok']} / 待复核 {report['needs_review']} / "
            f"失败 {report['failed']}\n报告：{report['report_path']}")

    def _refresh_convert_view(self, *, push: bool = False):
        """候选/意图变化后的缓存同步；图像显示统一走 _compose_view。"""
        if self.pr is not None and self.pr.candidates is not None:
            self.right.set_candidates(self.pr.candidates, self._candidate)

    # ------------------------------------------------------- 保存区域
    def save_regions_as_annotation(self):
        """用户显式确认：三类区域保存为**用户签字**标注（§6.2）。"""
        if self.current is None or not self.current.geometry.is_complete():
            self._set_banner("三类区域不完整，无法保存为确认标注", "warning")
            return
        rec = self.current
        geo = self._geometry_for_service()
        ann = Annotation(
            file=rec.source_path, roll=rec.roll_id or "misc",
            rotation_degrees=geo.rotation_degrees,
            fine_rotation_degrees=geo.fine_rotation_degrees,
            mirror_horizontal=geo.mirror_horizontal,
            output_crop_roi=geo.output_crop_roi,
            image_area_roi=geo.image_area_roi,
            base_sample_rois=list(geo.base_sample_rois),
            source_sha256=rec.source_sha256,
            verified=True, verified_by="user", notes="GUI 用户框选确认")
        try:
            p = save_annotation(rec.source_path, ann)
        except ValueError as e:
            self._set_banner(f"区域不完整：{e}", "danger")
            return
        self._ann = ann
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


def _annotation_for(rec: FrameRecord) -> Annotation | None:
    """用户确认标注优先；GUI 草稿其次（outputs/drafts/，重启可续）。"""
    try:
        a = load_annotation(rec.source_path)
        if a is not None and a.is_user_verified():
            return a
    except Exception:  # noqa: BLE001 —— 损坏标注不得阻断打开
        pass
    f = _DRAFTS_DIR / (ann_path_for(rec.source_path).name)
    try:
        if f.exists():
            return Annotation.from_dict(
                json.loads(f.read_text(encoding="utf-8")))
    except Exception:  # noqa: BLE001
        pass
    return None


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


def _status_of(pr, rec: FrameRecord) -> str:
    if pr.gate.failures:
        return FrameStatus.PROCESSING_ERROR.value
    if "needs_crop_review" in pr.roi_provenance.get("flags", []):
        return FrameStatus.NEEDS_CROP_REVIEW.value
    if pr.gate.review_reasons:
        return FrameStatus.NEEDS_REVIEW.value
    return FrameStatus.CONVERTED.value


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
