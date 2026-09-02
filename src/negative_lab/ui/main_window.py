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
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication, QButtonGroup, QFileDialog, QHBoxLayout, QLabel,
    QMainWindow, QMessageBox, QPushButton, QStackedWidget, QToolButton,
    QVBoxLayout, QWidget)

from .. import __version__
from ..calibration.annotations import Annotation, load_annotation, save_annotation
from ..io.images import ReadError, load_linear_rgb
from ..library.fingerprint import capture_fingerprint, fingerprints_compatible
from ..library.models import (FrameRecord, FrameStatus, GeometryRegions,
                              ImportSession, RollAnalysis)
from ..library.session import import_session, make_frame_record, suggest_groups_by_directory
from ..pipeline.geometry import OrientTransform
from ..pipeline.render import RenderIntent
from ..pipeline.roll_analysis import analyze_roll
from ..pipeline.service import candidate_positive, process_image
from . import theme
from .convert_page import ConvertPage
from .crop_page import CropPage
from .library_panel import LibraryPanel
from .preview_view import RoiKind

MAX_PREVIEW_DIM = 1400

KIND_LABELS = {RoiKind.OUTPUT_CROP: "输出裁切",
               RoiKind.IMAGE_AREA: "有效画面",
               RoiKind.BASE: "片基取样"}


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"负片裁切与去色罩 negative-lab v{__version__}")
        self.resize(1440, 900)
        self.setMinimumSize(1280, 780)
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

        self.setStyleSheet(theme.qss())
        central = QWidget()
        central.setObjectName("Root")
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(theme.SPACING["sm"], theme.SPACING["sm"],
                                theme.SPACING["sm"], 0)
        root.setSpacing(theme.SPACING["sm"])

        # ---- 顶部导航（44px，分段控件 §5.6.2）
        head = QHBoxLayout()
        head.setSpacing(theme.SPACING["sm"])
        self.btn_import = QPushButton("导入文件夹…")
        self.btn_import_file = QPushButton("导入文件…")
        self.btn_suggest = QPushButton("按目录建议分组")
        self.btn_confirm_rois = QPushButton("确认并保存标注")
        self.btn_confirm_rois.setToolTip("把当前三类区域保存为用户确认标注")
        for b in (self.btn_import, self.btn_import_file, self.btn_suggest,
                  self.btn_confirm_rois):
            head.addWidget(b)
        head.addStretch(1)
        self.btn_page_crop = QToolButton()
        self.btn_page_crop.setText("裁切")
        self.btn_page_crop.setCheckable(True)
        self.btn_page_crop.setChecked(True)
        self.btn_page_convert = QToolButton()
        self.btn_page_convert.setText("转换")
        self.btn_page_convert.setCheckable(True)
        for b in (self.btn_page_crop, self.btn_page_convert):
            b.setMinimumHeight(theme.SIZES["primary_target"])
            head.addWidget(b)
        self._page_group = QButtonGroup(self)
        self._page_group.setExclusive(True)
        self._page_group.addButton(self.btn_page_crop, 0)
        self._page_group.addButton(self.btn_page_convert, 1)
        root.addLayout(head)

        # ---- 三列主体
        body = QHBoxLayout()
        body.setSpacing(theme.SPACING["sm"])
        self.library = LibraryPanel()
        self.library.setFixedWidth(theme.SIZES["sidebar_w"])
        body.addWidget(self.library)

        self.stack = QStackedWidget()
        self.crop_page = CropPage()
        self.convert_page = ConvertPage()
        self.stack.addWidget(self.crop_page)
        self.stack.addWidget(self.convert_page)
        body.addWidget(self.stack, 1)
        root.addLayout(body, 1)

        # ---- 状态（36px）
        self.status = QLabel("就绪：导入包含 RAW / 16 位 TIFF / DNG 的文件夹")
        self.status.setProperty("caption", True)
        self.status.setFixedHeight(theme.SIZES["primary_target"])
        root.addWidget(self.status)

        self._viewer_background = theme.COLORS["viewer"]
        self.crop_page.view.setStyleSheet(
            f"background: {self._viewer_background}; border: none;")

        # ---- 信号
        self.btn_import.clicked.connect(self._pick_folder)
        self.btn_import_file.clicked.connect(self._pick_files)
        self.btn_suggest.clicked.connect(self._suggest_groups)
        self.btn_confirm_rois.clicked.connect(self.save_regions_as_annotation)
        self.btn_page_crop.clicked.connect(lambda: self._show_page(0))
        self.btn_page_convert.clicked.connect(lambda: self._show_page(1))
        self.library.frame_selected.connect(self.load_frame)
        self.library.grouping_changed.connect(self._on_grouping_changed)
        self.library.request_split.connect(self._on_split)

        cp = self.crop_page
        cp.rotate_requested.connect(self._rotate)
        cp.mirror_requested.connect(self._mirror)
        cp.auto_crop_requested.connect(self._auto_crop)
        cp.redetect_requested.connect(self._auto_crop)
        cp.fit_requested.connect(cp.view.fit)
        cp.zoom100_requested.connect(cp.view.zoom_100)
        cp.prev_requested.connect(lambda: self._step(-1))
        cp.next_requested.connect(lambda: self._step(1))
        cp.sync_requested.connect(self._sync_geometry)
        cp.reset_kind_requested.connect(self._reset_kind)
        cp.reset_all_requested.connect(self._reset_all_geometry)
        cp.delete_active_requested.connect(self._delete_active_roi)
        cp.view.roi_created.connect(self._on_roi_edited)
        cp.view.roi_changed.connect(self._on_roi_edited)
        cp.view.roi_deleted.connect(self._on_roi_deleted)
        cp.view.replace_hint.connect(
            lambda t: self.crop_page.set_banner(t, "info"))
        cp.view.draw_rejected.connect(
            lambda t: self.crop_page.set_banner(t, "warning"))

        cv = self.convert_page
        cv.candidate_chosen.connect(self._choose_candidate)
        cv.prev_requested.connect(lambda: self._step(-1))
        cv.next_requested.connect(lambda: self._step(1))
        cv.panel.changed.connect(self._intent_changed)
        cv.panel.committed.connect(self._intent_committed)
        cv.panel.sync_requested.connect(self._sync_intent)

    # ---------------------------------------------------------------- 页面
    def page_count(self) -> int:
        return self.stack.count()

    def page_title(self, index: int) -> str:
        return ("裁切", "转换")[index] if 0 <= index < self.stack.count() else ""

    def current_page(self) -> int:
        return self.stack.currentIndex()

    def _show_page(self, i: int):
        self.stack.setCurrentIndex(i)
        (self.btn_page_crop, self.btn_page_convert)[i].setChecked(True)
        if i == 1:
            self._refresh_convert_view()

    # ---------------------------------------------------------------- 导入
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
        return self._adopt_session(import_session(folder))

    def load_files(self, files: list[Path]) -> ImportSession:
        sess = ImportSession()
        root = Path(files[0]).parent if files else Path(".")
        for f in files:
            sess.files.append(make_frame_record(f, root=root, probe=False))
        return self._adopt_session(sess)

    def _adopt_session(self, sess: ImportSession) -> ImportSession:
        self.session = sess
        self.library.set_session(sess)
        if sess.roll_groups:
            self.library.select_roll(sess.roll_groups[0].id)
        self._set_banner(f"导入 {len(sess.files)} 个文件；请整理成卷后标定区域",
                         "info")
        return sess

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
        if self.session is None:
            return
        rec = self.session.frame(path)
        if rec is None:
            return
        try:
            loaded = self._load(rec)
        except ReadError as e:
            rec.status = FrameStatus.DECODE_ERROR.value
            QMessageBox.critical(self, "读取失败", str(e))
            return
        self.current = rec
        self.t_source = loaded.rgb
        self._load_meta = loaded.meta
        self._fingerprint = capture_fingerprint(
            loaded.meta, source_path=rec.source_path,
            batch_assumption=(self.session.batch_encoding_assumption
                              if self.session else None))
        rec.capture_fingerprint = self._fingerprint.value
        rec.input_facts = {**dict(loaded.meta), "kind": loaded.meta.get("kind", "")}
        self._ann = _annotation_for(rec)
        if self._ann is not None and not rec.geometry.is_complete():
            rec.geometry = _geometry_from_annotation(self._ann)
        self._analyze_current_roll()
        self._prepare_frame_views()
        self.library.set_current_frame(path)
        sh, sw = self.t_source.shape[:2]
        self._set_banner(f"{Path(path).name} | {sw}x{sh} | "
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
        self.roll_analysis = None
        if self.session is None or not self.library.current_roll:
            return
        frames = [f for f in self._frames_of(self.library.current_roll)
                  if f.capture_fingerprint]
        if not frames:
            return
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
        self.roll_analysis = analyze_roll(stats, fingerprint=main_fp)
        conflicts = [f.source_path for f in frames
                     if not fingerprints_compatible(f.capture_fingerprint, main_fp)]
        self.library.mark_conflicts(conflicts)
        if conflicts:
            self._set_banner(
                f"本卷有 {len(conflicts)} 帧采集指纹不一致：已停止共享卷分析，"
                "可用“拆分”一键分成不同卷", "warning")

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
        """加载公共后处理：工作副本缓存 + 预览服务 + 草稿上屏。"""
        rec = self.current
        k = max(1, int(np.ceil(max(self.t_source.shape[:2]) / MAX_PREVIEW_DIM)))
        self._small = self.t_source[::k, ::k, :].copy()
        self._small_scale = k
        # 已有完整草稿（历史标注）视为用户状态，不再被自动结果覆盖
        self._draft_touched = rec.geometry.is_complete()
        self._refresh_preview(seed_if_untouched=not self._draft_touched)
        self._show_frame()

    def _geometry_for_service(self) -> GeometryRegions:
        """把视图（ORIENTED 预览坐标）的三类 ROI 换算回 SOURCE 草稿。"""
        rec = self.current
        tr = OrientTransform(self.t_source.shape[:2],
                             rec.geometry.rotation_degrees,
                             rec.geometry.mirror_horizontal,
                             rec.geometry.fine_rotation_degrees)
        s = self._pscale
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
            output_crop_roi=to_src(self.crop_page.view.roi(RoiKind.OUTPUT_CROP)),
            image_area_roi=to_src(self.crop_page.view.roi(RoiKind.IMAGE_AREA)),
            base_sample_rois=[r for r in (
                to_src(self.crop_page.view.roi(RoiKind.BASE, i))
                for i in range(self.crop_page.view.roi_count(RoiKind.BASE))) if r])

    def _small_geometry(self, geo: GeometryRegions) -> GeometryRegions:
        """草稿 -> 降采样工作副本坐标（归一化相对位置映射）。"""
        return GeometryRegions.from_normalized(
            geo.normalized(self.t_source.shape[:2]), self._small.shape[:2])

    def _refresh_preview(self, *, seed_if_untouched: bool = False):
        """在降采样副本上调用同一 service 刷新预览（§5.2）。

        service 异常不阻断画布编辑与草稿保存：只给出轻量提示。
        """
        rec = self.current
        geo = rec.geometry
        try:
            self.pr = process_image(
                self._small, geometry=self._small_geometry(geo), ann=None,
                roll_analysis=self.roll_analysis,
                render_intent=self._intent, candidate=self._candidate,
                keep_rebate=self.crop_page.keep_rebate(),
                capture_fingerprint=self._fingerprint,
                film_type="unknown")
        except Exception as e:  # noqa: BLE001 —— §5.2：不阻断编辑
            self._set_banner(f"预览刷新失败：{type(e).__name__}: {e}"
                             "（区域编辑与草稿保存不受影响）", "danger")
            return
        rec.frame_analysis = {
            "base_dmin": self.pr.frame_base.get("dmin"),
            "base_confidence": self.pr.frame_base.get("confidence"),
            "contamination": self.pr.frame_base.get("contamination"),
            "black_level": self.pr.frame_base.get("black_level"),
            "spatial_trend": self.pr.frame_base.get("spatial_trend"),
            "regions": [list(r) for r in geo.base_sample_rois]}
        if seed_if_untouched and not geo.is_complete():
            rss = self.pr.geometry.roi_source_space
            rec.geometry = GeometryRegions(
                rotation_degrees=geo.rotation_degrees,
                fine_rotation_degrees=geo.fine_rotation_degrees,
                mirror_horizontal=geo.mirror_horizontal,
                output_crop_roi=_as_roi(rss.get("output_crop_roi")),
                image_area_roi=_as_roi(rss.get("image_area_roi")),
                base_sample_rois=[_as_roi(r) for r in
                                  (rss.get("base_sample_rois") or []) if r])
            self._draft_touched = False
            self._save_draft_annotation(rec, quiet=True)
        self.crop_page.set_status(_frame_status_text(rec, self.pr))
        self._refresh_convert_view()

    def _show_frame(self):
        """把**用户草稿**上屏（service 结果只用于图像与诊断）。"""
        if self.pr is None or self.current is None:
            return
        rec = self.current
        self._pscale = self._small_scale
        tr = OrientTransform(self.t_source.shape[:2],
                             rec.geometry.rotation_degrees,
                             rec.geometry.mirror_horizontal,
                             rec.geometry.fine_rotation_degrees)
        s = self._pscale
        view = self.crop_page.view
        draw_kind = view.draw_mode()
        view.clear_rois()

        def put(kind, roi_src, multiple=False):
            if not roi_src:
                return
            for r in (roi_src if multiple else [roi_src]):
                o = tr.bbox_source_to_oriented(tuple(r))
                view.add_roi(kind, _qrect(o[1] // s, o[0] // s,
                                          max(1, (o[3] - o[1]) // s),
                                          max(1, (o[2] - o[0]) // s)))

        rss = rec.geometry
        put(RoiKind.OUTPUT_CROP, rss.output_crop_roi)
        put(RoiKind.IMAGE_AREA, rss.image_area_roi)
        put(RoiKind.BASE, rss.base_sample_rois, multiple=True)
        view.set_draw_mode(draw_kind)
        view.set_image(self._small)
        view.fit()
        self.crop_page.set_fine_rotation(rec.geometry.fine_rotation_degrees)
        self.crop_page.set_keep_rebate(self.crop_page.keep_rebate())
        self.crop_page.set_status(_frame_status_text(rec, self.pr))
        self._refresh_convert_view()

    # ------------------------------------------------------- 编辑事件
    def _on_roi_edited(self, kind, index=0, _rect=None):
        """画布释放鼠标：保存草稿并刷新预览（service 失败不阻断）。"""
        self._update_draft_from_view()
        self.crop_page.roi_edited.emit(kind, index)

    def _on_roi_deleted(self, kind, index=0):
        label = KIND_LABELS.get(kind, str(kind))
        self._update_draft_from_view()
        if kind is not RoiKind.BASE:
            self._set_banner(f"已删除{label}：状态 = 需要裁切复核"
                             "（不会自动用其他框代替）", "warning")
        else:
            self._set_banner(f"已删除{label} {index + 1}", "info")

    def _update_draft_from_view(self):
        rec = self.current
        if rec is None or self.t_source is None:
            return
        rec.geometry = self._geometry_for_service()
        self._draft_touched = True
        if not rec.geometry.is_complete():
            rec.status = FrameStatus.NEEDS_CROP_REVIEW.value
        self._save_draft_annotation(rec, quiet=True)
        self.crop_page.set_status(_frame_status_text(rec, self.pr))
        self._refresh_preview()

    def _save_draft_annotation(self, rec: FrameRecord, *, quiet: bool = False):
        """编辑后把草稿写入现有 annotation 链路（原子、draft、不签字）。"""
        try:
            geo = rec.geometry
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
            save_annotation(rec.source_path, ann)
            self._ann = ann
        except Exception as e:  # noqa: BLE001 —— 保存失败可见但不阻断
            self._set_banner(f"草稿保存失败：{type(e).__name__}: {e}", "danger")

    def _reset_kind(self, kind: RoiKind):
        label = KIND_LABELS.get(kind, str(kind))
        self.crop_page.view.clear_rois(kind)
        self._update_draft_from_view()
        if kind is not RoiKind.BASE:
            self._set_banner(f"已清空{label}：状态 = 需要裁切复核", "warning")
        else:
            self._set_banner(f"已清空全部{label}区", "info")

    def _reset_all_geometry(self):
        self.crop_page.view.clear_rois()
        self._update_draft_from_view()
        self._set_banner("已清空全部几何：状态 = 需要裁切复核", "warning")

    def _delete_active_roi(self):
        if self.crop_page.view.delete_active():
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
        self.crop_page.set_banner("已按自动检测重新生成三类区域（可继续调整）",
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
            self.convert_page.panel.set_intent(self._intent, silent=True)
        self._refresh_convert_view()

    def _intent_changed(self, field: str, value: float):
        setattr(self._intent, field, float(value))
        self._refresh_convert_view()

    def _intent_committed(self, field: str, value: float):
        setattr(self._intent, field, float(value))
        if self.current is not None:
            self.current.render_intent_override = self._intent.to_dict()
        self._refresh_convert_view()

    # ---------------------------------------------------------------- 提示
    def _set_banner(self, text: str, level: str = "info"):
        """底部/右侧轻量 inline 提示（§5.6.7），非模态。"""
        self._banner_text = text if level != "info" else ""
        self.crop_page.set_banner(text, level)

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
        scope = self.convert_page.panel.combo_sync_scope.currentText()
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

    def _refresh_convert_view(self):
        if self.pr is None:
            return
        pos = candidate_positive(self.pr, self._candidate, self._intent)
        self.convert_page.set_images(self._small, pos)
        if self.pr.candidates is not None:
            self.convert_page.set_candidates(self.pr.candidates, self._candidate)
        roll_name = ""
        if self.session and self.current and self.current.roll_id:
            r = self.session.roll(self.current.roll_id)
            roll_name = r.display_name if r else ""
        self.convert_page.set_state(
            roll_name, Path(self.current.source_path).name,
            list(self.pr.gate.review_reasons),
            "" if self.pr.gate.status != "fail" else "质量门禁阻断导出")

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
    try:
        return load_annotation(rec.source_path)
    except Exception:  # noqa: BLE001 —— 损坏标注不得阻断打开
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
    theme.apply(app)
    win = MainWindow()
    win.show()
    if len(sys.argv) > 1:
        p = Path(sys.argv[1])
        if p.is_dir():
            win.load_folder(p)
        else:
            win.load_files([p])
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
