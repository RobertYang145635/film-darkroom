"""主窗口（v5 §1.3 / §5 / §7）：只有 裁切 与 转换 两个一级页面。

业务状态全部在 `library.models.ImportSession` 与 `pipeline.service` 中，
UI 只是视图：卷、帧、ROI、候选与 RenderIntent 都可脱离 GUI 测试。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication, QButtonGroup, QFileDialog, QHBoxLayout, QLabel,
    QMainWindow, QMessageBox, QPushButton, QStackedWidget, QVBoxLayout, QWidget)

from .. import __version__
from ..calibration.annotations import Annotation, load_annotation, save_annotation
from ..io.images import ReadError, load_linear_rgb
from ..library.fingerprint import (capture_fingerprint, fingerprints_compatible)
from ..library.models import (FrameRecord, FrameStatus, GeometryRegions,
                              ImportSession, RollAnalysis)
from ..library.session import import_session, suggest_groups_by_directory
from ..pipeline.geometry import OrientTransform
from ..pipeline.render import RenderIntent
from ..pipeline.roll_analysis import analyze_roll, frame_base_stats
from ..pipeline.service import candidate_positive, process_image
from .convert_page import ConvertPage
from .crop_page import CropPage
from .library_panel import LibraryPanel
from .preview_view import RoiKind

MAX_PREVIEW_DIM = 1400


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"负片裁切与去色罩 negative-lab v{__version__}")
        self.resize(1500, 920)
        self.session: ImportSession | None = None
        self.current: FrameRecord | None = None
        self.t_source: np.ndarray | None = None
        self.t_oriented: np.ndarray | None = None
        self.t_prev: np.ndarray | None = None
        self._pscale = 1
        self.pr = None
        self.roll_analysis: RollAnalysis | None = None
        self._intent = RenderIntent()
        self._candidate = "preserve"
        self._ann: Annotation | None = None
        self._load_meta: dict = {}
        self._fingerprint = None
        self._high_quality_pending = False

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        # ---- 顶部：导入 + 两个一级页面
        head = QHBoxLayout()
        self.btn_import = QPushButton("导入文件夹…")
        self.btn_import_file = QPushButton("导入文件…")
        self.btn_suggest = QPushButton("按目录建议分组")
        self.btn_export = QPushButton("整卷导出…")
        for b in (self.btn_import, self.btn_import_file, self.btn_suggest,
                  self.btn_export):
            head.addWidget(b)
        head.addStretch(1)
        self.btn_page_crop = QPushButton("裁切")
        self.btn_page_convert = QPushButton("转换")
        for b in (self.btn_page_crop, self.btn_page_convert):
            b.setCheckable(True)
            b.setAutoExclusive(True)
            head.addWidget(b)
        self._page_group = QButtonGroup(self)
        self._page_group.setExclusive(True)
        self._page_group.addButton(self.btn_page_crop, 0)
        self._page_group.addButton(self.btn_page_convert, 1)
        root.addLayout(head)

        body = QHBoxLayout()
        self.library = LibraryPanel()
        self.library.setFixedWidth(260)
        body.addWidget(self.library)

        self.stack = QStackedWidget()
        self.crop_page = CropPage()
        self.convert_page = ConvertPage()
        self.stack.addWidget(self.crop_page)
        self.stack.addWidget(self.convert_page)
        body.addWidget(self.stack, 1)
        root.addLayout(body, 1)

        self.status = QLabel("就绪：导入包含 RAW / 16 位 TIFF / DNG 的文件夹")
        f_small = QFont(self.font())
        f_small.setPointSize(max(7, f_small.pointSize() - 1))
        self.status.setFont(f_small)
        root.addWidget(self.status)

        # ---- 信号
        self.btn_import.clicked.connect(self._pick_folder)
        self.btn_import_file.clicked.connect(self._pick_files)
        self.btn_suggest.clicked.connect(self._suggest_groups)
        self.btn_export.clicked.connect(self.export_roll)
        self.convert_page.btn_export.clicked.connect(self.export_roll)
        self.btn_page_crop.clicked.connect(lambda: self._show_page(0))
        self.btn_page_convert.clicked.connect(lambda: self._show_page(1))
        self.library.frame_selected.connect(self.load_frame)
        self.library.grouping_changed.connect(self._on_grouping_changed)
        self.library.request_split.connect(self._on_split)

        cp = self.crop_page
        cp.rotate_requested.connect(self._rotate)
        cp.mirror_requested.connect(self._mirror)
        cp.auto_crop_requested.connect(lambda: self._recompute(redetect=True))
        cp.redetect_requested.connect(lambda: self._recompute(redetect=True))
        cp.fit_requested.connect(cp.view.fit)
        cp.zoom100_requested.connect(cp.view.zoom_100)
        cp.geometry_changed.connect(self._geometry_from_view)
        cp.prev_requested.connect(lambda: self._step(-1))
        cp.next_requested.connect(lambda: self._step(1))
        cp.sync_requested.connect(self._sync_geometry)

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
        sess = import_session(folder)
        return self._adopt_session(sess)

    def load_files(self, files: list[Path]) -> ImportSession:
        sess = ImportSession()
        root = Path(files[0]).parent if files else Path(".")
        for f in files:
            sess.files.append(_make_record(f, root))
        return self._adopt_session(sess)

    def _adopt_session(self, sess: ImportSession) -> ImportSession:
        self.session = sess
        self.library.set_session(sess)
        if sess.roll_groups:
            self.library.select_roll(sess.roll_groups[0].id)
        self.status.setText(
            f"导入 {len(sess.files)} 个文件；请整理成卷后开始裁切")
        return sess

    def _suggest_groups(self):
        if self.session is None:
            return
        suggest_groups_by_directory(self.session, apply=True)
        self.library.set_session(self.session)
        self.status.setText("已按目录建立卷分组（可继续改名、拆分、移动）")

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
            rec.geometry = GeometryRegions(
                rotation_degrees=self._ann.rotation_degrees,
                output_crop_roi=self._ann.output_crop_roi
                or self._ann.image_area_roi,
                image_area_roi=self._ann.image_area_roi,
                base_sample_rois=list(self._ann.base_sample_rois or []))
        self._analyze_current_roll()
        self._run_service()
        self._show_frame()
        self.library.set_current_frame(path)
        self.status.setText(
            f"{Path(path).name} | {self.t_source.shape[1]}x{self.t_source.shape[0]} | "
            f"片基来源 {self.pr.base_source} | 状态 {self.pr.gate.status}")

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
        """v5 §4.3.3：编码事实矛盾时，一次**导入批次级**选择（不是设备配置）。"""
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
        rid = self.library.current_roll
        frames = [f for f in self._frames_of(rid) if f.capture_fingerprint]
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
            self.status.setText(
                f"本卷有 {len(conflicts)} 帧采集指纹不一致：已停止共享卷分析，"
                "可用“拆分”一键分成不同卷")

    def _frames_of(self, rid: str) -> list[FrameRecord]:
        if self.session is None:
            return []
        if rid == "__ungrouped__":
            return self.session.ungrouped()
        r = self.session.roll(rid)
        return [f for f in (self.session.frame(p) for p in (r.frame_ids if r else []))
                if f is not None]

    # ---------------------------------------------------------------- 服务
    def _geometry_for_service(self) -> GeometryRegions:
        """把视图（ORIENTED 预览坐标）的三类 ROI 换算回 SOURCE。"""
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

    def _run_service(self):
        if self.t_source is None or self.current is None:
            return None
        geo = self._geometry_for_service()
        self.current.geometry = geo
        self.pr = process_image(
            self.t_source, geometry=geo, ann=None,
            roll_analysis=self.roll_analysis,
            render_intent=self._intent, candidate=self._candidate,
            keep_rebate=self.crop_page.keep_rebate(),
            capture_fingerprint=self._fingerprint,
            film_type="unknown")
        rec = self.current
        rec.frame_analysis = {
            "base_dmin": self.pr.frame_base.get("dmin"),
            "base_confidence": self.pr.frame_base.get("confidence"),
            "contamination": self.pr.frame_base.get("contamination"),
            "black_level": self.pr.frame_base.get("black_level"),
            "spatial_trend": self.pr.frame_base.get("spatial_trend"),
            "regions": [list(r) for r in geo.base_sample_rois]}
        rec.chosen_candidate = self.pr.chosen_candidate
        rec.status = _status_of(self.pr, rec)
        return self.pr

    def _show_frame(self):
        if self.pr is None:
            return
        self.t_oriented = self.pr.t if self.crop_page.keep_rebate() \
            else _orient(self.t_source, self.current.geometry)
        self._pscale = max(1, int(np.ceil(max(self.t_oriented.shape[:2])
                                          / MAX_PREVIEW_DIM)))
        self.t_prev = self.t_oriented[::self._pscale, ::self._pscale, :].copy()
        tr = OrientTransform(self.t_source.shape[:2],
                             self.current.geometry.rotation_degrees,
                             self.current.geometry.mirror_horizontal,
                             self.current.geometry.fine_rotation_degrees)
        s = self._pscale
        view = self.crop_page.view
        view.clear_rois()

        def put(kind, roi_src, multiple=False):
            if not roi_src:
                return
            rois = roi_src if multiple else [roi_src]
            for r in rois:
                o = tr.bbox_source_to_oriented(tuple(r))
                view.add_roi(kind, _qrect(o[1] // s, o[0] // s,
                                          max(1, (o[3] - o[1]) // s),
                                          max(1, (o[2] - o[0]) // s)))

        rss = self.pr.geometry.roi_source_space
        put(RoiKind.OUTPUT_CROP, rss.get("output_crop_roi"))
        put(RoiKind.IMAGE_AREA, rss.get("image_area_roi"))
        put(RoiKind.BASE, rss.get("base_sample_rois"), multiple=True)
        view.set_image(self.t_prev)
        view.fit()
        self.crop_page.set_fine_rotation(self.current.geometry.fine_rotation_degrees)
        self.crop_page.set_keep_rebate(self.crop_page.keep_rebate())
        self.crop_page.set_status(
            f"来源：{self.pr.roi_provenance.get('output_crop', '?')} / "
            f"{self.pr.roi_provenance.get('image_area', '?')} / "
            f"{self.pr.roi_provenance.get('base', '?')}"
            + ("（自动检测置信度低，请确认）"
               if "needs_crop_review" in self.pr.roi_provenance.get("flags", []) else ""))
        self._refresh_convert_view()

    def _refresh_convert_view(self):
        if self.pr is None:
            return
        pos = candidate_positive(self.pr, self._candidate, self._intent)
        self.convert_page.set_images(self.t_source, pos)
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

    # ---------------------------------------------------------------- 交互
    def _geometry_from_view(self):
        if self.t_source is None:
            return
        self._run_service()
        self._refresh_convert_view()

    def _rotate(self, delta: int):
        if self.current is None:
            return
        g = self.current.geometry
        g.rotation_degrees = (g.rotation_degrees + delta) % 360
        g.output_crop_roi = g.image_area_roi = None
        g.base_sample_rois = []
        self._run_service()
        self._show_frame()
        self.status.setText("方向已改变：三类区域重新检测，请确认")

    def _mirror(self):
        if self.current is None:
            return
        self.current.geometry.mirror_horizontal = \
            not self.current.geometry.mirror_horizontal
        self._run_service()
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
            own = GeometryRegions.from_normalized(norm, size)
            errs = own.validate()
            if errs:
                self.status.setText(f"同步被拒绝：{errs[0]}")
                continue
            own.rotation_degrees = src.rotation_degrees
            own.mirror_horizontal = src.mirror_horizontal
            rec.geometry = own
            rec.status = FrameStatus.NEEDS_CROP_REVIEW.value
        self.status.setText(f"方向与三类 ROI 已同步到 {len(targets)} 帧"
                            "（按归一化相对位置，逐帧重新钳制与验证）")

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
        self.status.setText(f"微调项 {', '.join(fields)} 已同步到 {len(targets)} 帧")

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
                progress=lambda msg: self.status.setText(msg))
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "导出失败（已回滚）",
                                 f"{type(e).__name__}: {e}\n源文件未改动")
            return
        QMessageBox.information(
            self, "整卷导出完成",
            f"成功 {report['ok']} / 待复核 {report['needs_review']} / "
            f"失败 {report['failed']}\n报告：{report['report_path']}")
        self.status.setText(f"整卷导出完成：{report['ok']} 帧")

    # ---------------------------------------------------------------- 保存区域
    def save_regions_as_annotation(self):
        if self.current is None or self.pr is None:
            return
        geo = self._geometry_for_service()
        rec = self.current
        ann = Annotation(
            file=rec.source_path, roll=rec.roll_id or "misc",
            rotation_degrees=geo.rotation_degrees,
            output_crop_roi=geo.output_crop_roi,
            image_area_roi=geo.image_area_roi,
            base_sample_rois=list(geo.base_sample_rois),
            source_sha256=rec.source_sha256,
            verified=True, verified_by="user", notes="GUI 用户框选")
        try:
            p = save_annotation(rec.source_path, ann)
        except ValueError as e:
            QMessageBox.critical(self, "区域不完整", str(e))
            return
        self.status.setText(f"三类区域已保存为用户确认标注：{p.name}")


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


def _make_record(path: Path, root: Path) -> FrameRecord:
    from ..library.session import make_frame_record

    return make_frame_record(path, root=root, probe=False)


def _annotation_for(rec: FrameRecord) -> Annotation | None:
    try:
        return load_annotation(rec.source_path)
    except Exception:  # noqa: BLE001 —— 损坏标注不得阻断打开
        return None


def _orient(t: np.ndarray, geo: GeometryRegions) -> np.ndarray:
    tr = OrientTransform(t.shape[:2], geo.rotation_degrees,
                         geo.mirror_horizontal, geo.fine_rotation_degrees)
    return tr.apply(t)


def _qrect(x, y, w, h):
    from PySide6.QtCore import QRectF

    return QRectF(x, y, w, h)


def _status_of(pr, rec: FrameRecord) -> str:
    if pr.gate.failures:
        return FrameStatus.PROCESSING_ERROR.value
    if "needs_crop_review" in pr.roi_provenance.get("flags", []):
        return FrameStatus.NEEDS_CROP_REVIEW.value
    if pr.gate.review_reasons:
        return FrameStatus.NEEDS_REVIEW.value
    return FrameStatus.CONVERTED.value


def main() -> int:
    import sys

    app = QApplication(sys.argv)
    app.setApplicationName("negative-lab")
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
