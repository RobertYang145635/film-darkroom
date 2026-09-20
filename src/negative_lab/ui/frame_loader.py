"""FrameLoader：导入与会话/帧加载生命周期（2026-09-08 MainWindow 拆分）。

只搬不改（交接 §12-(6)）：导入入口（文件夹/文件）、会话采纳、胶片条同步、
帧加载（后台解码 -> 卷分析 -> 预览数据装配）、批次编码询问、卷级分析
全部从 MainWindow 原文搬入本 mixin；共享状态（session/current/t_source/
_pr/_draft_flush_timer/...）仍在 MainWindow 实例上。

线程契约（ui/workers.py）：worker 只算并交数据，任何落控件都在主线程
（见 _analyze_roll_frames 交出 _pending_conflicts、_frame_done 落地）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import QThread
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

from ..io.images import ReadError, load_linear_rgb
from ..library.fingerprint import capture_fingerprint, fingerprints_compatible
from ..library.models import (FrameRecord, GeometryRegions, ImportSession,
                              RollAnalysis)
from ..library.session import (import_session, make_frame_record,
                               suggest_groups_by_directory)
from ..pipeline.roll_analysis import analyze_roll
from .state_store import (IMPORT_PROBE, MAX_PREVIEW_DIM, _CANVAS_STALE,
                          _abs, _geometry_for_job, _geometry_from_annotation,
                          _unscale_geometry, ui_process_image)


class FrameLoaderMixin:
    """导入、会话采纳、帧加载、卷级分析（数据装配部分）。"""

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

    def _start_import(self, job):
        """导入的唯一后台入口（文件夹与文件列表共用）。

        审查修复（2026-09-09）：此前 `self._busy_enter("import")` 的返回值被
        丢弃——忙碌守卫是装饰性的，连点两次「导入文件夹」会并发跑两遍导入，
        两个会话先后 `_adopt_session`，后到的那个无声覆盖前一个。
        现在忙就拒绝并说明（与 load_frame 的 L1-② 口径一致：不给不成立的承诺）。
        """
        if not self._busy_enter("import"):
            self._set_banner("上一次导入仍在进行中：请等待它完成后再导入", "warning")
            return None
        from .workers import run_in_thread
        return run_in_thread(
            job,
            on_done=self._import_done,
            on_progress=lambda m: self._set_banner(f"导入：{m}", "info"))

    def load_folder(self, folder: Path):
        """后台导入（§15.7 响应性）：只取元数据+哈希，不做完整解码探针
        （12 张 NEF 完整探针 20s——用户实测过久；解码失败在打开帧时
        给出完整错误即可）。

        返回后台线程句柄（导入已在进行则 None）；会话在 `_import_done`
        落地。此前注解写 `-> ImportSession` 是假的：调用方拿不到会话。
        """
        folder = Path(folder)
        self._set_banner(f"正在导入 {folder.name} …（读取元数据与哈希，"
                         "界面保持可用）", "info")
        return self._start_import(
            lambda cb: import_session(folder, probe=IMPORT_PROBE))

    def load_files(self, files: list[Path], *, background: bool = True):
        """导入显式选择的文件（「导入文件…」/ 双击打开）。

        审查修复（2026-09-09）：默认改到后台执行。此前逐文件哈希跑在 **GUI
        线程**，违反 ui/workers.py 自己立的契约「一切 >200ms 的工作都必须
        离开主线程」：`make_frame_record` 对每个文件做整文件 SHA-256，实测
        本仓库真实样片 NEF 55 MB ≈ 80 ms、16 位 TIFF 274 MB ≈ 400 ms ——
        选 20 个 TIFF 就是约 8 秒窗口「未响应」，且期间连横幅都不刷新。
        `background=False` 保留同步语义，供脚本与测试直接取会话。
        """
        files = [Path(f) for f in files]
        if not files:
            return None
        if not background:
            return self._adopt_session(self._build_files_session(files))
        return self._start_import(lambda cb: self._build_files_session(files))

    @staticmethod
    def _build_files_session(files: list[Path]) -> ImportSession:
        """文件列表 -> ImportSession（纯计算，可安全放后台线程）。"""
        sess = ImportSession()
        root = Path(files[0]).parent
        for f in files:
            sess.files.append(make_frame_record(f, root=root, probe=False))
        return sess

    def _import_done(self, result: dict):
        self._busy_leave("import")
        if not result.get("ok"):
            self._set_banner(f"导入失败：{result.get('error')}", "danger")
            return
        self._adopt_session(result["value"])
        # 导入即预览（兜底）：若自动选卷链路没触发加载，直接载第一帧
        if self.current is None and self.session is not None \
                and self.session.files:
            self._load_first_frame_of(self.library.current_roll)

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
        # 延迟导入：包装函数读 main_window._DRAFTS_DIR（测试 monkeypatch 面），
        # 而本模块被 main_window 导入，不能在 import 期成环。
        from .main_window import resolve_persistent_state
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
        pr = ui_process_image(
            small,
            geometry=(_geometry_for_job(None, small, loaded.rgb, seed_geo)
                      if seed_geo is not None else None),
            ann=None, roll_analysis=roll,
            render_intent=self._intent, candidate=self._candidate,
            keep_rebate=True, capture_fingerprint=fp)
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
