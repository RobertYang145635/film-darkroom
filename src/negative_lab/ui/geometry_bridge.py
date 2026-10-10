"""GeometryBridge：草稿 <-> 画布 <-> 管线的几何桥（2026-09-08 MainWindow 拆分）。

只搬不改（交接 §12-(6)）：预览管线（同步/后台）、即时合成、草稿读写与
防抖、ROI 编辑事件、方向/镜像/自动检测、高清分片（预留）从 MainWindow
原文搬入本 mixin；共享状态（current/_small/_positive_full/_draft_* 等）
仍在 MainWindow 实例上，mixin 只承载方法。

坐标契约（v6.2 F5）：画布坐标一律先加 _view_origin 偏移才是
_positive_full 索引；service 的 roi_source_space 是 small 空间坐标，
必须经 _unscale_geometry 归一化换算回 SOURCE。
"""

from __future__ import annotations

import json

import numpy as np
from PySide6.QtCore import QTimer

from ..calibration.annotations import Annotation
from ..library.models import FrameRecord, FrameStatus, GeometryRegions
from ..pipeline.geometry import OrientTransform, rotate_step
from . import theme
from .preview_view import RoiKind
from .state_store import (MAX_PREVIEW_DIM, _CANVAS_STALE, adopted_rotation,
                          _frame_status_text, _geometry_for_job, _qrect,
                          _shape_hw, _unscale_geometry, rotation_stale_regions,
                          ui_process_image)

KIND_LABELS = {RoiKind.OUTPUT_CROP: "输出裁切",
               RoiKind.IMAGE_AREA: "有效画面",
               RoiKind.BASE: "片基取样"}


class GeometryBridgeMixin:
    """草稿/画布/预览管线之间的几何桥（30 个方法，原文搬移）。"""

    def _prepare_frame_views(self):
        """（同步路径，测试/兜底用）：管线同步跑一次，状态与图一次到位。"""
        rec = self.current
        k = max(1, int(np.ceil(max(self.t_source.shape[:2]) / MAX_PREVIEW_DIM)))
        self._small = self.t_source[::k, ::k, :].copy()
        self._small_scale = k
        self._draft_touched = rec.geometry.is_complete()
        geo_snapshot = rec.geometry
        seed = not self._draft_touched
        self.pr = ui_process_image(
            self._small,
            geometry=(_geometry_for_job(None, self._small, self.t_source,
                                        geo_snapshot)
                      if rec.geometry.is_complete() else None),
            ann=None, roll_analysis=self.roll_analysis,
            render_intent=self._intent, candidate=self._candidate,
            keep_rebate=True, orientation_decided=rec.orientation_decided,
            capture_fingerprint=self._fingerprint)
        self._positive_full = self.pr.result.positive_linear
        if seed:
            src_geo = _unscale_geometry(self.pr.geometry.roi_source_space,
                                        self.t_source.shape[:2],
                                        self._small.shape[:2])
            rec.geometry = GeometryRegions(
                rotation_degrees=adopted_rotation(self.pr,
                                                  geo_snapshot.rotation_degrees),
                fine_rotation_degrees=geo_snapshot.fine_rotation_degrees,
                mirror_horizontal=geo_snapshot.mirror_horizontal,
                output_crop_roi=src_geo.get("output_crop_roi"),
                image_area_roi=src_geo.get("image_area_roi"),
                image_area_auto=bool(src_geo.get("image_area_auto")),
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
        decided_snapshot = rec.orientation_decided
        touched = self._draft_touched
        seed = bool(seed_if_untouched and not touched)
        small = self._small
        fp = self._fingerprint
        roll = self.roll_analysis
        intent_snapshot = self._intent.copy()
        cand_snapshot = self._candidate

        def job(cb):
            return ui_process_image(
                small,
                geometry=_geometry_for_job(None, small, self.t_source,
                                           geo_snapshot),
                ann=None, roll_analysis=roll,
                render_intent=intent_snapshot, candidate=cand_snapshot,
                keep_rebate=True,          # 整幅技术正片（裁切走即时合成）
                orientation_decided=decided_snapshot,
                capture_fingerprint=fp)

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
                        rotation_degrees=adopted_rotation(
                            pr, geo_snapshot.rotation_degrees),
                        fine_rotation_degrees=geo_snapshot.fine_rotation_degrees,
                        mirror_horizontal=geo_snapshot.mirror_horizontal,
                        output_crop_roi=src_geo.get("output_crop_roi"),
                        image_area_roi=src_geo.get("image_area_roi"),
                        image_area_auto=bool(src_geo.get("image_area_auto")),
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
        # 经 main_window 解析开关（冷读 minor-4，2026-09-08）：拆分后本模块
        # 的 by-value import 不再受 mw.ENABLE_HD_PREVIEW monkeypatch 控制；
        # 拆分前该 patch 点有效，恢复同一契约（延迟导入避免 import 期成环）。
        from .main_window import ENABLE_HD_PREVIEW as _hd_enabled
        if not _hd_enabled:
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
                pr = ui_process_image(
                    loaded.rgb, geometry=rec.geometry,
                    render_intent=self._intent, candidate=self._candidate,
                    keep_rebate=True, orientation_decided=rec.orientation_decided)
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
            # 延迟导入：_drafts_dir 读 main_window._DRAFTS_DIR（测试
            # monkeypatch 面），本模块被 main_window 导入，不能在 import 期成环。
            from .main_window import _drafts_dir
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
        # 审查修复（2026-09-09）：三类区域全清含统计范围（片基/有效画面），
        # 不重算就继续挂着按旧区域算出的分析结果——与 _on_area_clear 同一条
        # v6.2 F2 规则，此前一起漏了。
        self._update_draft_from_view(rerun=True)
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
        #: B-12 追加③：几何规则（竖画框 ⇒ 逆时针补 270°）可能已经把这帧转过来了。
        #: 记录里存的必须是**生效角度**，否则第一次按下算的是 `0+90=90`，
        #: 而画面已经在 90°——他看到的就是一条都没动（用户定的口径：按钮转的是我看到的）。
        # 片带横躺的帧上，±90 只会转出 35mm 不存在的竖幅，所以一次按键翻 180°
        # （在两个合法横幅之间来回）；正常横框的帧按键语义不变。
        trusted_area = (g.image_area_roi
                        if g.image_area_roi and not g.image_area_auto else None)
        g.rotation_degrees = ((adopted_rotation(self.pr, g.rotation_degrees)
                               + rotate_step(trusted_area, delta)) % 360)
        #: B-12：用户按下旋转键就是"方向已决定"，此后不再提示"未经确认"
        self.current.orientation_decided = True
        #: 只丢掉自动检测给出的区域；他亲手画的画框/片基框在 SOURCE 空间，不随旋转失效
        #: （旧实现三类一起清，等于把他为 1299 画的 11 个片基框一次按没，
        #: 还顺手让几何规则失去依据）。
        drop_crop, drop_area, drop_base = rotation_stale_regions(
            getattr(self.pr, "roi_provenance", None))
        if drop_crop:
            g.output_crop_roi = None
        if drop_area:
            g.image_area_roi = None
        if drop_base:
            g.base_sample_rois = []
        self._draft_touched = False
        self._refresh_preview(seed_if_untouched=True)
        self._show_frame()
        self._set_banner("方向已改变：自动检测的区域重新生成，你画的框保持不变", "info")

    def _mirror(self):
        if self.current is None:
            return
        self.current.geometry.mirror_horizontal = \
            not self.current.geometry.mirror_horizontal
        self._refresh_preview(seed_if_untouched=True)
        self._show_frame()
        self._update_draft_from_view()   # 审查修复（GUI 重要-5）：镜像此前只活在内存，
        self._set_banner("镜像已切换并写入草稿（重启可续）", "info")   #   几何完整时不落盘
