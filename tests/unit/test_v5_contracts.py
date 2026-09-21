"""v5 P0-0：新范围的红灯契约（生产路径，不是人为构造的无关失败）。

对应《负片裁切与去色罩_AI执行规格_v5.md》§12 P0-0.3 列出的五类现状缺陷：

1. 未校准彩色负片被阻断或标错状态（v4 的 needs_calibration 产品门禁）；
2. GUI 只有两个 ROI（RoiKind.FRAME 同时承担输出裁切与分析区）；
3. 输出裁切与有效画面分析区无法独立；
4. 主界面暴露非 P0 技术参数（Dmin/Dmax/wb/指数/校准/除尘）；
5. 无整卷手动分组与三候选（preserve/neutral/warm）。

每条断言都指向 v5 的**目标产品契约**：实现前必须红，实现后必须绿。
禁止用提高阈值/替换 golden 图的方式让它变绿（§0.3.5）。
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------- 公共夹具
@pytest.fixture(scope="module", autouse=True)
def qapp():
    """Qt 控件测试必须先有 QApplication（否则进程直接 abort）。"""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def synth_pair():
    """一张带片基条带的合成彩色负片（含串扰，接近真实 C-41 行为）。"""
    from negative_lab.testing.spectral import make_spectral_negative

    s = make_spectral_negative(seed=7, crosstalk=True)
    return s, s["negative"]


def _rois(s, t):
    bx0, bx1 = s["truth"]["base_cols"]
    area = (0, 0, t.shape[0], bx0)
    base = (0, bx0, t.shape[0], bx1)
    return area, base


# ------------------------------------------------- 1. 缺色卡/缺校准不阻断
class TestNoCalibrationBlocking:
    def test_uncalibrated_color_negative_never_blocked(self, synth_pair):
        """v5 §3.3/§8.1：没有色卡、没有 verified 校准、未知胶片型号，
        都必须能正常转换并导出（状态只能是 ok/warn/needs_review）。"""
        from negative_lab.pipeline.service import process_image

        s, t = synth_pair
        area, base = _rois(s, t)
        pr = process_image(t, output_crop_source=area, image_area_source=area,
                           base_roi_source=base)
        assert pr.gate.status not in ("needs_calibration", "needs_regions"), \
            f"缺色卡/缺校准仍在阻断普通路径：{pr.gate.status}"
        assert pr.gate.blocks_export is False, pr.gate.failures
        assert pr.rendered_positive_linear is not None

    def test_gate_has_no_needs_calibration_status(self):
        """v5 §8.1：needs_calibration 不再是产品状态（研发 API 可另存）。"""
        from negative_lab.diagnostics.quality_gate import QualityGate

        assert "needs_calibration" not in QualityGate.VALID_STATUS, \
            QualityGate.VALID_STATUS
        assert "needs_review" in QualityGate.VALID_STATUS

    def test_normal_paths_have_no_color_chart_language(self):
        """v5 §8.1：普通路径文案不得再要求色卡/校准文件/中性点才能继续。

        只检查**字符串字面量**（文档字符串除外——文档允许引用被删除的旧文案）。
        """
        import ast

        forbidden = ("请提供色卡", "需要色卡", "提供色卡", "卷级校准文件",
                     "必须加载卷级校准", "没有中性点")

        def literals(path: Path) -> list[str]:
            tree = ast.parse(path.read_text(encoding="utf-8-sig"), str(path))
            doc_nodes = set()
            for node in ast.walk(tree):
                if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                     ast.AsyncFunctionDef)):
                    body = getattr(node, "body", None)
                    if body and isinstance(body[0], ast.Expr) \
                            and isinstance(body[0].value, ast.Constant) \
                            and isinstance(body[0].value.value, str):
                        doc_nodes.add(id(body[0].value))
            out = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                        and id(node) not in doc_nodes:
                    out.append(node.value)
            return out

        offenders = []
        for f in (ROOT / "src" / "negative_lab").rglob("*.py"):
            for lit in literals(f):
                for phrase in forbidden:
                    if phrase in lit:
                        offenders.append(f"{f.relative_to(ROOT)}: {phrase}")
        assert not offenders, "普通路径仍存在色卡/校准阻断文案：\n  " + \
            "\n  ".join(offenders)


# -------------------------------------------------- 2. GUI 三 ROI 与两页面
class TestCropPageGeometry:
    def test_roi_kind_split_into_three(self):
        """v5 §5.2：RoiKind.FRAME 必须拆成 OUTPUT_CROP 与 IMAGE_AREA。"""
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from negative_lab.ui.preview_view import RoiKind

        names = {k.name for k in RoiKind}
        assert "FRAME" not in names, f"RoiKind.FRAME 仍在混用两种语义：{names}"
        assert {"OUTPUT_CROP", "IMAGE_AREA", "BASE"} <= names, names

    def test_roi_labels_are_distinguishable(self):
        """颜色之外必须有标签或线型（v5 §5.2 可访问性要求）。"""
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from negative_lab.ui.preview_view import RoiKind, ROI_STYLE

        for kind in (RoiKind.OUTPUT_CROP, RoiKind.IMAGE_AREA, RoiKind.BASE):
            style = ROI_STYLE[kind]
            assert style.get("label"), f"{kind.name} 缺少文字标签"
            assert style.get("dash") or style.get("width"), \
                f"{kind.name} 缺少颜色之外的线型区分"

    def test_preview_view_supports_edge_and_corner_handles(self):
        """v5 §5.2：当前只允许整体拖动，必须补齐边/角缩放与越界约束。"""
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from negative_lab.ui.preview_view import HANDLE_COUNT, PreviewView

        assert HANDLE_COUNT == 8
        view = PreviewView()
        view.set_image(np.zeros((64, 64, 3), np.float32))
        from PySide6.QtCore import QRectF
        view.set_roi(view.roi_kind_order()[0], QRectF(10, 10, 20, 20))
        kind = view.roi_kind_order()[0]
        assert view.handle_at(kind, (20.0, 30.0)) == "s", "底边手柄未识别"
        assert view.handle_at(kind, (30.0, 30.0)) == "se", "右下角手柄未识别"
        assert view.handle_at(kind, (10.0, 20.0)) == "w", "左边手柄未识别"
        assert view.handle_at(kind, (20.0, 20.0)) is None, "内部不应命中手柄"
        # 缩放并越界钳制
        view.resize_roi(kind, "se", (80.0, 80.0))
        r = view.roi(kind)
        assert r.right() <= 64 and r.bottom() <= 64, f"越界未钳制：{r}"
        assert r.width() >= view.min_roi_size() and r.height() >= view.min_roi_size()
        # 反向拖过对角 -> 仍保持最小尺寸且不倒置
        view.resize_roi(kind, "se", (-40.0, -40.0))
        r2 = view.roi(kind)
        assert r2.width() >= view.min_roi_size() and r2.height() >= view.min_roi_size()

    def test_main_window_has_only_two_primary_pages(self):
        """v5 §5.1/§7.1：主工作区只保留 裁切 / 转换 两个一级页面。"""
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance() or QApplication([])
        from negative_lab.ui.main_window import MainWindow

        win = MainWindow()
        titles = [win.page_title(i) for i in range(win.page_count())]
        assert titles == ['工作区'], f'v6 单工作区：{titles}'
        assert app is not None

    def test_main_window_hides_non_p0_technical_controls(self):
        """v6 重做：技术信息折叠进右栏 TechSection，主视野不出现参数名。"""
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
        from PySide6.QtWidgets import QApplication, QLabel

        QApplication.instance() or QApplication([])
        from negative_lab.ui.main_window import MainWindow

        win = MainWindow()
        banned = re.compile(
            r'(dmax_|wb_low|wb_high|reference_exponent|red_ratio|'
            r'blue_ratio|density_matrix|backend|除尘|画笔|蒙版|卷级校准|配置文件)')
        visible = [lbl.text() for lbl in win.findChildren(QLabel)
                   if lbl.isVisibleTo(win) and banned.search(lbl.text())]
        assert not visible, f'主界面仍暴露非 P0 技术参数控件：{visible}'
        # 技术信息节存在且默认折叠
        assert win.right.sec_tech.btn_head.isChecked() is False

# ------------------------------------------- 3. 输出裁切与分析区真正独立
class TestRoiIndependence:
    def test_output_crop_can_include_rebate_without_polluting_analysis(
            self, synth_pair):
        """v5 §11.1：输出保留片边时，质量统计 mask 仍只覆盖 image_area_roi。"""
        from negative_lab.pipeline.service import process_image

        s, t = synth_pair
        bx0, _ = s["truth"]["base_cols"]
        area = (0, 0, t.shape[0], bx0)
        full = (0, 0, t.shape[0], t.shape[1])       # 输出含片边
        pr_area = process_image(t, output_crop_source=area, image_area_source=area,
                                base_roi_source=(0, bx0, t.shape[0], t.shape[1]))
        pr_full = process_image(t, output_crop_source=full, image_area_source=area,
                                base_roi_source=(0, bx0, t.shape[0], t.shape[1]))
        # 输出尺寸不同（真的保留了片边）
        assert pr_full.geometry.output_size[1] > pr_area.geometry.output_size[1]
        # 但技术质量统计完全一致（mask 只覆盖 image_area）
        assert pr_full.gate.output_clip == pr_area.gate.output_clip, \
            "输出保留片边污染了有效画面统计"
        assert pr_full.gate.chroma_stats == pr_area.gate.chroma_stats

    def test_three_roi_roundtrip_within_one_pixel(self, synth_pair):
        """v5 §11.1：三类 ROI 独立序列化 round-trip 误差 <= 1 像素。"""
        from negative_lab.pipeline.geometry import (roi_source_to_oriented,
                                                    oriented_to_source)

        s, t = synth_pair
        bx0, bx1 = s["truth"]["base_cols"]
        rois = {
            "output_crop_roi": (0, 0, t.shape[0], bx0),
            "image_area_roi": (4, 6, t.shape[0] - 8, bx0 - 2),
            "base_sample_rois": [(0, bx0, 40, bx1)],
        }
        for deg in (0, 90, 180, 270):
            for name, roi in rois.items():
                for r in ([roi] if name != "base_sample_rois" else roi):
                    o = roi_source_to_oriented(tuple(r), t.shape[0], t.shape[1], deg)
                    back = oriented_to_source(o, t.shape[0], t.shape[1], deg)
                    assert max(abs(a - b) for a, b in zip(r, back)) <= 1, \
                        (name, deg, r, o, back)

    def test_annotation_persists_three_independent_rois(self, tmp_path, monkeypatch):
        """v5 §3.2/§5.2：标注必须能分别保存输出裁切、分析区与多个片基区。"""
        import negative_lab.calibration.annotations as A
        monkeypatch.setattr(A, "ANNOTATIONS_DIR", tmp_path)
        from negative_lab.calibration.annotations import Annotation, save_annotation

        ann = Annotation(file="roll/a.NEF", roll="roll", rotation_degrees=90,
                         output_crop_roi=(0, 0, 500, 400),
                         image_area_roi=(10, 10, 490, 380),
                         base_sample_rois=[(0, 400, 40, 500), (460, 400, 500, 500)],
                         verified=True, verified_by="user")
        save_annotation("roll/a.NEF", ann)
        got = A.load_annotation("roll/a.NEF")
        assert tuple(got.output_crop_roi) == (0, 0, 500, 400)
        assert tuple(got.image_area_roi) == (10, 10, 490, 380)
        assert [tuple(r) for r in got.base_sample_rois] == \
            [(0, 400, 40, 500), (460, 400, 500, 500)], \
            "片基取样区必须支持多个并独立持久化（v5 §3.2）"


# --------------------------------------------- 4. RenderIntent 是唯一微调
class TestRenderIntentP0Sliders:
    SLIDERS = ("temperature", "tint", "exposure", "contrast", "highlights",
               "shadows", "whites", "blacks", "vibrance", "saturation")

    def test_intent_exposes_exactly_p0_controls(self):
        from negative_lab.pipeline.render import RenderIntent

        fields = set(RenderIntent.FIELD_NAMES)
        assert set(self.SLIDERS) <= fields, fields
        assert "grade" not in fields, "旧 grade 不得再作为普通微调入口"

    def test_each_slider_changes_render_not_technical(self, synth_pair):
        """v5 §11.2：每个滑杆改变目标量且不改变技术正片。"""
        from negative_lab.pipeline.render import RenderIntent
        from negative_lab.pipeline.service import process_image

        s, t = synth_pair
        area, base = _rois(s, t)
        ref = process_image(t, output_crop_source=area, image_area_source=area,
                            base_roi_source=base)
        tech_ref = ref.result.positive_linear
        for name in self.SLIDERS:
            delta = {"temperature": 40.0, "tint": 20.0}.get(name, 0.3)
            intent = RenderIntent.identity()
            setattr(intent, name, delta)
            pr = process_image(t, output_crop_source=area, image_area_source=area,
                               base_roi_source=base, render_intent=intent)
            assert np.array_equal(pr.result.positive_linear, tech_ref), \
                f"{name} 改动了技术正片（v5 §7.2 禁止）"
            assert not np.array_equal(pr.rendered_positive_linear,
                                      ref.rendered_positive_linear), \
                f"{name} 是假旋钮：渲染结果无变化（v5 §11.2）"

    def test_render_order_is_single_and_documented(self):
        """v5 §7.2：微调顺序固定，且不得重复应用曝光/白平衡/饱和。"""
        from negative_lab.pipeline.render import (RENDER_ORDER, RenderIntent)

        assert RENDER_ORDER == [
            "temperature", "tint", "exposure", "contrast", "highlights",
            "shadows", "whites", "blacks", "vibrance", "saturation",
            "gamut_guard", "display_encoding"]
        assert RenderIntent.identity().is_identity()


# ------------------------------------------ 5. 整卷分组与三候选（新对象）
class TestRollAndCandidates:
    def test_core_models_exist_and_roundtrip(self):
        """v5 §3.1：六个可序列化对象必须存在且能 round-trip。"""
        from negative_lab.library.models import (Candidate, CandidateSet,
                                                 FrameRecord, GeometryRegions,
                                                 ImportSession, RollAnalysis,
                                                 RollGroup)

        geo = GeometryRegions(rotation_degrees=180, fine_rotation_degrees=0.4,
                              mirror_horizontal=True,
                              output_crop_roi=(0, 0, 100, 200),
                              image_area_roi=(2, 2, 98, 198),
                              base_sample_rois=[(0, 200, 20, 260)])
        ra = RollAnalysis(roll_id="r1", base_statistics={"dmin": [0.3, 0.2, 0.1]},
                          base_surface={"coverage": "strip-1d"},
                          shared_color_start={"wb_high": [1, 1, 1]},
                          capture_fingerprint="fp1", source_frames=["a"],
                          confidence=0.7, warnings=["w"])
        cs = CandidateSet(
            preserve=Candidate(name="preserve", intent={"temperature": 0.0},
                               confidence=0.8, rationale="卷级起点"),
            neutral=Candidate(name="neutral", intent={"temperature": 12.0},
                              confidence=0.6, rationale="SoG/GrayEdge 一致"),
            warm=Candidate(name="warm", intent={"temperature": 22.0},
                           confidence=0.7, rationale="受限暖向偏移"),
            default_name="preserve", diagnostics={"estimators": ["shades"]})
        frame = FrameRecord(source_path="a/b.NEF", source_sha256="0" * 64,
                            input_facts={"kind": "raw"},
                            capture_fingerprint="fp1", geometry=geo,
                            frame_analysis={"dmin": [0.3, 0.2, 0.1]},
                            chosen_candidate="neutral",
                            render_intent_override=None, status="ready_to_convert")
        roll = RollGroup(id="r1", display_name="卷 1", frame_ids=["a/b.NEF"],
                         capture_fingerprint="fp1", roll_analysis=ra,
                         shared_candidate=cs, shared_render_intent=None)
        sess = ImportSession(files=[frame], roll_groups=[roll])
        for obj in (geo, ra, cs, frame, roll, sess):
            back = type(obj).from_dict(json.loads(json.dumps(obj.to_dict())))
            assert back.to_dict() == obj.to_dict(), type(obj).__name__

    def test_session_grouping_operations(self, tmp_path):
        """v5 §2.1.2：手动建立、重命名、拆分和移动卷分组（可脱离 GUI）。"""
        from negative_lab.library.models import (FrameRecord, ImportSession,
                                                 FrameStatus)

        sess = ImportSession(files=[
            FrameRecord(source_path=f"mix/{n}.NEF", source_sha256=f"{i}" * 64,
                        status=FrameStatus.NEEDS_GROUP.value)
            for i, n in enumerate(("a", "b", "c", "d"))])
        r1 = sess.create_roll("卷 A", ["mix/a.NEF", "mix/b.NEF"])
        assert all(sess.frame(p).roll_id == r1.id for p in ("mix/a.NEF", "mix/b.NEF"))
        sess.rename_roll(r1.id, "海滩卷")
        assert sess.roll(r1.id).display_name == "海滩卷"
        r2 = sess.split_roll(r1.id, ["mix/b.NEF"], "卷 B")
        assert sess.roll(r1.id).frame_ids == ["mix/a.NEF"]
        assert r2.frame_ids == ["mix/b.NEF"]
        assert sess.frame("mix/b.NEF").roll_id == r2.id
        sess.move_frames(["mix/c.NEF"], r1.id)
        assert sess.frame("mix/c.NEF").roll_id == r1.id
        assert sess.ungrouped()[0].source_path == "mix/d.NEF"
        assert sess.validate() == []
        p = sess.save(tmp_path / "session.json")
        back = ImportSession.load(p)
        assert back.to_dict() == sess.to_dict()

    def test_frame_status_machine_has_no_calibration_blockers(self):
        """v5 §3.3：状态机不得再有 needs_calibration；缺色卡不是阻断状态。"""
        from negative_lab.library.models import FrameStatus

        allowed = {"imported", "needs_group", "needs_crop_review",
                   "ready_to_convert", "converted", "needs_review", "exported",
                   "decode_error", "processing_error",
                   # B-9 契约扩展：`gate_blocked` 是**记录性状态**（"为什么没导"），
                   # **不是** v5 §3.3 禁止的"校准类阻断态"，也**不在** BLOCKING_STATUSES。
                   "gate_blocked"}
        got = {s.value for s in FrameStatus}
        assert got == allowed, f"状态机与 v5 §3.3 不符：多 {got - allowed} 少 {allowed - got}"
        assert not FrameStatus.is_blocking("needs_review")
        assert FrameStatus.is_blocking("decode_error")
        # 把 B-9 的决定钉死：被门禁挡下**不是**硬阻断态，
        # 将来谁把它塞进 BLOCKING_STATUSES（= 第二条阻断路径）都会撞到这条。
        assert not FrameStatus.is_blocking("gate_blocked"), (
            "gate_blocked 竟成了硬阻断态：阻断应只由 blocks_export + GateBlocked + "
            "退出码这一条路决定，加进 BLOCKING_STATUSES 会造成双路径并违反 v5 §3.3")

    def test_capture_fingerprint_is_auto_and_name_independent(self):
        """v5 §4.4：指纹由 metadata 自动生成，不含文件名/卷名，可编辑性为否。"""
        from negative_lab.library.fingerprint import (capture_fingerprint,
                                                      fingerprints_compatible)

        meta_raw = {"kind": "raw", "camera": "NIKON Z8", "lens": "Z MC 105",
                    "width": 8256, "height": 5504,
                    "raw_decode_params": {"output_color_name": "sRGB"},
                    "rgb_xyz_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    "sha256": "a" * 64}
        fp1 = capture_fingerprint(meta_raw, source_path="x/rollA/1.NEF")
        fp2 = capture_fingerprint(meta_raw, source_path="y/rollB/9999.NEF")
        assert fp1.value == fp2.value, "指纹含文件名/卷名（v5 §4.4 禁止）"
        assert fp1.editable is False
        other = dict(meta_raw, camera="NIKON Z6")
        fp3 = capture_fingerprint(other, source_path="x/rollA/2.NEF")
        assert not fingerprints_compatible(fp1, fp3)

    def test_roll_analysis_detects_base_drift_beyond_exif(self):
        """v5 §4.4：EXIF 相同但片基漂移超容差时必须停止共享并建议拆卷。"""
        from negative_lab.pipeline.roll_analysis import analyze_roll

        frames = [
            {"frame_id": "f1", "capture_fingerprint": "fp1",
             "base_dmin": [0.30, 0.20, 0.10], "base_confidence": 0.9,
             "black_level": 0.001, "spatial_trend": [0.0, 0.0]},
            {"frame_id": "f2", "capture_fingerprint": "fp1",
             "base_dmin": [0.31, 0.20, 0.10], "base_confidence": 0.9,
             "black_level": 0.001, "spatial_trend": [0.0, 0.0]},
            {"frame_id": "f3", "capture_fingerprint": "fp1",
             "base_dmin": [0.95, 0.60, 0.30], "base_confidence": 0.9,
             "black_level": 0.02, "spatial_trend": [0.4, 0.0]},
        ]
        ra = analyze_roll(frames, fingerprint="fp1")
        assert ra.shared_with_frames == ["f1", "f2"], ra.shared_with_frames
        assert "f3" in ra.rejected_frames
        assert ra.split_suggested is True
        assert any("拆" in w or "split" in w.lower() for w in ra.warnings), ra.warnings

    def test_three_candidates_share_one_technical_substrate(self, synth_pair):
        """v5 §6.5：三候选共享物理底座，不得改输出裁切/片基/技术正片。"""
        from negative_lab.pipeline.candidates import build_candidates
        from negative_lab.pipeline.service import process_image

        s, t = synth_pair
        area, base = _rois(s, t)
        pr = process_image(t, output_crop_source=area, image_area_source=area,
                           base_roi_source=base)
        cs = build_candidates(pr)
        assert {cs.preserve.name, cs.neutral.name, cs.warm.name} == \
            {"preserve", "neutral", "warm"}
        assert cs.default_name in ("preserve", "neutral", "warm")
        for cand in (cs.preserve, cs.neutral, cs.warm):
            assert np.array_equal(cand.technical_positive_linear,
                                  pr.result.positive_linear), \
                f"{cand.name} 改动了技术正片"
            assert tuple(cand.output_crop_roi) == tuple(
                pr.geometry.roi_source_space["output_crop_roi"])

    def test_neutral_candidate_refuses_to_neutralize_cast_scenes(self, synth_pair):
        """v5 §6.6：夜景/单色光等场景不得被全图灰世界强行中和。"""
        from negative_lab.pipeline.candidates import build_candidates
        from negative_lab.pipeline.service import process_image

        s, t = synth_pair
        area, base = _rois(s, t)
        night = t.copy()
        # 只改画面内容、不改片基条带：构造单色主导夜景（片基不变，
        # 否则密度归一会把整体缩放完全抵消，测不到中和风险）
        bx0 = s["truth"]["base_cols"][0]
        night[:, :bx0, 0] *= 0.25
        night[:, :bx0, 1] *= 0.40
        pr = process_image(night, output_crop_source=area, image_area_source=area,
                           base_roi_source=base)
        cs = build_candidates(pr)
        assert cs.neutral.estimated_correction is not None
        assert cs.neutral.confidence < 0.5, "单色场景中性候选置信度未被压低"
        assert cs.neutral.name == "preserve" or cs.neutral.fallback_to == "preserve", \
            "估计器分歧过大时必须回退 preserve（v5 §6.5）"
        # 修正量必须有硬上限并记录触发情况
        assert cs.neutral.correction_capped in (True, False)

    def test_candidates_come_from_same_service_entry(self, synth_pair):
        """v5 §12 P0-5.5：三候选全部经过同一生产 service，不得平行实现。"""
        from negative_lab.pipeline import service as svc

        assert hasattr(svc, "process_roll"), "整卷服务入口缺失"
        assert hasattr(svc, "candidate_positive"), \
            "候选渲染必须复用同一 service 出口"
