"""**平铺单帧扫描的取景锁**（ECN-2 真数据，㊶ 追加④）。

缺陷：`detect_image_area` 的条带结构模型在单帧平铺扫描上会在**画面内部**找到一条亮带，
把带外全裁掉。看图凭据 `outputs/ref_look_probe/ecn2_frame_boxes_*.png`：
5207-6 交付框只剩整幅的 35.8%（用户样片里的 LAWSON 招牌整个在框外），
5219-20/15 只剩 52.2%。与用户正片做尺度×位移互相关（`tmp_ecn2_rebate_frame.py`）：
不裁 s=1.05 |r|=0.90~0.98（八帧全中），现 `image_area` s=0.60~1.05 |r|=0.63~0.98。

每条锁都是**两臂同帧**：同一份取向后数组上，
  新分支的交付框 ≥0.90 整幅（产品路径 `process_roll`，不是单帧 `process_image`），
  旧结构检测器的 `image_area` <0.75 整幅（非空控制：证明这帧真的坏过）。
两臂只差"取景由谁决定"这一个变量。

⚠ 不断言 `roi_provenance["image_area"]`：`process_roll` 两遍处理，第二遍把第一遍
检测出的框种回 `rec.geometry` 再传进 `process_image`，成品记录里的来源因此写的是
`manual_regions`——自动检测的证据在第一遍就丢了（观测缺口，登记 B-16 追加④）。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from negative_lab.calibration.border import detect_regions
from negative_lab.io.images import load_linear_rgb
from negative_lab.library.fingerprint import capture_fingerprint
from negative_lab.library.models import FrameRecord, GeometryRegions
from negative_lab.pipeline.geometry import OrientTransform
from negative_lab.pipeline.service import process_roll

ROOT = Path(__file__).resolve().parents[2]
NEG = ROOT / "20260506_SAMPLE-ID" / "ECN2"
MAX_DIM = 1200
#: 缺陷最重的两帧（一帧四条边全无片基证据、一帧只有一条边有）；旋转/镜像取
#: `outputs/ecn2_alignment_table_v2.json` 的逐帧实测值（探针产物不入库，故按字面钉住；
#: 它同时是本锁的输入条件，方向账改动就要回来重看，见 B-16 追加）。
CASES = (("5207-6", 180, True), ("5219-20", 90, True))

pytestmark = pytest.mark.skipif(
    not all((NEG / f"{s}.tif").exists() for s, _r, _m in CASES),
    reason="缺 ECN-2 扫描（`20260506_SAMPLE-ID/ECN2/`，用户媒体不入库）")


def _oriented(stem, rot, mir):
    loaded = load_linear_rgb(NEG / f"{stem}.tif")
    arr = loaded.rgb
    step = max(1, int(np.ceil(max(arr.shape[:2]) / float(MAX_DIM))))
    small = arr[::step, ::step, :].copy()
    return small, OrientTransform(small.shape[:2], rot, mir).apply(small), loaded


@pytest.mark.parametrize("stem,rot,mir", CASES)
def test_flush_scan_delivers_the_whole_frame(stem, rot, mir):
    small, orie, loaded = _oriented(stem, rot, mir)
    oh, ow = orie.shape[:2]
    rel = f"20260506_SAMPLE-ID/ECN2/{stem}.tif"
    rec = FrameRecord(source_path=rel, source_sha256="0" * 64,
                      input_facts=dict(loaded.meta),
                      capture_fingerprint=capture_fingerprint(
                          loaded.meta, source_path=rel).value,
                      geometry=GeometryRegions(rotation_degrees=rot, mirror_horizontal=mir),
                      orientation_decided=True)
    _, results = process_roll([(rec, small.copy())])
    pr = results[0]
    #: `pr.t` 就是"输出裁切区工作图"——交付框有多大，看它，不看中间量
    delivered = pr.t.shape[0] * pr.t.shape[1] / float(oh * ow)
    assert delivered >= 0.90, f"交付框只占整幅 {delivered:.3f}，取景仍在吃画面"

    # 非空控制：同一帧上旧的结构检测器确实把它裁残了
    det = detect_regions(orie)
    ia = det["image_area"]
    old_frac = ((ia[2] - ia[0]) * (ia[3] - ia[1]) / float(oh * ow)) if ia else 0.0
    assert old_frac < 0.75, (
        f"旧检测器在这帧并不裁（{old_frac:.3f}）⇒ 上面那条锁保护的是不存在的缺陷，"
        f"回去重看取景数据，别改阈值")


@pytest.mark.parametrize("stem,rot,mir", CASES)
def test_turning_off_the_flush_gate_reproduces_the_defect(stem, rot, mir, monkeypatch):
    """因果控制：只把 `FRAME_FLUSH_FRAC` 抬到取不到（1.01），同一帧必须重新被裁残。

    这条与上一条只差**一个常数**，所以"交付框回到整幅"只能归因于新分支本身，
    不是缓存、不是别处顺手改的东西。
    """
    from negative_lab.calibration import border
    monkeypatch.setattr(border, "FRAME_FLUSH_FRAC", 1.01)
    small, orie, loaded = _oriented(stem, rot, mir)
    oh, ow = orie.shape[:2]
    rel = f"20260506_SAMPLE-ID/ECN2/{stem}.tif"
    rec = FrameRecord(source_path=rel, source_sha256="0" * 64,
                      input_facts=dict(loaded.meta),
                      capture_fingerprint=capture_fingerprint(
                          loaded.meta, source_path=rel).value,
                      geometry=GeometryRegions(rotation_degrees=rot, mirror_horizontal=mir),
                      orientation_decided=True)
    _, results = process_roll([(rec, small.copy())])
    pr = results[0]
    delivered = pr.t.shape[0] * pr.t.shape[1] / float(oh * ow)
    assert delivered < 0.75, f"关掉新分支后交付框仍占 {delivered:.3f} ⇒ 缺陷另有来源"
