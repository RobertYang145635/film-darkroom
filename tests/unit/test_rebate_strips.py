"""扫描片边检测 `detect_rebate_strips` 的锁（㊵，ECN-2 逼出来的）。

三道闸各配一个**反例**，缺任何一道都会把不该当片基的东西当片基：
① 黑边框（5219 的四条边实测 t≈0.0003）；
② 扫描仪白边（5207 的右边实测 [0.86, 0.94, 1.00]）；
③ 画面内容比"候选片基"还亮（物理上说明这个候选太低）。
正例用 5207 上/下/左三条边实测到的 t 中位 [0.80, 0.45, 0.28] 造。
"""
from __future__ import annotations

import numpy as np

from negative_lab.calibration.border import detect_rebate_strips

BASE = np.array([0.80, 0.46, 0.30], np.float32)      # 5207 实测片基（R/B≈2.67）
H, W = 400, 300


def _scene(rng=7):
    """画面中心：整体比片基暗（负片里比片基亮是不可能的）。"""
    rs = np.random.RandomState(rng)
    a = np.clip(BASE[None, None, :] * rs.uniform(0.15, 0.75, (H, W, 3)).astype(np.float32),
                0, 1)
    return a


def _with_band(a, side):
    bh = max(4, int(H * 0.012 * 3))
    bw = max(4, int(W * 0.012 * 3))
    if side == "上":
        a[:bh] = BASE
    elif side == "下":
        a[-bh:] = BASE
    elif side == "左":
        a[:, :bw] = BASE
    else:
        a[:, -bw:] = BASE
    return a


def test_a_real_orange_rebate_is_found_and_is_the_right_band():
    rois, info = detect_rebate_strips(_with_band(_scene(), "上"))
    assert rois, "真有未曝光片边却没找到 ⇒ 这条路对扫描输入失效"
    assert info[0][0] == "上", info
    y0, x0, y1, x1 = rois[0]
    assert y0 == 0 and y1 <= H // 8, (y0, y1)


def test_black_frame_border_is_rejected():
    """5219 的实况：四条边是死黑，不能当片基（用它归一后 98% 像素"超基"）。"""
    a = _scene()
    a[:8] = 0.0004
    a[-8:] = 0.0004
    a[:, :8] = 0.0004
    a[:, -8:] = 0.0004
    rois, info = detect_rebate_strips(a)
    assert rois == [] and info == [], f"黑边框被当成了片基：{info}"


def test_scanner_white_overscan_is_rejected():
    """5207 右边的实况：[0.86, 0.94, 1.00] 是扫描仪白边，不是胶片橙罩。"""
    a = _with_band(_scene(), "右")
    a[:, -12:] = np.array([0.86, 0.94, 1.00], np.float32)
    rois, _info = detect_rebate_strips(a)
    for (y0, x0, y1, x1) in rois:
        assert x1 < W - 4, f"白边被当成片基取样条：{(y0, x0, y1, x1)}"


def test_band_darker_than_scene_is_rejected():
    """闸③：候选片基必须真的比画面亮，否则它就是画面里的某块暗东西。"""
    a = _scene()
    a[:12] = BASE * 0.5                       # "边带"比画面还暗
    a[H // 3: 2 * H // 3, W // 3: 2 * W // 3] = BASE   # 画面里有比它亮的
    rois, _info = detect_rebate_strips(a)
    assert rois == [], f"比画面还暗的候选被当成了片基：{rois}"


def test_no_false_positive_on_a_clean_recapture():
    """翻拍输入（G200/CN400 那类）最外圈是灯箱/黑背景 ⇒ 必须一条都不给，
    让调用方照旧走原检测，而不是凭空造一个片基。"""
    a = _scene()
    a[:20] = 0.02
    a[-20:] = 0.02
    a[:, :20] = 0.02
    a[:, -20:] = 0.02
    rois, info = detect_rebate_strips(a)
    assert rois == [] and info == []
