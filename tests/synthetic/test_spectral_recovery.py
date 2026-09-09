"""v3 P0-B：非循环合成色度测试 + 必败组。

色度数学测试显式传入真实片基 ROI（检测器行为由 test_border_detect 单独
覆盖，避免把两个失败模式混在一起）。串扰组记录对角模型的能力边界，
作为 P0-D 密度矩阵 backend 的动机证据（v3 F-ALG-6）。
"""

from __future__ import annotations

import numpy as np
import pytest

from negative_lab.pipeline.auto import auto_analyze
from negative_lab.pipeline.density import invert
from negative_lab.testing.spectral import PATCH_ORDER, hue_angle, make_spectral_negative


@pytest.fixture(scope="module")
def spec():
    return make_spectral_negative(seed=11, crosstalk=False)


@pytest.fixture(scope="module")
def spec_xt():
    return make_spectral_negative(seed=11, crosstalk=True)


def _base_roi(s):
    t = s["negative"]
    bx0, bx1 = s["truth"]["base_cols"]
    return t[:, bx0:bx1, :]


def _neutral_points(s):
    """v3 P0-E 层级：verified 中性点驱动指数/白平衡（不靠猜测）。"""
    roi = s["truth"]["patch_roi"]
    def center(name):
        rs, cs = roi[name]
        return (int((rs.start + rs.stop) / 2), int((cs.start + cs.stop) / 2))
    return [center("gray65")], [center("gray18")]


def _positive(s):
    t = s["negative"]
    nb, nd = _neutral_points(s)
    rep = auto_analyze(t, base_roi=_base_roi(s),
                       neutral_bright_points=nb, neutral_dark_points=nd)
    res = invert(t, rep.params)
    return res.positive_linear, rep


def _patch_mean(pos, s, name):
    rs, cs = s["truth"]["patch_roi"][name]
    return pos[rs, cs, :].reshape(-1, 3).mean(axis=0)


class TestChroma:
    def test_hue_order_preserved(self, spec):
        """红→橙→黄→绿→青→蓝→紫 的色相次序必须保留（v3 §6.1.6）。"""
        pos, _ = _positive(spec)
        hues = [float(hue_angle(_patch_mean(pos, spec, n))) for n in PATCH_ORDER]
        # sRGB hue: red~0, orange~30, yellow~60, green~120, cyan~180, blue~240, purple~300
        sorted(range(len(hues)), key=lambda i: hues[i])
        # red 必须与 green/blue 分离；次序沿色环单调（允许 ±1 邻居交换容忍度=0）
        assert hues[0] < 40 or hues[0] > 330, f"red hue 漂移: {hues}"
        assert 90 < hues[3] < 170, f"green hue 漂移: {hues}"
        assert 160 < hues[4] < 230, f"cyan hue 漂移: {hues}"
        assert 220 < hues[5] < 300, f"blue hue 漂移: {hues}"
        assert hues[6] > 260 or hues[6] < 20, f"purple hue 漂移: {hues}"
        # 沿色环次序单调（红->紫）
        assert hues == sorted(hues), f"色相次序破坏: {hues}"

    def test_chroma_not_collapsed(self, spec):
        """饱和色块彩度不得塌缩到灰（v3 §0 的失败模式）。"""
        pos, _ = _positive(spec)
        chroma = []
        for n in ("red", "yellow", "green", "blue"):
            m = _patch_mean(pos, spec, n)
            chroma.append(float((m.max() - m.min()) / max(m.mean(), 1e-6)))
        assert min(chroma) > 0.25, f"色度塌缩: {dict(zip(PATCH_ORDER, chroma))}"

    def test_neutral_no_hue_shift(self, spec):
        """灰块不得被染上颜色：gray 系列彩度低于饱和块。"""
        pos, _ = _positive(spec)
        gc = []
        for n in ("gray18", "gray35", "gray65"):
            m = _patch_mean(pos, spec, n)
            gc.append(float((m.max() - m.min()) / max(m.mean(), 1e-6)))
        sc = []
        for n in ("red", "blue"):
            m = _patch_mean(pos, spec, n)
            sc.append(float((m.max() - m.min()) / max(m.mean(), 1e-6)))
        assert max(gc) < min(sc), f"中性漂移: gray={gc} sat={sc}"


class TestRecovery:
    def test_dmin_recovers_base(self, spec):
        """Dmin 必须恢复真实片基（串扰不影响片基测量本身）。"""
        _, rep = _positive(spec)
        base = np.array(spec["truth"]["base"])
        got = np.array(rep.params.dmin)
        assert np.max(np.abs(got - base) / base) < 0.08, (got, base)

    def test_monotonic_in_density(self, spec):
        pos, _ = _positive(spec)
        a0, a1 = spec["truth"]["ramp_cols"]
        lum = pos.mean(axis=2)
        cols = np.arange(a0 + 10, a1 - 10, 8)
        prof = lum[:, cols].mean(axis=0)
        assert (np.diff(prof) > -1e-4).all()


class TestMustFailGroup:
    """v3 P0-B.3：错误实现必须被抓住。"""

    def test_channel_swap_detected(self, spec):
        """交换 R/B 通道 -> 红块色相必须跳出红区（证明测试有鉴别力；
        绿对 R/B 近似对称，不能用作鉴别样本）。"""
        s2 = make_spectral_negative(seed=11, crosstalk=False)
        t = s2["negative"][..., ::-1].copy()          # 交换通道
        bx0, bx1 = s2["truth"]["base_cols"]
        rep = auto_analyze(t, base_roi=t[:, bx0:bx1, :])
        pos = invert(t, rep.params).positive_linear
        rs, cs = s2["truth"]["patch_roi"]["red"]
        h_red = float(hue_angle(pos[rs, cs, :].reshape(-1, 3).mean(axis=0)
                                [None, None, :]).item())
        assert not (h_red < 40 or h_red > 330), \
            f"通道交换未被红块色相测试捕获（hue={h_red}）-> 测试无效"

    def test_scene_pixels_never_overwrite_dmin(self):
        """画面内注入比片基亮 30% 的漏光带：Dmin 仍必须等于片基条带测量。"""
        s2 = make_spectral_negative(seed=12, crosstalk=False)
        t = s2["negative"].copy()
        base = np.array(s2["truth"]["base"])
        t[:, 300:340, :] = np.clip(base * 1.3, 0, 1)   # 画面内部漏光带
        bx0, bx1 = s2["truth"]["base_cols"]
        rep = auto_analyze(t, base_roi=t[:, bx0:bx1, :])
        got = np.array(rep.params.dmin)
        assert np.all(got < base * 1.08), (got, base)

    def test_grayworld_suggestion_not_applied(self):
        """强色场景：统计 gray-world 只能建议，不得自动改写参数。"""
        s2 = make_spectral_negative(seed=13, crosstalk=False)
        t = s2["negative"]
        bx0, bx1 = s2["truth"]["base_cols"]
        # 只取红块区域（无中性物体）
        red_col = next(cs.start for n, (rs, cs) in s2["truth"]["patch_roi"].items()
                       if n == "red")
        t_red = t[:, max(0, red_col - 20):red_col + 40, :]
        rep = auto_analyze(t_red, base_roi=t[:, bx0:bx1, :][:, :t_red.shape[1]])
        assert rep.params.red_ratio == 1.0
        assert rep.params.blue_ratio == 1.0
        assert rep.params.wb_high == (1.0, 1.0, 1.0)
        assert any("建议" in w or "无中性" in w or "不足" in w
                   for w in rep.warnings), rep.warnings


class TestCrosstalkBoundary:
    """v3 F-ALG-6：对角模型对染料串扰的能力边界（P0-D 矩阵 backend 动机）。"""

    def test_red_blue_still_separable(self, spec_xt):
        pos, _ = _positive(spec_xt)
        mr = _patch_mean(pos, spec_xt, "red")
        mb = _patch_mean(pos, spec_xt, "blue")
        assert mr[0] > mr[2] and mb[2] > mb[0], "红蓝可分离性丧失"

    def test_crosstalk_degrades_saturation(self, spec, spec_xt):
        """串扰下饱和块彩度应低于无串扰——对角模型无法完全解混，
        这是 P0-D 密度矩阵 backend 的动机证据（不得掩盖）。"""
        def red_chroma(s):
            pos, _ = _positive(s)
            m = _patch_mean(pos, s, "red")
            return float((m.max() - m.min()) / max(m.mean(), 1e-6))
        c0, c1 = red_chroma(spec), red_chroma(spec_xt)
        assert c1 < c0, f"串扰组彩度未下降（{c0} vs {c1}）——前向模型或测试有问题"
