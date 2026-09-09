"""R2 单点中性救援算法（去色罩算法研究第二轮）。

锁定契约：
- neutral_pick_gain：给定渲染正片 + 拾取点（"这里应该是中性灰"），
  返回把该点拉向中性的线性 RGB 对角增益（灰世界假设的单点形式）；
- pick_to_intent：增益 -> RenderIntent(temperature, tint)（风格只在
  RenderIntent，绝不改 InvertParams——技术契约）；
- 合成验证：对注入已知色偏的图施加救援增益后，拾点色偏显著下降；
- 增益钳制与不可信输入（过暗/过亮/非有限点）的安全行为。
"""

from __future__ import annotations

import numpy as np
import pytest


def _gray_world_cast_img(h: int = 96, w: int = 96, cast=(0.10, 0.0, -0.06),
                         seed: int = 5) -> np.ndarray:
    """带已知均匀色偏的"场景"图：多数像素接近中性 + 纹理 + 色偏。"""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    scene = 0.45 + 0.10 * np.sin(xx / 9.0) * np.cos(yy / 7.0)
    base = np.stack([scene * 0.98, scene, scene * 1.02], axis=-1)
    return np.clip(base + np.array(cast, np.float32)
                   + rng.normal(0, 0.004, base.shape).astype(np.float32),
                   0, 1).astype(np.float32)


def _patch_median(img: np.ndarray, cy: int, cx: int, r: int = 3) -> np.ndarray:
    p = img[max(0, cy - r):cy + r + 1, max(0, cx - r):cx + r + 1]
    return np.median(p.reshape(-1, 3), axis=0)


class TestNeutralPickGain:
    def test_gain_pulls_point_to_neutral(self):
        from negative_lab.pipeline.rescue import neutral_pick_gain

        img = _gray_world_cast_img()
        pt = _patch_median(img, 48, 48)
        assert pt[0] > pt[2], "测试图应带暖/红偏"  # cast R+0.10 B-0.06
        gain = neutral_pick_gain(img, 48, 48)
        corrected = np.clip(img * gain[None, None, :], 0, 1)
        after = _patch_median(corrected, 48, 48)
        spread_before = float(pt.max() - pt.min())
        spread_after = float(after.max() - after.min())
        assert spread_after < 0.3 * spread_before, (pt, after)

    def test_gain_is_diagonal_and_positive(self):
        from negative_lab.pipeline.rescue import neutral_pick_gain

        gain = neutral_pick_gain(_gray_world_cast_img(), 48, 48)
        assert gain.shape == (3,)
        assert np.isfinite(gain).all() and (gain > 0).all()
        # 已中性的点 -> 单位增益（不乱动）
        neutral = np.full((16, 16, 3), 0.5, np.float32)
        assert np.allclose(neutral_pick_gain(neutral, 8, 8), [1, 1, 1],
                           atol=0.02)

    def test_gain_clamped(self):
        """拾到近黑区域时增益会爆——必须被上限钳制并给出警告
        （补丁中位数抗单像素噪声，所以暗区必须成片）。"""
        from negative_lab.pipeline.rescue import neutral_pick_gain

        img = np.full((32, 32, 3), 0.5, np.float32)
        img[14:19, 14:19] = (0.02, 0.002, 0.004)    # 成片极暗且强偏
        gain, warns = neutral_pick_gain(img, 16, 16, return_warnings=True)
        assert np.isfinite(gain).all() and (gain <= 8.0).all()
        assert warns, "必须给出不可信拾点警告"

    def test_out_of_range_point_rejected(self):
        from negative_lab.pipeline.rescue import neutral_pick_gain

        with pytest.raises(ValueError):
            neutral_pick_gain(_gray_world_cast_img(), -5, 5000)


class TestPickToIntent:
    def test_intent_matches_gain_direction(self):
        """红偏图拾点 -> 得到的 temp/tint 施加后拾点显著更中性。"""
        from negative_lab.pipeline.render import RenderIntent, apply_render
        from negative_lab.pipeline.rescue import (neutral_pick_gain,
                                                  neutral_pick_intent)

        img = _gray_world_cast_img()
        gain = neutral_pick_gain(img, 48, 48)
        assert gain[0] < gain[2], "红偏图的修正增益应压 R 抬 B"
        intent = neutral_pick_intent(img, 48, 48)
        assert isinstance(intent, RenderIntent)
        assert intent.temperature != 0.0 or intent.tint != 0.0
        # Render 施加后拾点应被拉近中性
        out = apply_render(img, intent)
        before = _patch_median(img, 48, 48)
        after = _patch_median(np.asarray(out), 48, 48)
        def spread(c):
            return float(c.max() - c.min())
        assert spread(after) < 0.3 * spread(before), (before, after)

    def test_identity_scene_identity_intent(self):
        from negative_lab.pipeline.render import apply_render
        from negative_lab.pipeline.rescue import neutral_pick_intent

        neutral = np.full((32, 32, 3), 0.5, np.float32)
        intent = neutral_pick_intent(neutral, 16, 16)
        out = apply_render(neutral, intent)
        assert np.allclose(out, neutral, atol=1e-6), \
            "已中性画面救援必须保持恒等（不硬给修正）"

    def test_rescue_roundtrip_on_cast(self):
        """端到端：强蓝偏图 -> 拾点 -> intent -> 施加后色偏收敛。"""
        from negative_lab.pipeline.render import apply_render
        from negative_lab.pipeline.rescue import neutral_pick_intent

        img = _gray_world_cast_img(cast=(-0.05, 0.02, 0.12), seed=11)
        intent = neutral_pick_intent(img, 48, 48)
        out = np.asarray(apply_render(img, intent))
        b0 = _patch_median(img, 48, 48)
        b1 = _patch_median(out, 48, 48)
        assert abs(b1.max() - b1.min()) < 0.25 * abs(b0.max() - b0.min()), \
            (b0, b1)

    def test_intent_stacks_on_base(self):
        """已有色温/色调时增量叠加（组合语义）。"""
        from negative_lab.pipeline.rescue import neutral_pick_intent

        img = _gray_world_cast_img()
        it0 = neutral_pick_intent(img, 48, 48)
        it1 = neutral_pick_intent(img, 48, 48,
                                  base_temperature=it0.temperature,
                                  base_tint=it0.tint)
        assert abs(it1.temperature) >= abs(it0.temperature) - 1e-6

    def test_implausible_pick_warns(self):
        """拾到通道比严重失衡的点（如深色树叶）必须警告部分收敛。"""
        from negative_lab.pipeline.rescue import neutral_pick_delta

        img = np.full((48, 48, 3), 0.45, np.float32)
        img[20:28, 20:28] = (0.03, 0.42, 0.55)   # R 极暗的"树叶"块
        warns = neutral_pick_delta(img, 24, 24, return_warnings=True)[2]
        assert any("不是中性" in w for w in warns), warns


class TestExactRescue:
    """R5a：含影调/彩度控件时的精确救援（对技术正片补丁逐点渲染求解）。"""

    def test_exact_converges_with_tone_stack(self):
        """base intent 带对比度/阴影时，绝对求解仍把拾点拉平。"""
        from negative_lab.pipeline.render import (RenderIntent, apply_render)
        from negative_lab.pipeline.rescue import neutral_pick_intent_exact

        img = _gray_world_cast_img(cast=(-0.04, 0.01, 0.10), seed=21)
        base = RenderIntent(contrast=25.0, shadows=-15.0)
        intent, warns = neutral_pick_intent_exact(
            img, 48, 48, base_intent=base, return_warnings=True)
        out = np.asarray(apply_render(img, intent))
        b0 = _patch_median(img, 48, 48)
        b1 = _patch_median(out, 48, 48)
        assert abs(b1.max() - b1.min()) < 0.25 * abs(b0.max() - b0.min()), \
            (b0, b1, intent.temperature, intent.tint)

    def test_exact_absorbs_base_temp_tint(self):
        """base 已有色温/色调时，绝对求解给出合成后的总值（不是增量）。"""
        from negative_lab.pipeline.render import (RenderIntent, apply_render)
        from negative_lab.pipeline.rescue import neutral_pick_intent_exact

        img = _gray_world_cast_img(seed=23)
        base = RenderIntent(temperature=-40.0, tint=-30.0)   # 故意偏冷/偏洋红
        intent = neutral_pick_intent_exact(img, 48, 48, base_intent=base)
        out = np.asarray(apply_render(img, intent))
        b1 = _patch_median(out, 48, 48)
        assert abs(b1.max() - b1.min()) < 0.05, \
            f"绝对求解必须吸收 base 的 temp/tint：{b1}"
        assert intent.temperature != base.temperature or \
            intent.tint != base.tint

    def test_exact_never_touches_technical(self):
        """救援只返回 RenderIntent：输入数组必须逐比特不变（技术契约）。"""
        from negative_lab.pipeline.render import RenderIntent
        from negative_lab.pipeline.rescue import neutral_pick_intent_exact

        img = _gray_world_cast_img(seed=29)
        before = img.copy()
        neutral_pick_intent_exact(img, 48, 48,
                                  base_intent=RenderIntent(contrast=10.0))
        assert np.array_equal(img, before), "技术正片/输入图像被修改"

    def test_exact_clipped_pick_warns(self):
        """拾点已被剪裁（通道=1.0）必须警告信息丢失。"""
        from negative_lab.pipeline.render import RenderIntent
        from negative_lab.pipeline.rescue import neutral_pick_intent_exact

        img = _gray_world_cast_img(seed=31)
        img[44:53, 44:53] = (0.9, 0.97, 1.0)     # B 通道已到 1.0
        _, warns = neutral_pick_intent_exact(
            img, 48, 48, base_intent=RenderIntent(), return_warnings=True)
        assert any("剪裁" in w for w in warns), warns

    def test_exact_identity_scene_untouched(self):
        from negative_lab.pipeline.render import apply_render
        from negative_lab.pipeline.rescue import neutral_pick_intent_exact

        neutral = np.full((32, 32, 3), 0.5, np.float32)
        intent = neutral_pick_intent_exact(neutral, 16, 16)
        out = apply_render(neutral, intent)
        assert np.allclose(out, neutral, atol=1e-6)


def _neutral_scene(h: int = 120, w: int = 160, seed: int = 7) -> np.ndarray:
    """中性参考图：亮度渐变 + 轻微通道差 + 噪声（模拟「颜色本来是对的」）。"""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    scene = 0.30 + 0.35 * ((xx / w) * 0.6 + (yy / h) * 0.4)
    base = np.stack([scene * 0.99, scene, scene * 1.01], axis=-1)
    return np.clip(base + rng.normal(0, 0.004, base.shape).astype(np.float32),
                   0, 1).astype(np.float32)


def _median_abs_err(img: np.ndarray, ref: np.ndarray) -> float:
    """整幅图相对参考的中位绝对误差（排除被裁到黑/白的像素）。"""
    m = (np.nanmin(img, axis=-1) > 0.02) & (np.nanmax(ref, axis=-1) < 0.98)
    if not m.any():
        return float("nan")
    d = img[m].astype(np.float64) - ref[m].astype(np.float64)
    return float(np.median(np.abs(d)))


def _whole_channel_spread(img: np.ndarray) -> float:
    """整幅图中位色的通道极差（log10）。"""
    m = np.median(img.reshape(-1, 3), axis=0).astype(np.float64)
    return float(np.log10(max(m.max(), 1e-9) / max(m.min(), 1e-9)))


class TestWholeImageRecovery:
    """审查 §3.5：只量「被求解的那个 patch」的残差是自证度量。

    求解目标就是让拾点 patch 的通道极差最小，再在同一 patch 上量极差必然
    变小——它只能证明求解器到达族内最优，证明不了整图变好。这里改成整幅图
    对照「注入前」的参考，同时验证救援没有把中性以外的区域带歪。
    """

    def test_rescue_improves_whole_image_against_reference(self):
        from negative_lab.pipeline.render import RenderIntent, apply_render
        from negative_lab.pipeline.rescue import neutral_pick_intent_exact

        ref = _neutral_scene()
        cast = np.array([1.00, 0.90, 1.15], np.float32)     # 已知对角增益翻车
        bad = np.clip(ref * cast[None, None, :], 0, 1)

        intent, _w = neutral_pick_intent_exact(
            bad, 60, 80, base_intent=RenderIntent(), return_warnings=True)
        fixed = np.asarray(apply_render(bad, intent), np.float32)

        e_before = _median_abs_err(bad, ref)
        e_after = _median_abs_err(fixed, ref)
        assert np.isfinite(e_before) and np.isfinite(e_after)
        assert e_after < e_before * 0.5, (
            f"整图误差没有明显下降：{e_before:.4f} -> {e_after:.4f}")
        assert (_whole_channel_spread(fixed) < _whole_channel_spread(bad)), \
            "整幅图的色偏没有减小"
        assert e_after < 0.02, f"救援后整图仍偏离参考 {e_after:.4f}"

    def test_patch_metric_is_not_a_sufficient_claim(self):
        """patch 与整图两种度量同时看：patch 归零是定义使然，整图才是结论。"""
        from negative_lab.pipeline.render import RenderIntent, apply_render
        from negative_lab.pipeline.rescue import neutral_pick_intent_exact

        ref = _neutral_scene(seed=13)
        bad = np.clip(ref * np.array([1.05, 0.93, 1.12],
                                     np.float32)[None, None, :], 0, 1)
        intent, _w = neutral_pick_intent_exact(
            bad, 60, 80, base_intent=RenderIntent(), return_warnings=True)
        fixed = np.asarray(apply_render(bad, intent), np.float32)
        pt_before = _patch_median(bad, 60, 80)
        pt_after = _patch_median(fixed, 60, 80)
        assert float(pt_after.max() - pt_after.min()) < float(
            pt_before.max() - pt_before.min())
        assert _median_abs_err(fixed, ref) < _median_abs_err(bad, ref), (
            "patch 变中性但整图变差：救援在自我证明")
