"""v3 P0-D：可插拔 backend 与密度仿射解混测试。"""

from __future__ import annotations

import numpy as np
import pytest

from negative_lab.calibration.base import robust_dmin
from negative_lab.pipeline.backends import (density_affine,
                                            fit_decrosstalk_supervised,
                                            get_backend, power_reciprocal)
from negative_lab.pipeline.density import transmittance_to_z
from negative_lab.pipeline.params import InvertParams
from negative_lab.testing.spectral import make_spectral_negative


@pytest.fixture(scope="module")
def xt():
    s = make_spectral_negative(seed=31, crosstalk=True)
    t = s["negative"]
    bx0, bx1 = s["truth"]["base_cols"]
    dmin = np.array(robust_dmin(t[:, bx0:bx1])["dmin"], np.float32)
    return s, t, dmin, bx0


def _neutral_z(t, dmin, bx0):
    z = transmittance_to_z(t[:, :bx0], dmin)
    zl = z.mean(axis=2)
    px = z[zl > -2.6]           # 排除死密度区
    rng = np.random.default_rng(3)
    idx = rng.choice(len(px), size=min(400, len(px)), replace=False)
    return px[idx]


def _chroma(pos, s, name="red"):
    rs, cs = s["truth"]["patch_roi"][name]
    m = pos[rs, cs, :].reshape(-1, 3).mean(axis=0)
    return float((m.max() - m.min()) / max(m.mean(), 1e-6))


class TestBackends:
    def test_identity_equals_power_reciprocal(self, xt):
        s, t, dmin, bx0 = xt
        p = InvertParams(dmin=tuple(float(v) for v in dmin), dmax=1.6)
        a = power_reciprocal(t[:, :bx0], p)
        b = density_affine(t[:, :bx0], p)
        assert np.allclose(a, b, atol=1e-6)

    def test_get_backend_unknown_raises(self):
        with pytest.raises(ValueError):
            get_backend("quantum-magic")

    def test_fit_insufficient_samples_stays_identity(self):
        out = fit_decrosstalk_supervised(
            np.random.default_rng(1).normal(size=(2, 3)),
            np.random.default_rng(1).normal(size=(2, 3)))
        assert np.allclose(np.array(out["matrix"]).reshape(3, 3), np.eye(3))
        assert out["confidence"] == 0.0

    def test_fit_single_density_stays_identity(self):
        z = np.tile(np.array([-1.0, -0.9, -1.1]), (50, 1))
        out = fit_decrosstalk_supervised(z, z)   # 秩不足（单射线）
        assert np.allclose(np.array(out["matrix"]).reshape(3, 3), np.eye(3),
                           atol=1e-6)

    def test_neutral_only_fit_is_unidentifiable_and_honest(self, xt):
        """v3 F-ALG-5：纯中性样本只张成一维射线 -> 必须返回单位阵+警告，
        绝不假装解混成功。"""
        s, t, dmin, bx0 = xt
        # 只取灰阶斜坡列（真中性，单射线）
        z = transmittance_to_z(t[:, 10:300], dmin)
        Z = z.reshape(-1, 3)[::37]
        fit = fit_decrosstalk_supervised(Z, Z)   # 中性目标=自身，秩仍不足
        assert np.allclose(np.array(fit["matrix"]).reshape(3, 3), np.eye(3))
        assert fit["confidence"] == 0.0
        assert fit["warnings"], fit

    def test_supervised_fit_recovers_crosstalk_matrix(self):
        """监督拟合（色卡等价物：已知目标密度）必须恢复串扰逆。"""
        C = np.array([[0.92, 0.09, 0.05],
                      [0.06, 0.90, 0.07],
                      [0.04, 0.08, 0.94]])
        rng = np.random.default_rng(5)
        D = rng.uniform(0.1, 2.4, size=(40, 3))          # 已知染料密度
        Z = -(D @ C.T)                                    # 测量 z（生成器同一关系）
        fit = fit_decrosstalk_supervised(Z, -D)           # 目标 = -真密度
        A = np.array(fit["matrix"]).reshape(3, 3)
        assert fit["confidence"] == 1.0
        rec = Z @ A.T                                     # 应回到 -D
        err = np.abs(rec + D).max(axis=None) / np.abs(D).max()
        assert err < 0.05, err
        # 应用到 density_affine：串扰数据的红块彩度回升
        s = make_spectral_negative(seed=31, crosstalk=True)
        t = s["negative"]
        bx0, bx1 = s["truth"]["base_cols"]
        dmin = np.array(robust_dmin(t[:, bx0:bx1])["dmin"], np.float32)
        p = InvertParams(dmin=tuple(float(v) for v in dmin), dmax=1.6,
                         density_matrix=tuple(float(v) for v in A.ravel().astype(np.float32)))
        pos_aff = density_affine(t[:, :bx0], p)
        pos_raw = power_reciprocal(t[:, :bx0],
                                   InvertParams(dmin=tuple(float(v) for v in dmin),
                                                dmax=1.6))
        assert _chroma(pos_aff, s) > _chroma(pos_raw, s) * 1.15

    def test_supervised_rank_deficient_stays_identity(self):
        Z = np.tile(np.array([-1.0, -0.9, -1.1]), (20, 1))
        T = np.tile(np.array([-1.0, -1.0, -1.0]), (20, 1))
        fit = fit_decrosstalk_supervised(Z, T)
        assert np.allclose(np.array(fit["matrix"]).reshape(3, 3), np.eye(3))
        assert fit["confidence"] == 0.0
