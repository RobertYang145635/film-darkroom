"""锁：**密度域解混资产必须走通整卷路径**（B-13 追加①）。

被修的缺陷不是算法而是**可达性**：`process_image` 有 `model_asset` 形参，
`ModelAsset.apply_to()` 会把 3×3 矩阵与 backend 写进 `InvertParams`，
`fixtures/models/` 里还躺着一个训练好的资产——但生产主路径是 `process_roll`，
而它两遍调用 `process_image` 时**都不转发** `model_asset`。
于是"跨通道校正能力"在整卷里根本进不去，`apply_to` 与那个资产全是死路；
更糟的是默认路径连 `load_asset()` 都不调，直接 `deterministic_asset()`，
所以"我们评估过这个矩阵"这句话在修复前**不成立**（本轮才第一次量到，
结论是它让颜色更差：见 B-13 追加① 的 50.9°→90.4°）。

三件事必须钉住：
1. 给了启用的资产 ⇒ 每一帧的 `backend_name` 与 `density_matrix` 真的换掉了；
2. 负控制一：不给资产 ⇒ 保持 `power-reciprocal` + 单位阵（默认行为没被改坏）；
3. 负控制二：资产 `enabled=False` ⇒ 整卷路径必须**尊重这个开关**，
   不得"既然传进来了就应用"（那个训练资产正是靠这个开关留在 disabled）。

外加一条 AST 锁：`process_roll` 里**每一处** `process_image` 调用都必须转发
`model_asset`——两遍都要，否则第一遍与第二遍用不同 backend，卷级片基与帧结果对不上。
"""

from __future__ import annotations

import ast
import pathlib

import numpy as np

from negative_lab.library.models import FrameRecord, GeometryRegions
from negative_lab.pipeline.model_assets import ModelAsset, canonical_hash
from negative_lab.pipeline.service import process_roll

SERVICE = pathlib.Path(__file__).resolve().parents[2] / "src" / "negative_lab" \
    / "pipeline" / "service.py"

MATRIX = ((0.90, -0.05, -0.03), (-0.08, 0.95, -0.05), (-0.02, -0.10, 1.12))


def _neg(seed: int) -> np.ndarray:
    """合成负片：上方一条亮片边 + 下方随机场景（透射率都在 (0,1)）。"""
    rng = np.random.default_rng(seed)
    img = np.full((64, 48, 3), 0.30, np.float32)
    img[:6, :, :] = 0.55
    img[8:, :, :] *= (0.25 + 0.6 * rng.random((56, 48, 3))).astype(np.float32)
    return np.clip(img, 1e-3, 0.9)


def _roll_frames():
    recs, imgs = [], []
    for i in range(3):
        recs.append(FrameRecord(source_path=f"unit_roll::F{i}",
                                source_sha256="", input_facts={},
                                geometry=GeometryRegions()))
        imgs.append(_neg(1000 + i))
    return list(zip(recs, imgs))


def _asset(*, enabled: bool = True) -> ModelAsset:
    payload = {"density_matrix": [[float(v) for v in row] for row in MATRIX],
               "channel_curves": [None, None, None],
               "red_ratio": None, "blue_ratio": None,
               "reference_exponent": None, "dmax": None}
    return ModelAsset(asset_version="unit-supervised", payload=payload,
                      sha256=canonical_hash(payload), enabled=enabled,
                      meta={"kind": "unit"})


def test_enabled_asset_reaches_every_frame_of_a_roll():
    _, results = process_roll(_roll_frames(), model_asset=_asset())
    assert len(results) == 3
    want = tuple(float(v) for row in MATRIX for v in row)
    for i, pr in enumerate(results):
        assert pr.backend_name == "density-affine", (
            f"第 {i} 帧的 backend 没换：{pr.backend_name} —— 资产没走通整卷路径")
        got = tuple(float(v) for v in pr.report.params.density_matrix)
        assert got == want, f"第 {i} 帧 density_matrix 不是资产里的矩阵：{got}"
        assert pr.model_asset.get("applied") is True, pr.model_asset


def test_default_roll_path_keeps_the_deterministic_backend():
    """**负控制一**：不传资产 ⇒ 单位阵 + 旧 backend（默认行为没被顺手改掉）。"""
    _, results = process_roll(_roll_frames())
    ident = tuple(float(v) for v in np.eye(3).ravel())
    for pr in results:
        assert pr.backend_name == "power-reciprocal", pr.backend_name
        assert tuple(float(v) for v in pr.report.params.density_matrix) == ident
        assert pr.model_asset.get("applied") is False, pr.model_asset


def test_disabled_asset_is_not_applied_through_the_roll_path():
    """**负控制二**：`enabled=False` 的资产传进来也不许应用。

    这条不是多余的：`fixtures/models/generic-color-negative-v1` 正是
    `enabled=false`，而它现在被证明**更差**（色相角差 50.9°→90.4°、
    十一帧四统计距离 0.0905→0.1816）。如果整卷路径忽略这个开关，
    将来有人调 `load_asset()` 就会把它静默打开。
    """
    _, results = process_roll(_roll_frames(), model_asset=_asset(enabled=False))
    ident = tuple(float(v) for v in np.eye(3).ravel())
    for pr in results:
        assert pr.backend_name == "power-reciprocal", pr.backend_name
        assert tuple(float(v) for v in pr.report.params.density_matrix) == ident
        assert pr.model_asset.get("applied") is False


# ---------------------------------------------------------------- AST 锁
def _unforwarded(src: str) -> list[int]:
    """`process_roll` 函数体里没转发 `model_asset` 的 `process_image` 调用行号。"""
    tree = ast.parse(src)
    fn = next((n for n in tree.body
               if isinstance(n, ast.FunctionDef) and n.name == "process_roll"),
              None)
    assert fn is not None, "找不到 process_roll，检查器在空转"
    out = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "process_image":
            if not any(kw.arg == "model_asset" for kw in node.keywords):
                out.append(node.lineno)
    return out


def test_every_pass_of_process_roll_forwards_the_asset():
    lines = _unforwarded(SERVICE.read_text(encoding="utf-8"))
    body = ast.parse(SERVICE.read_text(encoding="utf-8"))
    fn = next(n for n in body.body if isinstance(n, ast.FunctionDef)
              and n.name == "process_roll")
    n_calls = sum(1 for n in ast.walk(fn) if isinstance(n, ast.Call)
                  and getattr(n.func, "id", "") == "process_image")
    assert n_calls >= 2, f"只扫到 {n_calls} 处调用，两遍转发的锁没有意义"
    assert not lines, f"这些 `process_image` 调用点没转发 model_asset：行号 {lines}"


def test_the_ast_checker_is_not_vacuous():
    """**负控制**：缺转发的写法必须被抓到，带转发的必须放过。"""
    bad = ("def process_roll(frames):\n"
           "    return process_image(frames[0], geometry=None)\n")
    good = ("def process_roll(frames, model_asset=None):\n"
            "    return process_image(frames[0], geometry=None,\n"
            "                           model_asset=model_asset)\n")
    assert _unforwarded(bad) == [2]
    assert _unforwarded(good) == []
