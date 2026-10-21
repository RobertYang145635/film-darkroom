"""跨卷低自由度通用颜色模型资产（v5 §6.3 / §9.4）。

模型是**应用内资产**，不是要求用户为每卷拟合的校准：

- 主界面只要求用户确认"这是彩色负片"，不要求选择 C-41/ECN-2 或具体型号；
- 资产缺失、版本不兼容或校验失败时，自动回退到经过测试的确定性密度模型，
  **仍可转换**，不进入任何"缺校准"状态（v5 §6.3 末段、§8.1）。

资产内容（全部写入 sidecar）：`A_global` 3x3 密度变换、每通道受约束单调
曲线节点、技术端点先验、通道斜率、模型版本与哈希、训练数据清单哈希、
随机种子与超参数。
"""

from __future__ import annotations

from .. import storage

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ASSET_DIR = storage.repo_path("fixtures", "models")
SCHEMA = 1
MAX_CURVE_NODES = 7          # v5 §6.3：节点数受限
MATRIX_COND_LIMIT = 1e6


class ModelAssetError(ValueError):
    pass


def canonical_hash(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass
class ModelAsset:
    asset_version: str
    payload: dict
    sha256: str
    enabled: bool = True
    meta: dict | None = None

    # ---- 校验
    def validated(self) -> "ModelAsset":
        p = self.payload
        A = np.asarray(p.get("density_matrix", np.eye(3)), np.float64)
        if A.shape != (3, 3):
            raise ModelAssetError(f"density_matrix 必须是 3x3，得到 {A.shape}")
        if not np.isfinite(A).all():
            raise ModelAssetError("density_matrix 含非有限值")
        cond = float(np.linalg.cond(A))
        if cond > MATRIX_COND_LIMIT:
            raise ModelAssetError(f"density_matrix 病态（cond={cond:.3g}）")
        for c, knots in enumerate(p.get("channel_curves") or []):
            if not knots:
                continue
            if len(knots) > MAX_CURVE_NODES:
                raise ModelAssetError(
                    f"通道 {c} 曲线节点数 {len(knots)} 超上限 {MAX_CURVE_NODES}")
            ys = [float(k[1]) for k in knots]
            if any(b - a < -1e-9 for a, b in zip(ys, ys[1:])):
                raise ModelAssetError(f"通道 {c} 曲线非单调（v5 §6.3）")
        if canonical_hash(p) != self.sha256:
            raise ModelAssetError("模型资产哈希不一致（内容被改动或写入不完整）")
        return self

    # ---- 应用
    def apply_to(self, params) -> dict:
        """把资产写进 InvertParams（返回实际应用的字段说明，供 sidecar）。"""
        p = self.payload
        if not self.enabled:
            return {"applied": False, "reason": p.get("disabled_reason",
                                                      "资产未启用"),
                    "asset_version": self.asset_version,
                    "asset_sha256": self.sha256}
        params.density_matrix = tuple(float(v) for v in
                                      np.asarray(p["density_matrix"]).ravel())
        curves = p.get("channel_curves") or [None, None, None]
        params.channel_curves = tuple(
            tuple(tuple(map(float, k)) for k in c) if c else () for c in curves)
        for key in ("red_ratio", "blue_ratio", "reference_exponent", "dmax"):
            if p.get(key) is not None:
                setattr(params, key, float(p[key]))
        params.backend_name = ("density-affine-curves"
                               if any(c for c in curves) else "density-affine")
        info = {"applied": True, "asset_version": self.asset_version,
                "asset_sha256": self.sha256, "backend": params.backend_name,
                "training_manifest_sha256": (self.meta or {}).get(
                    "training_manifest_sha256", "")}
        #: 片种资产的溯源字段一并回传。交付记录必须能回答
        #: "这一帧用的是哪个片种、打印中灰目标是多少、值是在几帧上定的"——
        #: 只有 `asset_version` 的话，`print_aim` 这类**不改反相参数**的字段
        #: 在 sidecar 里完全隐形（㊶ 追加⑯ 建探针时实测 `film_type` 读出来是 None，
        #: 只能靠解析版本字符串反推）。
        for key in ("film_type", "provenance", "frames", "print_aim", "white_balance"):
            if p.get(key) is not None:
                info[key] = p[key]
        return info

    def to_dict(self) -> dict:
        return {"schema_version": SCHEMA, "asset_version": self.asset_version,
                "enabled": bool(self.enabled), "sha256": self.sha256,
                "payload": self.payload, "meta": self.meta or {}}

    @classmethod
    def from_dict(cls, d: dict) -> "ModelAsset":
        if int(d.get("schema_version", 0)) != SCHEMA:
            raise ModelAssetError(
                f"模型资产 schema_version={d.get('schema_version')} "
                f"与当前 {SCHEMA} 不兼容")
        payload = d.get("payload") or {}
        return cls(asset_version=str(d.get("asset_version", "unknown")),
                   payload=payload, sha256=str(d.get("sha256", "")),
                   enabled=bool(d.get("enabled", True)),
                   meta=d.get("meta") or {}).validated()


def deterministic_asset(*, reason: str = "无训练资产：使用确定性密度模型") -> ModelAsset:
    """经过测试的确定性回退：A=I、无曲线、不改动端点。

    **但它不是"什么都不做"**：`red_ratio`/`blue_ratio` 带产品默认值
    `print_math.LOOK_RED_EXPONENT = 1.30` 与 `LOOK_BLUE_EXPONENT = 0.7`
    （都是对着用户手工成品量出来、留出卷逐帧复验过的通道指数，与
    `LOOK_CHROMA_GAIN` 同地位；两者**联立定出**，出处写在 `print_math` 那两段上）。
    矩阵仍是单位阵，
    所以 `density-affine` 与 `power-reciprocal` 在这里逐比特等价
    （`tests/unit/test_v4_contracts.py::test_unit_matrix_affine_matches_reciprocal_allclose`）。
    `reason` 只在**资产被显式禁用**时才有意义，保留给调用方说明来历。
    """
    from ..print_math import LOOK_BLUE_EXPONENT, LOOK_RED_EXPONENT

    payload = {
        "density_matrix": np.eye(3).tolist(),
        "channel_curves": [None, None, None],
        "red_ratio": LOOK_RED_EXPONENT, "blue_ratio": LOOK_BLUE_EXPONENT,
        "reference_exponent": None, "dmax": None,
        "disabled_reason": reason,
    }
    return ModelAsset(asset_version="deterministic-fallback-v1",
                      payload=payload, sha256=canonical_hash(payload),
                      enabled=True,
                      meta={"kind": "fallback", "note": reason})


def save_asset(asset: ModelAsset, directory: Path | None = None) -> Path:
    asset.validated()
    d = directory or ASSET_DIR
    d.mkdir(parents=True, exist_ok=True)
    out = d / f"{asset.asset_version}.model.json"
    from ..io.atomic import atomic_write_bytes        # 原子写（v6.2 审查批次 3）
    atomic_write_bytes(out, json.dumps(asset.to_dict(), ensure_ascii=False,
                                       indent=2).encode("utf-8"),
                       overwrite=True)
    return out


def load_asset(path: str | Path | None = None, *,
               prefer: str = "generic-color-negative") -> ModelAsset:
    """加载并校验模型资产；任何失败都回退到确定性模型（不抛给普通路径）。"""
    candidates: list[Path] = []
    if path:
        candidates.append(Path(path))
    else:
        if ASSET_DIR.is_dir():
            candidates.extend(sorted(ASSET_DIR.glob("*.model.json")))
    for c in candidates:
        if not c.exists():
            continue
        try:
            asset = ModelAsset.from_dict(
                json.loads(c.read_text(encoding="utf-8")))
        except Exception as e:  # noqa: BLE001 —— 损坏资产必须可见但可回退
            LAST_LOAD_NOTE = f"模型资产 {c.name} 不可用（{type(e).__name__}: {e}）"
            globals()["_LAST_NOTE"] = LAST_LOAD_NOTE
            continue
        if prefer and not asset.asset_version.startswith("deterministic") \
                and prefer not in asset.asset_version:
            continue
        return asset
    reason = "没有可用模型资产：使用经过测试的确定性密度模型"
    if globals().get("_LAST_NOTE"):
        reason = globals()["_LAST_NOTE"] + "；已回退确定性密度模型"
    return deterministic_asset(reason=reason)


def last_load_note() -> str:
    return globals().get("_LAST_NOTE", "")


# ------------------------------------------------------------------ 训练
def fit_decorrelation(densities: list[np.ndarray], *, ridge: float = 0.35,
                      seed: int = 20260901) -> dict:
    """无监督密度解混：让跨帧采样的密度通道互相关下降。

    A = (1-λ)·I + λ·C^{-1/2}（λ=ridge 的补），保持对角占优与条件数；
    不使用任何参考正片，因此不构成"追参考图"的高自由度拟合。
    """
    rng = np.random.default_rng(seed)
    mats = []
    for d in densities:
        d = np.asarray(d, np.float64).reshape(-1, 3)
        d = d[np.isfinite(d).all(axis=1)]
        if d.shape[0] > 4000:
            idx = rng.choice(d.shape[0], 4000, replace=False)
            d = d[idx]
        if d.shape[0] >= 64:
            mats.append(d)
    if not mats:
        return {"matrix": np.eye(3).tolist(), "confidence": 0.0,
                "warnings": ["没有可用密度样本：保持单位阵"]}
    D = np.concatenate(mats, axis=0)
    D = D - D.mean(axis=0)
    C = np.cov(D.T)
    if not np.isfinite(C).all():
        return {"matrix": np.eye(3).tolist(), "confidence": 0.0,
                "warnings": ["协方差含非有限值：保持单位阵"]}
    evals, evecs = np.linalg.eigh(C)
    evals = np.clip(evals, 1e-12, None)
    inv_sqrt = evecs @ np.diag(evals ** -0.5) @ evecs.T
    lam = float(np.clip(ridge, 0.0, 1.0))
    A = (1.0 - lam) * np.eye(3) + lam * inv_sqrt
    # 行归一：保持整体密度量级，避免顺带改变反差
    A = A / np.maximum(np.abs(A).sum(axis=1, keepdims=True), 1e-9) * \
        np.abs(np.eye(3)).sum(axis=1, keepdims=True)
    cond = float(np.linalg.cond(A))
    warn = []
    if cond > MATRIX_COND_LIMIT:
        A = np.eye(3)
        cond = 1.0
        warn.append("解混矩阵病态，已回退单位阵")
    before = float(np.abs(np.corrcoef(D.T) - np.eye(3)).sum())
    after = float(np.abs(np.corrcoef((D @ A.T).T) - np.eye(3)).sum())
    return {"matrix": A.tolist(), "confidence": float(np.clip(
        (before - after) / max(before, 1e-9), 0.0, 1.0)),
        "correlation_before": round(before, 4),
        "correlation_after": round(after, 4),
        "cond": round(cond, 3), "n_samples": int(D.shape[0]),
        "warnings": warn}


def build_asset_from_fit(fit: dict, *, asset_version: str,
                         manifest_sha256: str, hyper: dict,
                         enabled: bool = True) -> ModelAsset:
    payload = {
        "density_matrix": [[float(v) for v in row] for row in fit["matrix"]],
        "channel_curves": [None, None, None],
        "red_ratio": None, "blue_ratio": None,
        "reference_exponent": None, "dmax": None,
    }
    meta = {
        "kind": "trained",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "training_manifest_sha256": manifest_sha256,
        "hyperparameters": hyper,
        "confidence": fit.get("confidence"),
        "correlation_before": fit.get("correlation_before"),
        "correlation_after": fit.get("correlation_after"),
        "n_samples": fit.get("n_samples"),
        "warnings": fit.get("warnings", []),
    }
    return ModelAsset(asset_version=asset_version, payload=payload,
                      sha256=canonical_hash(payload), enabled=enabled,
                      meta=meta)
