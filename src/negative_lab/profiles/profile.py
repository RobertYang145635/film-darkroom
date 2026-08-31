"""胶片/设备配置文件（规格 §2.2.2、§5.3.1）。

配置文件只能作为单张分析的**初始先验**，不得强行覆盖逐张分析结果
（用户决策：批处理策略）。字段带 schema_version 与 profile version，
未来字段变更走 migrate()，不得静默改变旧文件含义。
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

SCHEMA_VERSION = 1


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@dataclass
class FilmProfile:
    name: str = "untitled"
    profile_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    version: int = 1
    created_utc: str = field(default_factory=_utc_now)
    modified_utc: str = field(default_factory=_utc_now)
    film_type: str = "unknown"          # c41_color|bw|ecn2_color|unknown
    device: str = "unknown"             # 相机/扫描仪型号
    light_source: str = "unknown"       # 光源（翻拍台）
    # 先验（None = 不提供，交给逐张自动分析）
    base_prior: list | None = None      # [dminR, dminG, dminB]
    exponent_prior: dict | None = None  # {reference_exponent, red_ratio, blue_ratio}
    tone_prior: dict | None = None      # {dmax, offset, black, gamma, exposure}
    notes: str = ""
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "profile_id": self.profile_id,
            "name": self.name,
            "version": self.version,
            "created_utc": self.created_utc,
            "modified_utc": self.modified_utc,
            "film_type": self.film_type,
            "device": self.device,
            "light_source": self.light_source,
            "base_prior": self.base_prior,
            "exponent_prior": self.exponent_prior,
            "tone_prior": self.tone_prior,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FilmProfile":
        d = cls.migrate(d)
        known = {f for f in ("name", "profile_id", "version", "created_utc",
                             "modified_utc", "film_type", "device", "light_source",
                             "base_prior", "exponent_prior", "tone_prior", "notes")}
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def migrate(cls, d: dict) -> dict:
        """旧 schema -> 当前 schema。目前只有 v1；未来版本在此追加链式迁移。"""
        v = int(d.get("schema_version", 1))
        if v > SCHEMA_VERSION:
            raise ValueError(f"配置文件 schema_version={v} 新于本软件支持"
                             f"的 {SCHEMA_VERSION}，请升级软件")
        d = dict(d)
        d["schema_version"] = SCHEMA_VERSION
        return d

    # ---- 与反相参数的桥接（只作初始值）

    def to_initial_params(self):
        from ..pipeline.params import InvertParams
        p = InvertParams()
        if self.base_prior:
            p.dmin = tuple(float(v) for v in self.base_prior)
        if self.exponent_prior:
            e = self.exponent_prior
            p.reference_exponent = float(e.get("reference_exponent", 1.0))
            p.red_ratio = float(e.get("red_ratio", 1.0))
            p.blue_ratio = float(e.get("blue_ratio", 1.0))
        if self.tone_prior:
            t = self.tone_prior
            for k in ("dmax", "offset", "black", "gamma", "exposure"):
                if k in t:
                    setattr(p, k, float(t[k]))
        return p


def profiles_dir() -> Path:
    import os

    base = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
    d = Path(base) / "negative-lab" / "profiles"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_profile(prof: FilmProfile, directory: Path | None = None) -> Path:
    directory = directory or profiles_dir()
    directory.mkdir(parents=True, exist_ok=True)
    prof.modified_utc = _utc_now()
    out = directory / f"{prof.name}.profile.json"
    out.write_text(json.dumps(prof.to_dict(), ensure_ascii=False, indent=2),
                   encoding="utf-8")
    return out


def load_profile(path: str | Path) -> FilmProfile:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return FilmProfile.from_dict(d)


def list_profiles(directory: Path | None = None) -> list[FilmProfile]:
    directory = directory or profiles_dir()
    out = []
    for f in sorted(directory.glob("*.profile.json")):
        try:
            out.append(load_profile(f))
        except Exception:  # noqa: BLE001 —— 坏文件不拖垮列表
            continue
    return out
