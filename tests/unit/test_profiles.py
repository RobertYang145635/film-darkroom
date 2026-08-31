"""配置文件往返、迁移与先验桥接测试（§2.2.2、§6 可迁移性）。"""

import json

import pytest

from negative_lab.profiles.profile import (FilmProfile, load_profile,
                                           save_profile, SCHEMA_VERSION)


def test_roundtrip(tmp_path):
    p = FilmProfile(name="gold200-z8", film_type="c41_color",
                    device="NIKON Z 8", light_source="LED 5600K",
                    base_prior=[0.28, 0.11, 0.03],
                    exponent_prior={"reference_exponent": 1.0,
                                    "red_ratio": 0.98, "blue_ratio": 1.05},
                    tone_prior={"gamma": 1.1}, notes="测试")
    f = save_profile(p, tmp_path)
    q = load_profile(f)
    assert q.to_dict() == p.to_dict()
    assert q.version == 1 and q.created_utc == p.created_utc


def test_initial_params_bridge(tmp_path):
    p = FilmProfile(base_prior=[0.3, 0.1, 0.04],
                    exponent_prior={"reference_exponent": 0.95,
                                    "red_ratio": 1.1, "blue_ratio": 0.9},
                    tone_prior={"gamma": 1.2, "exposure": 1.1})
    ip = p.to_initial_params()
    assert ip.dmin == (0.3, 0.1, 0.04)
    assert ip.red_ratio == 1.1 and ip.gamma == 1.2
    # 未提供的先验保持默认
    assert ip.black == 0.0


def test_future_schema_rejected_not_silently_misread():
    d = {"schema_version": SCHEMA_VERSION + 1, "name": "x"}
    with pytest.raises(ValueError):
        FilmProfile.from_dict(d)


def test_unknown_fields_ignored(tmp_path):
    d = FilmProfile(name="t").to_dict()
    d["some_future_field"] = 42
    f = tmp_path / "t.profile.json"
    f.write_text(json.dumps(d), encoding="utf-8")
    p = load_profile(f)
    assert p.name == "t"
