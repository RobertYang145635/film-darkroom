"""§6：旁车参数字段可迁移（schema_version=1 往返一致）。"""

from negative_lab.pipeline.params import InvertParams


def test_parameters_roundtrip():
    p = InvertParams(dmin=(0.62, 0.45, 0.30), dmax=2.3, offset=0.1,
                     wb_low=(1.0, 1.0, 1.02), wb_high=(0.98, 1.0, 1.05),
                     reference_exponent=1.0, red_ratio=0.87, blue_ratio=1.18,
                     black=0.05, gamma=1.1, exposure=1.2)
    d = p.to_parameters_dict()
    q = InvertParams.from_parameters_dict(d)
    assert q.to_parameters_dict() == d
    assert q.dmin == (0.62, 0.45, 0.30)
    assert isinstance(q.dmin, tuple)


def test_unknown_fields_ignored_forward_compat():
    d = InvertParams().to_parameters_dict()
    d["future_control"] = 123
    p = InvertParams.from_parameters_dict(d)
    assert p.gamma == 1.0
