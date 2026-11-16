import pytest
from abm_banks.config import ModelConfig, preset, PRESET_LABELS


def test_roundtrip_dict_json_yaml(tmp_path):
    cfg = preset("rate_shock_panic")
    assert ModelConfig.from_dict(cfg.to_dict()) == cfg
    assert ModelConfig.from_json(cfg.to_json(tmp_path / "c.json")) == cfg
    assert ModelConfig.from_yaml(cfg.to_yaml(tmp_path / "c.yaml")) == cfg


def test_replace_nested_and_tuple_index():
    cfg = ModelConfig()
    new = cfg.replace(**{"banks.car_min": 0.09, "demand.loan_rate_elasticity.1": 4.2})
    assert new.banks.car_min == 0.09
    assert new.demand.loan_rate_elasticity == (cfg.demand.loan_rate_elasticity[0], 4.2)
    assert cfg.banks.car_min == 0.08                     # исходный конфиг не изменён


def test_unknown_parameter_rejected():
    with pytest.raises(AttributeError):
        ModelConfig().replace(**{"banks.no_such": 1})
    with pytest.raises(ValueError):
        ModelConfig.from_dict({"bogus": 1})


@pytest.mark.parametrize("bad", [{"sector.n_banks": 2}, {"banks.car_target_mean": 0.05},
                                 {"scenario.kind": "nope"}, {"n_months": 1}])
def test_validation(bad):
    with pytest.raises(ValueError):
        ModelConfig().replace(**bad).validate()


def test_all_presets_valid():
    for name in PRESET_LABELS:
        preset(name).validate()
