import json

import numpy as np
import pandas as pd
import pytest

from abm_banks import ModelConfig, preset
from abm_banks.batch import generate_synthetic, run_many, sample_param_sets
from abm_banks.config import ScenarioConfig
from abm_banks.scenarios import (Scenario, build_scenario, random_jump_path, step_path,
                                 ramp_path)
from abm_banks.synthetic import ARJumpControl


def test_step_and_ramp_paths():
    p = step_path(0.08, 0.07, 10, 30)
    assert p[9] == pytest.approx(0.08) and p[10] == pytest.approx(0.15)
    d = step_path(0.08, 0.07, 10, 40, decay_months=5)
    assert d[10] == pytest.approx(0.15) and d[15] == pytest.approx(0.08 + 0.035)
    r = ramp_path(0.08, 0.06, 5, 6, 30)
    assert r[4] == pytest.approx(0.08) and r[10] == pytest.approx(0.14)


def test_random_jump_paths_bounds_and_reproducibility():
    cfg = ScenarioConfig(kind="random_jumps", jump_rate_per_year=1.0)
    a, ev_a = random_jump_path(np.random.default_rng(1), 240, cfg)
    b, _ = random_jump_path(np.random.default_rng(1), 240, cfg)
    assert np.array_equal(a, b)
    assert a.min() >= cfg.key_rate_floor and a.max() <= cfg.key_rate_cap
    assert len(ev_a) > 5
    assert all(cfg.jump_min <= abs(e["size"]) <= cfg.jump_max for e in ev_a)


def test_scenario_padding_and_validation():
    s = Scenario("x", key_rate=np.array([0.1, 0.12, 0.12]))
    p = s.padded(4)
    assert len(p) == 7 and (p.key_rate[:4] == 0.1).all()
    with pytest.raises(ValueError):
        Scenario("bad", key_rate=np.array([0.1, -0.01]))


def test_macro_shocks_follow_jump():
    cfg = ScenarioConfig(kind="step", shock_size=0.08, shock_month=10, confidence_shock_peak=0.5,
                         credit_stress_peak=1.0, deposit_shock_peak=-0.05)
    s = build_scenario(cfg, 40)
    assert s.confidence[9] == 0 and s.confidence[10] > 0.4
    assert s.credit_stress[10] > s.credit_stress[25] > 0
    assert s.deposit_shock[10] < 0


def test_run_many_shapes_and_reproducibility():
    cfg = ModelConfig(n_months=24)
    a = run_many(cfg, [1, 2, 3])
    b = run_many(cfg, [1, 2, 3])
    assert set(a["run"]) == {0, 1, 2} and len(a) == 3 * 24
    pd.testing.assert_frame_equal(a, b)
    assert a.groupby("run")["loans_total"].first().nunique() == 3


def test_run_many_parallel_equals_serial():
    cfg = ModelConfig(n_months=18)
    serial = run_many(cfg, list(range(6)), n_jobs=1)
    par = run_many(cfg, list(range(6)), n_jobs=2)
    pd.testing.assert_frame_equal(serial, par)


def test_run_many_param_sets_change_results():
    cfg = preset("rate_shock").replace(n_months=48)
    sets = [{"demand.loan_rate_elasticity.0": 0.5}, {"demand.loan_rate_elasticity.0": 6.0}]
    out = run_many(cfg, [3, 3], param_sets=sets)
    end = out.groupby("run")["loans_corp"].last()
    assert end.iloc[0] > end.iloc[1]            # чувствительнее спрос к ставке → кредит ниже


def test_latin_hypercube_covers_ranges():
    sets = sample_param_sets(50, {"a.b": (0.0, 1.0)}, 1)
    vals = np.sort([s["a.b"] for s in sets])
    assert vals[0] < 0.05 and vals[-1] > 0.95 and len(np.unique(np.floor(vals * 50))) == 50


def test_generate_synthetic_writes_files(tmp_path):
    cfg = ModelConfig(n_months=36)
    data, meta = generate_synthetic(cfg, n_runs=6, base_seed=5, outdir=tmp_path)
    assert data["run"].nunique() == 6 and len(meta) == 6
    assert (tmp_path / "synthetic.parquet").exists() and (tmp_path / "meta.parquet").exists()
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["n_runs"] == 6
    again, _ = generate_synthetic(cfg, n_runs=6, base_seed=5)
    pd.testing.assert_frame_equal(data, again)


def test_control_generator_matches_moments():
    cfg = ModelConfig(n_months=60).replace(**{"scenario.kind": "random_jumps"})
    data = run_many(cfg, list(range(12)))
    ctl = ARJumpControl().fit(data)
    rep = ctl.moments_report(data)
    for _, row in rep.iterrows():
        scale = max(abs(row["mean_src"]), 1.0)
        assert abs(row["mean_ctrl"] - row["mean_src"]) / scale < 0.35, row["variable"]
        if row["variable"] != "cb_debt":      # сильно скошенный ряд: хвосты контроль воспроизводит лишь приближённо
            assert row["std_ctrl"] > 0.3 * row["std_src"], row["variable"]
    sim = ctl.simulate(40, 3, seed=2)
    assert len(sim) == 120 and sim["run"].nunique() == 3
