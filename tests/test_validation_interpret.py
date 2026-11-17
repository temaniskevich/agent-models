import numpy as np
import pandas as pd

from abm_banks import BankingModel, ModelConfig, preset
from abm_banks.interpret import (ensemble_summary, findings_markdown, interpret,
                                 interpret_ensemble)
from abm_banks.batch import run_many
from abm_banks.validation import (checks_table, detect_shock, mbk_network_stats,
                                  mbk_topology_series, size_distribution_stability,
                                  stylized_facts, zipf_slope)


def test_detect_shock():
    res = BankingModel(preset("rate_shock").replace(n_months=60), seed=1).run()
    m, size = detect_shock(res.sector)
    assert abs(m - 24) <= 1 and 6.5 < size < 7.5
    assert detect_shock(BankingModel(ModelConfig(n_months=40), seed=1).run().sector) is None


def test_stylized_facts_pass_in_shock_with_panic():
    cfg = preset("rate_shock_panic").replace(n_months=72)
    res = BankingModel(cfg, seed=1).run()
    checks = stylized_facts(res.sector)
    failed = [c.name for c in checks if c.passed is False]
    assert not failed, failed
    assert len(checks_table(checks)) == len(checks)


def test_interpret_runs_for_shock_and_baseline():
    for name in ("rate_shock_panic", "baseline"):
        res = BankingModel(preset(name).replace(n_months=60), seed=2).run()
        text = findings_markdown(interpret(res.sector, res.events))
        assert "Сценарий" in text and "Прибыльность" in text
    assert "Передача ставки" in findings_markdown(
        interpret(BankingModel(preset("rate_shock").replace(n_months=60), seed=2).run().sector))


def test_ensemble_interpretation():
    runs = run_many(preset("rate_shock").replace(n_months=48), list(range(5)))
    s = ensemble_summary(runs)
    assert len(s) == 5 and (s["max_key_rate"] > 10).all()
    assert len(interpret_ensemble(runs)) >= 3


def test_size_distribution_stability_baseline():
    res = BankingModel(ModelConfig(n_months=60), seed=3).run()
    st = size_distribution_stability(res.banks)
    assert st["rank_corr_to_start"].iloc[-1] > 0.8                 # ранги банков сохраняются
    assert abs(st["zipf_slope"].iloc[-1] - st["zipf_slope"].iloc[0]) < 0.5


def test_zipf_slope_exact():
    sizes = 1.0 / np.arange(1, 21)
    assert abs(zipf_slope(sizes) + 1.0) < 1e-9


def test_network_stats_and_series():
    cfg = ModelConfig(n_months=24, record_network=True)
    res = BankingModel(cfg, seed=4).run()
    assert res.network is not None and len(res.network) > 0
    ser = mbk_topology_series(res.network, cfg.sector.n_banks)
    assert (ser["density"] <= 1).all() and ser["edges"].max() > 0
    st = mbk_network_stats(np.array([0, 1]), np.array([1, 0]), np.array([1.0, 2.0]), 4)
    assert st["reciprocity"] == 1.0
