import numpy as np
import pandas as pd
import pytest

from abm_banks import BankingModel, ModelConfig, preset
from abm_banks.scenarios import Scenario, step_path


def run(cfg, seed=1):
    return BankingModel(cfg, seed=seed).run()


def test_determinism_same_seed_identical():
    cfg = preset("rate_shock").replace(n_months=48)
    a, b = run(cfg, 5).sector, run(cfg, 5).sector
    pd.testing.assert_frame_equal(a, b)


def test_different_seeds_differ():
    cfg = ModelConfig(n_months=24)
    assert not run(cfg, 1).sector["loans_total"].equals(run(cfg, 2).sector["loans_total"])


@pytest.mark.parametrize("name", ["baseline", "rate_shock", "rate_shock_panic",
                                  "spike_and_retreat", "random_jumps"])
def test_balance_identities_every_step(name):
    cfg = preset(name).replace(n_months=60, check_identities=True)
    for seed in (1, 2, 3):
        run(cfg, seed)                       # AssertionError, если тождество нарушено


def test_interbank_network_closed_and_nonnegative():
    cfg = preset("rate_shock_panic").replace(n_months=60)
    m = BankingModel(cfg, seed=4)
    for _ in m.iter_months():
        act = m.active
        assert (m.L[act] >= -1e-9).all() and (m.D[act] >= -1e-9).all()
        assert (m.sec[act] >= -1e-9).all() and (m.liq[act] >= -1e-6).all()
        assert (m.cb_debt >= 0).all() and (m.cb_dep >= 0).all() and (m.M >= 0).all()
        assert np.allclose(np.diag(m.M), 0.0)


def test_inactive_slots_are_zeroed():
    cfg = ModelConfig(n_months=120).replace(**{"resolution.entry_enabled": False,
                                               "credit.idio_event_prob": 0.05})
    m = BankingModel(cfg, seed=3)
    m.run()
    dead = ~m.active
    assert dead.any(), "ожидались выбывшие банки"
    for arr in (m.K, m.liq, m.sec, m.res, m.cb_debt, m.cb_dep):
        assert (arr[dead] == 0).all()
    assert (m.L[dead] == 0).all() and (m.D[dead] == 0).all()


def test_exit_and_entry_keep_number_of_slots():
    cfg = ModelConfig(n_months=120).replace(**{"credit.idio_event_prob": 0.05, "cb.bailout_enabled": False,
                                               "resolution.entry_delay_months": 2, "n_months": 120,
                                               "check_identities": True})
    res = BankingModel(cfg, seed=7).run()
    assert res.sector["failures"].sum() > 0
    assert res.sector["entries"].sum() > 0
    assert res.sector["n_banks"].max() <= cfg.sector.n_banks
    assert len(res.events) == res.sector["failures"].sum()


def test_rate_shock_qualitative_effects():
    """Скачок ставки: ставки банков растут с лагом, кредит замедляется, вклады притекают, дефолты растут."""
    cfg = preset("rate_shock").replace(n_months=72)
    shock = cfg.scenario.shock_month
    dfs = [run(cfg, s).sector for s in range(6)]
    avg = lambda col, a, b: np.mean([d[col].iloc[a:b].mean() for d in dfs])      # noqa: E731
    # ставки
    assert avg("loan_rate", shock, shock + 3) > avg("loan_rate", shock - 6, shock) + 1.0
    assert avg("deposit_rate", shock, shock + 12) > avg("deposit_rate", shock - 6, shock) + 2.0
    # лаг: через месяц передано меньше, чем через год
    lr = lambda k: np.mean([d["loan_rate"].iloc[shock + k] for d in dfs])           # noqa: E731
    base = avg("loan_rate", shock - 6, shock)
    assert (lr(0) - base) < (lr(12) - base)
    # кредит замедляется
    assert avg("loans_growth_yoy", shock + 9, shock + 15) < avg("loans_growth_yoy", shock - 12, shock) - 3
    # дефолты растут (с лагом: растёт доходность портфеля) — к концу второго года на треть выше исходного
    peak = np.mean([d["npl_share"].iloc[shock + 1:shock + 25].max() for d in dfs])
    assert peak > 1.3 * avg("npl_share", shock - 12, shock)
    assert avg("npl_share", shock + 13, shock + 24) > avg("npl_share", shock + 1, shock + 6)


def test_panic_freezes_interbank_and_triggers_cb():
    cfg = preset("rate_shock_panic").replace(n_months=48)
    m = cfg.scenario.shock_month
    dfs = [run(cfg, s).sector for s in range(6)]
    mbk_before = np.mean([d["mbk_volume"].iloc[m - 6:m].mean() for d in dfs])
    mbk_after = np.mean([d["mbk_volume"].iloc[m:m + 2].mean() for d in dfs])
    cb_before = np.mean([d["cb_debt"].iloc[m - 6:m].mean() for d in dfs])
    cb_after = np.mean([d["cb_debt"].iloc[m:m + 3].max() for d in dfs])
    assert mbk_after < 0.6 * mbk_before
    assert cb_after > cb_before + 1.0
    assert np.mean([d["panic"].iloc[m] for d in dfs]) > 0.3


def test_scenario_length_checked():
    cfg = ModelConfig(n_months=24)
    bad = Scenario("x", key_rate=np.full(10, 0.08))
    with pytest.raises(ValueError):
        BankingModel(cfg, scenario=bad)


def test_custom_scenario_used():
    cfg = ModelConfig(n_months=36)
    scn = Scenario("manual", key_rate=step_path(0.06, 0.05, 12, 36))
    out = BankingModel(cfg, scenario=scn, seed=1).run().sector
    assert out["key_rate"].iloc[0] == pytest.approx(6.0) and out["key_rate"].iloc[-1] == pytest.approx(11.0)


def test_record_banks_toggle_and_columns():
    cfg = ModelConfig(n_months=12)
    res = BankingModel(cfg, seed=1).run()
    assert res.banks is not None and len(res.banks) == 12 * cfg.sector.n_banks
    res2 = BankingModel(cfg.replace(record_banks=False), seed=1).run()
    assert res2.banks is None
    for col in ("key_rate", "loans_total", "deposits_total", "cb_debt", "n_banks", "car", "mbk_volume"):
        assert col in res.sector.columns


def test_iter_months_streaming_matches_run():
    cfg = ModelConfig(n_months=18)
    m = BankingModel(cfg, seed=9)
    streamed = [r["loans_total"] for r in m.iter_months()]
    full = BankingModel(cfg, seed=9).run().sector["loans_total"].tolist()
    assert streamed == full


def test_burn_in_excluded():
    a = BankingModel(ModelConfig(n_months=10, burn_in=0), seed=1).run().sector
    b = BankingModel(ModelConfig(n_months=10, burn_in=12), seed=1).run().sector
    assert len(a) == len(b) == 10


@pytest.mark.parametrize("dist", ["top20", "uniform", "zipf", "lognormal"])
@pytest.mark.parametrize("n", [8, 20, 45])
def test_size_distributions_and_sector_size(dist, n):
    cfg = ModelConfig(n_months=24, check_identities=True).replace(
        **{"sector.size_distribution": dist, "sector.n_banks": n})
    out = run(cfg, 2).sector
    assert out["n_banks"].iloc[0] <= n
    assert np.isfinite(out[["loans_total", "deposits_total", "car"]].to_numpy()).all()


def test_initialization_robust_over_many_seeds_and_sizes():
    for n in (3, 8, 20, 60, 150):
        for seed in range(120):
            BankingModel(ModelConfig(n_months=2).replace(**{"sector.n_banks": n}), seed=seed)


def test_extreme_initial_structure_is_repaired_not_crashed():
    cfg = ModelConfig(n_months=12, check_identities=True).replace(**{
        "sector.init_loans_corp": 0.6, "sector.init_loans_retail": 0.25, "cb.reserve_ratio": 0.12,
        "banks.liq_target_mean": 0.12})
    out = run(cfg, 3).sector
    assert np.isfinite(out["loans_total"]).all()


def test_sector_collapse_is_reported_not_raised():
    cfg = ModelConfig(n_months=60).replace(**{
        "cb.bailout_enabled": False, "resolution.entry_enabled": False,
        "credit.idio_event_prob": 0.5, "credit.idio_loss_median": 0.5})
    res = BankingModel(cfg, seed=1).run()
    assert res.meta["collapsed"] is not None
    assert len(res.sector) < 60


def test_random_parameter_fuzz_keeps_identities():
    from abm_banks.config import set_param
    rng = np.random.default_rng(11)
    ranges = {"banks.rate_speed_loan": (0.05, 0.9), "demand.loan_rate_elasticity.0": (0.2, 8),
              "credit.rate_sensitivity.0": (0, 40), "confidence.run_panic": (0.0, 0.5),
              "interbank.structural_share": (0.0, 0.15), "cb.reserve_ratio": (0.0, 0.12),
              "credit.idio_event_prob": (0.0, 0.03), "resolution.deposit_leak": (0.0, 0.6)}
    for i in range(12):
        cfg = preset(["rate_shock_panic", "random_jumps"][i % 2]).replace(n_months=40, check_identities=True)
        for k, (lo, hi) in ranges.items():
            set_param(cfg, k, float(rng.uniform(lo, hi)))
        res = BankingModel(cfg, seed=int(rng.integers(1e6))).run()
        assert np.isfinite(res.sector["loans_total"]).all()


def test_partial_pass_through_scales_rate_response():
    base = preset("rate_shock").replace(n_months=48)
    full = run(base, 3).sector
    half = run(base.replace(**{"banks.pass_through_loan.0": 0.5, "banks.pass_through_loan.1": 0.5}), 3).sector
    m = base.scenario.shock_month
    d_full = full["loan_rate_corp"].iloc[m + 8] - full["loan_rate_corp"].iloc[m - 2]
    d_half = half["loan_rate_corp"].iloc[m + 8] - half["loan_rate_corp"].iloc[m - 2]
    assert 0.3 * d_full < d_half < 0.75 * d_full


# --- структура предложения кредита (шаг 3б): уход спроса с рынка и надбавки к весам риска ----------
def _lending_run(extra=None, mult=None, n=60, seed=3):
    cfg = ModelConfig().replace(n_months=n, record_banks=False, check_identities=True, **(extra or {}))
    scn = Scenario("b", key_rate=np.full(n, 0.08), rw_retail_mult=mult)
    return BankingModel(cfg, scenario=scn, seed=seed).run().sector


def test_backlog_exit_off_by_default_and_reproduces_old_behaviour():
    cfg = ModelConfig().replace(n_months=36)
    assert cfg.demand.backlog_exit == 0.0
    a = run(cfg, 4).sector
    b = run(cfg.replace(**{"demand.backlog_exit": 0.0}), 4).sector
    pd.testing.assert_frame_equal(a, b)


def test_backlog_exit_reduces_accumulated_unmet_demand():
    tight = {"demand.loan_trend_growth.1": 0.28, "demand.loan_trend_growth.0": 0.13}
    base = _lending_run(tight)
    exit_ = _lending_run({**tight, "demand.backlog_exit": 0.3})
    rel = lambda s: s["credit_demand_unmet"].iloc[-1] / s["loans_total"].iloc[-1]       # noqa: E731
    assert rel(base) > 0.03                      # без ухода спрос копится
    assert rel(exit_) < 0.5 * rel(base)


def test_rw_retail_mult_slows_retail_lending_and_keeps_identities():
    n = 60
    mult = np.ones(n)
    mult[30:] = 1.5
    free, loaded = _lending_run(n=n), _lending_run(mult=mult, n=n)
    g = lambda s: s["loans_retail"].iloc[-1] / s["loans_retail"].iloc[29]               # noqa: E731
    assert g(loaded) < g(free)


def test_rw_retail_mult_validated_and_padded():
    with pytest.raises(ValueError):
        Scenario("x", key_rate=np.full(5, 0.08), rw_retail_mult=np.ones(4))
    with pytest.raises(ValueError):
        Scenario("x", key_rate=np.full(5, 0.08), rw_retail_mult=np.zeros(5))
    s = Scenario("x", key_rate=np.full(5, 0.08), rw_retail_mult=np.full(5, 1.3)).padded(3)
    assert len(s.rw_retail_mult) == 8 and np.allclose(s.rw_retail_mult[:3], 1.0)
