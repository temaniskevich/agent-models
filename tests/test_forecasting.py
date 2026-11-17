import numpy as np
import pandas as pd

from abm_banks import forecasting as fc


def test_dm_test_detects_clear_difference_and_ignores_zero():
    rng = np.random.default_rng(0)
    d = -1.0 + 0.2 * rng.standard_normal(24)
    stat, p = fc.dm_test(d, 3)
    assert stat < -3 and p < 0.01
    assert np.isnan(fc.dm_test(np.zeros(24), 1)[0])


def test_fit_predict_recovers_linear_relation_and_shrinks():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(200, 2))
    df = pd.DataFrame({"a": x[:, 0], "b": x[:, 1]}); df["y"] = 2 * df["a"] + 1
    x_new = pd.Series({"a": 1.0, "b": 0.0})
    assert abs(fc.fit_predict(df, np.ones(200), x_new) - 3.0) < 0.1
    assert fc.fit_predict(df, np.ones(200), x_new, ridge=100.0) < 2.0       # сильное сжатие к среднему


def test_real_series_harmonized_and_key_rate_continuous():
    real = fc.load_real()
    assert set(fc.COLUMNS) <= set(real.columns)
    assert abs(real["key_rate"].diff().loc[pd.Period("2013-09", "M")]) < 1e-9
    assert real["g_lc"].dropna().abs().max() < 200


def test_small_pilot_runs_end_to_end():
    real = fc.load_real()
    theta = fc.load_theta("A")
    abm = fc.make_abm_panel(theta, n_runs=12, months=48, seed=1)
    ctl = fc.make_control_panel(abm, 12, 48)
    res = fc.run_pilot(real, abm, ctl, ["g_lc"], ("2014-01", "2014-12"), horizons=(1, 3), shares=(0.5,))
    assert {"naive", "real", "real_shrunk", "abm@0.5", "control@0.5"} <= set(res.summary["variant"])
    assert np.isfinite(res.errors["error"]).all() and len(res.dm) > 0
