import numpy as np
import pandas as pd
import pytest

from abm_banks import ModelConfig
from abm_banks.calibration import (SMMCalibrator, WINDOWS, compute_moments, load_cbr_monthly,
                                   slice_window)


@pytest.fixture(scope="module")
def data():
    return load_cbr_monthly()


def test_data_loaded_and_sane(data):
    assert data.index[0] == pd.Period("2010-01") and len(data) > 150
    w = slice_window(data, "2017-01", "2021-12")
    assert w["key_rate"].between(4, 10.5).all()
    assert (w["loans_corp"] > 20_000).all() and w["loan_rate_corp"].notna().all()
    # объёмы в млрд руб.: кредиты организациям в 2021 г. — десятки трлн руб.
    assert 40_000 < data.loc["2021-12", "loans_corp"] < 80_000


def test_moments_defined_for_data_and_model(data):
    md = compute_moments(slice_window(data, *WINDOWS["B"].calib))
    assert np.isfinite(md["loan_rate_corp.pass_through"]) and md["loans_corp.growth_mean"] > 0
    # окно A: ставок по кредитам нет, но моменты объёмов определены
    ma = compute_moments(slice_window(data, *WINDOWS["A"].calib))
    assert np.isnan(ma["loan_rate_corp.mean"]) and np.isfinite(ma["loans_corp.growth_mean"])


def test_smm_objective_and_fit_smoke(data):
    cal = SMMCalibrator(ModelConfig(), data, "B", n_seeds=2)
    assert len(cal.keys) >= 10
    mid = np.array([(p.low + p.high) / 2 for p in cal.params])
    q0 = cal.objective(mid)
    assert np.isfinite(q0) and q0 < 1e8
    assert cal.objective(mid) == q0                      # общие случайные числа → детерминированная цель
    res = cal.fit("lhs_nm", budget=40, seed=1)
    assert res.objective <= q0 + 1e-9
    for p in cal.params:
        assert p.low <= res.theta[p.name] <= p.high
    rep = cal.validate(res.theta, pre_months=12)
    assert len(rep.actual) == 12 and "loan_rate_corp" in rep.summary().index
    assert rep.paths("loan_rate_corp").shape == (12, 4)


def test_calibration_result_roundtrip(tmp_path, data):
    from abm_banks.calibration import CalibrationResult
    cal = SMMCalibrator(ModelConfig(), data, "B", n_seeds=2)
    res = cal.fit("lhs_nm", budget=30, seed=2)
    f = tmp_path / "theta.json"
    res.save(f)
    assert CalibrationResult.load_theta(f) == res.theta


def test_default_params_per_window():
    from abm_banks.calibration import DEFAULT_PARAMS, default_params
    assert len(default_params("B")) == len(DEFAULT_PARAMS)
    a = default_params("A")
    assert len(a) == 4 and all("_trend_growth" in p.name for p in a)
