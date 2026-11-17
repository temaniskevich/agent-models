import numpy as np
import pandas as pd

from abm_banks import irf


def test_local_projection_recovers_known_response():
    rng = np.random.default_rng(0)
    n = 400
    dk = rng.normal(0, 0.5, n)
    k = 8 + np.cumsum(dk)
    # y реагирует на Δk(t) мгновенно на 0.4 и ещё на 0.3 в следующем месяце (кумулятивно 0.7 после h≥1)
    y = np.zeros(n)
    for t in range(2, n):
        y[t] = y[t - 1] + 0.4 * dk[t] + 0.3 * dk[t - 1] + 0.05 * rng.normal()
    df = pd.DataFrame({"key_rate": k, "loan_rate_corp": y})
    lp = irf.local_projection(df, "loan_rate_corp", H=6).set_index("h")
    assert abs(lp.loc[0, "beta"] - 0.4) < 0.1 and abs(lp.loc[3, "beta"] - 0.7) < 0.15
    assert (lp["se"] > 0).all() and (lp["lo90"] < lp["beta"]).all()


def test_data_irf_and_model_irf_shapes_and_verdict():
    data, df = irf.data_irf()
    assert set(data) == set(irf.OUTCOMES) and len(data["loan_rate_corp"]) == 13
    from abm_banks.forecasting import load_theta
    theta = load_theta("B")
    mod = irf.model_lp_irf(theta, df["key_rate"].to_numpy()[:60], n_seeds=3)
    step = irf.model_step_irf(theta, n_seeds=3, n_months=48)
    assert step["loan_rate_corp"].set_index("h").loc[6, "beta"] > 0.2          # ставка по кредитам растёт после повышения
    v = irf.evaluate(data, mod)
    assert len(v.table) == 8 and 0 <= v.passed <= 8
