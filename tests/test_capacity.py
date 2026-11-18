"""Тесты эксперимента «сложность × точность симулятора» (docs/CAPACITY_PROTOCOL.md, раздел 7).

M3 (torch) проверяется в подпроцессе: в одном процессе с lightgbm он падает из-за конфликта OpenMP на macOS.
"""
import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm")

from abm_banks import capacity as cp                      # noqa: E402
from abm_banks import capacity_analysis as ca             # noqa: E402
from abm_banks import forecasting as fc                   # noqa: E402

WIN = cp.WINDOWS["P19"]


@pytest.fixture(scope="module")
def data():
    real = fc.load_real()
    abm = fc.make_abm_panel(fc.load_theta("B"), n_runs=6, months=48, seed=7, n_jobs=1)
    ctl = fc.make_control_panel(abm, 6, 48, seed=8)
    return real, abm, ctl


def _key(df):
    return df.set_index(["variant", "target", "h", "month"])["forecast"]


@pytest.mark.parametrize("fn", [cp.forecast_m1, cp.forecast_m2])
def test_deterministic(data, fn):
    real, abm, ctl = data
    pd.testing.assert_frame_equal(fn(WIN, real, abm, ctl, shares=(0.5,)), fn(WIN, real, abm, ctl, shares=(0.5,)))


@pytest.mark.parametrize("fn", [cp.forecast_m1, cp.forecast_m2])
def test_no_lookahead_first_window_month(data, fn):
    """Изменение реальных данных начиная с первого месяца окна не меняет прогнозы на этот месяц (любой горизонт):
    обучающие пары и вход используют только данные до начала окна."""
    real, abm, ctl = data
    real2 = real.copy()
    real2.loc[WIN.start:, :] = real2.loc[WIN.start:, :] * 3.0 + 5.0
    a = _key(fn(WIN, real, abm, ctl, shares=(0.5,)))
    b = _key(fn(WIN, real2, abm, ctl, shares=(0.5,)))
    first = [i for i in a.index if i[3] == str(WIN.start)]
    assert len(first) >= 20
    np.testing.assert_allclose(a.loc[first].to_numpy(), b.loc[first].to_numpy(), rtol=1e-9, atol=1e-9)


def test_train_cut_excludes_window_targets():
    for h in cp.HORIZONS:
        assert cp.train_cut(WIN, h) + h < WIN.start


def test_forecast_cases_use_only_window_months(data):
    real, _, _ = data
    cases = cp.forecast_cases(WIN, real, "g_lc", 3)
    assert cases and all(WIN.start <= m <= pd.Period(WIN.test[1], "M") and t0 == m - 3 for m, t0, _ in cases)


def test_synth_weights_total():
    w = cp.synth_weights(100, 5000, 0.5)
    assert w.sum() == pytest.approx(100.0) and cp.synth_weights(100, 10, 0.75).sum() == pytest.approx(300.0)


def test_holm_monotone_and_bounded():
    p = np.array([0.01, 0.04, 0.03, np.nan])
    adj = ca.holm(p)
    assert np.isnan(adj[3]) and adj[0] == pytest.approx(0.03) and np.all(adj[:3] >= p[:3]) and adj[:3].max() <= 1


def test_verdict_rules():
    assert ca.verdict(0.9, 0.7) == "помогает" and ca.verdict(0.9, 0.4) == "эффекта не обнаружено"
    assert ca.verdict(1.06, 0.2) == "вредит" and ca.verdict(0.97, 0.9) == "эффекта не обнаружено"


M3_SCRIPT = textwrap.dedent("""
    import sys
    import numpy as np
    from abm_banks import capacity as cp, capacity_m3 as m3, forecasting as fc
    m3.MAX_STEPS, m3.EVAL_EVERY, m3.NO_VAL_STEPS = 40, 20, 20
    real = fc.load_real()
    w = cp.WINDOWS["P19"]
    abm = fc.make_abm_panel(fc.load_theta("B"), n_runs=6, months=48, seed=7, n_jobs=1)
    ctl = fc.make_control_panel(abm, 6, 48, seed=8)
    def run(r):
        return m3.forecast_m3(w, r, abm, ctl, shares=(0.5,), seeds=(0,))
    a, b = run(real), run(real)
    assert len(a) > 100 and set(a["variant"]) == {"real", "abm@0.5", "control@0.5"}, set(a["variant"])
    assert np.allclose(a["forecast"], b["forecast"]), "недетерминизм"
    real2 = real.copy()
    real2.loc[w.start:, :] = real2.loc[w.start:, :] * 3.0 + 5.0
    c = run(real2)
    k = lambda d: d[d["month"] == str(w.start)].set_index(["variant", "target", "h"])["forecast"].sort_index()
    assert len(k(a)) >= 20 and np.allclose(k(a).to_numpy(), k(c).to_numpy(), atol=1e-6), "заглядывание вперёд"
    print("ok")
""")


def test_m3_in_subprocess_deterministic_and_no_lookahead():
    r = subprocess.run([sys.executable, "-c", M3_SCRIPT], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0 and r.stdout.strip().endswith("ok"), r.stderr[-1500:]


def test_noise_decision_rules():
    assert ca.noise_decision([0.90, 0.92, 0.95, 0.91, 0.93, 0.94]) == "устойчиво"
    assert ca.noise_decision([1.10, 1.00, 0.90, 1.05, 0.95, 1.00]) == "шум"
    assert ca.noise_decision([0.90, 0.95, 0.97, 1.02, 1.03, 0.99]) == "не определён"
    assert ca.noise_decision([1.10, 1.12, 1.08, 1.10, 1.09, 1.11], "harm") == "устойчиво (вред)"
    assert ca.noise_decision([1.0, 1.0]).startswith("недостаточно")
