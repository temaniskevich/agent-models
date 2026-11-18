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


# --------------------------------------------------------------------------
# Линейный двойник (поправка 18.11.2026, раздел 12 протокола)
# --------------------------------------------------------------------------
from abm_banks.synthetic import LinearTwin, TWIN_STATE   # noqa: E402


def _linear_panel(n_runs=40, T=150, seed=0):
    """Панель из известного VARX(2, 12): двойник должен восстановить коэффициенты и выбрать p = 2."""
    rng = np.random.default_rng(seed)
    k = len(TWIN_STATE)
    A1 = np.diag(np.full(k, 0.5)); A1[0, 4] = 0.1
    A2 = np.diag(np.full(k, 0.2))
    B = np.zeros((13, k)); B[0, :4] = 0.3; B[1, :4] = 0.2; B[3, 4:] = -0.5
    rows = []
    for r in range(n_runs):
        a = rng.normal(0, 1, k)
        K = 8 + np.cumsum(rng.normal(0, 0.3, T + 12))
        W = np.zeros((T, k))
        for t in range(2, T):
            W[t] = a + A1 @ W[t - 1] + A2 @ W[t - 2] + K[t + 12 - np.arange(13)] @ B + rng.normal(0, 0.2, k)
        d = pd.DataFrame(W, columns=TWIN_STATE)
        d["key_rate"] = K[12:]
        d["run"], d["t"] = r, np.arange(1, T + 1)
        rows.append(d)
    return pd.concat(rows, ignore_index=True), A1, A2, B


def test_twin_recovers_known_varx_and_selects_p():
    panel, A1, A2, B = _linear_panel()
    tw = LinearTwin().fit(panel)
    assert tw.p_bic_ == 2 and tw.p == 2 and tw.max_root_ < 1
    np.testing.assert_allclose(tw.A_[0], A1, atol=0.05)
    np.testing.assert_allclose(tw.A_[1], A2, atol=0.05)
    np.testing.assert_allclose(tw.B_[:4], B[:4], atol=0.05)


def test_twin_reduces_p_until_stable():
    A = np.zeros((2, 2, 2)); A[0] = np.eye(2) * 1.2
    assert LinearTwin.companion_max_root(A) >= 1
    A[0] = np.eye(2) * 0.5
    assert LinearTwin.companion_max_root(A) < 1


def test_twin_simulation_shares_key_path_and_initial_conditions(data):
    _, abm, _ = data
    tw = LinearTwin(p_max=3, q=6).fit(abm)
    a = tw.simulate(abm, seed=1)
    b = tw.simulate(abm, seed=1)
    c = tw.simulate(abm, seed=2)
    pd.testing.assert_frame_equal(a, b)                                         # детерминизм
    assert not np.allclose(a["g_lc"], c["g_lc"])
    src = abm.sort_values(["run", "t"]).reset_index(drop=True)
    np.testing.assert_array_equal(a["key_rate"].to_numpy(), src["key_rate"].to_numpy())  # тот же путь ставки
    np.testing.assert_array_equal(a["t"].to_numpy(), src["t"].to_numpy())
    first = a.groupby("run").head(tw.p).index
    np.testing.assert_allclose(a.loc[first, TWIN_STATE[4:]].to_numpy(), src.loc[first, TWIN_STATE[4:]].to_numpy())
    assert np.isfinite(a[TWIN_STATE].to_numpy()).all() and (a[["lr_c", "lr_r", "dr_c", "dr_r"]] >= 0.1).all().all()


@pytest.mark.parametrize("fn", [cp.forecast_m1, cp.forecast_m2])
def test_twin_trained_exactly_like_control(data, fn):
    """twin@w обучается тем же кодом, что control@w: панель, поданная как twin, даёт те же прогнозы, что поданная как control."""
    real, abm, ctl = data
    as_twin = fn(WIN, real, None, None, shares=(0.5,), twin=ctl, only_src="twin")
    as_ctl = fn(WIN, real, None, ctl, shares=(0.5,), only_src="control")
    assert set(as_twin["variant"]) == {"twin@0.5"} and set(as_ctl["variant"]) == {"control@0.5"}
    np.testing.assert_allclose(as_twin["forecast"].to_numpy(), as_ctl["forecast"].to_numpy())


@pytest.mark.parametrize("fn", [cp.forecast_m1, cp.forecast_m2])
def test_default_call_unchanged_by_twin_option(data, fn):
    """Прежний вызов (abm, control) даёт те же варианты, что до поправки: twin не появляется без явного запроса."""
    real, abm, ctl = data
    out = fn(WIN, real, abm, ctl, shares=(0.5,))
    assert not out["variant"].str.startswith("twin").any()
    assert {"real", "abm@0.5", "control@0.5"} <= set(out["variant"])


def _ttable(t1, t2, t5b, t5e, model="M3"):
    rows = []
    for sid, vals in (("T1 abm/twin ставки B+E", t1), ("T2 twin/control ставки B+E", t2),
                      ("T5 abm/twin ставки B", t5b), ("T5 abm/twin ставки E", t5e)):
        rows.append({"stat": sid, "model": model, **{f"r{k}": v for k, v in enumerate(vals)}})
    return pd.DataFrame(rows)


def test_twin_decision_rules():
    lin = _ttable([1.0, 0.99, 1.02, 0.98, 1.01, 1.0], [0.9, 0.91, 0.92, 0.9, 0.93, 0.92], [1] * 6, [1] * 6)
    assert ca.twin_decision(lin).startswith("сигнал объясняется линейной динамикой")
    beyond = _ttable([0.90, 0.91, 0.92, 0.90, 0.93, 0.92], [1.0] * 6, [0.9] * 6, [0.95] * 6)
    assert ca.twin_decision(beyond).startswith("сигнал выходит за пределы")
    small = _ttable([0.97, 0.98, 0.97, 0.96, 0.98, 0.97], [1.0] * 6, [0.9] * 6, [0.95] * 6)
    assert ca.twin_decision(small).startswith("нелинейный вклад устойчив по знаку, но мал")
    harm = _ttable([1.10, 1.12, 1.08, 1.10, 1.09, 1.11], [0.9] * 6, [1.1] * 6, [1.1] * 6)
    d = ca.twin_decision(harm)
    assert "вредит" in d and d.startswith("сигнал объясняется линейной динамикой")


M3_TWIN_SCRIPT = textwrap.dedent("""
    import numpy as np
    from abm_banks import capacity as cp, capacity_m3 as m3, forecasting as fc
    m3.MAX_STEPS, m3.EVAL_EVERY, m3.NO_VAL_STEPS = 40, 20, 20
    real = fc.load_real()
    w = cp.WINDOWS["P19"]
    abm = fc.make_abm_panel(fc.load_theta("B"), n_runs=6, months=48, seed=7, n_jobs=1)
    ctl = fc.make_control_panel(abm, 6, 48, seed=8)
    a = m3.forecast_m3(w, real, None, None, shares=(0.5,), seeds=(0,), only_src="twin", twin=ctl)
    b = m3.forecast_m3(w, real, None, ctl, shares=(0.5,), seeds=(0,), only_src="control")
    assert set(a["variant"]) == {"twin@0.5"} and set(b["variant"]) == {"control@0.5"}
    assert np.allclose(a["forecast"], b["forecast"]), "twin обучается не так, как control"
    print("ok")
""")


def test_m3_twin_trained_exactly_like_control():
    r = subprocess.run([sys.executable, "-c", M3_TWIN_SCRIPT], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0 and r.stdout.strip().endswith("ok"), r.stderr[-1500:]
