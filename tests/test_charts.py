import numpy as np
import pandas as pd
import plotly.graph_objects as go

from abm_banks import BankingModel, ModelConfig, preset
from abm_banks.batch import run_many
from ui import charts
from ui.params import all_specs, apply_widgets, fill_widgets, to_widget


def _res():
    cfg = preset("rate_shock_panic").replace(n_months=40, record_network=True)
    m = BankingModel(cfg, seed=2)
    res = m.run()
    return m, res


def test_all_figures_build_and_are_serializable():
    m, res = _res()
    df = res.sector
    figs = [charts.fig_rates(df), charts.fig_credit_deposits(df, "growth"), charts.fig_credit_deposits(df, "levels"),
            charts.fig_liquidity(df), charts.fig_risk(df, 8.0), charts.fig_exit(df), charts.fig_profit(df),
            charts.fig_structure(m.network_snapshot(), m.car_now, 8.0), charts.fig_network(m.network_snapshot()),
            charts.fig_bank_heatmap(res.banks)]
    for f in figs:
        assert isinstance(f, go.Figure)
        f.to_json()                                   # валидный JSON для браузера


def test_fan_and_validation_figures():
    runs = run_many(ModelConfig(n_months=24), [1, 2, 3, 4])
    f = charts.fig_fan(runs, "loan_rate", "t")
    assert len(f.data) >= 3
    paths = pd.DataFrame({"actual": [1.0, 2, 3], "model_mean": [1.1, 2, 2.9], "model_p05": [0.9, 1.8, 2.5],
                          "model_p95": [1.3, 2.2, 3.4]}, index=pd.period_range("2022-01", periods=3, freq="M"))
    assert len(charts.fig_validation(paths, "x").data) == 3


def test_network_figure_handles_empty_network_and_dead_banks():
    snap = {"lender": np.array([], int), "borrower": np.array([], int), "amount": np.array([]),
            "assets": np.array([5.0, 3.0, 0.0]), "active": np.array([True, True, False]),
            "reliability": np.array([0.8, 0.5, 0.0]), "bank_id": np.arange(3), "cb_debt": np.zeros(3)}
    assert isinstance(charts.fig_network(snap), go.Figure)


def test_widget_roundtrip_preserves_config():
    for name in ("baseline", "rate_shock_panic", "random_jumps"):
        cfg = preset(name)
        state = {}
        fill_widgets(cfg, state)
        back = apply_widgets(ModelConfig(), state)
        assert back.to_dict() == cfg.to_dict() or all(
            abs(float(a) - float(b)) < 1e-9
            for a, b in zip(
                [v for v in _flat(back.to_dict())], [v for v in _flat(cfg.to_dict())])
            if isinstance(a, (int, float)) and isinstance(b, (int, float)))


def _flat(d):
    for k in sorted(d):
        v = d[k]
        if isinstance(v, dict):
            yield from _flat(v)
        elif isinstance(v, list):
            yield from v
        else:
            yield v


def test_every_param_spec_exists_in_config_and_in_range():
    cfg = ModelConfig()
    for spec in all_specs():
        v = to_widget(spec, cfg)                      # AttributeError, если параметра нет в конфиге
        if spec.kind in ("float", "int"):
            assert spec.lo <= v <= spec.hi, (spec.path, v, spec.lo, spec.hi)


def test_live_altair_charts_build_and_are_valid_vega_lite():
    from ui import live_charts as lc
    m, res = _res()
    df = res.sector.assign(t=lambda d: range(len(d)))
    iv = [{"month": 5, "type": "key_rate", "value": 0.01}]
    charts_ = [lc.rates(df, iv), lc.credit_deposits(df, "growth", iv), lc.credit_deposits(df, "levels"),
               lc.liquidity(df, iv), lc.risk(df, 8.0, iv), lc.exit_panic(df, iv), lc.profit(df, iv),
               lc.structure(m.network_snapshot(), m.car_now, 8.0), lc.network(m.network_snapshot())]
    for ch in charts_:
        spec = lc.finish(ch).to_dict()                # валидация по схеме Vega-Lite
        assert "$schema" in spec
        assert "selection" not in str(spec).lower() or "params" not in spec   # без выделений: иначе у элемента появится id


def test_live_network_handles_empty_network():
    from ui import live_charts as lc
    snap = {"lender": np.array([], int), "borrower": np.array([], int), "amount": np.array([]),
            "assets": np.array([5.0, 3.0, 0.0]), "active": np.array([True, True, False]),
            "reliability": np.array([0.8, 0.5, 0.0]), "bank_id": np.arange(3), "cb_debt": np.zeros(3)}
    lc.finish(lc.network(snap)).to_dict()


def test_live_chart_spec_is_stable_between_updates_only_data_changes():
    """Защита от мигания: если спецификация меняется между тиками, Vega перерисовывает график целиком."""
    import copy
    from ui import live_charts as lc
    cfg = preset("rate_shock_panic").replace(n_months=60, record_network=True)
    m = BankingModel(cfg, seed=2)
    frames = {}
    snaps = {}
    for rec in m.iter_months():
        pass
    full = m.result().sector.assign(t=lambda d: range(len(d)))
    snap = m.network_snapshot()
    builders = {
        "rates": lambda d: lc.rates(d), "growth": lambda d: lc.credit_deposits(d, "growth"),
        "liq": lambda d: lc.liquidity(d), "risk": lambda d: lc.risk(d, 8.0), "exit": lambda d: lc.exit_panic(d),
        "profit": lambda d: lc.profit(d)}
    for name, f in builders.items():
        a = lc.finish(f(full.iloc[:30])).to_dict()
        b = lc.finish(f(full.iloc[:31])).to_dict()
        a.pop("datasets"); b.pop("datasets")
        assert a == b, f"спецификация графика {name} зависит от данных → будет мигать"
    a = lc.finish(lc.network(snap)).to_dict(); b = lc.finish(lc.network(snap)).to_dict()
    assert a == b


def test_to_spec_keeps_constant_dataset_names_and_data():
    from ui import live_charts as lc
    m, res = _res()
    df = res.sector.assign(t=lambda d: range(len(d)))
    a = lc.to_spec(lc.rates(df.iloc[:20]))
    b = lc.to_spec(lc.rates(df.iloc[:21]))
    assert set(a["datasets"]) == set(b["datasets"]) == {"rates"}
    assert len(b["datasets"]["rates"]) > len(a["datasets"]["rates"]) > 0
    assert a["datasets"]["rates"][0].keys() >= {"t", "series", "value"}
    # в слоях ссылки на именованные наборы, а не встроенные данные
    assert all(layer["data"] == {"name": "rates"} for layer in a["layer"] if "data" in layer)
