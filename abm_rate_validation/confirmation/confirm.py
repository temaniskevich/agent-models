"""Прогноз розничной кредитной ставки по протоколу от 21.11.2026."""

import argparse
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from backtest import ridge_fit, ridge_predict, train_transformer
from data_sources import K, fc, generate_synthetic, model_config, cache_name

PROTOCOL_PATH = HERE / "protocol.json"
RULES = json.loads(PROTOCOL_PATH.read_text())
OUT = HERE / "results"
TARGET = "lr_r"
COLS = ["run", "t", K, TARGET]
FIRST_ISSUE = pd.Timestamp(RULES["prospective"]["first_issue"])


def release_ledger(path):
    ledger = pd.read_csv(path, parse_dates=["available_by"])
    ledger["period"] = pd.PeriodIndex(ledger.period, freq="M")
    assert not ledger.period.duplicated().any()
    assert (ledger.available_by > ledger.period.dt.to_timestamp(how="end").dt.normalize()).all()
    assert ledger.source_url.str.startswith(("https://www.cbr.ru/", "https://cbr.ru/")).all()
    return ledger.set_index("period").sort_index()


def available_by(ledger, issue):
    eligible = ledger[ledger.available_by <= issue]
    if eligible.empty:
        raise ValueError(f"Нет документированной даты публикации на момент выпуска {issue.date()}")
    last = eligible.index.max()
    return last, eligible.at[last, "available_by"], eligible.at[last, "source_url"]


def issues_for(name, prospective_issue=None):
    if name == "prospective":
        issue = FIRST_ISSUE if prospective_issue is None else pd.Timestamp(prospective_issue)
        target = issue.to_period("M") + 1
        assert str(target) in RULES["prospective"]["targets"]
        if issue != FIRST_ISSUE and (target == FIRST_ISSUE.to_period("M") + 1 or not issue.is_month_end):
            raise ValueError("После первого выпуска используются только последние дни следующих месяцев")
        return [issue], pd.PeriodIndex([target])
    targets = pd.period_range(*RULES["historical"][name], freq="M")
    issues = (targets - 1).to_timestamp(how="end").normalize().tolist()
    return issues, targets


def source_data(issues, snapshot=None):
    if snapshot is not None:
        source = pd.read_csv(snapshot)
        source.index = pd.PeriodIndex(source.pop("period"), freq="M")
        source = source[[K, TARGET]].sort_index()
    else:
        source = fc.load_real().loc["2014-01":, [K, TARGET]].copy()
        if issues == [FIRST_ISSUE]:
            source = source.loc[:RULES["prospective"]["first_issue_latest_bank_rate_period"]]
            for period, field in (("2026-10", "october_key_rate"), ("2026-11", "november_key_rate_projection_at_first_issue")):
                source.loc[pd.Period(period, "M"), K] = RULES["prospective"][field]
    assert source.index.equals(pd.period_range(source.index[0], source.index[-1], freq="M"))
    return source


def estimate_partial(data, cutoff):
    history = data.loc["2014-01":cutoff, [K, TARGET]].dropna()
    assert len(history) >= 24 and history.index[-1] == cutoff
    assert history.index.equals(pd.period_range(history.index[0], cutoff, freq="M"))
    k = history[K].to_numpy()[1:]
    rate = history[TARGET].to_numpy()

    def residual(theta):
        a, beta, speed = theta
        return rate[1:] - (1 - speed) * rate[:-1] - speed * (a + beta * k)

    fit = least_squares(residual, [np.mean(rate[1:] - k), 1, 0.4],
                        bounds=([-np.inf, 0, 0.05], [np.inf, 1.5, 0.95]))
    assert fit.success
    return fit.x, fit.fun


def partial_synthesis(base, params, eps, seed):
    a, beta, speed = params
    rng = np.random.default_rng(seed)
    frames = []
    for _, path in base.groupby("run", sort=True):
        cur = path[["run", "t", K]].copy()
        k = cur[K].to_numpy()
        x = np.empty(len(k))
        x[0] = max(a + beta * k[0], 0.1)
        shocks = eps[rng.integers(len(eps), size=len(k))]
        for t in range(1, len(k)):
            x[t] = max((1 - speed) * x[t - 1] + speed * (a + beta * k[t]) + shocks[t], 0.1)
        frames.append(cur.assign(**{TARGET: x}))
    return pd.concat(frames, ignore_index=True)


def matched_panels(out, params, errors, rep, panel_jobs):
    base = pd.read_parquet(cache_name("abm", rep), columns=COLS)
    assert base.run.nunique() == RULES["matched_panel"]["histories"]
    filename = out / f"retail_calibrated_r{rep}.parquet"
    if filename.exists():
        revised = pd.read_parquet(filename, columns=COLS)
        row_beta = pd.read_parquet(filename, columns=["retail_beta"]).retail_beta
        assert row_beta.eq(params[1]).all()
    else:
        theta = fc.load_theta("B")
        months = RULES["matched_panel"]["months"]
        cfg = model_config(theta, months).replace(**{"banks.pass_through_loan.1": float(params[1])})
        sample, _ = generate_synthetic(cfg, RULES["matched_panel"]["histories"],
                                       RULES["matched_panel"]["seeds_abm"][rep], fc.pilot_ranges(theta),
                                       n_jobs=panel_jobs)
        frames = []
        for run, group in sample.groupby("run", sort=True):
            rates = fc.harmonize(group, "model")[[K, TARGET]].copy()
            rates["run"], rates["t"] = run, np.arange(len(rates))
            frames.append(rates.iloc[1:][COLS])
        revised = pd.concat(frames, ignore_index=True)
        tmp = filename.with_suffix(".tmp")
        revised.assign(retail_beta=params[1]).to_parquet(tmp, index=False)
        tmp.replace(filename)
    np.testing.assert_equal(base[["run", "t", K]].to_numpy(), revised[["run", "t", K]].to_numpy())
    return {"real": None, "abm": base, "retail_calibrated": revised,
            "partial": partial_synthesis(base, params, errors, 349_193 + rep)}


def arx_features(data, lag):
    k, observed = data[K], data[TARGET].shift(lag)
    x = {f"key_{h}": k.diff(h) for h in (1, 3, 6)}
    x.update({f"own_{h}": observed.diff(h) for h in (1, 3, 6)})
    x["spread"] = k - observed
    return pd.DataFrame(x, index=data.index).assign(y=data[TARGET].shift(-1) - observed)


def neural_examples(data, lag, context=24):
    k, observed = data[K], data[TARGET].shift(lag)
    x = np.column_stack([k.diff(), observed.diff(), k - observed])
    contexts = sliding_window_view(x, (context, x.shape[1]))[:, 0]
    ends = np.arange(context - 1, len(x))
    valid = np.isfinite(contexts).all(axis=(1, 2))
    labels = (data[TARGET].shift(-1) - observed).to_numpy()[ends[valid]]
    return contexts[valid].astype(np.float32), labels[:, None].astype(np.float32), data.index[ends[valid]]


def arx_predictions(data, panels, periods, last_label_issue, lag):
    features = arx_features(data, lag)
    train, test = features.loc[:last_label_issue].dropna(), features.loc[periods]
    assert np.isfinite(test.drop(columns="y")).all().all()
    predictions = {}
    for name, panel in panels.items():
        syn = None if panel is None else pd.concat([arx_features(path, lag).dropna()
                                                    for _, path in panel.groupby("run", sort=True)])
        predictions[name] = ridge_predict(ridge_fit(train, syn), test)
    return predictions


def transformer_predictions(data, panels, periods, last_label_issue, lag):
    from abm_banks import capacity_m3 as module

    # Тот же цикл оптимизации. У сети один выход под заранее выбранный ряд.
    optimizer_settings = {p: getattr(module, p) for p in (
        "BATCH", "CLIP", "EVAL_EVERY", "LR", "MAX_STEPS", "MIN_VAL", "NO_VAL_STEPS")}
    architecture = SimpleNamespace(Net=lambda features, _: module.Net(features, 1), **optimizer_settings)
    X, Y, dates = neural_examples(data, lag)
    val_start = last_label_issue - 11
    valid_y = np.isfinite(Y[:, 0])
    train = valid_y & (dates <= val_start - lag - 1)
    val = valid_y & (dates >= val_start) & (dates <= last_label_issue)
    indices = dates.get_indexer(periods)
    assert np.all(indices >= 0)
    predictions = {}
    for name, panel in panels.items():
        synth = None
        if panel is not None:
            features = [neural_examples(path, lag)[:2] for _, path in panel.groupby("run", sort=True)]
            x = np.concatenate([x[np.isfinite(y[:, 0])] for x, y in features])
            y = np.concatenate([y[np.isfinite(y[:, 0])] for _, y in features])
            synth = (x, y)
        preds = train_transformer(architecture, (X[train], Y[train]), (X[val], Y[val]), synth, X[indices], seed=200)
        predictions[name] = preds[:, 0]
    return predictions


def direct_partial(data, params, issue_period, released):
    a, beta, speed = params
    r = float(data.at[released, TARGET])
    key_path = data.loc[released + 1:issue_period, K].to_list()
    key_path.append(float(data.at[issue_period, K]))
    for k in key_path:
        r = max((1 - speed) * r + speed * (a + beta * k), 0.1)
    return r


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_case(model, name, rep, panel_jobs, ledger_path, snapshot, issue_requested):
    issues, targets = issues_for(name, issue_requested)
    prospective = name == "prospective"
    cohort = f"prospective_{issues[0].date()}" if prospective else name
    out = OUT / cohort
    file = out / f"{model}_r{rep}.csv"
    if file.exists():
        print(f"Сохранённый выпуск {file}", flush=True)
        return
    out.mkdir(parents=True, exist_ok=True)
    periods = pd.PeriodIndex(issues, freq="M")
    ledger = release_ledger(ledger_path)
    timing = [available_by(ledger, issue) for issue in issues]
    last_bank = pd.PeriodIndex([items[0] for items in timing])
    cutoff = last_bank[0]
    lags = periods.asi8 - last_bank.asi8
    assert np.all(lags >= 1)
    raw = source_data(issues, snapshot)
    if prospective:
        data = raw.loc[:periods[0]].copy()
        data.loc[data.index > cutoff, TARGET] = np.nan
        fixed = out / "input_snapshot.csv"
        payload = data.rename_axis("period").to_csv()
        provenance_path = out / "provenance.json"
        if fixed.exists():
            assert fixed.read_text() == payload, "Срез уже зафиксирован. Смена входов требует нового протокола."
            metadata = json.loads(provenance_path.read_text())
            assert metadata["protocol_sha256"] == digest(PROTOCOL_PATH)
            assert metadata["ledger_sha256"] == digest(ledger_path)
            assert metadata["snapshot_sha256"] == digest(fixed)
        else:
            fixed.write_text(payload)
            provenance_path.write_text(json.dumps({
                "issue": str(issues[0].date()), "target": str(targets[0]),
                "last_rate_month": str(cutoff), "documented_by": str(timing[0][1].date()),
                "official_source": timing[0][2], "protocol_sha256": digest(PROTOCOL_PATH),
                "ledger_sha256": digest(ledger_path), "snapshot_sha256": digest(fixed),
            }, ensure_ascii=False, indent=2) + "\n")
    else:
        data = raw.loc[:targets[-1]]
    assert pd.notna(data.at[cutoff, TARGET]) and np.isfinite(data.loc[:periods[-1], K]).all()
    params, residuals = estimate_partial(data, cutoff)
    panels = matched_panels(out, params, residuals, rep, panel_jobs)
    preds = {arm: np.empty(len(issues)) for arm in RULES["variants"]}
    predict = arx_predictions if model == "ARX" else transformer_predictions
    for lag in np.unique(lags):
        ix = lags == lag
        batch = predict(data, panels, periods[ix], cutoff - 1, int(lag))
        for variant, values in batch.items():
            preds[variant][ix] = values + data.loc[last_bank[ix], TARGET].to_numpy()
    preds["naive"] = data.loc[last_bank, TARGET].to_numpy()
    preds["partial_direct"] = [direct_partial(data, params, period, released)
                               for period, released in zip(periods, last_bank)]
    rows = []
    used_history = data.loc[:cutoff, [K, TARGET]].dropna()
    for j, (issue, month, (released, public_by, url)) in enumerate(zip(issues, targets, timing)):
        assert public_by <= issue and used_history.index[-1] <= released < month
        for variant, values in preds.items():
            row = {"cohort": cohort, "model": model, "rep": rep, "variant": variant,
                   "issue": str(issue.date()), "target": str(month), "last_bank_rate_month": str(released),
                   "bank_rate_documented_by": str(public_by.date()), "source": url,
                   "fit_through": str(cutoff), "partial_slope_b": float(params[1]),
                   "partial_speed_lambda": float(params[2]), "forecast": float(values[j])}
            if not prospective:
                row["actual"] = float(raw.at[month, TARGET])
            rows.append(row)
    result = pd.DataFrame(rows)
    assert np.isfinite(result.forecast).all()
    tmp = file.with_suffix(".tmp")
    result.to_csv(tmp, index=False)
    tmp.replace(file)
    print(f"{cohort} {model} r{rep}: β={params[1]:.4f}, календарные лаги {lags.tolist()}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["ARX", "M3"])
    parser.add_argument("cohort", choices=[*RULES["historical"], "prospective"])
    parser.add_argument("--rep", type=int, choices=[0, 1, 2], required=True)
    parser.add_argument("--panel-jobs", type=int, default=3)
    parser.add_argument("--ledger", type=Path, default=HERE / "cbr_release_dates.csv")
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--issue")
    opts = parser.parse_args()
    selected = RULES["matched_panel"]["repetitions_arx" if opts.model == "ARX" else "repetitions_transformer"]
    if opts.rep not in selected:
        parser.error("Этот повтор отсутствует в протоколе для выбранного прогнозиста")
    if opts.issue and pd.Timestamp(opts.issue) > FIRST_ISSUE and (opts.snapshot is None or opts.cohort != "prospective"):
        parser.error("Для будущего выпуска требуется отдельный файл доступного на его дату среза")
    run_case(opts.model, opts.cohort, opts.rep, opts.panel_jobs, opts.ledger, opts.snapshot, opts.issue)


if __name__ == "__main__":
    main()
