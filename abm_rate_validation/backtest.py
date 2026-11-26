"""Псевдореальный прогноз следующего месяца с задержкой публикации ставок."""

import argparse
import copy
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from data_sources import HERE, K, PROTOCOL, RATES, base_panels, fc, partial_fit, partial_panel


def aligned(series, lag):
    """Индекс — месяц выпуска прогноза; r в признаках доступна с лагом."""
    observed_rates = series[RATES].shift(lag)
    known_key = series[K]
    values = series[RATES].shift(-PROTOCOL["forecast_horizon_calendar_months"]) - observed_rates
    return known_key, observed_rates, values


def ridge_frame(series, target, lag=2, nonlinear=False):
    k, observed, values = aligned(series, lag)
    own = observed[target]
    x = {f"key_{h}": k.diff(h) for h in (1, 3, 6)}
    x.update({f"own_{h}": own.diff(h) for h in (1, 3, 6)})
    x["spread"] = k - own
    if nonlinear:
        x["positive_jump"] = k.diff().clip(lower=0.0)
        x["positive_jump_squared"] = x["positive_jump"] ** 2
    return pd.DataFrame(x, index=series.index).assign(y=values[target])


def ridge_fit(real, synthetic=None):
    X = real.drop(columns="y").to_numpy()
    center = X.mean(axis=0)
    scale = np.maximum(X.std(axis=0), 0.25)
    pools = (real,) if synthetic is None else (real, synthetic)
    p = X.shape[1] + 1
    xx, xy = np.zeros((p, p)), np.zeros(p)
    w = PROTOCOL["synthetic_share"] if synthetic is not None else 0.0
    for frame, weight in zip(pools, (1 - w, w)):
        z = (frame.drop(columns="y").to_numpy() - center) / scale
        z = np.column_stack([np.ones(len(z)), z])
        xx += weight * (z.T @ z) / len(z)
        xy += weight * (z.T @ frame["y"].to_numpy()) / len(z)
    penalty = 0.01 * np.eye(p)
    penalty[0, 0] = 0
    beta = np.linalg.solve(xx + penalty, xy)
    return center, scale, beta


def ridge_predict(model, frame):
    center, scale, beta = model
    return beta[0] + ((frame.drop(columns="y").to_numpy() - center) / scale) @ beta[1:]


def arx_predictions(real, panels, issue_dates, last_label_issue, lag):
    predictions = {name: [] for name in PROTOCOL["generators"]}
    for rate in RATES:
        data = ridge_frame(real, rate, lag)
        train = data.loc[:last_label_issue].dropna()
        for name in predictions:
            synthetic = None
            if name != "real":
                synthetic = pd.concat([ridge_frame(run, rate, lag).dropna()
                                       for _, run in panels[name].groupby("run", sort=True)], ignore_index=True)
            model = ridge_fit(train, synthetic)
            predictions[name].append(ridge_predict(model, data.loc[issue_dates]))
    return {name: np.column_stack(values) for name, values in predictions.items()}


def neural_samples(series, lag, ctx):
    k, observed, y = aligned(series, lag)
    x = np.column_stack([k.diff(), observed.diff(), observed.rsub(k, axis=0)])
    contexts = sliding_window_view(x, (ctx, x.shape[1]))[:, 0]
    ends = np.arange(ctx - 1, len(x))
    valid = np.isfinite(contexts).all(axis=(1, 2)) & np.isfinite(y.to_numpy()[ends]).all(axis=1)
    return contexts[valid].astype(np.float32), y.to_numpy()[ends[valid]].astype(np.float32), series.index[ends[valid]]


def train_transformer(module, train, validation, synthetic, Xnew, seed):
    import torch

    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng_real = np.random.default_rng(seed)
    rng_syn = np.random.default_rng(seed + 5000)
    Xr, Yr = train
    Xv, Yv = validation
    mu_x, mu_y = Xr.mean(axis=(0, 1)), Yr.mean(axis=0)
    sd_x, sd_y = np.maximum(Xr.std(axis=(0, 1)), 0.25), np.maximum(Yr.std(axis=0), 0.25)

    def inputs(x):
        return torch.from_numpy(np.clip((x - mu_x) / sd_x, -module.CLIP, module.CLIP).astype(np.float32))

    def targets(y):
        return torch.from_numpy(((y - mu_y) / sd_y).astype(np.float32))

    xr, yr, xv, yv = inputs(Xr), targets(Yr), inputs(Xv), targets(Yv)
    n_syn = 0
    if synthetic is not None:
        xs, ys = inputs(synthetic[0]), targets(synthetic[1])
        n_syn = round(module.BATCH * PROTOCOL["synthetic_share"])
    n_real = module.BATCH - n_syn
    net = module.Net(Xr.shape[-1], len(RATES))
    optimizer = torch.optim.Adam(net.parameters(), lr=module.LR)
    best, best_state = np.inf, None
    use_val = len(xv) >= module.MIN_VAL
    steps = module.MAX_STEPS if use_val else module.NO_VAL_STEPS
    for step in range(1, steps + 1):
        net.train()
        rows = rng_real.integers(0, len(xr), n_real)
        x, y = xr[rows], yr[rows]
        if n_syn:
            rows = rng_syn.integers(0, len(xs), n_syn)
            x, y = torch.cat([x, xs[rows]]), torch.cat([y, ys[rows]])
        optimizer.zero_grad()
        loss = torch.nn.functional.huber_loss(net(x), y)
        loss.backward()
        optimizer.step()
        if use_val and step % module.EVAL_EVERY == 0:
            net.eval()
            with torch.no_grad():
                score = float(torch.nn.functional.huber_loss(net(xv), yv))
            if score < best:
                best, best_state = score, copy.deepcopy(net.state_dict())
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        predictions = net(inputs(Xnew)).numpy() * sd_y + mu_y
    return predictions


def transformer_predictions(real, panels, issues, last_label_issue, lag, seed):
    from abm_banks import capacity_m3 as module
    from abm_banks.capacity import CTX

    X, Y, dates = neural_samples(real, lag, CTX)
    first_validation = last_label_issue - 11
    # К первой валидационной дате должны быть опубликованы все цели обучающих пар.
    train = dates <= first_validation - lag - PROTOCOL["forecast_horizon_calendar_months"]
    val = (dates >= first_validation) & (dates <= last_label_issue)
    score = dates.get_indexer(issues)
    if np.any(score < 0):
        raise ValueError("Есть даты пробного прогноза без полного контекста")
    predictions = {}
    for name in PROTOCOL["generators"]:
        syn = None
        if name != "real":
            samples = [neural_samples(run, lag, CTX) for _, run in panels[name].groupby("run", sort=True)]
            syn = tuple(np.concatenate([s[i] for s in samples], axis=0) for i in (0, 1))
        predictions[name] = train_transformer(module, (X[train], Y[train]), (X[val], Y[val]), syn, X[score], seed)
    return predictions


def dates_for_window(real, name, lag):
    targets = pd.period_range(*PROTOCOL["windows"][name], freq="M")
    horizon = PROTOCOL["forecast_horizon_calendar_months"]
    cutoff_calibration = pd.Period("2021-12", "M")
    targets = targets[(targets - horizon - lag >= cutoff_calibration) & (targets <= real.index[-1])]
    issues = targets - horizon
    latest_train_target = issues[0] - lag
    return targets, issues, latest_train_target, latest_train_target - horizon


def one_job(args):
    out, model, name, lag, rep = args
    final = out / f"forecast_{model}_{name}_lag{lag}_r{rep}.csv"
    if final.exists():
        return f"Готовый результат {final.name}"
    real = fc.load_real().loc["2014-01":, [K, *RATES]]
    targets, issues, train_cut, last_label_issue = dates_for_window(real, name, lag)
    assert last_label_issue + PROTOCOL["forecast_horizon_calendar_months"] + lag == issues[0]
    panels = base_panels(rep, out, 1)
    params, eps = partial_fit(real, train_cut)
    panels["partial"] = partial_panel(panels["abm"], params, eps, 349_193 + rep)
    assert panels["partial"][["run", "t", K]].reset_index(drop=True).equals(
        panels["abm"][["run", "t", K]].reset_index(drop=True))
    if model == "ARX":
        preds = arx_predictions(real, panels, issues, last_label_issue, lag)
    else:
        preds = transformer_predictions(real, panels, issues, last_label_issue, lag, seed=200 + rep)
    preds["naive"] = np.zeros_like(preds["real"])
    current = real.loc[issues - lag, RATES].to_numpy()
    actual = real.loc[targets, RATES].to_numpy()
    rows = []
    for variant, changes in preds.items():
        forecast = current + changes
        for i, (month, issue) in enumerate(zip(targets, issues)):
            for j, rate in enumerate(RATES):
                assert issue - lag < month and train_cut <= issue - lag
                rows.append((model, name, lag, rep, variant, rate, str(month), str(issue), str(issue - lag),
                             str(train_cut), float(forecast[i, j]), float(actual[i, j])))
    frame = pd.DataFrame(rows, columns=["model", "window", "lag", "rep", "variant", "rate", "month", "issue",
                                        "last_bank_rate", "train_cut", "forecast", "actual"])
    assert np.isfinite(frame[["forecast", "actual"]]).all().all()
    tmp = final.with_suffix(".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(final)
    return f"{model}, {name}, лаг {lag}, повтор {rep}: {len(frame)} прогнозов"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["ARX", "M3"])
    parser.add_argument("--out", type=Path, default=HERE / "results")
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--reps", type=int, nargs="+", default=PROTOCOL["synthetic_repetitions"])
    parser.add_argument("--windows", nargs="+", default=list(PROTOCOL["windows"]))
    parser.add_argument("--lag", type=int, default=PROTOCOL["bank_rate_release_lag_months"])
    args = parser.parse_args()
    # Все панели должны быть готовы до параллельного расчёта прогнозов.
    for rep in args.reps:
        if not (args.out / f"weak_{rep}.parquet").is_file():
            parser.error(f"Сначала подготовьте панели, отсутствует weak_{rep}.parquet")
    jobs = [(args.out, args.model, name, args.lag, rep) for rep in args.reps for name in args.windows]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for status in pool.map(one_job, jobs):
            print(status, flush=True)


if __name__ == "__main__":
    main()
