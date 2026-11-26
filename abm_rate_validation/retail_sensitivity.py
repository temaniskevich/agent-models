"""Поисковая проверка после ARX: ослабить только перенос в кредитах населению."""

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from backtest import dates_for_window, ridge_fit, ridge_frame, ridge_predict
from data_sources import HERE, K, PROTOCOL, RATES, Scenario, base_panels, fc, generate_synthetic, irf, model_config

OUT = HERE / "results"


def retail_config(theta, length):
    cfg = model_config(theta, length)
    return cfg.replace(**{"banks.pass_through_loan.1": 0.5 * cfg.banks.pass_through_loan[1]})


def history_seed(arg):
    theta, path, seed = arg
    sim = irf._run(retail_config(theta, len(path)), Scenario("historical", key_rate=path), seed).iloc[24:]
    return pd.concat([irf.local_projection(sim, full).assign(rate=short, seed=seed)
                      for full, short in zip(irf.RATES, RATES)], ignore_index=True)


def synth(theta, rep, abm):
    cfg = retail_config(theta, PROTOCOL["synthetic_history_months"])
    panel, _ = generate_synthetic(cfg, PROTOCOL["synthetic_histories_per_repetition"], 12345 + 1000 * rep,
                                  fc.pilot_ranges(theta), n_jobs=2)
    frames = []
    for run, sim in panel.groupby("run", sort=True):
        frame = fc.harmonize(sim, "model")[[K, *RATES]].copy()
        frame["run"], frame["t"] = run, np.arange(len(frame))
        frames.append(frame.iloc[1:])
    rules = pd.concat(frames, ignore_index=True)
    np.testing.assert_equal(rules[["run", "t", K]].to_numpy(), abm[["run", "t", K]].to_numpy())
    return rules


def main():
    real = fc.load_real().loc["2014-01":, [K, *RATES]]
    theta = fc.load_theta("B")
    start = pd.Period("2022-01", "M") - 24
    path = real.loc[start:"2026-09", K].to_numpy() / 100
    with ProcessPoolExecutor(max_workers=2) as pool:
        profiles = list(pool.map(history_seed, ((theta, path, seed) for seed in range(50))))
    transfer = pd.concat(profiles, ignore_index=True).groupby(["rate", "h"]).beta.mean()
    transfer.rename("beta").to_csv(OUT / "exploratory_retail_transfer.csv")

    lag = PROTOCOL["bank_rate_release_lag_months"]
    results = []
    for rep in PROTOCOL["synthetic_repetitions"]:
        panel = synth(theta, rep, base_panels(rep, OUT, 1)["abm"])
        for rate in RATES:
            synthetic = pd.concat([ridge_frame(path, rate, lag).dropna()
                                   for _, path in panel.groupby("run", sort=True)], ignore_index=True)
            data = ridge_frame(real, rate, lag)
            for window in PROTOCOL["windows"]:
                months, issues, _, last_label = dates_for_window(real, window, lag)
                model = ridge_fit(data.loc[:last_label].dropna(), synthetic)
                guesses = real.loc[issues - lag, rate].to_numpy() + ridge_predict(model, data.loc[issues])
                actual = real.loc[months, rate].to_numpy()
                results.extend((window, rate, rep, str(month), got, truth)
                               for month, got, truth in zip(months, guesses, actual))
    frame = pd.DataFrame(results, columns=["window", "rate", "rep", "month", "forecast", "actual"])
    frame.to_csv(OUT / "exploratory_retail_forecast.csv", index=False)
    print("Готов поисковый тест ослабления переноса только по розничным кредитам.", flush=True)


if __name__ == "__main__":
    main()
