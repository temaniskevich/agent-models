"""Сценарии, историческая проверка переноса и источники синтетики."""

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
SOURCE = Path(os.environ.get("ABM_BANKS_REPO", Path.home() / "Desktop/Coding/ABM_Banks"))
sys.path.insert(0, str(SOURCE))
PROTOCOL = json.loads((HERE / "protocol.json").read_text())
RATES = PROTOCOL["rates"]
K = "key_rate"

from abm_banks import forecasting as fc, irf
from abm_banks.batch import generate_synthetic
from abm_banks.config import ModelConfig
from abm_banks.scenarios import Scenario, step_path


def model_config(theta, length, weak=False):
    cfg = ModelConfig().replace(n_months=length, record_banks=False, record_network=False, **theta)
    if weak:
        scale = PROTOCOL["direct_pass_through_multiplier"]
        cfg = cfg.replace(**{
            "banks.pass_through_loan": tuple(x * scale for x in cfg.banks.pass_through_loan),
            "banks.pass_through_deposit": tuple(x * scale for x in cfg.banks.pass_through_deposit),
        })
    return cfg


def historical_seed(job):
    theta, historical, seed = job
    profiles = {"historical": (historical, 24), "flat": (np.full(60, 0.08), 24),
                "step": (step_path(0.08, 0.02, 24, 60), 24)}
    records = []
    for variant in ("abm", "weak"):
        frames = {}
        for name, (path, warmup) in profiles.items():
            cfg = model_config(theta, len(path), weak=variant == "weak")
            sim = irf._run(cfg, Scenario(name, key_rate=path), seed)
            frames[name] = sim.iloc[warmup:]
        for short, target in zip(RATES, irf.RATES):
            fitted = irf.local_projection(frames["historical"], target)
            for row in fitted.itertuples():
                records.append((variant, "lp", short, seed, row.h, row.beta))
            response = ((frames["step"][target] - frames["flat"][target]) / 2).iloc[:13]
            for h, beta in enumerate(response):
                records.append((variant, "step", short, seed, h, beta))
    return records


def direct_validation(out, jobs):
    real = fc.load_real()
    sample = real.loc["2022-01":"2026-09", [K, *RATES]].dropna()
    path = real.loc[sample.index[0] - 24:sample.index[-1], K].to_numpy() / 100
    observed = pd.concat([irf.local_projection(sample, rate).assign(rate=rate) for rate in RATES])
    theta = fc.load_theta("B")
    args = [(theta, path, seed) for seed in range(PROTOCOL["direct_seeds"])]
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        simulated = [row for batch in pool.map(historical_seed, args) for row in batch]
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(simulated, columns=["variant", "method", "rate", "seed", "h", "beta"]).to_csv(
        out / "transfer_runs.csv", index=False)
    observed.to_csv(out / "transfer_data.csv", index=False)


def cache_name(variant, rep):
    suffix = "" if rep == 0 else f"_r{rep}"
    return SOURCE / "outputs/capacity" / f"panel_{variant}_B{suffix}.parquet"


def weak_panel(rep, path, jobs):
    theta = fc.load_theta("B")
    cfg = model_config(theta, PROTOCOL["synthetic_history_months"], weak=True)
    panel, _ = generate_synthetic(cfg, PROTOCOL["synthetic_histories_per_repetition"], 12345 + 1000 * rep,
                                  fc.pilot_ranges(theta), n_jobs=jobs)
    frames = []
    for run, sim in panel.groupby("run", sort=True):
        frame = fc.harmonize(sim, "model")[[K, *RATES]].copy()
        frame["run"], frame["t"] = run, np.arange(len(frame))
        frames.append(frame.iloc[1:])
    result = pd.concat(frames, ignore_index=True)
    result.to_parquet(path, index=False)
    return result


def base_panels(rep, out, jobs):
    cols = ["run", "t", K, *RATES]
    abm_path, twin_path = cache_name("abm", rep), cache_name("twin", rep)
    if abm_path.exists() and twin_path.exists():
        abm = pd.read_parquet(abm_path, columns=cols)
        twin = pd.read_parquet(twin_path, columns=cols)
    else:
        theta = fc.load_theta("B")
        abm = fc.make_abm_panel(theta, PROTOCOL["synthetic_histories_per_repetition"],
                                PROTOCOL["synthetic_history_months"], 12345 + 1000 * rep, jobs)
        twin, _ = fc.make_twin_panel(abm, 12348 + 1000 * rep)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"weak_{rep}.parquet"
    weak = pd.read_parquet(path, columns=cols) if path.exists() else weak_panel(rep, path, jobs)
    panels = {"abm": abm[cols], "twin": twin[cols], "weak": weak[cols]}
    for name, panel in panels.items():
        if not np.array_equal(panel[["run", "t", K]].to_numpy(), abm[["run", "t", K]].to_numpy()):
            raise ValueError(f"Несовпадающие пути в панели {name}, повтор {rep}")
    return panels


def partial_fit(real, cutoff):
    history = real.loc["2014-01":cutoff, [K, *RATES]].dropna()
    k = history[K].iloc[1:].to_numpy()
    params, errors = [], []
    for rate in RATES:
        r = history[rate].to_numpy()

        def residual(theta):
            a, b, speed = theta
            return r[1:] - (1 - speed) * r[:-1] - speed * (a + b * k)

        start = [np.mean(r[1:] - k), 1, 0.4]
        fit = least_squares(residual, start, bounds=([-np.inf, 0, 0.05], [np.inf, 1.5, 0.95]))
        params.append(fit.x)
        errors.append(fit.fun)
    return np.asarray(params), np.column_stack(errors)


def partial_panel(paths, params, errors, seed):
    a, b, speed = params.T
    rng = np.random.default_rng(seed)
    frames = []
    for _, path in paths.groupby("run", sort=True):
        frame = path[["run", "t", K]].copy()
        k = frame[K].to_numpy()
        rate = np.empty((len(k), len(RATES)))
        rate[0] = np.maximum(a + b * k[0], 0.1)
        noise = errors[rng.integers(len(errors), size=len(k))]
        for t in range(1, len(k)):
            rate[t] = np.maximum((1 - speed) * rate[t - 1] + speed * (a + b * k[t]) + noise[t], 0.1)
        frame[RATES] = rate
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("part", choices=["transfer", "panels"])
    parser.add_argument("--out", type=Path, default=HERE / "results")
    parser.add_argument("--jobs", type=int, default=4)
    cli = parser.parse_args()
    if cli.part == "transfer":
        direct_validation(cli.out, cli.jobs)
        print("Готовы локальные проекции и парные сценарии ставки.", flush=True)
    else:
        for rep in PROTOCOL["synthetic_repetitions"]:
            base_panels(rep, cli.out, cli.jobs)
            print(f"Готовы панели, повтор {rep}.", flush=True)
