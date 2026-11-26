"""Калибровка и проверка скорости подстройки на данных и парных сценариях."""

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get("ABM_BANKS_REPO", HERE.parents[1])).resolve()
sys.path.insert(0, str(REPO))
RULES = json.loads((HERE / "protocol.json").read_text())
OUT = HERE / "results"

from abm_banks import BankingModel, ModelConfig, irf
from abm_banks.calibration import load_cbr_monthly
from abm_banks.forecasting import load_theta
from abm_banks.scenarios import Scenario, step_path

THETA = load_theta(RULES["base_calibration"])
CORP, RETAIL = RULES["target_outcomes"]
HORIZONS = tuple(RULES["primary_horizons"])
VARIANTS = RULES["variants"]
SHIFT = "banks.rate_speed_loan_retail_shift"
SPEED = "banks.rate_speed_loan"
COLS = ["outcome", "h", "beta", "lo90", "hi90"]


def design_config(variant, ratio, **updates):
    params = THETA.copy()
    delta = THETA[SPEED] * (ratio - 1)
    if variant == "shared_speed":
        params[SPEED] += delta
    elif variant == "retail_speed":
        params[SHIFT] = delta
    return ModelConfig().replace(**params, **updates)


def simulate(variant, ratio, path, seed):
    cfg = design_config(variant, ratio, n_months=len(path), check_identities=True,
                        record_banks=False, record_network=False)
    return BankingModel(cfg, scenario=Scenario("key_path", key_rate=path), seed=seed).run().sector


def project(data, outcome, horizons=None):
    return irf.local_projection(data, outcome, H=RULES["max_horizon"], nlags=RULES["lp_lags"], hs=horizons)


def fit_seed(job):
    variant, ratio, path, seed = job
    sim = simulate(variant, ratio, path, seed)
    return np.array([project(sim, rate, HORIZONS).beta for rate in (CORP, RETAIL)])


def calibrate(pool, data):
    observed = np.array([project(data, rate, HORIZONS).beta for rate in (CORP, RETAIL)])
    path = data.key_rate.to_numpy() / 100
    seeds = list(range(RULES["fit_seed_start"], RULES["fit_seed_start"] + RULES["fit_seed_count"]))
    ratios = RULES["grid_speed_ratios"]
    candidates = [("original", 1.0)] + [(variant, ratio) for variant in VARIANTS[1:]
                                      for ratio in ratios if ratio != 1]
    tasks = [(variant, ratio, path, seed) for variant, ratio in candidates for seed in seeds]
    profiles = np.asarray(list(pool.map(fit_seed, tasks, chunksize=2)))
    assert np.isfinite(profiles).all() and np.isfinite(observed).all()
    profiles = profiles.reshape(len(candidates), len(seeds), 2, len(HORIZONS)).mean(axis=1)
    rows = []
    for (variant, ratio), profile in zip(candidates, profiles):
        errors = (profile - observed) ** 2
        rows.append({"variant": variant, "ratio": ratio, "delta": THETA[SPEED] * (ratio - 1),
                     "rmse_joint": np.sqrt(errors.mean()), "rmse_corp": np.sqrt(errors[0].mean()),
                     "rmse_retail": np.sqrt(errors[1].mean())})
    rows.extend(dict(rows[0], variant=variant) for variant in VARIANTS[1:])
    table = pd.DataFrame(rows).sort_values(["variant", "ratio"])
    table.to_csv(OUT / "calibration_grid.csv", index=False)
    table["dist_to_original"] = (table.ratio - 1).abs()
    best = table.sort_values(["rmse_joint", "dist_to_original", "ratio"]).groupby("variant").first()
    best.to_csv(OUT / "selected.csv")
    chosen = best.ratio.to_dict()
    for variant, ratio in chosen.items():
        design_config(variant, ratio).to_json(OUT / f"config_{variant}.json")
    return chosen


def historical_seed(job):
    variant, ratio, path, seed, first, end = job
    sim = simulate(variant, ratio, path, seed)
    sim.index = pd.period_range(first, end, freq="M")
    rows = []
    for window in ("validation_window", "validation_without_2022"):
        sample = sim.loc[slice(*RULES[window])]
        for outcome in irf.OUTCOMES:
            rows.extend((window, variant, seed, outcome, int(row.h), row.beta)
                        for row in project(sample, outcome).itertuples())
    return rows


def step_seed(job):
    variant, ratio, seed = job
    spec = RULES["step"]
    k, month, months = spec["initial_rate_percent"] / 100, spec["month_zero_based"], spec["months"]
    base = simulate(variant, ratio, np.full(months, k), seed)
    rows = []
    for size in spec["size_pp"]:
        sim = simulate(variant, ratio, step_path(k, size / 100, month, months), seed)
        pd.testing.assert_frame_equal(sim.iloc[:month], base.iloc[:month], check_exact=True)
        response = (sim[[CORP, RETAIL]] - base[[CORP, RETAIL]]).iloc[month:month + RULES["max_horizon"] + 1] / size
        rows.extend((variant, seed, size, outcome, h, beta)
                    for outcome in (CORP, RETAIL) for h, beta in enumerate(response[outcome]))
    return rows


def summarize_profiles(runs, groups):
    return (runs.groupby(groups).beta.agg(beta="mean", lo90=lambda a: a.quantile(0.05),
                                         hi90=lambda a: a.quantile(0.95)).reset_index())


def validation_metrics(observed, history):
    keys = ["window", "outcome", "h"]
    errors = history.merge(observed[keys + ["beta"]], on=keys, suffixes=("", "_observed"), validate="many_to_one")
    errors = errors.loc[errors.h.isin(HORIZONS)].assign(sq_error=lambda d: (d.beta - d.beta_observed) ** 2)
    by_outcome = errors.groupby(["window", "variant", "outcome"]).sq_error.mean().pow(0.5).unstack("outcome")
    by_outcome["joint_loan_rates"] = ((by_outcome[CORP] ** 2 + by_outcome[RETAIL] ** 2) / 2) ** 0.5
    by_outcome.to_csv(OUT / "validation_rmse.csv")
    primary = by_outcome.loc["validation_window"]
    original, shared, retail = (primary.loc[v] for v in VARIANTS)
    verdicts = {
        "retail_min_gain_met": bool(retail[RETAIL] <= original[RETAIL] * (1 - RULES["retail_min_rmse_gain"])),
        "corp_guardrail_met": bool(retail[CORP] - original[CORP] <= max(
            RULES["corp_max_absolute_rmse_loss"], original[CORP] * RULES["corp_max_relative_rmse_loss"])),
        "joint_better_than_shared": bool(retail.joint_loan_rates < shared.joint_loan_rates),
    }
    verdicts["primary_rule_met"] = all(verdicts.values())
    (OUT / "decision.json").write_text(json.dumps(verdicts, indent=2) + "\n")
    print(primary[[CORP, RETAIL, "joint_loan_rates"]].round(4).to_string(), verdicts, flush=True)


def validate(pool, data, chosen):
    observed = pd.concat([project(data.loc[slice(*RULES[window])], outcome)[COLS[1:]]
                          .assign(window=window, outcome=outcome)
                          for window in ("validation_window", "validation_without_2022")
                          for outcome in irf.OUTCOMES], ignore_index=True)
    observed.to_csv(OUT / "observed_lp.csv", index=False)
    start, end = RULES["validation_window"]
    first = pd.Period(start, "M") - RULES["historical_warmup_months"]
    path = data.loc[first:end, "key_rate"].to_numpy() / 100
    seeds = range(RULES["validation_seed_start"], RULES["validation_seed_start"] + RULES["validation_seed_count"])
    jobs = [(v, chosen[v], path, seed, first, end) for v in VARIANTS for seed in seeds]
    history = pd.DataFrame([row for block in pool.map(historical_seed, jobs, chunksize=2) for row in block],
                           columns=["window", "variant", "seed", "outcome", "h", "beta"])
    assert np.isfinite(observed.beta).all() and np.isfinite(history.beta).all()
    history.to_csv(OUT / "historical_by_seed.csv", index=False)
    historical = summarize_profiles(history, ["window", "variant", "outcome", "h"])
    historical.to_csv(OUT / "historical_lp.csv", index=False)
    validation_metrics(observed, historical)
    jobs = [(v, chosen[v], seed) for v in VARIANTS for seed in seeds]
    steps = pd.DataFrame([row for block in pool.map(step_seed, jobs, chunksize=2) for row in block],
                         columns=["variant", "seed", "shock_pp", "outcome", "h", "beta"])
    steps.to_csv(OUT / "step_by_seed.csv", index=False)
    summarize_profiles(steps, ["variant", "shock_pp", "outcome", "h"]).to_csv(OUT / "step.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    raw = load_cbr_monthly()
    fit = raw.loc[slice(*RULES["calibration_window"])].copy()
    assert fit.index.equals(pd.period_range(*RULES["calibration_window"], freq="M"))
    assert np.isfinite(fit[["key_rate", CORP, RETAIL]]).all().all()
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        chosen = calibrate(pool, fit)
        print(f"Подобранные отношения к скорости B: {chosen}", flush=True)
        validate(pool, raw, chosen)


if __name__ == "__main__":
    main()
