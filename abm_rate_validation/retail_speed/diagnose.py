"""Дополнительная диагностика уровней ставок и чувствительности к скорости."""

import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from validate_transfer import CORP, HORIZONS, OUT, RETAIL, RULES, load_cbr_monthly, project, simulate


def replay(job):
    variant, ratio, seed, timeline, path = job
    sim = simulate(variant, ratio, path, seed)
    return sim[[CORP, RETAIL]].set_axis(timeline)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=6)
    args = parser.parse_args()
    chosen = pd.read_csv(OUT / "selected.csv", index_col="variant").ratio
    start, end = RULES["validation_window"]
    warmup = pd.Period(start, "M") - RULES["historical_warmup_months"]
    real = load_cbr_monthly().loc[warmup:end, ["key_rate", CORP, RETAIL]]
    path = real.key_rate.to_numpy() / 100
    seeds = range(RULES["validation_seed_start"], RULES["validation_seed_start"] + RULES["validation_seed_count"])
    cases = [(v, v, ratio) for v, ratio in chosen.items()]
    cases += [("shared_at_retail_ratio", "shared_speed", chosen.retail_speed),
              ("retail_ratio_0.50", "retail_speed", 0.5)]
    sample = real.loc[start:end]
    levels, sensitivity = [], []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for case, variant, ratio in cases:
            jobs = [(variant, ratio, seed, real.index, path) for seed in seeds]
            replays = list(pool.map(replay, jobs, chunksize=2))
            expected = pd.concat(replays).groupby(level=0).mean().loc[start:end]
            for outcome in (CORP, RETAIL):
                error = expected[outcome] - sample[outcome]
                levels.append({"case": case, "outcome": outcome, "ratio": ratio,
                               "level_rmse_pp": np.sqrt((error ** 2).mean()),
                               "bias_pp": error.mean(), "monthly_change_rmse_pp": np.sqrt((error.diff() ** 2).mean())})
                observed = project(sample, outcome, HORIZONS).beta.to_numpy()
                modeled = np.asarray([project(sim.assign(key_rate=real.key_rate).loc[start:end], outcome, HORIZONS).beta
                                      for sim in replays]).mean(axis=0)
                sensitivity.append({"case": case, "outcome": outcome, "ratio": ratio,
                                    "lp_rmse": np.sqrt(np.mean((modeled - observed) ** 2))})
    pd.DataFrame(levels).to_csv(OUT / "exploratory_rate_levels.csv", index=False)
    pd.DataFrame(sensitivity).to_csv(OUT / "exploratory_speed_sensitivity.csv", index=False)


if __name__ == "__main__":
    main()
