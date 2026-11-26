"""Парная диагностика вмешательств в штатном стресс-сценарии АБМ."""

import argparse
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from generate_panels import HERE, ARMS, config, BankingModel
from abm_banks import preset


def run(job):
    arm, seed = job
    spec = preset("rate_shock_panic")
    cfg = config(arm, spec.n_months).replace(scenario=spec.scenario)
    data = BankingModel(cfg, seed=seed).run().sector
    assert len(data) == spec.n_months, job
    shock = spec.scenario.shock_month
    before, after = data.iloc[shock - 1], data.iloc[shock:shock + 12]
    if arm == "no_interbank":
        assert after.mbk_volume.eq(0).all()
    return {"arm": arm, "seed": seed,
            "mbk_pct_assets": 100 * (after.mbk_volume / after.assets).mean(),
            "cb_debt_pct_assets": 100 * (after.cb_debt / after.assets).mean(),
            "peak_cb_debt_pct_assets": 100 * (after.cb_debt / after.assets).max(),
            "worst_loan_change_pct": 100 * (after.loans_total.min() / before.loans_total - 1),
            "min_sector_car_pct": after.car.min(), "exits": after.failures.sum()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    jobs = [(arm, seed) for arm in ARMS for seed in range(801, 831)]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        pairs = pd.DataFrame(pool.map(run, jobs, chunksize=3))
    pairs.to_csv(HERE / "stress_by_seed.csv", index=False)
    metrics = pairs.drop(columns="seed").groupby("arm").mean()
    metrics.to_csv(HERE / "stress_summary.csv")
    print(metrics.round(3).to_string(), flush=True)


if __name__ == "__main__":
    main()
