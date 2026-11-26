"""Воспроизведение месячных ставок и выдач из проверки розничной подстройки."""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "retail_speed"
REPO = Path(os.environ.get("ABM_BANKS_REPO", HERE.parents[1])).resolve()
sys.path.insert(0, str(REPO))

from abm_banks.batch import run_many
from abm_banks.calibration import load_cbr_monthly
from abm_banks.config import ModelConfig
from abm_banks.scenarios import Scenario


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    rules = json.loads((PREVIOUS / "protocol.json").read_text())
    start, end = rules["validation_window"]
    first = pd.Period(start, "M") - rules["historical_warmup_months"]
    observed = load_cbr_monthly().loc[first:end, ["key_rate", "loan_rate_retail"]]
    scenario = Scenario("retail_mix_replay", key_rate=observed.key_rate.to_numpy() / 100)
    seed0, n_seeds = rules["validation_seed_start"], rules["validation_seed_count"]
    rows, runs = [], []
    for variant in ("original", "retail_speed"):
        cfg = ModelConfig.from_json(PREVIOUS / "results" / f"config_{variant}.json").replace(
            n_months=len(observed), start_date=str(first), check_identities=True,
            record_banks=False, record_network=False,
        )
        frame = run_many(cfg, range(seed0, seed0 + n_seeds), n_jobs=args.jobs, scenarios=[scenario] * n_seeds)
        assert frame.groupby("date").size().eq(n_seeds).all() and frame.date.nunique() == len(observed)
        individual = frame.loc[frame.date.between(start, end), ["date", "seed", "loan_rate_retail"]]
        runs.append(individual.assign(variant=variant).rename(columns={"date": "month"}))
        frame = frame.groupby("date")[["loan_rate_retail", "new_loans_retail"]].mean().rename(columns={
            "loan_rate_retail": f"{variant}_rate_pct", "new_loans_retail": f"{variant}_new_loans_bn",
        })
        frame.index = pd.PeriodIndex(frame.index, freq="M", name="month")
        rows.append(frame)
    replay = pd.concat(rows, axis=1).join(observed.rename(columns={"loan_rate_retail": "project_cbr_rate_pct"}))
    replay = replay.loc[start:end]
    levels = pd.read_csv(PREVIOUS / "results" / "exploratory_rate_levels.csv")
    for variant in ("original", "retail_speed"):
        previous = levels.loc[levels["case"].eq(variant) & levels.outcome.eq("loan_rate_retail"), "bias_pp"].item()
        bias = (replay[f"{variant}_rate_pct"] - replay.project_cbr_rate_pct).mean()
        np.testing.assert_allclose(bias, previous, atol=1e-10, rtol=0)
    replay.to_csv(HERE / "model_replay.csv")
    pd.concat(runs, ignore_index=True).to_csv(HERE / "model_replay_by_seed.csv", index=False)


if __name__ == "__main__":
    main()
