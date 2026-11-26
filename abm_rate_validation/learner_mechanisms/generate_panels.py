"""Парные панели АБМ, двойника и отдельных вмешательств."""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get("ABM_BANKS_REPO", HERE.parents[1])).resolve()
sys.path.insert(0, str(REPO))
RULES = json.loads((HERE / "protocol.json").read_text())
PANELS = HERE / "panels"
ARMS = ("abm", *RULES["ablation_interventions"])

from abm_banks import ModelConfig, BankingModel
from abm_banks.batch import generate_synthetic
from abm_banks import forecasting as fc

THETA = fc.load_theta("B")
TRAJECTORY = ["run", "t", "key_rate"]


def config(arm, length):
    intervention = RULES["ablation_interventions"].get(arm, {})
    return ModelConfig().replace(n_months=length, record_banks=False, record_network=False, check_identities=True,
                                 **(THETA | intervention))


def build_arm(arm, seed, jobs):
    n, length = RULES["histories"], RULES["months_per_history"]
    cfg = config(arm, length)
    frame, draws = generate_synthetic(cfg, n, seed, fc.pilot_ranges(THETA), n_jobs=jobs)
    counts = frame.groupby("run").size()
    assert len(counts) == n and counts.eq(length).all(), (arm, seed, counts.min())
    if arm == "no_interbank":
        assert frame.mbk_volume.eq(0).all()
    elif arm == "no_liquidity_pricing":
        assert cfg.banks.liq_pressure_loan == cfg.banks.liq_pressure_deposit == 0.0
    elif arm == "equal_banks":
        model = BankingModel(cfg, seed=seed)
        for value in (model.assets0, model.car_target, model.liq_ratio, model.quality, model.speed_loan,
                      model.speed_dep, model.appetite, model.appeal, model.risk_aversion):
            assert np.ptp(value) == 0.0
    elif arm == "weak_transfer":
        assert tuple(cfg.banks.pass_through_loan) == tuple(cfg.banks.pass_through_deposit) == (0.5, 0.5)

    panel = fc.harmonize(frame, "model")
    panel["run"] = frame.run
    panel["t"] = frame.groupby("run", sort=False).cumcount()
    panel = panel.loc[panel.t > 0].reset_index(drop=True)
    assert np.isfinite(panel.to_numpy()).all(), arm
    stats = {
        "seed": seed, "arm": arm, "histories": len(counts), "months": length,
        "mbk_pct_of_assets": 100 * (frame.mbk_volume / frame.assets).mean(),
        "cb_debt_pct_of_assets": 100 * (frame.cb_debt / frame.assets).mean(),
        "hhi": frame.hhi.mean(), "exits_per_history": frame.failures.sum() / n,
    }
    for rate in fc.RATE_T:
        stats[f"{rate}_mean"] = panel[rate].mean()
    return panel, draws, stats


def save(panel, name):
    dest = PANELS / name
    tmp = dest.with_suffix(".tmp")
    panel.to_parquet(tmp, index=False)
    tmp.replace(dest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=10)
    args = parser.parse_args()
    PANELS.mkdir(exist_ok=True)
    stats = []
    for seed in RULES["seeds"]:
        original, draws, diag = build_arm("abm", seed, args.jobs)
        stats.append(diag)
        save(original, f"{seed}_abm.parquet")
        print(f"Готова исходная АБМ для {seed}", flush=True)
        twin, fitted = fc.make_twin_panel(original, seed + 70_000)
        assert fitted.max_root_ < 1
        np.testing.assert_array_equal(original[TRAJECTORY], twin[TRAJECTORY])
        save(twin, f"{seed}_twin.parquet")
        for arm in ARMS[1:]:
            panel, matched, diag = build_arm(arm, seed, args.jobs)
            pd.testing.assert_frame_equal(draws, matched, check_exact=True)
            np.testing.assert_array_equal(original[TRAJECTORY], panel[TRAJECTORY])
            stats.append(diag)
            save(panel, f"{seed}_{arm}.parquet")
            print(f"Проверены сценарии и балансы для {seed}, {arm}", flush=True)
        pd.DataFrame(stats).to_csv(HERE / "panel_diagnostics.csv", index=False)


if __name__ == "__main__":
    main()
