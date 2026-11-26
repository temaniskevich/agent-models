"""Проверка отклика на одном и том же подокне с двумя рядами сравнения."""

import numpy as np
import pandas as pd

from replay import HERE
from abm_banks.irf import local_projection

HORIZONS = (0, 1, 3, 6)


def project(df, col):
    return local_projection(df, col, hs=HORIZONS).set_index("h").beta


def main():
    d = pd.read_csv(HERE / "monthly_comparison.csv", dtype={"month": str}).set_index("month")
    sims = pd.read_csv(HERE / "model_replay_by_seed.csv", dtype={"month": str})
    observed = project(d, "cbr_rate_pct")
    ex_subsidy = project(d, "implied_ex_subsidy_pct")
    profiles = pd.DataFrame({"cbr_rate": observed, "cbr_ex_subsidy": ex_subsidy})
    errors = []
    for variant, rows in sims.groupby("variant"):
        beta, beta_mixed = [], []
        for _, run in rows.groupby("seed"):
            frame = d.join(run.set_index("month").loan_rate_retail, how="left", validate="one_to_one")
            assert frame.loan_rate_retail.notna().all()
            w = frame.subsidized_share
            frame["mixed"] = (1 - w) * frame.loan_rate_retail + w * frame.subsidized_mortgage_pct
            beta.append(project(frame, "loan_rate_retail"))
            beta_mixed.append(project(frame, "mixed"))
        raw, mixed = pd.concat(beta, axis=1).mean(axis=1), pd.concat(beta_mixed, axis=1).mean(axis=1)
        profiles[variant] = raw
        profiles[f"{variant}_mixed"] = mixed
        errors.append({"variant": variant, "raw_vs_all_rmse": np.sqrt(((raw - observed) ** 2).mean()),
                       "raw_vs_ex_subsidized_rmse": np.sqrt(((raw - ex_subsidy) ** 2).mean()),
                       "mixed_vs_all_rmse": np.sqrt(((mixed - observed) ** 2).mean())})
    profiles.to_csv(HERE / "exploratory_lp.csv")
    pd.DataFrame(errors).to_csv(HERE / "exploratory_lp_rmse.csv", index=False)


if __name__ == "__main__":
    main()
