"""Проверка синтетики трёх версий АБМ с прежним ARX и архивом публикаций ЦБ."""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

from validate_transfer import HERE, OUT, REPO, RETAIL, RULES, SPEED, THETA, VARIANTS, design_config

os.environ["ABM_BANKS_REPO"] = str(REPO)
sys.path.insert(0, str(HERE.parent))

from abm_banks.batch import generate_synthetic
from abm_banks.forecasting import pilot_ranges
from confirmation.confirm import TARGET, arx_predictions, available_by, release_ledger, source_data
from data_sources import PROTOCOL as ARX_RULES

FORECAST = RULES["forecast"]
K = "key_rate"


def make_panel(variant, ratio, seed, jobs):
    cfg = design_config(variant, ratio, n_months=FORECAST["months_per_history"],
                        record_banks=False, record_network=False)
    ranges = pilot_ranges(THETA)
    if variant == "shared_speed":
        delta = THETA[SPEED] * (ratio - 1)
        ranges[SPEED] = tuple(x + delta for x in ranges[SPEED])
    data, draws = generate_synthetic(cfg, FORECAST["histories_per_panel"], seed, ranges, n_jobs=jobs)
    counts = data.groupby("run", sort=False).size()
    assert len(counts) == FORECAST["histories_per_panel"] and counts.eq(cfg.n_months).all()
    panel = data[["run", K, RETAIL]].rename(columns={RETAIL: TARGET})
    panel["t"] = panel.groupby("run", sort=False).cumcount()
    panel = panel.loc[panel.t > 0]
    return panel, draws


def forecast_cohort(windows, panels, ledger, seed):
    for window, bounds in windows.items():
        targets = pd.period_range(*bounds, freq="M")
        issues = (targets - 1).to_timestamp(how="end").normalize()
        periods = issues.to_period("M")
        dates = [available_by(ledger, issue) for issue in issues]
        last_rates = pd.PeriodIndex([row[0] for row in dates])
        lags = periods.asi8 - last_rates.asi8
        assert np.all(lags >= 1)
        data = source_data(issues.tolist()).loc[:targets[-1]]
        naive = data.loc[last_rates, TARGET].to_numpy()
        prediction = {v: np.empty(len(targets)) for v in panels}
        for lag in np.unique(lags):
            use = lags == lag
            fitted = arx_predictions(data, panels, periods[use], last_rates[0] - 1, int(lag))
            for variant, change in fitted.items():
                prediction[variant][use] = naive[use] + change
        prediction["naive"] = naive
        for variant, values in prediction.items():
            for i, (target, pred) in enumerate(zip(targets, values)):
                last_rate_month, available, url = dates[i]
                actual = data.at[target, TARGET]
                assert np.isfinite([actual, pred]).all()
                yield {"window": window, "seed": seed, "variant": variant, "target": str(target),
                       "issue": str(issues[i].date()), "last_rate_month": str(last_rate_month),
                       "available_by": str(available.date()), "source_url": url, "lag": int(lags[i]),
                       "training_labels_through": str(last_rates[0]), "actual": actual, "prediction": pred,
                       "error": pred - actual}


def summarize(forecasts):
    data = forecasts.assign(sq_error=forecasts.error ** 2)
    grouped = pd.concat([data, data.assign(window="pooled")], ignore_index=True)
    by_window = grouped.groupby(["window", "variant"]).sq_error.mean().pow(0.5).unstack("variant")
    by_window.to_csv(OUT / "forecast_rmse.csv")
    by_seed = grouped.groupby(["window", "seed", "variant"]).sq_error.mean().pow(0.5).unstack("variant")
    by_seed.to_csv(OUT / "forecast_rmse_by_seed.csv")
    pairs = data.groupby(["target", "variant"]).sq_error.mean().unstack("variant")
    sensitivity = {target: 1 - np.sqrt(rest.retail_speed.mean() / rest.original.mean())
                   for target in pairs.index for rest in (pairs.drop(target),)}
    pd.Series(sensitivity, name="retail_speed_rmse_gain").rename_axis("excluded_target").to_csv(
        OUT / "forecast_leave_one_month.csv")
    pooled = by_window.loc["pooled"]
    gain = float(1 - pooled.retail_speed / pooled.original)
    result = {"retail_speed_rmse_gain_over_original": gain,
              "minimum_gain_met": gain >= FORECAST["minimum_practical_rmse_gain"],
              "with_retail_speed_beats_real_only": bool(pooled.retail_speed < pooled.real),
              "leave_one_month_gain_range": [min(sensitivity.values()), max(sensitivity.values())]}
    (OUT / "forecast_decision.json").write_text(json.dumps(result, indent=2) + "\n")
    print(by_window.round(4).to_string(), result, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=10)
    args = parser.parse_args()
    assert ARX_RULES["synthetic_share"] == FORECAST["synthetic_share"]
    chosen = pd.read_csv(OUT / "selected.csv", index_col="variant").ratio
    ledger = release_ledger(HERE / FORECAST["ledger"])
    forecasts = []
    for seed in FORECAST["generator_seeds"]:
        original, draws = make_panel("original", 1., seed, args.jobs)
        panels = {"real": None, "original": original}
        for variant in VARIANTS[1:]:
            ratio = chosen[variant]
            if ratio == 1:
                panels[variant] = original
                continue
            panel, shifted = make_panel(variant, ratio, seed, args.jobs)
            pd.testing.assert_frame_equal(shifted.drop(columns=SPEED), draws.drop(columns=SPEED), check_exact=True)
            pd.testing.assert_frame_equal(panel[["run", "t", K]], original[["run", "t", K]], check_exact=True)
            delta = THETA[SPEED] * (ratio - 1) if variant == "shared_speed" else 0.0
            np.testing.assert_allclose(shifted[SPEED] - draws[SPEED], delta, rtol=0, atol=1e-14)
            panels[variant] = panel
        forecasts.extend(forecast_cohort(FORECAST["windows"], panels, ledger, seed))
        pd.DataFrame(forecasts).to_csv(OUT / "forecast_errors.csv", index=False)
        print(f"Панели и прогнозы проверены для seed={seed}", flush=True)
    summarize(pd.DataFrame(forecasts))


if __name__ == "__main__":
    main()
