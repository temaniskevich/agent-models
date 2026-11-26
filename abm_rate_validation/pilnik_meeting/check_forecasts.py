"""Псевдореальный прогноз после пересчёта состава розничных выдач."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RULES = json.loads((HERE / "protocol.json").read_text())
sys.path.insert(0, str(HERE.parent / "retail_speed"))

from validate_forecast import ARX_RULES, FORECAST, K, TARGET, make_panel
from abm_banks.synthetic import LinearTwin
from backtest import ridge_fit, ridge_predict
from confirmation.confirm import arx_features, fc

MIX = "subsidized_share"
SUB = "subsidized_mortgage_pct"
MIXED = ("real_mix", "original_mix", "retail_speed_mix", "twin_mix", "twin_retail_speed_mix")


def inputs():
    source = pd.read_csv(HERE.parent / "retail_mix" / "monthly_comparison.csv")
    source.index = pd.PeriodIndex(source.pop("month"), freq="M")
    source = source[[MIX, SUB, "cbr_rate_pct"]].rename(columns={"cbr_rate_pct": TARGET})
    observed = fc.load_real().loc[RULES["first_composition_month"]:RULES["targets"][1], [K, TARGET]]
    np.testing.assert_array_equal(source[TARGET], observed[TARGET])
    return observed.join(source[[MIX, SUB]])


def synthetic_features(panels, lag):
    return {name: pd.concat([arx_features(path, lag).dropna()
                             for _, path in panel.groupby("run", sort=True)], ignore_index=True)
            for name, panel in panels.items()}


def forecast(data, issue, last, training_synthetic, subsidized_rate):
    lag = issue.ordinal - last.ordinal
    known = data.loc[:issue].copy()
    known.loc[known.index > last, [TARGET, MIX, SUB]] = np.nan
    anchor = known.loc[last]
    raw_frame = arx_features(known, lag)
    cleaned = known.assign(**{TARGET: (known[TARGET] - known[MIX] * subsidized_rate) / (1 - known[MIX])})
    market = arx_features(cleaned, lag)
    train, test = market.loc[:last - 1].dropna(), market.loc[[issue]]
    assert len(train) > train.shape[1] and test.drop(columns="y").notna().all().all()
    assert train.index.max() + 1 == last
    predictions = {}
    for variant in MIXED:
        change = ridge_predict(ridge_fit(train, training_synthetic.get(variant)), test)[0]
        rate = cleaned.at[last, TARGET] + change
        predictions[variant] = ((1 - anchor[MIX]) * rate + anchor[MIX] * subsidized_rate, rate)
    raw = ridge_fit(raw_frame.loc[:last - 1].dropna())
    predictions["real_raw"] = (float(anchor[TARGET] + ridge_predict(raw, raw_frame.loc[[issue]])[0]), None)
    predictions["naive"] = (float(anchor[TARGET]), None)
    return predictions, len(train)


def long_arx(data, issue, last):
    known = data.loc[:issue].copy()
    known.loc[known.index > last, TARGET] = np.nan
    features = arx_features(known, issue.ordinal - last.ordinal)
    train = features.loc[:last - 1].dropna()
    result = known.at[last, TARGET] + ridge_predict(ridge_fit(train), features.loc[[issue]])[0]
    return (float(result), None), len(train)


def score(rows):
    forecasts = pd.DataFrame(rows)
    sensitivity = forecasts.assign(sq=forecasts.error_pp ** 2).groupby(
        ["predicted_subsidized_rate", "variant"]).sq.mean().pow(.5).unstack("variant")
    sensitivity.to_csv(HERE / "forecast_rate_sensitivity.csv")
    forecasts = forecasts.loc[forecasts.predicted_subsidized_rate.eq(RULES["subsidized_rate_assumption"])]
    forecasts.to_csv(HERE / "forecast_errors.csv", index=False)
    f = forecasts.assign(sq=forecasts.error_pp ** 2, sq_market=forecasts.market_error_pp ** 2)
    score = f.groupby("variant").agg(rmse_pp=("sq", "mean"), rmse_ex_subsidy_pp=("sq_market", "mean"),
                                     bias_pp=("error_pp", "mean"), dates=("target", "nunique"))
    score[["rmse_pp", "rmse_ex_subsidy_pp"]] = score[["rmse_pp", "rmse_ex_subsidy_pp"]].pow(.5)
    score.to_csv(HERE / "forecast_summary.csv")
    f.groupby(["seed", "variant"]).sq.mean().pow(.5).unstack().to_csv(HERE / "forecast_by_seed.csv")
    by_date = f.groupby(["target", "variant"]).sq.mean().unstack()
    loo = []
    for target in by_date.index:
        rmse = by_date.drop(target).mean().pow(.5)
        loo.append({"excluded_target": target,
                    "gain_original_vs_real": 1 - rmse.original_mix / rmse.real_mix,
                    "gain_original_vs_twin": 1 - rmse.original_mix / rmse.twin_mix,
                    "gain_retail_speed_vs_twin": 1 - rmse.retail_speed_mix / rmse.twin_retail_speed_mix,
                    "gain_retail_speed_vs_naive": 1 - rmse.retail_speed_mix / rmse.naive})
    pd.DataFrame(loo).to_csv(HERE / "forecast_leave_one_month.csv", index=False)
    full = score.rmse_pp
    decision = {"gain_original_vs_real": float(1 - full.original_mix / full.real_mix),
                "gain_original_vs_twin": float(1 - full.original_mix / full.twin_mix),
                "gain_retail_speed_vs_twin": float(1 - full.retail_speed_mix / full.twin_retail_speed_mix),
                "gain_retail_speed_vs_original": float(1 - full.retail_speed_mix / full.original_mix),
                "composition_only_rmse_pp": float(np.sqrt(np.mean(f.composition_error_pp.dropna() ** 2)))}
    decision["practical_gain_over_real_met"] = decision["gain_original_vs_real"] >= RULES["minimum_gain"]
    (HERE / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    print(score.round(4).to_string(), decision, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=10)
    args = parser.parse_args()
    assert RULES["synthetic_share"] == ARX_RULES["synthetic_share"]
    assert RULES["synthetic_histories"] == FORECAST["histories_per_panel"]
    assert RULES["synthetic_months"] == FORECAST["months_per_history"]
    chosen = pd.read_csv(HERE.parent / "retail_speed/results/selected.csv", index_col="variant").ratio
    ledger = pd.read_csv(HERE / "release_dates.csv", parse_dates=["available_by"])
    ledger.index = pd.PeriodIndex(ledger.pop("period"), freq="M")
    live = inputs()
    long_raw = fc.load_real().loc["2014-01":RULES["targets"][1], [K, TARGET]]
    assert live.notna().all().all()
    assert live[MIX].between(0, 1, inclusive="neither").all()
    targets = pd.period_range(*RULES["targets"], freq="M")
    cases = []
    for target in targets:
        issue = target - 1
        issue_date = issue.to_timestamp(how="end").normalize()
        last = ledger.loc[ledger.available_by <= issue_date].index.max()
        assert last < issue
        cases.append((target, issue, issue_date, last))
    rows, twins = [], []
    assumptions = [RULES["subsidized_rate_assumption"], *RULES["subsidized_rate_sensitivity"]]
    for seed, twin_seed in zip(RULES["generator_seeds"], RULES["twin_seeds"], strict=True):
        original, draws = make_panel("original", 1.0, seed, args.jobs)
        slower, other_draws = make_panel("retail_speed", chosen.retail_speed, seed, args.jobs)
        pd.testing.assert_frame_equal(draws, other_draws, check_exact=True)
        panels = {"original_mix": original, "retail_speed_mix": slower}
        for name, source in (("twin_mix", original), ("twin_retail_speed_mix", slower)):
            twin = LinearTwin(state=[TARGET], q=12, p_max=6).fit(source)
            assert twin.max_root_ < 1
            panels[name] = twin.simulate(source, twin_seed)
            twins.append({"variant": name, "seed": seed, "p": twin.p, "q": twin.q, "max_root": twin.max_root_})
        for other in panels.values():
            np.testing.assert_array_equal(original[["run", "t", K]], other[["run", "t", K]])
        features = {lag: synthetic_features(panels, lag) for lag in {b.ordinal - d.ordinal for _, b, _, d in cases}}
        for target, issue, issue_date, last in cases:
            prepared = features[issue.ordinal - last.ordinal]
            long_prediction, long_count = long_arx(long_raw, issue, last)
            long_mutated = long_raw.copy()
            long_mutated.loc[long_mutated.index > last, TARGET] += 1.0
            long_mutated.loc[long_mutated.index > issue, K] += 10.0
            assert long_arx(long_mutated, issue, last) == (long_prediction, long_count)
            mutated = live.copy()
            mutated.loc[mutated.index > last, [TARGET, MIX, SUB]] += 1.0
            mutated.loc[mutated.index > issue, K] += 10.0
            actual, anchor = live.loc[target], live.loc[last]
            market_actual = (actual[TARGET] - actual[MIX] * actual[SUB]) / (1 - actual[MIX])
            for subsidized_rate in assumptions:
                predictions, n = forecast(live, issue, last, prepared, subsidized_rate)
                assert forecast(mutated, issue, last, prepared, subsidized_rate) == (predictions, n)
                predictions["real_raw_long"] = long_prediction
                composition_error = (1 - anchor[MIX]) * market_actual + anchor[MIX] * subsidized_rate - actual[TARGET]
                for variant, (aggregate, ex_subsidized) in predictions.items():
                    market_error = ex_subsidized - market_actual if ex_subsidized is not None else np.nan
                    if variant in MIXED:
                        np.testing.assert_allclose(aggregate - actual[TARGET],
                                                   (1 - anchor[MIX]) * market_error + composition_error, atol=1e-10)
                    rows.append({"variant": variant, "seed": seed, "target": str(target), "issue": str(issue_date.date()),
                                 "data_through": str(last), "available_by": str(ledger.at[last, "available_by"].date()),
                                 "train_count": long_count if variant == "real_raw_long" else n,
                                 "predicted_share": anchor[MIX], "actual_share": actual[MIX],
                                 "predicted_subsidized_rate": subsidized_rate, "actual_subsidized_rate": actual[SUB],
                                 "prediction": aggregate, "actual": actual[TARGET], "error_pp": aggregate - actual[TARGET],
                                 "prediction_ex_subsidy": ex_subsidized, "actual_ex_subsidy": market_actual,
                                 "market_error_pp": market_error,
                                 "composition_error_pp": composition_error if variant in MIXED else np.nan})
        print(f"Проверены хронология, синтетические пути и отсечение входов для seed={seed}", flush=True)
    pd.DataFrame(twins).to_csv(HERE / "linear_twin.csv", index=False)
    score(rows)


if __name__ == "__main__":
    main()
