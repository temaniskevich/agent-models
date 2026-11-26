"""Проверка полноты, ретроспективные ошибки и оценка зафиксированных выпусков."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from confirm import HERE, OUT, RULES, PROTOCOL_PATH, TARGET, arx_features, available_by, digest, estimate_partial, issues_for, release_ledger

SEARCH_PROTOCOL = json.loads((HERE.parent / "protocol.json").read_text())


def ensemble(folder, prospective=False, require_complete=False):
    frames = []
    for model, key in (("ARX", "repetitions_arx"), ("M3", "repetitions_transformer")):
        for rep in RULES["matched_panel"][key]:
            path = folder / f"{model}_r{rep}.csv"
            if not path.exists():
                if require_complete:
                    raise FileNotFoundError(path)
                continue
            frames.append(pd.read_csv(path))
    raw = pd.concat(frames, ignore_index=True)
    keys = ["model", "target", "variant"]
    cols = {"forecast": ("forecast", "mean"), "issue": ("issue", "first"),
            "fit_through": ("fit_through", "first"), "beta": ("partial_slope_b", "first"),
            "replicates": ("rep", "nunique")}
    assert not raw.duplicated([*keys, "rep"]).any()
    assert ((pd.PeriodIndex(raw.target, freq="M").asi8 - pd.PeriodIndex(raw.issue, freq="M").asi8) == 1).all()
    assert (pd.to_datetime(raw.bank_rate_documented_by) <= pd.to_datetime(raw.issue)).all()
    assert (pd.PeriodIndex(raw.fit_through, freq="M") <= pd.PeriodIndex(raw.last_bank_rate_month, freq="M")).all()
    assert set(raw.variant) == set(RULES["variants"] + RULES["benchmarks"])
    if prospective:
        assert "actual" not in raw
    else:
        assert raw.groupby("target").actual.nunique().eq(1).all()
        cols["actual"] = ("actual", "first")
    return raw.groupby(keys, as_index=False).agg(**cols)


def verify_leakage():
    ledger = release_ledger(HERE / "cbr_release_dates.csv")
    from confirm import source_data

    previous = {str(p) for dates in SEARCH_PROTOCOL["windows"].values() for p in pd.period_range(*dates, freq="M")}
    for name in RULES["historical"]:
        issues, targets = issues_for(name)
        assert not previous.intersection(str(p) for p in targets)
        data = source_data(issues)
        cutoff = available_by(ledger, issues[0])[0]
        for issue, target in zip(issues, targets):
            observed, _, _ = available_by(ledger, issue)
            t = issue.to_period("M")
            lag = t.ordinal - observed.ordinal
            fake = data.copy()
            fake.loc[observed + 1:, TARGET] += 250
            fake.loc[t + 1:, "key_rate"] -= 200
            ref, alt = arx_features(data, lag), arx_features(fake, lag)
            pd.testing.assert_frame_equal(ref.loc[:cutoff - 1], alt.loc[:cutoff - 1])
            pd.testing.assert_series_equal(ref.loc[t].drop("y"), alt.loc[t].drop("y"))
            np.testing.assert_array_equal(estimate_partial(data, cutoff)[0], estimate_partial(fake, cutoff)[0])
            assert ref.loc[t, "y"] != alt.loc[t, "y"] and target == t + 1


def summaries(ensemble_results, out):
    frame = ensemble_results.copy()
    frame["error"] = frame.forecast - frame.actual
    rmse = frame.groupby(["model", "cohort", "variant"]).error.agg(lambda x: np.sqrt(np.mean(x ** 2)))
    pooled = frame.groupby(["model", "variant"]).error.agg(lambda x: np.sqrt(np.mean(x ** 2)))
    clinical = pd.concat([rmse.rename("rmse").reset_index(), pooled.rename("rmse").reset_index().assign(cohort="both")])
    tableau = clinical.pivot(index=["model", "cohort"], columns="variant", values="rmse")
    tableau["calibrated_to_original"] = tableau.retail_calibrated / tableau.abm
    tableau.to_csv(out / "historical_summary.csv")
    frame.to_csv(out / "historical_ensemble.csv", index=False)
    # Две короткие траектории. Сохраняем парные ошибки, без асимптотического теста.
    pairs = frame.pivot(index=["model", "cohort", "target"], columns="variant", values="error") ** 2
    diagnostics = []
    for model, errors in pairs.groupby(level="model"):
        pair = errors[["retail_calibrated", "abm"]]
        omit = pair.sum() - pair
        ratios = np.sqrt(omit.retail_calibrated / omit.abm)
        diagnostics.append({"model": model, "months": len(pair), "calibrated_wins": int((pair.retail_calibrated < pair.abm).sum()),
                            "leave_one_month_min": float(ratios.min()), "leave_one_month_max": float(ratios.max())})
    (out / "historical_sensitivity.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    print(tableau.round(4).to_string())
    draw_paths(frame, out)


def draw_paths(frame, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = sorted(frame.model.unique())
    cohorts = list(RULES["historical"])
    fig, axes = plt.subplots(len(models), len(cohorts), figsize=(10.5, 3.1 * len(models) + 0.6),
                             squeeze=False, layout="constrained")
    labels = {"abm": "Исходная АБМ", "retail_calibrated": "АБМ с оценённым переносом",
              "partial": "Синтетика уравнения", "real": "Только реальные данные"}
    colors = {"abm": "#0072B2", "retail_calibrated": "#D55E00", "partial": "#009E73", "real": "#888888"}
    for i, model in enumerate(models):
        for j, cohort in enumerate(cohorts):
            ax = axes[i, j]
            subset = frame[(frame.model == model) & (frame.cohort == cohort)]
            piv = subset.pivot(index="target", columns="variant", values="forecast")
            actual = subset.groupby("target").actual.first().reindex(piv.index)
            ax.plot(actual.index, actual, color="#161616", lw=2.4, marker="o", label="Данные ЦБ")
            for variant, label in labels.items():
                ax.plot(piv.index, piv[variant], color=colors[variant], lw=1.5, label=label)
            ax.set_title(f"{model} · {cohort[-4:]}")
            ax.set_ylabel("Ставка, % годовых")
            ax.set_xticks(piv.index, [s[-2:] for s in piv.index])
            ax.grid(axis="y", alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=3, fontsize=8, frameon=False)
    fig.suptitle("Ставка по новым рублёвым кредитам населению\nРетроспективная проверка на дополнительных прогнозных датах")
    fig.savefig(out / "historical_retail_forecasts.png", dpi=155)
    plt.close(fig)


def current_release(folder):
    provenance = json.loads((folder / "provenance.json").read_text())
    assert digest(PROTOCOL_PATH) == provenance["protocol_sha256"]
    assert digest(folder / "input_snapshot.csv") == provenance["snapshot_sha256"]
    data = pd.read_csv(folder / "input_snapshot.csv")
    assert data.loc[data.period > provenance["last_rate_month"], TARGET].isna().all()
    scored = ensemble(folder, prospective=True, require_complete=True)
    assert (scored.target == provenance["target"]).all()
    frozen = folder / "frozen_forecasts.csv"
    hashes_path = folder / "forecast_hashes.json"
    files = sorted(folder.glob("*_r*.csv")) + [frozen, folder / "input_snapshot.csv", PROTOCOL_PATH]
    if frozen.exists():
        assert frozen.read_text() == scored.to_csv(index=False)
        assert {p.name: digest(p) for p in files} == json.loads(hashes_path.read_text())
    else:
        frozen.write_text(scored.to_csv(index=False))
        hashes_path.write_text(json.dumps({p.name: digest(p) for p in files}, indent=2) + "\n")
    print(f"Срез {provenance['issue']}, цель {provenance['target']}")
    print(scored.pivot(index="model", columns="variant", values="forecast").round(4).to_string())
    return scored


def score_prospective(facts):
    actual = pd.read_csv(facts, dtype={"target": str}, parse_dates=["published_on"])
    assert not actual.target.duplicated().any()
    assert np.isfinite(actual.actual).all()
    assert actual.source_url.str.startswith(("https://www.cbr.ru/", "https://cbr.ru/")).all()
    assert (actual.published_on > pd.PeriodIndex(actual.target, freq="M").to_timestamp(how="end").normalize()).all()
    assert set(actual.target).issubset(RULES["prospective"]["targets"])
    forecasts = []
    for folder in sorted(OUT.glob("prospective_*")):
        frozen = folder / "frozen_forecasts.csv"
        if not frozen.exists():
            continue
        recorded = json.loads((folder / "forecast_hashes.json").read_text())
        assert digest(frozen) == recorded[frozen.name]
        forecasts.append(pd.read_csv(frozen))
    frame = pd.concat(forecasts, ignore_index=True)
    assert not frame.duplicated(["model", "target", "variant"]).any()
    scored = frame.merge(actual, on="target", how="inner", validate="many_to_one")
    if scored.empty:
        raise ValueError("Нет опубликованных фактов для зафиксированных прогнозов")
    assert (pd.to_datetime(scored.issue) < scored.published_on).all()
    scored["error"] = scored.forecast - scored.actual
    scored.to_csv(OUT / "prospective_scored.csv", index=False)
    print(scored.groupby(["model", "variant"]).error.agg(n="size", rmse=lambda x: np.sqrt(np.mean(x ** 2))).round(4))
    n = scored.target.nunique()
    if n < len(RULES["prospective"]["targets"]):
        print(f"Промежуточная оценка, опубликованных месяцев {n} из {len(RULES['prospective']['targets'])}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["historical", "freeze", "score"])
    parser.add_argument("--facts", type=Path)
    parser.add_argument("--issue", default=RULES["prospective"]["first_issue"])
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    if args.mode == "historical":
        verify_leakage()
        results = [ensemble(OUT / cohort, require_complete=args.require_complete).assign(cohort=cohort)
                   for cohort in RULES["historical"]]
        summaries(pd.concat(results, ignore_index=True), OUT)
    elif args.mode == "freeze":
        current_release(OUT / f"prospective_{args.issue}")
    else:
        if args.facts is None:
            parser.error("Укажите CSV с target,actual,published_on,source_url")
        score_prospective(args.facts)
