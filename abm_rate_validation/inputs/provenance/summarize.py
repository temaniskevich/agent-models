"""Сводка по парным прогнозам и локальным проекциям."""

import hashlib
import json
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_sources import HERE, PROTOCOL, RATES, SOURCE, cache_name

OUT = HERE / "results"
LABELS = dict(zip(RATES, ("Кредиты организациям", "Кредиты населению", "Вклады организаций", "Вклады населения")))
VARIANTS = ["naive", *PROTOCOL["generators"]]
LABEL_VARIANT = {"real": "Только реальные", "naive": "Последняя ставка", "abm": "АБМ", "twin": "Двойник",
                 "partial": "Уравнение переноса", "weak": "Ослабленный перенос"}
COLORS = {"naive": "#777777", "real": "#3C3C3C", "abm": "#0072B2", "twin": "#56B4E9",
          "partial": "#009E73", "weak": "#D55E00"}
COMPARISONS = [("abm", x) for x in ("real", "naive", "twin", "partial", "weak")] + [("partial", "real")]


def transfer_tables(out):
    raw = pd.read_csv(out / "transfer_runs.csv")
    actual = pd.read_csv(out / "transfer_data.csv")
    fits = raw.groupby(["method", "variant", "rate", "h"], as_index=False).agg(
        beta=("beta", "mean"), lo90=("beta", lambda x: x.quantile(0.05)),
        hi90=("beta", lambda x: x.quantile(0.95)), seeds=("seed", "nunique"))
    joined = fits[fits.method.eq("lp")].merge(actual, on=["rate", "h"], suffixes=("_sim", "_data"),
                                             validate="many_to_one")
    joined["difference"] = joined.beta_sim - joined.beta_data
    joined["inside_data_90"] = joined.beta_sim.between(joined.lo90_data, joined.hi90_data)
    joined["within_025"] = joined.difference.abs() <= 0.25
    joined.to_csv(out / "transfer_comparison.csv", index=False)
    subset = joined[joined.h.isin(PROTOCOL["direct_primary_horizons"])]
    aggregate = subset.groupby("variant").agg(
        rmse=("difference", lambda x: np.sqrt(np.mean(x ** 2))),
        within_025=("within_025", "sum"), inside_data_90=("inside_data_90", "sum"), cells=("rate", "size"))
    aggregate.to_csv(out / "transfer_summary.csv")
    fits.to_csv(out / "transfer_profiles.csv", index=False)
    return fits, actual, aggregate


def draw_transfer(fits, actual, out):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.1), sharex=True, sharey=True, layout="constrained")
    for rate, ax in zip(RATES, axes.flat):
        obs = actual[actual.rate.eq(rate)]
        ax.fill_between(obs.h, obs.lo90, obs.hi90, color="#dce1e6", label="90% ДИ данных")
        ax.plot(obs.h, obs.beta, color="#202124", lw=2.1, label="Данные ЦБ")
        for variant in ("abm", "weak"):
            sim = fits.loc[fits.variant.eq(variant) & fits.method.eq("lp") & fits.rate.eq(rate)]
            ax.plot(sim.h, sim.beta, color=COLORS[variant], lw=2.1, label=LABEL_VARIANT[variant])
        ax.axhline(0, color="#999999", lw=0.6)
        ax.set_title(LABELS[rate])
        ax.set_xticks([0, 1, 3, 6, 9, 12])
        ax.set_xlabel("Месяцев с изменения ключевой ставки")
        ax.set_ylabel("Ставка банка, п.п. на 1 п.п.")
    axes[0, 1].legend(loc="upper left", fontsize=8, frameon=False)
    fig.suptitle("Перенос ключевой ставки на одном историческом пути\nЯнварь 2022 — сентябрь 2026, локальные проекции")
    fig.savefig(out / "pass_through.png", dpi=180)
    plt.close(fig)


def bootstrap_ratios(squared, rng):
    draws, block = 2000, 3
    sums = np.zeros((draws, len(VARIANTS)))
    for loss in squared:
        length = len(loss)
        starts = rng.integers(length, size=(draws, (length + block - 1) // block))
        indexes = ((starts[:, :, None] + np.arange(block)) % length).reshape(draws, -1)[:, :length]
        sums += loss[indexes].sum(axis=1)
    return sums


def evaluate_forecasts(out, require_complete=False):
    raw = pd.concat([pd.read_csv(path) for path in sorted(out.glob("forecast_*_*_lag*_r*.csv"))], ignore_index=True)
    check_cols = ["model", "window", "lag", "rep", "variant", "rate", "month"]
    assert not raw.duplicated(check_cols).any()
    assert raw.groupby(["window", "month", "rate"]).actual.nunique().eq(1).all()
    if require_complete:
        required = [(model, lag) for model, lag in (("ARX", 2), ("ARX", 3), ("M3", 2))]
        for model, lag in required:
            for window, (start, end) in PROTOCOL["windows"].items():
                months = pd.period_range(start, end, freq="M")
                if window == "B" and lag == 3:
                    months = months[1:]
                records = raw.loc[raw.model.eq(model) & raw.window.eq(window) & raw.lag.eq(lag)]
                assert set(records.month) == set(months.astype(str))
                assert set(records.rate) == set(RATES)
                assert set(records.variant) == set(VARIANTS)
                assert set(records.rep) == set(PROTOCOL["synthetic_repetitions"])
                assert len(records) == len(months) * len(RATES) * len(VARIANTS) * len(PROTOCOL["synthetic_repetitions"])
                issue = pd.PeriodIndex(records.issue, freq="M")
                assert ((pd.PeriodIndex(records.month, freq="M").asi8 - issue.asi8) == 1).all()
                assert ((issue.asi8 - pd.PeriodIndex(records.last_bank_rate, freq="M").asi8) == lag).all()
                assert (pd.PeriodIndex(records.train_cut, freq="M") + lag <= issue).all()
    ensemble = raw.groupby(["model", "lag", "window", "month", "rate", "variant"], as_index=False).agg(
        actual=("actual", "first"), forecast=("forecast", "mean"), reps=("rep", "size"))
    ensemble["se"] = (ensemble.forecast - ensemble.actual) ** 2
    ensemble.to_csv(out / "forecast_ensemble.csv", index=False)
    segment = ensemble.groupby(["model", "lag", "window", "rate", "variant"], as_index=False).agg(
        mse=("se", "mean"), n_months=("month", "size"))
    segment["rmse"] = np.sqrt(segment.pop("mse"))
    segment.to_csv(out / "forecast_by_rate.csv", index=False)
    raw["se"] = (raw.forecast - raw.actual) ** 2
    repetitions = raw.groupby(["model", "lag", "window", "rep", "variant"], as_index=False).se.mean()
    repetitions["rmse"] = np.sqrt(repetitions.pop("se"))
    repetitions.to_csv(out / "forecast_repetitions.csv", index=False)
    rows, comparisons, influences = [], [], []
    for (model, lag), subset in ensemble.groupby(["model", "lag"]):
        for name in [*PROTOCOL["windows"], "all"]:
            display = subset if name == "all" else subset[subset.window.eq(name)]
            if display.empty:
                continue
            errors = display.pivot(index=["window", "month", "rate"], columns="variant", values="se")[VARIANTS]
            assert not errors.isna().any().any()
            mean_by_month = errors.groupby(level=["window", "month"]).mean()
            rmse = np.sqrt(mean_by_month.mean())
            for variant in VARIANTS:
                count = display[display.variant.eq(variant)].reps.min()
                rows.append((model, lag, name, variant, rmse[variant], len(mean_by_month), int(count)))
            blocks = [v.to_numpy() for _, v in mean_by_month.groupby(level="window")]
            sampled = bootstrap_ratios(blocks, np.random.default_rng(4104))
            for a, b in COMPARISONS:
                idx_a, idx_b = VARIANTS.index(a), VARIANTS.index(b)
                draws = np.sqrt(sampled[:, idx_a] / sampled[:, idx_b])
                lo, hi = np.quantile(draws, [0.05, 0.95])
                comparisons.append((model, lag, name, a, b, rmse[a] / rmse[b], lo, hi, len(mean_by_month)))
            if len(mean_by_month) > 1:
                more = mean_by_month.sum() - mean_by_month
                ratios = np.sqrt(more.abm / more.real)
                influences.extend((model, lag, name, window, month, ratio)
                                  for (window, month), ratio in ratios.items())
    summary = pd.DataFrame(rows, columns=["model", "lag", "window", "variant", "rmse", "n_months", "reps"])
    ratios = pd.DataFrame(comparisons, columns=["model", "lag", "window", "variant", "base", "rmse_ratio", "lo90",
                                                "hi90", "n_months"])
    summary.to_csv(out / "forecast_summary.csv", index=False)
    ratios.to_csv(out / "forecast_ratios.csv", index=False)
    pd.DataFrame(influences, columns=["model", "lag", "window", "omitted_window", "omitted_month", "rmse_ratio"]).to_csv(
        out / "forecast_leave_one_month.csv", index=False)
    return summary, ratios


def draw_forecasts(summary, ratios, out):
    s = summary[summary.lag.eq(PROTOCOL["bank_rate_release_lag_months"])]
    available = s.model.unique()
    fig, axes = plt.subplots(1, len(available), figsize=(6.1 * len(available), 4.9), squeeze=False, layout="constrained")
    panels = [*PROTOCOL["windows"], "all"]
    for model, ax in zip(available, axes.flat):
        values = s[s.model.eq(model)].pivot(index="window", columns="variant", values="rmse").reindex(panels)
        for j, variant in enumerate(VARIANTS):
            ax.bar(np.arange(len(values)) + (j - 2.5) * 0.13, values[variant], width=0.12,
                   color=COLORS[variant], label=LABEL_VARIANT[variant])
        ax.set_title("Трансформер" if model == "M3" else model)
        ax.set_xticks(np.arange(len(values)), ["2022", "2023–24", "2025–26", "Все месяцы"])
        ax.set_ylabel("RMSE, п.п.")
        ax.grid(axis="y", alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncols=3, loc="outside lower center", fontsize=8, frameon=False)
    fig.suptitle("Прогнозные ошибки на реальных данных ЦБ\nЗадержка публикации банковских ставок — два месяца")
    fig.savefig(out / "forecast_rmse.png", dpi=180)
    plt.close(fig)

    r = ratios.loc[ratios.lag.eq(PROTOCOL["bank_rate_release_lag_months"]) & ratios.variant.eq("abm")
                   & ratios.base.eq("real")].copy()
    fig, ax = plt.subplots(figsize=(8.4, 4.6), layout="constrained")
    ax.axvspan(0.95, 1.05, color="#dce1e6", alpha=0.6, label="Различие в пределах 5%")
    ax.axvline(1, color="#3C3C3C", lw=1, ls="--")
    for j, (model, group) in enumerate(r.groupby("model")):
        rr = group.set_index("window").reindex(panels).dropna()
        yy = np.arange(len(rr)) + (j - 0.5) * 0.24
        ax.hlines(yy, rr.lo90, rr.hi90, color=COLORS["abm"] if j else COLORS["partial"], lw=1.5)
        ax.plot(rr.rmse_ratio, yy, "o", color=COLORS["abm"] if j else COLORS["partial"],
                label="Трансформер" if model == "M3" else model)
    ax.set_yticks(np.arange(len(panels)), ["2022", "2023–24", "2025–26", "Все месяцы"])
    ax.invert_yaxis()
    ax.set_xlabel("Отношение RMSE обучения с АБМ к обучению только на реальных данных")
    ax.set_title("Польза агентной синтетики и неопределённость по тестовым месяцам")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncols=3, loc="outside lower center", fontsize=8)
    fig.savefig(out / "forecast_effect.png", dpi=180)
    plt.close(fig)


def draw_monthly_difference(out):
    ensemble = pd.read_csv(out / "forecast_ensemble.csv")
    x = ensemble[ensemble.lag.eq(PROTOCOL["bank_rate_release_lag_months"])]
    x = x.groupby(["model", "month", "variant"]).se.mean().unstack("variant")
    diff = (x.abm - x.real).rename("difference")
    fig, axes = plt.subplots(x.index.get_level_values("model").nunique(), 1, figsize=(10, 5.8),
                             squeeze=False, sharex=True, layout="constrained")
    for (model, frame), ax in zip(diff.groupby(level="model"), axes.flat):
        months = pd.PeriodIndex(frame.index.get_level_values("month"), freq="M").to_timestamp()
        delta = frame.to_numpy()
        ax.bar(months, delta, width=22, color=np.where(delta > 0, COLORS["weak"], COLORS["abm"]))
        ax.axhline(0, color="#74777a", lw=0.8)
        ax.set_title("Трансформер" if model == "M3" else model)
        ax.set_ylabel("Разность MSE, п.п.²")
    fig.suptitle("Вклад отдельных месяцев\nОбучение с синтетикой АБМ минус обучение только на реальных данных")
    fig.savefig(out / "forecast_monthly_difference.png", dpi=180)
    plt.close(fig)


def exploratory_retail(estimates, actual, out):
    result = pd.read_csv(out / "exploratory_retail_forecast.csv")
    agg = result.groupby(["window", "rate", "month"], as_index=False).agg(forecast=("forecast", "mean"),
                                                                         actual=("actual", "first"))
    agg["se"] = (agg.forecast - agg.actual) ** 2
    baseline = pd.read_csv(out / "forecast_ensemble.csv").query('model == "ARX" and lag == 2 and variant == "abm"')
    paired = agg.merge(baseline[["window", "rate", "month", "actual", "se"]],
                       on=["window", "rate", "month", "actual"], suffixes=("_retail", "_abm"), validate="one_to_one")
    assert len(paired) == len(agg) == len(baseline)
    result_rows = []
    for (window, rate), v in paired.groupby(["window", "rate"]):
        dropped = v[["se_retail", "se_abm"]].sum() - v[["se_retail", "se_abm"]]
        leave_one_out = np.sqrt(dropped.se_retail / dropped.se_abm)
        result_rows.append((window, rate, np.sqrt(v.se_retail.mean()), np.sqrt(v.se_abm.mean()),
                            leave_one_out.min(), leave_one_out.max()))
    ref = pd.DataFrame(result_rows, columns=["window", "rate", "rmse_retail", "rmse_abm", "leave_one_out_min",
                                             "leave_one_out_max"])
    ref.to_csv(out / "exploratory_retail_summary.csv", index=False)
    retail_fit = pd.read_csv(out / "exploratory_retail_transfer.csv").query('rate == "lr_r"')
    empirical = actual[actual.rate.eq("lr_r")]
    abm = estimates.query('method == "lp" and variant == "abm" and rate == "lr_r"')

    fig, (a, b) = plt.subplots(1, 2, figsize=(10.3, 4.3), layout="constrained")
    a.fill_between(empirical.h, empirical.lo90, empirical.hi90, color="#dce1e6", label="90% ДИ данных")
    a.plot(empirical.h, empirical.beta, color="#202124", lw=2, label="Данные ЦБ")
    a.plot(abm.h, abm.beta, color=COLORS["abm"], lw=2, label="АБМ")
    a.plot(retail_fit.h, retail_fit.beta, color=COLORS["weak"], lw=2, label="Перенос в рознице × 0,5")
    a.set_ylim(top=2.72)
    a.set_title("Локальные проекции, 2022–2026")
    a.set_ylabel("Ставка, п.п. на 1 п.п. ключевой")
    a.set_xlabel("Месяцев с изменения ключевой ставки")
    a.legend(fontsize=8, loc="upper left", frameon=False)
    s = ref[ref.rate.eq("lr_r")].set_index("window").reindex(PROTOCOL["windows"])
    b.bar(np.arange(len(s)) - 0.18, s.rmse_abm, width=0.35, color=COLORS["abm"])
    b.bar(np.arange(len(s)) + 0.18, s.rmse_retail, width=0.35, color=COLORS["weak"])
    b.set_xticks(np.arange(len(s)), ["2022", "2023–24", "2025–26"])
    b.set_title("Прогноз ARX с синтетикой")
    b.set_ylabel("RMSE, п.п.")
    fig.suptitle("Розничные кредиты\nПоисковый тест ослабления одного канала")
    fig.savefig(out / "exploratory_retail_effect.png", dpi=180)
    plt.close(fig)


def manifest(out):
    inputs = [SOURCE / "abm_banks/data" / f for f in ("cbr_monthly.csv", "calibrated_window_B.json")]
    panels = [cache_name(name, rep) for name in ("abm", "twin") for rep in PROTOCOL["synthetic_repetitions"]]
    files = sorted(HERE.glob("*.py")) + [HERE / "protocol.json"] + inputs + panels
    hashes = {str(file): hashlib.sha256(file.read_bytes()).hexdigest() for file in files if file.is_file()}
    (out / "manifest.json").write_text(json.dumps(hashes, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-complete", action="store_true")
    options = parser.parse_args()
    estimates, empirical, validation = transfer_tables(OUT)
    print(validation.round(3).to_string())
    draw_transfer(estimates, empirical, OUT)
    summary, diagnostics = evaluate_forecasts(OUT, require_complete=options.require_complete)
    print(summary[summary.window.eq("all")].round(3).to_string(index=False))
    draw_forecasts(summary, diagnostics, OUT)
    draw_monthly_difference(OUT)
    if (OUT / "exploratory_retail_forecast.csv").exists():
        exploratory_retail(estimates, empirical, OUT)
    manifest(OUT)
