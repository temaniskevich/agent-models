"""Рисунки к проверке розничной скорости, автономно от прогонов модели."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS, FIGURES = HERE / "results", HERE / "figures"
STYLES = {
    "original": ("Исходная АБМ", "#ad633e", "--"),
    "shared_speed": ("Общая скорость, переоценена", "#70689b", ":"),
    "retail_speed": ("Отдельная розничная скорость", "#00766a", "-"),
}
OUTCOMES = {"loan_rate_retail": "Кредиты населению", "loan_rate_corp": "Кредиты организациям"}
TICKS = [0, 1, 3, 6, 9, 12]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": "#bdc6ce", "text.color": "#203142",
                     "axes.labelcolor": "#203142", "xtick.color": "#344858", "ytick.color": "#344858"})


def setup_axes(ax, title):
    ax.grid(axis="y", color="#e4e9ed", lw=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", fontweight="bold")


def draw_profiles(ax, data):
    for variant, (label, color, ls) in STYLES.items():
        part = data.loc[data.variant.eq(variant)]
        ax.plot(part.h, part.beta, ls=ls, color=color, lw=2.5, label=label)


def historic():
    data = pd.read_csv(RESULTS / "observed_lp.csv").query('window == "validation_window"')
    models = pd.read_csv(RESULTS / "historical_lp.csv").query('window == "validation_window"')
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, (rate, label) in zip(axs, OUTCOMES.items()):
        setup_axes(ax, label)
        observed = data.loc[data.outcome.eq(rate)]
        ax.fill_between(observed.h, observed.lo90, observed.hi90, color="#c9d4df", alpha=0.64, lw=0)
        ax.plot(observed.h, observed.beta, color="#203142", lw=2, marker="o", ms=3.5, label="ЦБ, 90%-й ДИ")
        draw_profiles(ax, models.loc[models.outcome.eq(rate)])
        ax.axhline(0, color="#73808c", lw=0.7)
        ax.set_xticks(TICKS)
        ax.set_xlabel("Месяцев после изменения ключевой ставки")
    axs[0].set_ylabel("Отклик, п.п. на 1 п.п. ключевой")
    fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False, fontsize=10)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig(FIGURES / "historical_transfer.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def stepped():
    data = pd.read_csv(RESULTS / "step.csv").query('outcome == "loan_rate_retail"')
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, size in zip(axs, (2, -2)):
        label = "Повышение на 2 п.п." if size > 0 else "Понижение на 2 п.п."
        setup_axes(ax, label)
        draw_profiles(ax, data.loc[data.shock_pp.eq(size)])
        ax.set_xticks(TICKS)
        ax.set_xlabel("Месяцев после ступени ключевой ставки")
    axs[0].set_ylabel("Отклик розничной ставки\nп.п. на 1 п.п. ключевой")
    fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False, fontsize=10)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig(FIGURES / "paired_step.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def forecast():
    scores = RESULTS / "forecast_rmse.csv"
    if not scores.exists():
        return
    rmse = pd.read_csv(scores, index_col="window")
    windows = {"G2023": "Январь–июнь 2023", "G2025": "Январь–июнь 2025", "pooled": "Оба окна"}
    bars = {"real": ("ARX на реальных данных", "#48667c"),
            **{k: (label, c) for k, (label, c, _) in STYLES.items()},
            "naive": ("Последняя известная ставка", "#9ba7b0")}
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.7), sharey=True)
    for ax, (window, label) in zip(axs, windows.items()):
        setup_axes(ax, label)
        for x, (variant, (name, color)) in enumerate(bars.items()):
            error = rmse.at[window, variant]
            bar = ax.bar(x, error, color=color, width=0.73, label=name)
            ax.bar_label(bar, labels=[f"{error:.2f}".replace(".", ",")], fontsize=9, padding=4, color="#203142")
        ax.set_ylim(0, np.nanmax(rmse.to_numpy()) * 1.18)
        ax.set_xticks([])
    axs[0].set_ylabel("RMSE прогноза уровня ставки, п.п.")
    fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False, fontsize=10)
    fig.tight_layout(rect=(0, 0.18, 1, 1))
    fig.savefig(FIGURES / "forecast_rmse.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    FIGURES.mkdir(exist_ok=True)
    historic()
    stepped()
    forecast()
