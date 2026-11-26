"""Рисунки для обсуждения результатов проверки прогнозистов."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ORDER = {"ARX": "ARX", "LightGBM": "LightGBM", "MLP": "Полносвязная сеть",
         "M3_32": "Трансформер, 26,6 тыс.", "M3_64": "Трансформер, 102,4 тыс."}
TITLES = {"E": "Июль 2023 — декабрь 2024", "F": "Июль 2025 — сентябрь 2026"}
OBS, SYN = "#a55d38", "#00776b"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": "#aab8c2", "text.color": "#14293b"})


def predictors(data):
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.5), sharex=True, sharey=True, layout="constrained")
    for ax, (scope, label) in zip(axs, TITLES.items()):
        result = data.loc[data.scope.eq(scope) & data["size"].eq("all")].set_index("model").loc[list(ORDER)]
        y = np.arange(len(result))
        ax.hlines(y, result.rmse_real, result.rmse_abm, color="#aab8c2", lw=1.8)
        ax.scatter(result.rmse_real, y, s=54, color=OBS, label="Только реальные данные", zorder=3)
        ax.scatter(result.rmse_abm, y, s=54, color=SYN, label="50% веса синтетики АБМ", zorder=3)
        ax.set_title(label, loc="left", fontweight="bold")
        ax.set_xlabel("RMSE уровня ставки, п.п.")
        ax.grid(axis="x", color="#e5eaef")
        ax.set_axisbelow(True)
        ax.set_yticks(y, list(ORDER.values()))
    axs[0].invert_yaxis()
    axs[1].legend(loc="lower right", frameon=False, fontsize=9)
    fig.savefig(HERE / "predictors_by_window.png", dpi=215, bbox_inches="tight")
    plt.close(fig)


def capacity(data):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True, layout="constrained")
    for ax, (scope, label) in zip(axs, TITLES.items()):
        sub = data.loc[data.scope.eq(scope)].set_index(["model", "size"])
        x = np.arange(2)
        for j, (model, color) in enumerate([("M3_32", "#00776b"), ("M3_64", "#64598e")]):
            values = sub.loc[[(model, "all"), (model, "few")]]
            lo, hi, gain = 100 * values.lo90, 100 * values.hi90, 100 * values.gain
            err = np.vstack((gain - lo, hi - gain))
            ax.bar(x + (2*j - 1)*0.16, gain, 0.28, yerr=err, capsize=3, label=ORDER[model], color=color)
        count = 71 if scope == "E" else 95
        ax.set_xticks(x, [f"Все пары ({count})", "24 пары"])
        ax.axhline(0, color="#516170", lw=1)
        ax.set_title(label, loc="left", fontweight="bold")
        ax.grid(axis="y", color="#e5eaef")
        ax.set_axisbelow(True)
        ax.set_ylim(-15, 35)
    axs[0].set_ylabel("Снижение RMSE с синтетикой, %")
    axs[0].legend(loc="upper left", frameon=False, fontsize=9)
    fig.savefig(HERE / "transformer_pairs.png", dpi=215, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    summary = pd.read_csv(HERE / "augmentation_effects.csv")
    predictors(summary)
    capacity(summary)
