"""Розничная ставка и доля льготных выдач по данным Банка России."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent


def main():
    frame = pd.read_csv(HERE / "monthly_comparison.csv", index_col="month")
    x = pd.PeriodIndex(frame.index, freq="M").to_timestamp()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#aebac3"})
    fig, axs = plt.subplots(2, 1, height_ratios=(1.7, 1), figsize=(10.4, 6.8), sharex=True, layout="constrained")
    ax = axs[0]
    ax.set_title("Ставка по новым кредитам населению", fontweight="bold", loc="left")
    ax.plot(x, frame.cbr_rate_pct, color="#263546", lw=2.4, label="Банк России")
    ax.plot(x, frame.retail_speed_rate_pct, color="#ad633e", lw=2, ls="--", label="АБМ, раздельная скорость")
    ax.plot(x, frame.retail_speed_mixed_pct, color="#00766a", lw=2.2, label="АБМ с фактической долей льготной ипотеки")
    ax.set_ylabel("% годовых")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.03), fontsize=9, ncol=3)
    ax = axs[1]
    ax.set_title("Доля льготной ипотеки в новых выдачах", fontweight="bold", loc="left")
    ax.plot(x, 100 * frame.subsidized_bn / frame.mortgage_bn, color="#526786", lw=1.8, ls="--", label="В ипотеке")
    ax.plot(x, 100 * frame.subsidized_share, color="#00766a", lw=2.2, label="Во всех рублёвых выдачах без карт")
    ax.set_ylim(0, 102)
    ax.set_ylabel("% объёма выдач")
    ax.legend(frameon=False, loc="lower left", fontsize=9, ncol=2)
    ticks = x[::4]
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m.%Y"))
    for ax in axs:
        ax.grid(axis="y", color="#e1e8ed", linewidth=0.8)
        ax.set_axisbelow(True)
    fig.savefig(HERE / "retail_mix.png", dpi=190)
    plt.close(fig)


if __name__ == "__main__":
    main()
