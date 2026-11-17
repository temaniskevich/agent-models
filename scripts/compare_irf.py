"""Сравнение реакций на ключевую ставку: данные ЦБ и модель (протокол: docs/IRF_PROTOCOL.md).

    python scripts/compare_irf.py --seeds 50 --out outputs/irf --fig docs/figures/irf_comparison.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib                                                       # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                         # noqa: E402
import pandas as pd                                                     # noqa: E402

from abm_banks import irf                                               # noqa: E402
from abm_banks.forecasting import load_theta                            # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--theta", default="B")
    ap.add_argument("--out", default="outputs/irf")
    ap.add_argument("--fig", default="docs/figures/irf_comparison.png")
    a = ap.parse_args()
    theta = load_theta(a.theta)
    data, df = irf.data_irf()
    mod_lp = irf.model_lp_irf(theta, df["key_rate"].to_numpy(), a.seeds)
    mod_step = irf.model_step_irf(theta, n_seeds=a.seeds)
    v = irf.evaluate(data, mod_lp)
    pd.set_option("display.width", 200)
    print(f"выборка данных: {df.index[0]} … {df.index[-1]} ({len(df)} мес.)")
    print(v.table.round(3).to_string(index=False))
    print(f"\nпроходят {v.passed} из {len(irf.OUTCOMES)} показателей (адекватно при ≥ 6; неадекватно при ≤ 3)")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for name, d in (("data", data), ("model_lp", mod_lp), ("model_step", mod_step)):
        pd.concat({c: x for c, x in d.items()}, names=["outcome"]).reset_index(level=0).to_csv(out / f"{name}.csv", index=False)
    v.table.to_csv(out / "verdict.csv", index=False)
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.5))
    for ax, c in zip(axes.ravel(), irf.OUTCOMES):
        d, m, s = data[c], mod_lp[c], mod_step[c]
        ax.fill_between(d["h"], d["lo90"], d["hi90"], color="#9ca3af", alpha=0.35, label="данные ЦБ, 90 % ИД")
        ax.plot(d["h"], d["beta"], color="#1f2937", lw=2, label="данные ЦБ")
        ax.fill_between(m["h"], m["lo90"], m["hi90"], color="#2563eb", alpha=0.15)
        ax.plot(m["h"], m["beta"], color="#2563eb", lw=2, label="модель (тот же путь ставки)")
        ax.plot(s["h"], s["beta"], color="#f59e0b", lw=2, ls="--", label="модель, ступень +2 п.п.")
        ax.axhline(0, color="#d1d5db", lw=0.8); ax.set_title(irf.LABELS[c], fontsize=10); ax.set_xlabel("месяцев после изменения")
    axes[0, 0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Реакция на изменение ключевой ставки на 1 п.п.: данные и модель", fontsize=13)
    fig.tight_layout()
    Path(a.fig).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.fig, dpi=110)
    print("рисунок:", a.fig)


if __name__ == "__main__":
    main()
