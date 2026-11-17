"""Пилот шага 5 (протокол: docs/PILOT_PROTOCOL.md).

    python scripts/run_pilot.py --jobs 4 --out outputs/pilot

Сохраняет ошибки по прогнозам, сводку, ячейки и тест Диболда–Мариано для окон B (2022) и A (2014–15).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd                                                    # noqa: E402

from abm_banks import forecasting as fc                                # noqa: E402

WINDOWS = {"B": (("2022-01", "2022-12"), fc.RATE_T + fc.GROWTH_T),
           "A": (("2014-01", "2015-12"), fc.GROWTH_T)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=300)
    ap.add_argument("--months", type=int, default=120)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out", default="outputs/pilot")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)
    real = fc.load_real()
    for w, (test, targets) in WINDOWS.items():
        theta = fc.load_theta(w)
        print(f"\n=== окно {w}: тест {test}, параметры {len(theta)} ===")
        abm = fc.make_abm_panel(theta, a.runs, a.months, a.seed, a.jobs)
        control = fc.make_control_panel(abm, a.runs, a.months, a.seed + 1)
        res = fc.run_pilot(real, abm, control, targets, test)
        for name, df in (("errors", res.errors), ("summary", res.summary), ("cells", res.cells), ("dm", res.dm)):
            df.to_csv(out / f"{w}_{name}.csv", index=False)
        print(res.summary.round(3).to_string(index=False))
        print(res.dm[res.dm["variant"].isin(["abm@0.5", "control@0.5"])].round(3).to_string(index=False))
        print("\nсводка по целям (RMSE, w=0.5):")
        print(res.cells[res.cells["variant"].isin(["naive", "real", "real_shrunk", "control@0.5", "abm@0.5"])]
              .pivot_table(index=["target", "h"], columns="variant", values="rmse").round(2).to_string())


if __name__ == "__main__":
    main()
