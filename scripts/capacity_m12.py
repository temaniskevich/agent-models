"""Прогнозы M1 (ARX) и M2 (бустинг) для эксперимента docs/CAPACITY_PROTOCOL.md. Без torch (конфликт OpenMP).

    python scripts/capacity_m12.py --windows P19          # отладка (псевдоокно 2019)
    python scripts/capacity_m12.py --windows B A E        # основной запуск
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd                                                    # noqa: E402

from abm_banks import capacity as cp                                   # noqa: E402
from abm_banks import forecasting as fc                                # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", nargs="+", default=["P19"], choices=list(cp.WINDOWS))
    ap.add_argument("--panels", default="outputs/capacity")
    ap.add_argument("--out", default="outputs/capacity")
    ap.add_argument("--models", nargs="+", default=["M1", "M2"], choices=["M1", "M2"])
    a = ap.parse_args()
    real = fc.load_real()
    for name in a.windows:
        w = cp.WINDOWS[name]
        abm, ctl = cp.load_panels(w.theta, a.panels)
        for model in a.models:
            fn = cp.forecast_m1 if model == "M1" else cp.forecast_m2
            df = fn(w, real, abm, ctl)
            path = Path(a.out) / f"{name}_{model}.csv"
            df.to_csv(path, index=False)
            print(f"{name} {model}: {len(df)} прогнозов → {path}", flush=True)


if __name__ == "__main__":
    main()
