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
    ap.add_argument("--rep", type=int, default=0, help="повтор проверки шума: другая синтетика, зерно бустинга = rep")
    ap.add_argument("--twin", action="store_true", help="только варианты twin@w (раздел 12 протокола) → файлы *_twin*.csv")
    a = ap.parse_args()
    real = fc.load_real()
    for name in a.windows:
        w = cp.WINDOWS[name]
        if a.twin:
            abm, ctl, twin, src, tag = None, None, cp.load_twin(w.theta, a.panels, a.rep), "twin", "_twin"
        else:
            (abm, ctl), twin, src, tag = cp.load_panels(w.theta, a.panels, a.rep), None, None, ""
        for model in a.models:
            df = (cp.forecast_m1(w, real, abm, ctl, twin=twin, only_src=src) if model == "M1"
                  else cp.forecast_m2(w, real, abm, ctl, lgb_seed=a.rep, twin=twin, only_src=src))
            path = Path(a.out) / f"{name}_{model}{tag}{cp.rep_suffix(a.rep)}.csv"
            df.to_csv(path, index=False)
            print(f"{name} {model}: {len(df)} прогнозов → {path}", flush=True)


if __name__ == "__main__":
    main()
