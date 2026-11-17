"""Синтетические панели для эксперимента «сложность × точность симулятора» (docs/CAPACITY_PROTOCOL.md).

    python scripts/capacity_make_panels.py --theta B --jobs 8

Агентная синтетика: 300 историй × 120 мес., зёрна 12345; контроль без экономики того же объёма: зёрна 12346 (как в пилоте).
Панели кэшируются в outputs/capacity/panel_{abm,control}_{A,B}.parquet; окно E и отладочное окно используют панели B.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from abm_banks import forecasting as fc                                # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--theta", choices=["A", "B"], required=True)
    ap.add_argument("--runs", type=int, default=300)
    ap.add_argument("--months", type=int, default=120)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out", default="outputs/capacity")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    abm = fc.make_abm_panel(fc.load_theta(a.theta), a.runs, a.months, a.seed, a.jobs)
    control = fc.make_control_panel(abm, a.runs, a.months, a.seed + 1)
    abm.to_parquet(out / f"panel_abm_{a.theta}.parquet")
    control.to_parquet(out / f"panel_control_{a.theta}.parquet")
    print(f"θ={a.theta}: агентная панель {abm.shape}, контроль {control.shape} → {out}")


if __name__ == "__main__":
    main()
