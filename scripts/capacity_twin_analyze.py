"""Анализ поправки 18.11.2026 (docs/CAPACITY_PROTOCOL.md, раздел 12): линейный двойник как второй контроль.

    python scripts/capacity_twin_analyze.py              # запуски 0…5
    python scripts/capacity_twin_analyze.py --reps 0     # только основной запуск (отладка механики)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd                                                    # noqa: E402

from abm_banks import capacity_analysis as ca                          # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="outputs/capacity")
    ap.add_argument("--reps", type=int, default=5)
    a = ap.parse_args()
    pd.set_option("display.width", 250)
    d = Path(a.dir)
    for theta in ("A", "B"):
        p = d / f"twin_check_{theta}.json"
        if p.exists():
            ch = json.loads(p.read_text(encoding="utf-8"))
            print(f"Двойник θ={theta}: p по BIC {ch['p_bic']}, p {ch['p']}, наибольший корень {ch['max_root']:.4f}, наблюдений {ch['n_obs']}")
    per_rep = ca._rep_cells(d, range(0, a.reps + 1))
    t = ca.twin_table(d, per_rep=per_rep)
    t.to_csv(d / "twin_table.csv", index=False)
    print("\nСтатистики T1–T5 (w = 0,5):")
    print(t.round(3).to_string(index=False))
    print("\nПравило вывода (раздел 12):")
    for m in ca.MODELS:
        print(f"  {m}: {ca.twin_decision(t, m)}")
    c0 = per_rep[0]
    summ0 = ca.summary_table(c0)
    summ0.to_csv(d / "twin_summary_r0.csv", index=False)
    h0 = ca.hypotheses(c0, summ0, ctl="twin")
    print("\nH2 с двойником вместо контроля:")
    print(f"  запуск 0: {h0['H2']}")
    if a.reps > 0:
        print(f"  среднее по запускам: {ca.h2_averaged(per_rep, ctl='twin')}")
    print(f"\nH3 с двойником (запуск 0): {h0['H3']}")


if __name__ == "__main__":
    main()
