"""Анализ эксперимента docs/CAPACITY_PROTOCOL.md по сохранённым прогнозам (ничего не обучает).

    python scripts/capacity_analyze.py --windows P19          # отладка: только таблицы, гипотезы не оцениваются
    python scripts/capacity_analyze.py --windows B A E        # основной запуск: таблицы + решения по H1–H3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd                                                    # noqa: E402

from abm_banks import capacity_analysis as ca                          # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", nargs="+", default=["P19"])
    ap.add_argument("--dir", default="outputs/capacity")
    a = ap.parse_args()
    pd.set_option("display.width", 220)
    e = ca.load_errors(a.dir, a.windows)
    c = ca.cells(e)
    summ = ca.summary_table(c)
    out = Path(a.dir)
    c.to_csv(out / "analysis_cells.csv", index=False)
    summ.to_csv(out / "analysis_summary.csv", index=False)
    dm = ca.dm_table(e)
    dm.to_csv(out / "analysis_dm.csv", index=False)
    main_w = summ[summ["share_w"] == 0.5].pivot_table(index=["group", "model"], columns="comparison", values="ratio")
    print("Отношения RMSE при w = 0,5 (< 1 — первое лучше второго):")
    print(main_w.round(3).to_string())
    print("\nВердикты при w = 0,5:")
    print(summ[summ["share_w"] == 0.5].pivot_table(index=["group", "model"], columns="comparison", values="verdict",
                                                    aggfunc="first").to_string())
    full = {"B", "A", "E"} <= set(a.windows)
    print("\nРешения по гипотезам:" if full else "\nГипотезы не оцениваются (нужны окна B, A, E).")
    if full:
        for k, v in ca.hypotheses(c, summ).items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
