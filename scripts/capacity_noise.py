"""Анализ проверки шума (поправка 19.11.2026, docs/CAPACITY_PROTOCOL.md раздел 11): статистики S1–S4 по запускам 0…5.

    python scripts/capacity_noise.py
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
    ap.add_argument("--dir", default="outputs/capacity")
    ap.add_argument("--reps", type=int, default=5, help="число повторов (запуск 0 — основной)")
    a = ap.parse_args()
    pd.set_option("display.width", 250)
    t = ca.noise_table(a.dir, reps=range(0, a.reps + 1))
    t.to_csv(Path(a.dir) / "noise_table.csv", index=False)
    print(t.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
