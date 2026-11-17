"""Прогнозы M3 (трансформер) для эксперимента docs/CAPACITY_PROTOCOL.md. Только torch, без lightgbm.

    python scripts/capacity_m3.py --windows P19 --jobs 6          # отладка (псевдоокно 2019)
    python scripts/capacity_m3.py --windows B A E --jobs 6        # основной запуск

Каждое задание (окно, источник синтетики) выполняется отдельным процессом; внутри — все доли и три зерна.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd                                                    # noqa: E402


def run_job(args):
    name, src, panels = args
    from abm_banks import capacity as cp
    from abm_banks import capacity_m3 as m3
    from abm_banks import forecasting as fc
    w = cp.WINDOWS[name]
    abm, ctl = cp.load_panels(w.theta, panels)
    df = m3.forecast_m3(w, fc.load_real(), abm, ctl, only_src=src)
    print(f"{name} {src}: {len(df)} прогнозов", flush=True)
    return name, df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", nargs="+", default=["P19"])
    ap.add_argument("--panels", default="outputs/capacity")
    ap.add_argument("--out", default="outputs/capacity")
    ap.add_argument("--jobs", type=int, default=1)
    a = ap.parse_args()
    jobs = [(n, s, a.panels) for n in a.windows for s in ("real", "abm", "control")]
    with mp.get_context("spawn").Pool(a.jobs) as pool:
        results = pool.map(run_job, jobs, chunksize=1)
    for name in a.windows:
        df = pd.concat([d for n, d in results if n == name], ignore_index=True)
        path = Path(a.out) / f"{name}_M3.csv"
        df.to_csv(path, index=False)
        print(f"{name} M3: {len(df)} прогнозов → {path}")


if __name__ == "__main__":
    main()
