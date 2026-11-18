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

from abm_banks.capacity import rep_suffix                              # noqa: E402


def run_job(args):
    name, src, panels, rep = args
    from abm_banks import capacity as cp
    from abm_banks import capacity_m3 as m3
    from abm_banks import forecasting as fc
    w = cp.WINDOWS[name]
    seeds = (0, 1, 2) if rep == 0 else (10 * rep, 10 * rep + 1, 10 * rep + 2)
    if src == "twin":
        df = m3.forecast_m3(w, fc.load_real(), None, None, seeds=seeds, only_src="twin",
                            twin=cp.load_twin(w.theta, panels, rep))
    else:
        abm, ctl = cp.load_panels(w.theta, panels, rep)
        df = m3.forecast_m3(w, fc.load_real(), abm, ctl, seeds=seeds, only_src=src)
    print(f"{name} {src}: {len(df)} прогнозов", flush=True)
    return name, df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", nargs="+", default=["P19"])
    ap.add_argument("--panels", default="outputs/capacity")
    ap.add_argument("--out", default="outputs/capacity")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--rep", type=int, default=0, help="повтор проверки шума: другая синтетика, зёрна сети 10·rep…10·rep+2")
    ap.add_argument("--twin", action="store_true", help="только варианты twin@w (раздел 12 протокола) → файлы *_M3_twin*.csv")
    a = ap.parse_args()
    srcs = ("twin",) if a.twin else ("real", "abm", "control")
    tag = "_twin" if a.twin else ""
    jobs = [(n, s, a.panels, a.rep) for n in a.windows for s in srcs]
    with mp.get_context("spawn").Pool(a.jobs) as pool:
        results = pool.map(run_job, jobs, chunksize=1)
    for name in a.windows:
        df = pd.concat([d for n, d in results if n == name], ignore_index=True)
        path = Path(a.out) / f"{name}_M3{tag}{rep_suffix(a.rep)}.csv"
        df.to_csv(path, index=False)
        print(f"{name} M3: {len(df)} прогнозов → {path}")


if __name__ == "__main__":
    main()
