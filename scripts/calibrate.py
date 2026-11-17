"""Калибровка методом симулированных моментов по данным ЦБ (шаг 3 плана).

Пример:
    python -I scripts/fetch_cbr_data.py          # один раз: скачать данные
    python scripts/calibrate.py --window B --budget 400 --seeds 6 --out outputs/theta_window_B.json

Окна: A — калибровка 2010–2013 (только объёмы) → проверка 2014–15; B — калибровка 2017–2021 → проверка 2022;
C — калибровка по реакциям на ключевую ставку 2014–2019 → проверка реакций 2020–2026.
Печатает таблицу моментов и ошибки вне выборки, сохраняет параметры в JSON.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd                                                  # noqa: E402

from abm_banks import ModelConfig                                    # noqa: E402
from abm_banks.calibration import SMMCalibrator, load_cbr_monthly  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--window", choices=["A", "B", "C"], default="B")
    ap.add_argument("--budget", type=int, default=300)
    ap.add_argument("--seeds", type=int, default=4, help="зёрен на одно вычисление цели")
    ap.add_argument("--method", choices=["lhs_nm", "de", "optuna"], default="lhs_nm")
    ap.add_argument("--config", help="базовый конфиг JSON/YAML")
    ap.add_argument("--out", default="outputs/theta.json")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    base = ModelConfig()
    if a.config:
        base = ModelConfig.from_json(a.config) if a.config.endswith(".json") else ModelConfig.from_yaml(a.config)
    cal = SMMCalibrator(base.replace(record_banks=False, record_network=False), load_cbr_monthly(), a.window,
                        n_seeds=a.seeds)               # параметры — по умолчанию для окна (A: только тренды роста)
    print(f"окно {a.window}: калибровка {cal.window.calib}, моментов в цели: {len(cal.keys)}")
    res = cal.fit(a.method, budget=a.budget, seed=a.seed,
                  callback=lambda i, q: print(f"\rвычисление {i}, цель {q:10.2f}", end="", flush=True))
    print(f"\nцель {res.objective:.3f}, вычислений {res.n_evals}")
    pd.set_option("display.width", 200)
    print(pd.Series(res.theta).round(4).to_string())
    print(res.moments.round(3).to_string())
    if cal.window.lp:
        from abm_banks import irf
        _, _, v = irf.evaluate_out_of_sample(res.theta, cal.window.test and ("2020-01", "2026-09"))
        print("\nпроверка реакций на отложенной выборке 2020-01…2026-09 (протокол IRF, итерация 2)")
        print(v.table.round(3).to_string(index=False))
        print(f"проходят {v.passed} из 8")
    else:
        rep = cal.validate(res.theta)
        print("\nпроверка вне выборки", cal.window.test)
        print(rep.summary().round(3).to_string())
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    res.save(a.out)
    print("сохранено:", a.out)


if __name__ == "__main__":
    main()
