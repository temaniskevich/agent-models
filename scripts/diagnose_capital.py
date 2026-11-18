"""Диагностика предложения кредита: капитал и леверидж в данных ЦБ и в модели (шаг 3б, docs/CAPITAL_DIAGNOSTIC.md).

    python scripts/diagnose_capital.py --theta C --seeds 4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np                                                      # noqa: E402
import pandas as pd                                                     # noqa: E402

from abm_banks import BankingModel, ModelConfig                         # noqa: E402
from abm_banks.calibration import load_cbr_monthly                      # noqa: E402
from abm_banks.forecasting import load_theta                            # noqa: E402
from abm_banks.scenarios import Scenario                                # noqa: E402


def data_table() -> pd.DataFrame:
    """Декабрьские значения: капитал сектора, рублёвые кредиты, леверидж, темпы роста (% за год)."""
    d = load_cbr_monthly()
    d = d.assign(loans_rub=d["loans_corp"] + d["loans_retail"])
    d = d[d.index.month == 12]
    out = pd.DataFrame({"equity": d["equity"], "loans_rub": d["loans_rub"],
                        "loans_to_equity": d["loans_rub"] / d["equity"],
                        "retail_to_equity": d["loans_retail"] / d["equity"],
                        "equity_g": d["equity"].pct_change() * 100, "loans_g": d["loans_rub"].pct_change() * 100,
                        "retail_g": d["loans_retail"].pct_change() * 100, "corp_g": d["loans_corp"].pct_change() * 100})
    return out.dropna()


def model_row(extra: dict, theta: dict, seeds: int, n: int = 96) -> dict:
    cfg = ModelConfig().replace(n_months=n, record_banks=False, record_network=False, **{**theta, **extra})
    scn = Scenario("base", key_rate=np.full(n, 0.08))
    runs = []
    for s in range(seeds):
        sec = BankingModel(cfg, scenario=scn, seed=s).run().sector
        runs.append(pd.DataFrame({"K": sec.capital, "L": sec.loans_total, "Lr": sec.loans_retail, "Lc": sec.loans_corp,
                                  "car": sec.car, "unmet": sec.credit_demand_unmet,
                                  "new": sec.new_loans_corp + sec.new_loans_retail}))
    d = sum(runs) / len(runs)
    g = lambda c: (d[c].iloc[-1] / d[c].iloc[-13] - 1) * 100            # noqa: E731
    return {"K_г/г": g("K"), "кредиты_г/г": g("L"), "розн_г/г": g("Lr"), "орг_г/г": g("Lc"),
            "L/K": d.L.iloc[-1] / d.K.iloc[-1], "CAR": d.car.iloc[-1],
            "невыполн_%кредитов": d.unmet.iloc[-1] / d.L.iloc[-1] * 100,
            "выдано_%желаемого": d.new.iloc[-1] / (d.new.iloc[-1] + d.unmet.iloc[-1]) * 100}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--theta", default="C")
    ap.add_argument("--seeds", type=int, default=4)
    a = ap.parse_args()
    pd.set_option("display.width", 200)
    print("Данные ЦБ (рублёвые кредиты, балансовый капитал сектора):")
    print(data_table().round(2).to_string())
    theta = load_theta(a.theta)
    variants = {"прежний режим": {}, "backlog_exit=0.1": {"demand.backlog_exit": 0.1},
                "backlog_exit=0.3": {"demand.backlog_exit": 0.3},
                "backlog 0.3, тренд розницы 0.15": {"demand.backlog_exit": 0.3, "demand.loan_trend_growth.1": 0.15}}
    print(f"\nМодель (theta {a.theta}, {a.seeds} зёрен, значения на 96-м месяце):")
    print(pd.DataFrame({k: model_row(v, theta, a.seeds) for k, v in variants.items()}).T.round(1).to_string())


if __name__ == "__main__":
    main()
