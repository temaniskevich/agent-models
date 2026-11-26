"""Поисковый контроль: то же уравнение ставок как прямой прогнозист."""

import numpy as np
import pandas as pd

from backtest import dates_for_window
from data_sources import HERE, K, PROTOCOL, RATES, fc, partial_fit


def reference_forecasts():
    real = fc.load_real().loc["2014-01":, [K, *RATES]]
    rows = []
    for lag in (2, 3):
        for name in PROTOCOL["windows"]:
            targets, issues, cutoff, _ = dates_for_window(real, name, lag)
            params, _ = partial_fit(real, cutoff)
            a, b, speed = params.T
            for month, issue in zip(targets, issues):
                observed = issue - lag
                forecast = real.loc[observed, RATES].to_numpy()
                key = real.loc[observed + 1:issue, K].to_list()
                key.extend([key[-1]] * PROTOCOL["forecast_horizon_calendar_months"])
                for k in key:
                    forecast = np.maximum((1 - speed) * forecast + speed * (a + b * k), 0.1)
                actual = real.loc[month, RATES].to_numpy()
                rows.extend((name, lag, str(month), str(issue), rate, f, y)
                            for rate, f, y in zip(RATES, forecast, actual))
    return pd.DataFrame(rows, columns=["window", "lag", "month", "issue", "rate", "forecast", "actual"])


if __name__ == "__main__":
    results = reference_forecasts()
    results.to_csv(HERE / "results/reference_partial_direct.csv", index=False)
    results["squared_error"] = (results.forecast - results.actual) ** 2
    by_window = np.sqrt(results.groupby(["lag", "window"]).squared_error.mean())
    print("Прямое уравнение ставок, RMSE, п.п.")
    print(by_window.round(3))
    print("Все окна:", np.sqrt(results.groupby("lag").squared_error.mean()).round(3).to_dict())
