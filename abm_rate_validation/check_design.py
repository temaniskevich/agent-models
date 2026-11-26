"""Проверка отсечек и контроль с известным сигналом."""

from pathlib import Path

import numpy as np
import pandas as pd

from backtest import dates_for_window, neural_samples, ridge_fit, ridge_frame, ridge_predict
from data_sources import HERE, K, PROTOCOL, RATES, fc


def check_timing():
    real = fc.load_real().loc["2014-01":, [K, *RATES]]
    for lag in (2, 3):
        for name in PROTOCOL["windows"]:
            targets, issues, cutoff, train_end = dates_for_window(real, name, lag)
            assert ((targets.asi8 - issues.asi8) == 1).all()
            assert cutoff == issues[0] - lag and train_end + 1 == cutoff
            fake = real.copy()
            fake.loc[cutoff + 1:, RATES] += np.linspace(20, 40, len(fake.loc[cutoff + 1:]))[:, None]
            fake.loc[issues[0] + 1:, K] -= 1000
            for target in RATES:
                a, b = ridge_frame(real, target, lag), ridge_frame(fake, target, lag)
                pd.testing.assert_frame_equal(a.loc[:train_end], b.loc[:train_end])
                pd.testing.assert_series_equal(a.loc[issues[0]].drop("y"), b.loc[issues[0]].drop("y"))
                assert a.loc[issues[0], "y"] != b.loc[issues[0], "y"]
            x, y, dates = neural_samples(real, lag, 24)
            x2, y2, dates2 = neural_samples(fake, lag, 24)
            assert dates.equals(dates2)
            np.testing.assert_equal(x[dates <= issues[0]], x2[dates <= issues[0]])
            np.testing.assert_equal(y[dates <= train_end], y2[dates <= train_end])
            assert not np.array_equal(y[dates == issues[0]], y2[dates == issues[0]])
    print("Проверены отсечки для ARX и трансформера при задержке 2 и 3 месяца.")


def observed_world(key, noise, nonlinear):
    jump = np.maximum(np.diff(key, prepend=key[0]), 0)
    r = np.empty_like(key)
    r[0] = 2 + 0.75 * key[0]
    for t in range(1, len(r)):
        r[t] = (0.45 * r[t - 1] + 0.55 * (2 + 0.75 * key[t - 1])
                + nonlinear * jump[t - 1] ** 2 + noise[t])
    return pd.DataFrame({K: key, RATES[0]: r})


def positive_control(out: Path):
    # Для одновариантного контроля сохраняем полный интерфейс признаков четырёх банковских ставок.
    def features(frame):
        frame = frame.assign(**{rate: frame[RATES[0]] for rate in RATES[1:]})
        return ridge_frame(frame, RATES[0], lag=2, nonlinear=True)

    rng = np.random.default_rng(2493)
    rows = []
    for length in (10, 18, 15):
        for world in range(20):
            altered, unchanged = [], []
            for _ in range(60):
                delta = rng.normal(0, 0.25, 120) + rng.binomial(1, 0.1, 120) * rng.choice([-2, 2], 120)
                key = 8 + np.cumsum(delta)
                noise = rng.normal(0, 0.15, 120)
                altered.append(features(observed_world(key, noise, 0.45)).dropna())
                unchanged.append(features(observed_world(key, noise, 0.0)).dropna())
            key_train = 8 + rng.normal(0, 0.15, 100).cumsum()
            delta = rng.normal(0, 0.25, length + 1)
            jumps = sorted({1, length // 2})
            delta[jumps] += 2
            key = np.r_[key_train, key_train[-1] + delta.cumsum()]
            noise = rng.normal(0, 0.15, len(key))
            real = features(observed_world(key, noise, 0.45))
            issues = np.arange(100, 100 + length)
            train, test = real.loc[:issues[0] - 3].dropna(), real.loc[issues]
            for label, syn in (("correct", altered), ("missing", unchanged), ("real", None)):
                pred = ridge_predict(ridge_fit(train, None if syn is None else pd.concat(syn)), test)
                rmse = float(np.sqrt(np.mean((pred - test["y"]) ** 2)))
                rows.append((length, world, label, rmse))
    frame = pd.DataFrame(rows, columns=["n_months", "world", "generator", "rmse"])
    frame.to_csv(out / "positive_control.csv", index=False)
    relative = frame.pivot(index=["n_months", "world"], columns="generator", values="rmse")
    print("Контроль с известным сигналом, медиана RMSE верный / пропустивший сигнал:",
          (relative.correct / relative.missing).groupby(level=0).median().round(3).to_dict())


if __name__ == "__main__":
    check_timing()
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    positive_control(out)
