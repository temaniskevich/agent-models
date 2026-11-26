"""Общий набор дат, входов и меток для разных прогнозистов."""

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from generate_panels import HERE, PANELS, RULES, ARMS, fc

RATES = RULES["target_rates"]
KEY = "key_rate"
SOURCES = ("real", "abm", "twin", *ARMS[1:])


def features(data, lag):
    anchor = data[RATES].shift(lag)
    key = data[KEY]
    signals = np.column_stack([key.diff(), anchor.diff(), anchor.rsub(key, axis=0)]).astype(np.float32)
    context = RULES["context"]
    x = sliding_window_view(signals, (context, signals.shape[1]))[:, 0]
    ends = np.arange(context - 1, len(signals))
    valid = np.isfinite(x).all(axis=(1, 2))
    ends = ends[valid]
    y = data[RATES].shift(-1) - anchor
    return x[valid], y.to_numpy(dtype=np.float32)[ends], data.index[ends]


def tabular(x):
    changes = [x[:, -h:, :1 + len(RATES)].sum(axis=1) for h in (1, 3, 6)]
    return np.column_stack([*changes, x[:, -1, 1 + len(RATES):]])


def release_lag(issues, rates):
    dates = issues.to_timestamp(how="end").normalize()
    for lag in range(1, 9):
        release = rates.reindex(issues - lag).to_numpy(dtype="datetime64[ns]")
        if np.all(release <= dates.to_numpy(dtype="datetime64[ns]")):
            return lag
    raise ValueError(f"Нет общего документированного лага для выпусков {issues[0]}…{issues[-1]}")


def real_case(window):
    targets = pd.period_range(*RULES["target_windows"][window], freq="M")
    issues = targets - 1
    ledger = pd.read_csv(HERE / "release_dates.csv", parse_dates=["available_by"])
    ledger.index = pd.PeriodIndex(ledger.period, freq="M")
    lag = release_lag(issues, ledger.available_by)
    latest = issues[0] - lag
    data = fc.load_real().loc["2014-01":targets[-1], [KEY, *RATES]]
    assert data.index.equals(pd.period_range(data.index[0], data.index[-1], freq="M"))
    x, y, dates = features(data, lag)
    last_val = latest - 1
    first_val = last_val - (RULES["validation_count"] - 1)
    valid_y = np.isfinite(y).all(axis=1)
    validation = valid_y & (dates >= first_val) & (dates <= last_val)
    train = valid_y & (dates + 1 <= first_val - lag)
    test_idx = dates.get_indexer(issues)
    assert validation.sum() == RULES["validation_count"] and train.sum() > RULES["few_real_pairs"]
    assert np.all(test_idx >= 0)
    for i, issue in enumerate(issues):
        masked = data.copy()
        masked.loc[masked.index > issue - lag, RATES] += 100.0
        masked.loc[masked.index > issue, KEY] += 100.0
        altered, _, alternative_dates = features(masked, lag)
        np.testing.assert_array_equal(x[test_idx[i]], altered[alternative_dates.get_loc(issue)])
    return {"train": (x[train], y[train]), "val": (x[validation], y[validation]), "new": x[test_idx],
            "actual": data.loc[targets, RATES].to_numpy(), "anchor": data.loc[issues - lag, RATES].to_numpy(),
            "lag": lag, "targets": targets, "last_train_label": str(dates[train][-1] + 1),
            "first_release_through": str(latest)}


def synthetic(seed, source, lag):
    table = pd.read_parquet(PANELS / f"{seed}_{source}.parquet", columns=["run", "t", KEY, *RATES])
    parts = []
    for _, path in table.groupby("run", sort=False):
        x, y, _ = features(path, lag)
        ok = np.isfinite(y).all(axis=1)
        # Как в прежней синтетике, последняя строка остаётся без будущей метки.
        parts.append((x[ok], y[ok]))
    return tuple(np.concatenate([part[i] for part in parts]) for i in (0, 1))
