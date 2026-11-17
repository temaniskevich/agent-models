"""Эксперимент «сложность модели прогноза × точность симулятора» (протокол — docs/CAPACITY_PROTOCOL.md).

Этот модуль не импортирует torch (конфликт OpenMP с lightgbm на macOS: M1+M2 и M3 запускаются отдельными процессами).
Содержит окна, признаки, обучающие выборки и модели M1 (ARX) и M2 (градиентный бустинг). M3 — ``capacity_m3.py``.

Режим основного эксперимента — одна точка обучения на окно: обучающие пары — реальные (вход, цель), у которых месяц цели
раньше первого месяца окна; прогнозы на все месяцы окна делаются по реальному контексту на момент прогноза, без переобучения.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import forecasting as fc

SHARES = (0.25, 0.5, 0.75, 0.9)          # доли синтетики; основная — 0,5
MAIN_SHARE = 0.5
HORIZONS = (1, 3, 6)
CTX = 24                                  # контекст M3, мес.
ROW_WARMUP = 12                           # первые месяцы каждой синтетической истории — прогрев предикторов (как в пилоте)

# параметры бустинга зафиксированы протоколом, по тесту не настраиваются
LGB_PARAMS = {"objective": "regression", "learning_rate": 0.05, "num_leaves": 15, "min_data_in_leaf": 20,
              "bagging_fraction": 0.8, "bagging_freq": 1, "feature_fraction": 0.8, "seed": 0, "verbose": -1,
              "deterministic": True, "force_row_wise": True, "num_threads": 1}
LGB_ROUNDS = 300


@dataclass(frozen=True)
class CWindow:
    name: str
    test: tuple[str, str]               # месяцы цели
    targets: tuple[str, ...]
    theta: str                          # какой калибровке соответствует синтетика (окно A или B)
    debug: bool = False

    @property
    def start(self) -> pd.Period:
        return pd.Period(self.test[0], "M")

    @property
    def months(self) -> pd.PeriodIndex:
        return pd.period_range(self.test[0], self.test[1], freq="M")


ALL = tuple(fc.RATE_T + fc.GROWTH_T)
WINDOWS = {
    "B": CWindow("B", ("2022-01", "2022-12"), ALL, "B"),
    "A": CWindow("A", ("2014-01", "2015-12"), tuple(fc.GROWTH_T), "A"),
    "E": CWindow("E", ("2023-07", "2024-12"), ALL, "B"),
    # псевдоокно только для отладки механики (протокол, раздел 7); результаты не интерпретируются
    "P19": CWindow("P19", ("2019-01", "2019-12"), ALL, "B", debug=True),
}
GROUPS = {"ставки": fc.RATE_T, "объёмы": fc.GROWTH_T}


def group_of(target: str) -> str:
    return "ставки" if target in fc.RATE_T else "объёмы"


def variants(shares=SHARES) -> list[str]:
    return ["real"] + [f"{s}@{w}" for s in ("abm", "control") for w in shares]


# --------------------------------------------------------------------------
# Обучающие пары и признаки
# --------------------------------------------------------------------------
def train_cut(window: CWindow, h: int) -> pd.Period:
    """Последний допустимый месяц ПРОГНОЗА для обучающей пары: месяц цели (t0 + h) раньше начала окна."""
    return window.start - 1 - h


def m2_features(df: pd.DataFrame, target: str, series: tuple[str, ...] | list[str]) -> pd.DataFrame:
    """Признаки бустинга на момент прогноза t0 (индекс — t0). Уровень ключевой ставки не используется:
    реальный ряд до 2013-09 сдвинут на константу (см. ``forecasting.load_real``), поэтому только разности."""
    k = df["key_rate"]
    X = {"dk1": k.diff(), "dk3": k.diff(3), "dk6": k.diff(6), "dk12": k.diff(12)}
    own = df[target].diff() if target in fc.RATE_T else df[target]
    for j in range(1, 7):
        X[f"own_l{j}"] = own.shift(j - 1)                  # lag 1 — последнее известное значение
    for s in series:
        X[f"x3_{s}"] = df[s].diff(3) if s in fc.RATE_T else df[s].rolling(3).mean()
    return pd.DataFrame(X, index=df.index)


def m2_pool(panel: pd.DataFrame, target: str, series) -> dict[int, pd.DataFrame]:
    out: dict[int, list] = {h: [] for h in HORIZONS}
    for _, g in panel.groupby("run"):
        g = g.reset_index(drop=True)
        X = m2_features(g, target, series)
        for h in HORIZONS:
            d = X.copy()
            d["y"] = fc.outcome(g, target, h)
            out[h].append(d.iloc[ROW_WARMUP:].dropna(subset=["y"]))
    return {h: pd.concat(v, ignore_index=True) for h, v in out.items()}


def forecast_cases(window: CWindow, real: pd.DataFrame, target: str, h: int) -> list[tuple[pd.Period, pd.Period, float]]:
    """(месяц цели, момент прогноза t0, факт) для окна; только случаи, где факт и t0 есть в реальных данных."""
    y = fc.outcome(real, target, h)
    cases = []
    for m in window.months:
        t0 = m - h
        if t0 in real.index and m in real.index and np.isfinite(y.loc[t0]):
            cases.append((m, t0, float(y.loc[t0])))
    return cases


def _row(model, window, variant, target, h, month, forecast, actual) -> dict:
    return {"model": model, "window": window.name, "variant": variant, "target": target, "h": h,
            "month": str(month), "forecast": float(forecast), "actual": float(actual),
            "error": float(forecast - actual)}


def synth_weights(n_real: int, n_syn: int, w: float) -> np.ndarray:
    """Вес каждой синтетической строки: суммарный вес синтетики = w/(1−w) × число реальных наблюдений."""
    return np.full(n_syn, n_real * w / (1 - w) / n_syn)


# --------------------------------------------------------------------------
# M1: ARX (как в пилоте, но одна точка обучения)
# --------------------------------------------------------------------------
def forecast_m1(window: CWindow, real: pd.DataFrame, abm: pd.DataFrame, control: pd.DataFrame,
                shares=SHARES) -> pd.DataFrame:
    rows = []
    panels = {"abm": abm, "control": control}
    for t in window.targets:
        Xnew_all = fc.predictors(real, t)
        for h in HORIZONS:
            cases = forecast_cases(window, real, t, h)
            train = fc.supervised(real, t, h).loc[: train_cut(window, h)].dropna()
            if len(train) < 12 or not cases:
                continue
            pools = {src: fc.pool_supervised(p, t, h) for src, p in panels.items()}
            for m, t0, truth in cases:
                x_new = Xnew_all.loc[t0]
                if x_new.isna().any():
                    continue
                ones = np.ones(len(train))
                preds = {"naive": fc.naive_forecast(real, t, t0),
                         "real": fc.fit_predict(train, ones, x_new),
                         "real_shrunk": fc.fit_predict(train, ones, x_new, ridge=1.0)}
                for src, syn in pools.items():
                    for w in shares:
                        comb = pd.concat([train, syn], ignore_index=True)
                        wt = np.concatenate([ones, synth_weights(len(train), len(syn), w)])
                        preds[f"{src}@{w}"] = fc.fit_predict(comb, wt, x_new)
                for v, p in preds.items():
                    rows.append(_row("M1", window, v, t, h, m, p, truth))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# M2: градиентный бустинг (родной интерфейс lightgbm; sklearn не нужен)
# --------------------------------------------------------------------------
def forecast_m2(window: CWindow, real: pd.DataFrame, abm: pd.DataFrame, control: pd.DataFrame,
                shares=SHARES) -> pd.DataFrame:
    import lightgbm as lgb                                   # только здесь: не смешивать с torch в одном процессе

    series = list(window.targets)
    rows = []
    panels = {"abm": abm, "control": control}
    for t in window.targets:
        feats = m2_features(real, t, series)
        pools = {src: m2_pool(p, t, series) for src, p in panels.items()}
        cols = list(feats.columns)
        for h in HORIZONS:
            cases = forecast_cases(window, real, t, h)
            d = feats.copy()
            d["y"] = fc.outcome(real, t, h)
            train = d.loc[: train_cut(window, h)].dropna(subset=["y"])
            if len(train) < 12 or not cases:
                continue
            X_new = feats.loc[[t0 for _, t0, _ in cases]].to_numpy(float)
            Xr, yr = train[cols].to_numpy(float), train["y"].to_numpy(float)
            specs = {"real": (Xr, yr, np.ones(len(yr)))}
            for src in ("abm", "control"):
                syn = pools[src][h]
                Xs, ys = syn[cols].to_numpy(float), syn["y"].to_numpy(float)
                for w in shares:
                    specs[f"{src}@{w}"] = (np.vstack([Xr, Xs]), np.concatenate([yr, ys]),
                                           np.concatenate([np.ones(len(yr)), synth_weights(len(yr), len(ys), w)]))
            for v, (X, y, wt) in specs.items():
                booster = lgb.train(LGB_PARAMS, lgb.Dataset(X, y, weight=wt, free_raw_data=True), num_boost_round=LGB_ROUNDS)
                for (m, t0, truth), p in zip(cases, booster.predict(X_new)):
                    rows.append(_row("M2", window, v, t, h, m, p, truth))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Синтетические панели (кэш на диске)
# --------------------------------------------------------------------------
def load_panels(theta: str, cache_dir) -> tuple[pd.DataFrame, pd.DataFrame]:
    from pathlib import Path
    d = Path(cache_dir)
    abm = pd.read_parquet(d / f"panel_abm_{theta}.parquet")
    ctl = pd.read_parquet(d / f"panel_control_{theta}.parquet")
    return abm, ctl
