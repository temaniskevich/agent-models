"""Реакции на ключевую ставку: локальные проекции по данным ЦБ и по модели (протокол — docs/IRF_PROTOCOL.md)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from .calibration import load_cbr_monthly, slice_window
from .config import ModelConfig
from .model import BankingModel
from .scenarios import Scenario, step_path

RATES = ["loan_rate_corp", "loan_rate_retail", "deposit_rate_corp", "deposit_rate_retail"]
VOLUMES = ["loans_corp", "loans_retail", "deposits_corp", "deposits_retail"]
OUTCOMES = RATES + VOLUMES
LABELS = {"loan_rate_corp": "ставка по кредитам орг., п.п.", "loan_rate_retail": "ставка по кредитам нас., п.п.",
          "deposit_rate_corp": "ставка по вкладам орг., п.п.", "deposit_rate_retail": "ставка по вкладам нас., п.п.",
          "loans_corp": "кредиты орг., %", "loans_retail": "кредиты нас., %",
          "deposits_corp": "вклады орг., %", "deposits_retail": "вклады нас., %"}


def _base(df: pd.DataFrame, col: str) -> pd.Series:
    return 100.0 * np.log(df[col]) if col in VOLUMES else df[col].astype(float)


def local_projection(df: pd.DataFrame, col: str, H: int = 12, nlags: int = 3, hs: tuple[int, ...] | None = None) -> pd.DataFrame:
    """β_h при Δk(t) для h = 0…H, ошибки Ньюи–Уэста. ``df`` — колонки ``key_rate`` (в %) и ``col``."""
    k = df["key_rate"].astype(float)
    y = _base(df, col)
    dk, dy = k.diff(), y.diff()
    ctrl = {f"dk{j}": dk.shift(j) for j in range(1, nlags + 1)} | {f"dy{j}": dy.shift(j) for j in range(1, nlags + 1)}
    rows = []
    for h in (range(H + 1) if hs is None else hs):
        d = pd.DataFrame({"y": y.shift(-h) - y.shift(1), "dk": dk, **ctrl}).dropna()
        X = np.column_stack([np.ones(len(d)), d.drop(columns="y").to_numpy()])
        Y = d["y"].to_numpy()
        n, p = X.shape
        if n < p + 8:
            rows.append((h, np.nan, np.nan, n)); continue
        beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
        u = Y - X @ beta
        XtXi = np.linalg.pinv(X.T @ X)
        L = h + nlags
        S = (X * u[:, None]).T @ (X * u[:, None])
        for l in range(1, L + 1):
            G = (X[l:] * u[l:, None]).T @ (X[:-l] * u[:-l, None])
            S += (1 - l / (L + 1)) * (G + G.T)
        se = float(np.sqrt(max((XtXi @ S @ XtXi)[1, 1], 0.0)))
        rows.append((h, float(beta[1]), se, n))
    out = pd.DataFrame(rows, columns=["h", "beta", "se", "n"])
    z = stats.norm.ppf(0.95)
    out["lo90"], out["hi90"] = out["beta"] - z * out["se"], out["beta"] + z * out["se"]
    return out


def data_irf(window: tuple[str, str] = ("2014-01", "2026-12"), H: int = 12) -> dict[str, pd.DataFrame]:
    df = slice_window(load_cbr_monthly(), *window).dropna(subset=["key_rate", *RATES])
    return {c: local_projection(df, c, H) for c in OUTCOMES}, df


def _run(cfg: ModelConfig, scn: Scenario, seed: int) -> pd.DataFrame:
    return BankingModel(cfg, scenario=scn, seed=seed).run().sector


def model_lp_irf(theta: dict[str, float], key_path_pct: np.ndarray, n_seeds: int = 50, H: int = 12,
                 seed0: int = 0) -> dict[str, pd.DataFrame]:
    """Та же регрессия на выходе модели, управляемой реальным путём ключевой ставки."""
    cfg = ModelConfig().replace(n_months=len(key_path_pct), record_banks=False, record_network=False, **theta)
    cfg.scenario.key_rate_start = float(key_path_pct[0] / 100)
    scn = Scenario("historical", key_rate=key_path_pct / 100.0)
    per = {c: [] for c in OUTCOMES}
    for s in range(seed0, seed0 + n_seeds):
        sector = _run(cfg, scn, s)
        for c in OUTCOMES:
            per[c].append(local_projection(sector, c, H)["beta"].to_numpy())
    out = {}
    for c, arr in per.items():
        a = np.vstack(arr)
        out[c] = pd.DataFrame({"h": np.arange(H + 1), "beta": np.nanmean(a, axis=0),
                               "lo90": np.nanpercentile(a, 5, axis=0), "hi90": np.nanpercentile(a, 95, axis=0)})
    return out


def model_step_irf(theta: dict[str, float], size_pp: float = 2.0, n_seeds: int = 50, H: int = 12,
                   shock_month: int = 24, n_months: int = 60, k0: float = 0.08) -> dict[str, pd.DataFrame]:
    """Структурная реакция: ступень ``size_pp`` против сценария без скачка, общие зёрна; на 1 п.п."""
    cfg = ModelConfig().replace(n_months=n_months, record_banks=False, record_network=False, **theta)
    base = Scenario("base", key_rate=np.full(n_months, k0))
    shock = Scenario("shock", key_rate=step_path(k0, size_pp / 100, shock_month, n_months))
    diffs = {c: [] for c in OUTCOMES}
    for s in range(n_seeds):
        a, b = _run(cfg, base, s), _run(cfg, shock, s)
        for c in OUTCOMES:
            ya, yb = _base(a, c), _base(b, c)
            diffs[c].append(((yb - ya).iloc[shock_month:shock_month + H + 1].to_numpy()) / size_pp)
    out = {}
    for c, arr in diffs.items():
        m = np.vstack(arr)
        out[c] = pd.DataFrame({"h": np.arange(H + 1), "beta": m.mean(axis=0),
                               "lo90": np.percentile(m, 5, axis=0), "hi90": np.percentile(m, 95, axis=0)})
    return out


@dataclass
class Verdict:
    table: pd.DataFrame
    passed: int


def evaluate(data: dict[str, pd.DataFrame], model: dict[str, pd.DataFrame]) -> Verdict:
    """Критерии протокола: доля горизонтов 1…12 внутри 90 % ИД данных ≥ 70 % и совпадение знака на h = 6, 12."""
    rows = []
    for c in OUTCOMES:
        d, m = data[c].set_index("h"), model[c].set_index("h")
        hs = [h for h in range(1, 13) if h in d.index and np.isfinite(d.loc[h, "beta"])]
        inside = np.mean([d.loc[h, "lo90"] <= m.loc[h, "beta"] <= d.loc[h, "hi90"] for h in hs]) if hs else np.nan
        sign_ok = True
        for h in (6, 12):
            if h in d.index and np.isfinite(d.loc[h, "se"]) and d.loc[h, "se"] > 0:
                signif = abs(d.loc[h, "beta"] / d.loc[h, "se"]) > stats.norm.ppf(0.95)
                if signif and np.sign(d.loc[h, "beta"]) != np.sign(m.loc[h, "beta"]):
                    sign_ok = False
        ok = bool(inside >= 0.7 and sign_ok)
        rows.append({"outcome": c, "share_inside": inside, "sign_ok": sign_ok, "passed": ok,
                     "data_h12": d.loc[12, "beta"], "data_h12_se": d.loc[12, "se"], "model_h12": m.loc[12, "beta"]})
    t = pd.DataFrame(rows)
    return Verdict(t, int(t["passed"].sum()))


def evaluate_out_of_sample(theta: dict[str, float], sample: tuple[str, str] = ("2020-01", "2026-09"), n_seeds: int = 50,
                           warmup: int = 24, H: int = 12):
    """Итерация 2 протокола: проверка на выборке, не участвовавшей в калибровке.

    Данные — проекции на ``sample``; модель — реальный путь ставки с ``warmup`` месяцами разогрева до выборки."""
    full = load_cbr_monthly()
    dsample = slice_window(full, *sample).dropna(subset=["key_rate", *RATES])
    data = {c: local_projection(dsample, c, H) for c in OUTCOMES}
    start = dsample.index[0]
    path = full.loc[start - warmup:dsample.index[-1], "key_rate"].to_numpy()
    cfg = ModelConfig().replace(n_months=len(path), record_banks=False, record_network=False, **theta)
    cfg.scenario.key_rate_start = float(path[0] / 100)
    scn = Scenario("historical", key_rate=path / 100.0)
    per = {c: [] for c in OUTCOMES}
    for sd in range(n_seeds):
        sector = _run(cfg, scn, sd).iloc[warmup:]
        for c in OUTCOMES:
            per[c].append(local_projection(sector, c, H)["beta"].to_numpy())
    model = {}
    for c, arr in per.items():
        a = np.vstack(arr)
        model[c] = pd.DataFrame({"h": np.arange(H + 1), "beta": np.nanmean(a, axis=0),
                                 "lo90": np.nanpercentile(a, 5, axis=0), "hi90": np.nanpercentile(a, 95, axis=0)})
    return data, model, evaluate(data, model)
