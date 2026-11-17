"""Пилот шага 5: помогает ли синтетика прогнозу вне выборки (протокол — docs/PILOT_PROTOCOL.md).

Прямой ARX-прогноз (взвешенный МНК с небольшой гребневой регуляризацией) на трёх вариантах обучающей
выборки: только реальные данные; плюс контрольная синтетика «без экономики»; плюс синтетика агентной модели.
Предикторы — изменения ключевой ставки и собственная недавняя динамика, поэтому модель различает сценарии
скачков только если синтетика содержит правильную реакцию на скачки.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .batch import DEFAULT_PARAM_RANGES, generate_synthetic
from .calibration import DATA_DIR, load_cbr_monthly
from .config import ModelConfig
from .synthetic import ARJumpControl

RATE_T = ["lr_c", "lr_r", "dr_c", "dr_r"]            # ставки: кредиты/вклады × организации/население
GROWTH_T = ["g_lc", "g_lr", "g_dc", "g_dr"]          # рост, % годовых (1200·Δln)
COLUMNS = ["key_rate"] + RATE_T + GROWTH_T
LABELS = {"lr_c": "ставка по кредитам орг.", "lr_r": "ставка по кредитам нас.", "dr_c": "ставка по вкладам орг.",
          "dr_r": "ставка по вкладам нас.", "g_lc": "рост кредитов орг.", "g_lr": "рост кредитов нас.",
          "g_dc": "рост вкладов орг.", "g_dr": "рост вкладов нас."}


# --------------------------------------------------------------------------
# Приведение реальных и модельных рядов к общему виду
# --------------------------------------------------------------------------
def harmonize(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Таблица с колонками COLUMNS. ``source``: ``real`` (данные ЦБ) или ``model`` (``result.sector``)."""
    out = pd.DataFrame(index=df.index)
    out["key_rate"] = df["key_rate"]
    for new, col in zip(RATE_T, ["loan_rate_corp", "loan_rate_retail", "deposit_rate_corp", "deposit_rate_retail"]):
        out[new] = df[col]
    for new, col in zip(GROWTH_T, ["loans_corp", "loans_retail", "deposits_corp", "deposits_retail"]):
        out[new] = 1200.0 * np.log(df[col]).diff()
    return out


def load_real() -> pd.DataFrame:
    real = harmonize(load_cbr_monthly(), "real")
    # стык ставки рефинансирования (до 2013-08) и ключевой (с 2013-09): сдвиг, чтобы разность была нулевой
    k = real["key_rate"].copy()
    a, b = pd.Period("2013-08", "M"), pd.Period("2013-09", "M")
    if a in k.index and b in k.index:
        k.loc[:a] += k.loc[b] - k.loc[a]
    real["key_rate"] = k
    return real


# --------------------------------------------------------------------------
# Признаки и обучающие выборки
# --------------------------------------------------------------------------
def predictors(df: pd.DataFrame, target: str) -> pd.DataFrame:
    k = df["key_rate"]
    X = pd.DataFrame({"dk1": k.diff(), "dk3": k.diff(3), "dk6": k.diff(6)}, index=df.index)
    if target in RATE_T:
        X["own3"] = df[target].diff(3)
    else:
        X["g3"] = df[target].rolling(3).mean()
        X["dk12"] = k.diff(12)
    return X


def outcome(df: pd.DataFrame, target: str, h: int) -> pd.Series:
    if target in RATE_T:
        return df[target].shift(-h) - df[target]
    return df[target].rolling(h).mean().shift(-h)           # средний рост за t+1…t+h


def supervised(df: pd.DataFrame, target: str, h: int) -> pd.DataFrame:
    d = predictors(df, target)
    d["y"] = outcome(df, target, h)
    return d


def naive_forecast(df: pd.DataFrame, target: str, t0: pd.Period) -> float:
    if target in RATE_T:
        return 0.0
    return float(df[target].rolling(3).mean().loc[t0])


def fit_predict(train: pd.DataFrame, weights: np.ndarray, x_new: pd.Series, ridge: float = 1e-2) -> float:
    cols = [c for c in train.columns if c != "y"]
    w = weights / weights.sum()
    X = train[cols].to_numpy(float)
    mu = (w[:, None] * X).sum(axis=0)
    sd = np.sqrt((w[:, None] * (X - mu) ** 2).sum(axis=0))
    sd = np.maximum(sd, 0.25)                                # мало вариации (ставка постоянна) → без раздувания
    Z = np.column_stack([np.ones(len(X)), (X - mu) / sd])
    y = train["y"].to_numpy(float)
    ww = w * len(w)
    pen = np.eye(Z.shape[1]) * ridge * len(w)
    pen[0, 0] = 0.0
    beta = np.linalg.solve((Z * ww[:, None]).T @ Z + pen, (Z * ww[:, None]).T @ y)
    z = np.concatenate([[1.0], (x_new[cols].to_numpy(float) - mu) / sd])
    return float(z @ beta)


# --------------------------------------------------------------------------
# Синтетика
# --------------------------------------------------------------------------
def pilot_ranges(theta: dict[str, float], spread: float = 0.25) -> dict[str, tuple[float, float]]:
    """Диапазоны вокруг откалиброванных значений (±spread) для параметров калибровки; прочие — по умолчанию."""
    ranges = dict(DEFAULT_PARAM_RANGES)
    for k, v in theta.items():
        ranges[k] = (v * (1 - spread), v * (1 + spread)) if v > 0 else (v - 0.01, v + 0.01)
    return ranges


def make_abm_panel(theta: dict[str, float], n_runs: int = 300, months: int = 120, seed: int = 12345,
                   n_jobs: int = 1) -> pd.DataFrame:
    """Длинная таблица синтетических историй агентной модели в общем виде (колонки COLUMNS, run, t)."""
    cfg = ModelConfig().replace(n_months=months, record_banks=False, record_network=False)
    cfg = cfg.replace(**{k: v for k, v in theta.items()})
    data, _ = generate_synthetic(cfg, n_runs, seed, pilot_ranges(theta), n_jobs=n_jobs)
    frames = []
    for run, g in data.groupby("run"):
        g = g.set_index(pd.PeriodIndex(g["date"], freq="M"))
        h = harmonize(g, "model")
        h["run"] = run
        h["t"] = np.arange(len(h))
        frames.append(h.iloc[1:])                            # первая строка без роста
    return pd.concat(frames, ignore_index=True)


def make_control_panel(abm_panel: pd.DataFrame, n_runs: int, months: int, seed: int = 777) -> pd.DataFrame:
    ctl = ARJumpControl(columns=COLUMNS, log_columns=()).fit(abm_panel.dropna(subset=COLUMNS))
    sim = ctl.simulate(months, n_runs, seed=seed)
    sim["key_rate"] = sim["key_rate"].clip(lower=0.5)
    for c in RATE_T:
        sim[c] = sim[c].clip(lower=0.1)
    return sim


def pool_supervised(panel: pd.DataFrame, target: str, h: int) -> pd.DataFrame:
    parts = []
    for _, g in panel.groupby("run"):
        s = supervised(g.reset_index(drop=True), target, h).dropna()
        parts.append(s.iloc[12:])                            # первые месяцы — прогрев предикторов
    return pd.concat(parts, ignore_index=True)


# --------------------------------------------------------------------------
# Оценка
# --------------------------------------------------------------------------
@dataclass
class PilotResult:
    errors: pd.DataFrame            # строка = (вариант, цель, h, месяц цели): прогноз, факт, ошибка
    summary: pd.DataFrame           # сводка по вариантам
    cells: pd.DataFrame             # RMSE и отношение к real по ячейкам
    dm: pd.DataFrame                # тест Диболда–Мариано


def run_pilot(real: pd.DataFrame, abm: pd.DataFrame, control: pd.DataFrame, targets: list[str], test: tuple[str, str],
              horizons=(1, 3, 6), shares=(0.25, 0.5, 0.75)) -> PilotResult:
    start, end = pd.Period(test[0], "M"), pd.Period(test[1], "M")
    pools = {(src, t, h): pool_supervised(panel, t, h)
             for src, panel in (("abm", abm), ("control", control)) for t in targets for h in horizons}
    rows = []
    for t in targets:
        for h in horizons:
            sup = supervised(real, t, h)
            for target_month in pd.period_range(start, end, freq="M"):
                t0 = target_month - h
                if t0 not in real.index or target_month not in real.index:
                    continue
                truth = outcome(real, t, h).loc[t0]
                x_new = predictors(real, t).loc[t0]
                if not np.isfinite(truth) or x_new.isna().any():
                    continue
                train = sup.loc[:t0 - h].dropna()
                if len(train) < 12:
                    continue
                preds = {"naive": naive_forecast(real, t, t0),
                         "real": fit_predict(train, np.ones(len(train)), x_new),
                         # нейтральный базис: те же реальные данные, но сильное сжатие коэффициентов к нулю
                         "real_shrunk": fit_predict(train, np.ones(len(train)), x_new, ridge=1.0)}
                for src in ("control", "abm"):
                    syn = pools[(src, t, h)]
                    for w in shares:
                        comb = pd.concat([train, syn], ignore_index=True)
                        wt = np.concatenate([np.ones(len(train)),
                                             np.full(len(syn), len(train) * w / (1 - w) / len(syn))])
                        preds[f"{src}@{w}"] = fit_predict(comb, wt, x_new)
                for name, p in preds.items():
                    rows.append({"variant": name, "target": t, "h": h, "month": str(target_month),
                                 "forecast": p, "actual": truth, "error": p - truth})
    errors = pd.DataFrame(rows)
    return PilotResult(errors, *_summarize(errors, targets))


def _summarize(errors: pd.DataFrame, targets: list[str]):
    cells = (errors.assign(se=errors["error"] ** 2).groupby(["variant", "target", "h"])
             .agg(rmse=("se", lambda x: float(np.sqrt(x.mean()))), n=("se", "size")).reset_index())
    base = cells[cells["variant"] == "real"].set_index(["target", "h"])["rmse"]
    cells["ratio_vs_real"] = [r / base.loc[(t, h)] if base.loc[(t, h)] > 0 else np.nan
                              for r, t, h in zip(cells["rmse"], cells["target"], cells["h"])]
    summary = (cells.groupby("variant")["ratio_vs_real"]
               .agg(pooled_ratio=lambda x: float(np.exp(np.log(x).mean())), share_better=lambda x: float((x < 1).mean()),
                    cells="size").reset_index())
    summary["verdict"] = [_verdict(v, r, s) for v, r, s in zip(summary["variant"], summary["pooled_ratio"], summary["share_better"])]
    c = cells.set_index(["variant", "target", "h"])["rmse"]
    vs = {}
    for v in summary["variant"]:
        if v.startswith("abm@"):
            ctl = "control@" + v.split("@")[1]
            ratios = [c[(v, t, h)] / c[(ctl, t, h)] for (vv, t, h) in c.index if vv == v and (ctl, t, h) in c.index and c[(ctl, t, h)] > 0]
            vs[v] = float(np.exp(np.mean(np.log(ratios))))
    summary["abm_vs_control"] = summary["variant"].map(vs)
    return summary, cells, _dm_table(errors)


def _verdict(variant: str, ratio: float, share: float) -> str:
    if variant in ("real", "naive", "real_shrunk") or variant.startswith("control"):
        return "—"
    if ratio <= 0.95 and share >= 0.6:
        return "помогает"
    if ratio >= 1.05:
        return "вредит"
    return "эффекта не обнаружено"


def dm_test(d: np.ndarray, h: int) -> tuple[float, float]:
    """Диболд–Мариано с поправкой Харви–Леборна–Ньюбольда; ``d`` — разности потерь (вариант − база)."""
    n = len(d)
    if n < 4 or np.allclose(d, 0):
        return np.nan, np.nan
    dm = d - d.mean()
    lags = max(h - 1, 0)
    var = float((dm @ dm) / n)
    for k in range(1, lags + 1):
        var += 2 * (1 - k / (lags + 1)) * float((dm[k:] @ dm[:-k]) / n)
    var = max(var / n, 1e-18)
    stat = d.mean() / np.sqrt(var)
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat *= hln
    return float(stat), float(2 * stats.t.sf(abs(stat), df=n - 1))


def _dm_table(errors: pd.DataFrame) -> pd.DataFrame:
    """DM по среднему по целям нормированному различию квадратичных потерь, для каждого h.

    Пары: каждый вариант против ``real``; а также ``abm@w`` против ``control@w`` и против ``real_shrunk``
    (последние пары добавлены после пробного запуска — см. протокол)."""
    rows = []
    e = errors.assign(se=errors["error"] ** 2)
    variants = sorted(set(e["variant"]))
    pairs = [(v, "real") for v in variants if v != "real"]
    pairs += [(v, "control@" + v.split("@")[1]) for v in variants if v.startswith("abm@")]
    pairs += [(v, "real_shrunk") for v in variants if v.startswith("abm@")]
    for h in sorted(e["h"].unique()):
        eh = e[e["h"] == h]
        piv = {v: eh[eh["variant"] == v].pivot_table(index="month", columns="target", values="se") for v in variants}
        for v, base_v in pairs:
            base, cur = piv[base_v], piv[v]
            diff = ((cur - base) / base.mean().replace(0, np.nan)).mean(axis=1).dropna().sort_index().to_numpy()
            stat, p = dm_test(diff, h)
            rows.append({"variant": v, "vs": base_v, "h": h, "mean_norm_loss_diff": float(diff.mean()) if len(diff) else np.nan,
                         "dm_stat": stat, "p_value": p, "n_months": len(diff)})
    return pd.DataFrame(rows)


def load_theta(window: str) -> dict[str, float]:
    return json.loads((DATA_DIR / f"calibrated_window_{window}.json").read_text(encoding="utf-8"))["theta"]
