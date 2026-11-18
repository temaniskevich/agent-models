"""Анализ эксперимента «сложность × точность симулятора»: сводки, правила решения H1–H3, тест Диболда–Мариано.

Правила — docs/CAPACITY_PROTOCOL.md, раздел 6 (и «Уточнения реализации»). Модуль только читает прогнозы (CSV) и ничего не обучает.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import forecasting as fc
from .capacity import MAIN_SHARE, group_of

MODELS = ("M1", "M2", "M3")
HELP, HARM = 0.95, 1.05
SHARE_BETTER = 0.6


def load_errors(directory, windows, models=MODELS, rep: int = 0) -> pd.DataFrame:
    parts = []
    sfx = "" if rep == 0 else f"_r{rep}"
    for w in windows:
        for m in models:
            p = Path(directory) / f"{w}_{m}{sfx}.csv"
            if p.exists():
                parts.append(pd.read_csv(p))
    if not parts:
        raise FileNotFoundError("нет файлов прогнозов")
    e = pd.concat(parts, ignore_index=True)
    e["group"] = e["target"].map(group_of)
    return e


def common_months(e: pd.DataFrame) -> pd.DataFrame:
    """Внутри (модель, окно, цель, h) оставляем месяцы, для которых есть прогнозы ВСЕХ вариантов: сравнение на одних и тех же случаях."""
    key = ["model", "window", "target", "h"]
    nvar = e.groupby(key)["variant"].transform("nunique")
    cnt = e.groupby(key + ["month"])["variant"].transform("nunique")
    return e[cnt == nvar].copy()


def cells(e: pd.DataFrame) -> pd.DataFrame:
    e = common_months(e).assign(se=lambda d: d["error"] ** 2)
    return (e.groupby(["model", "window", "group", "target", "h", "variant"])["se"]
            .agg(rmse=lambda x: float(np.sqrt(x.mean())), n="size").reset_index())


def ratios(c: pd.DataFrame, num: str, den: str) -> pd.DataFrame:
    """Отношение RMSE ``num``/``den`` по ячейкам (модель, окно, группа, цель, h)."""
    key = ["model", "window", "group", "target", "h"]
    a = c[c["variant"] == num].set_index(key)["rmse"]
    b = c[c["variant"] == den].set_index(key)["rmse"]
    r = (a / b.where(b > 0)).dropna().rename("ratio").reset_index()
    return r


def _gm(x) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x) & (x > 0)]
    return float(np.exp(np.log(x).mean())) if len(x) else np.nan


def pooled(r: pd.DataFrame, model: str, group: str, windows=None) -> tuple[float, float, int]:
    """Сводка группы: среднее геометрическое по окнам (равный вес) средних геометрических по ячейкам; доля ячеек с отношением < 1."""
    d = r[(r["model"] == model) & (r["group"] == group)]
    if windows is not None:
        d = d[d["window"].isin(windows)]
    if d.empty:
        return np.nan, np.nan, 0
    per_win = d.groupby("window")["ratio"].apply(_gm)
    return _gm(per_win.to_numpy()), float((d["ratio"] < 1).mean()), len(d)


def verdict(ratio: float, share: float) -> str:
    if not np.isfinite(ratio):
        return "нет данных"
    if ratio <= HELP and share >= SHARE_BETTER:
        return "помогает"
    if ratio >= HARM:
        return "вредит"
    return "эффекта не обнаружено"


def summary_table(c: pd.DataFrame, shares=(0.25, 0.5, 0.75, 0.9)) -> pd.DataFrame:
    rows = []
    comps = {"abm/real": ("abm@{w}", "real"), "control/real": ("control@{w}", "real"),
             "abm/control": ("abm@{w}", "control@{w}")}
    for w in shares:
        for name, (num, den) in comps.items():
            r = ratios(c, num.format(w=w), den.format(w=w))
            for (model, group), _ in r.groupby(["model", "group"]):
                ratio, share, n = pooled(r, model, group)
                rows.append({"model": model, "group": group, "share_w": w, "comparison": name,
                             "ratio": ratio, "share_better": share, "cells": n, "verdict": verdict(ratio, share)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Гипотезы (раздел 6 протокола)
# --------------------------------------------------------------------------
def _get(summ: pd.DataFrame, model, group, comp, w=MAIN_SHARE):
    d = summ[(summ["model"] == model) & (summ["group"] == group) & (summ["comparison"] == comp) & (summ["share_w"] == w)]
    return (float(d["ratio"].iloc[0]), str(d["verdict"].iloc[0])) if len(d) else (np.nan, "нет данных")


def _window_verdict(c: pd.DataFrame, model: str, group: str, window: str, w=MAIN_SHARE) -> str:
    r = ratios(c, f"abm@{w}", f"control@{w}")
    d = r[(r["model"] == model) & (r["group"] == group) & (r["window"] == window)]
    if d.empty:
        return "нет данных"
    return verdict(_gm(d["ratio"]), float((d["ratio"] < 1).mean()))


def hypotheses(c: pd.DataFrame, summ: pd.DataFrame) -> dict[str, str]:
    out = {}
    # H1: R(abm/real) у M3 не больше, чем у M1, минус 0,05 — для ставок и объёмов
    conf, refu, det = [], [], []
    for g in ("ставки", "объёмы"):
        r1, _ = _get(summ, "M1", g, "abm/real")
        r3, _ = _get(summ, "M3", g, "abm/real")
        det.append(f"{g}: M1 {r1:.3f}, M3 {r3:.3f}")
        conf.append(np.isfinite(r1) and np.isfinite(r3) and r3 <= r1 - 0.05)
        refu.append(np.isfinite(r1) and np.isfinite(r3) and r3 >= r1)
    out["H1"] = ("подтверждена" if all(conf) else "опровергнута" if all(refu) else "не определена") + " (" + "; ".join(det) + ")"
    # H2: ставки, abm против control
    helps = {m: _get(summ, m, "ставки", "abm/control")[1] == "помогает" for m in MODELS}
    win_ok = {m: all(_window_verdict(c, m, "ставки", w) == "помогает" for w in ("B", "E")) for m in ("M2", "M3")}
    r1, _ = _get(summ, "M1", "ставки", "abm/control")
    r3, _ = _get(summ, "M3", "ставки", "abm/control")
    detail = f"R(abm/control) на ставках: M1 {r1:.3f}, M2 {_get(summ, 'M2', 'ставки', 'abm/control')[0]:.3f}, M3 {r3:.3f}"
    if any(helps[m] and win_ok[m] for m in ("M2", "M3")) and np.isfinite(r1) and r3 < r1:
        h2 = "подтверждена"
    elif not any(helps.values()) and np.isfinite(r1) and np.isfinite(r3) and r3 >= r1:
        h2 = "опровергнута"
    else:
        h2 = "не определена: зависит от ступени/окна"
    out["H2"] = h2 + f" ({detail})"
    # H3: объёмы
    helps_v = {m: _get(summ, m, "объёмы", "abm/control")[1] == "помогает" for m in MODELS}
    n_win = {m: sum(_window_verdict(c, m, "объёмы", w) == "помогает" for w in ("A", "B", "E")) for m in MODELS}
    r1, _ = _get(summ, "M1", "объёмы", "abm/control")
    r3, _ = _get(summ, "M3", "объёмы", "abm/control")
    detail = f"R(abm/control) на объёмах: M1 {r1:.3f}, M2 {_get(summ, 'M2', 'объёмы', 'abm/control')[0]:.3f}, M3 {r3:.3f}"
    if any(n >= 2 for n in n_win.values()):
        h3 = "опровергнута"
    elif not any(helps_v.values()) and np.isfinite(r1) and np.isfinite(r3) and r3 > r1:
        h3 = "подтверждена"
    else:
        h3 = "не определена"
    out["H3"] = h3 + f" ({detail})"
    return out


# --------------------------------------------------------------------------
# Диболд–Мариано с поправкой Холма
# --------------------------------------------------------------------------
def holm(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    order = np.argsort(np.where(np.isfinite(p), p, 2.0))
    m = int(np.isfinite(p).sum())
    adj = np.full(len(p), np.nan)
    run = 0.0
    for rank, i in enumerate(order[:m]):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(run, 1.0)
    return adj


def dm_table(e: pd.DataFrame, w: float = MAIN_SHARE) -> pd.DataFrame:
    e = common_months(e).assign(se=lambda d: d["error"] ** 2)
    rows = []
    for (model, window, group, h), g in e.groupby(["model", "window", "group", "h"]):
        piv = {v: g[g["variant"] == v].pivot_table(index="month", columns="target", values="se")
               for v in ("real", f"abm@{w}", f"control@{w}")}
        for cur, base in ((f"abm@{w}", f"control@{w}"), (f"abm@{w}", "real")):
            a, b = piv[cur], piv[base]
            if a.empty or b.empty:
                continue
            diff = ((a - b) / b.mean().replace(0, np.nan)).mean(axis=1).dropna().sort_index().to_numpy()
            stat, p = fc.dm_test(diff, h)
            rows.append({"model": model, "window": window, "group": group, "h": h, "variant": cur, "vs": base,
                         "mean_norm_loss_diff": float(diff.mean()) if len(diff) else np.nan, "dm_stat": stat,
                         "p_value": p, "n_months": len(diff)})
    t = pd.DataFrame(rows)
    if not t.empty:
        t["p_holm"] = t.groupby("model")["p_value"].transform(lambda s: holm(s.to_numpy()))
    return t


# --------------------------------------------------------------------------
# Проверка шума (поправка 19.11.2026 к протоколу, раздел 11)
# --------------------------------------------------------------------------
NOISE_WINDOWS = ("B", "E")


def noise_decision(values, mode: str = "help") -> str:
    """Правило поправки. ``help``: устойчиво, если n(<1) ≥ 5 и m + 2·s/√n < 1; шум, если n(<1) ≤ 3 или m ≥ 1.
    ``harm``: зеркально — устойчиво (вред), если n(>1) ≥ 5 и m − 2·s/√n > 1; шум, если n(>1) ≤ 3 или m ≤ 1."""
    v = np.asarray([x for x in values if np.isfinite(x)], float)
    n = len(v)
    if n < 6:
        return f"недостаточно запусков ({n})"
    m, s = float(v.mean()), float(v.std(ddof=1))
    se = s / np.sqrt(n)
    if mode == "help":
        k = int((v < 1).sum())
        return "устойчиво" if (k >= 5 and m + 2 * se < 1) else "шум" if (k <= 3 or m >= 1) else "не определён"
    k = int((v > 1).sum())
    return "устойчиво (вред)" if (k >= 5 and m - 2 * se > 1) else "шум" if (k <= 3 or m <= 1) else "не определён"


def noise_table(directory, reps=range(0, 6), w: float = MAIN_SHARE) -> pd.DataFrame:
    """Статистики S1–S4 по запускам: по строке на (статистика, модель)."""
    per_rep = {}
    for k in reps:
        e = load_errors(directory, NOISE_WINDOWS + (("A",) if True else ()), rep=k)
        per_rep[k] = cells(e)
    comps = {"abm/control": (f"abm@{w}", f"control@{w}"), "abm/real": (f"abm@{w}", "real"),
             "control/real": (f"control@{w}", "real")}
    specs = []                                     # (id, comparison, group, windows, mode, models)
    specs += [("S1 abm/control ставки B+E", "abm/control", "ставки", NOISE_WINDOWS, "help", MODELS)]
    specs += [("S2a abm/real ставки B+E", "abm/real", "ставки", NOISE_WINDOWS, "help", MODELS)]
    specs += [("S2b control/real ставки B+E", "control/real", "ставки", NOISE_WINDOWS, "help", MODELS)]
    specs += [("S3a abm/control объёмы B+E", "abm/control", "объёмы", NOISE_WINDOWS, "harm", MODELS)]
    specs += [("S3b abm/real объёмы B+E", "abm/real", "объёмы", NOISE_WINDOWS, "help", MODELS)]
    specs += [("S3c abm/control объёмы A", "abm/control", "объёмы", ("A",), "harm", ("M1", "M2"))]
    specs += [(f"S4 abm/control ставки {w_}", "abm/control", "ставки", (w_,), "help", ("M3",)) for w_ in NOISE_WINDOWS]
    rows = []
    for sid, comp, group, wins, mode, models in specs:
        num, den = comps[comp]
        for model in models:
            vals = []
            for k, c in per_rep.items():
                r = ratios(c, num, den)
                vals.append(pooled(r, model, group, windows=wins)[0])
            v = np.asarray(vals, float)
            rows.append({"stat": sid, "model": model, **{f"r{k}": x for k, x in zip(per_rep, vals)},
                         "mean": float(np.nanmean(v)), "sd": float(np.nanstd(v, ddof=1)), "min": float(np.nanmin(v)),
                         "max": float(np.nanmax(v)), "n_lt1": int((v < 1).sum()), "decision": noise_decision(v, mode)})
    return pd.DataFrame(rows)
