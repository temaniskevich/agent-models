"""Проверка качественного поведения модели и сравнение с российскими работами.

* ``stylized_facts`` — критерий готовности шага 2: при скачке ключевой ставки модель
  качественно показывает ожидаемое (рост ставок банков с лагом, замедление кредита,
  рост дефолтов, обращения к ЦБ, замирание МБК при панике, снижение капитала).
* ``size_distribution_stability`` — аналог проверки Малахова, Пильника, Радионова (2015):
  стабильность распределения банков по размеру.
* ``mbk_network_stats`` — аналог проверки Леонидова, Нечитайло, Серебрянниковой (2018):
  топология сети межбанковского кредитования.

Числа из этих статей в код не зашиты: функции возвращают показатели модели, которые
нужно сопоставить с таблицами статей при написании раздела.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Check:
    name: str
    passed: bool | None          # None — проверка неприменима (нет скачка/паники)
    value: str
    expectation: str

    def as_dict(self) -> dict:
        icon = {True: "✅", False: "❌", None: "➖"}[self.passed]
        return {"": icon, "Проверка": self.name, "Результат модели": self.value,
                "Ожидание": self.expectation}


def detect_shock(df: pd.DataFrame, min_jump_pp: float = 2.0, window: int = 3) -> tuple[int, float] | None:
    """Месяц и размер (п.п.) самого резкого роста ключевой ставки за ``window`` месяцев."""
    k = df["key_rate"].to_numpy()
    if len(k) <= window + 6:
        return None
    jump = k[window:] - k[:-window]
    i = int(np.argmax(jump))
    if jump[i] < min_jump_pp:
        return None
    # месяц, с которого началось движение
    start = i + 1
    while start < len(k) - 1 and k[start] - k[start - 1] < 0.2 * jump[i] / window and start < i + window:
        start += 1
    return int(start), float(jump[i])


def _mean(x: pd.Series | np.ndarray, a: int, b: int) -> float:
    arr = np.asarray(x)[max(a, 0):max(b, 0)]
    return float(np.nanmean(arr)) if len(arr) else float("nan")


def stylized_facts(df: pd.DataFrame, pre: int = 6, post: int = 18) -> list[Check]:
    """Проверки на одном прогоне (или на усреднении прогонов с колонками как у ``result.sector``)."""
    shock = detect_shock(df)
    checks: list[Check] = []
    if shock is None:
        return [Check("Скачок ключевой ставки", None, "в сценарии нет скачка ≥ 2 п.п.",
                      "нужен скачок для проверки реакции")]
    m, size = shock
    c = lambda col: df[col].to_numpy()          # noqa: E731
    pre_loan = _mean(c("loan_rate"), m - pre, m)
    pre_dep = _mean(c("deposit_rate"), m - pre, m)
    d_loan0 = _mean(c("loan_rate"), m, m + 1) - pre_loan
    d_loan_long = _mean(c("loan_rate"), m + 6, m + 12) - pre_loan
    d_dep_long = _mean(c("deposit_rate"), m + 3, m + 12) - pre_dep
    checks.append(Check("Скачок ставки", True, f"+{size:.1f} п.п. в месяце {m}", "—"))
    checks.append(Check(
        "Кредитные ставки растут и переносятся с лагом",
        bool(d_loan_long >= 0.4 * size and d_loan0 < d_loan_long),
        f"в месяц скачка +{d_loan0:.1f} п.п., через 6–12 мес. +{d_loan_long:.1f} п.п.",
        "перенос ≥ 40 % скачка, полный эффект позже"))
    checks.append(Check(
        "Ставки по вкладам растут",
        bool(d_dep_long >= 0.3 * size), f"+{d_dep_long:.1f} п.п. через 3–12 мес.", "≥ 30 % скачка"))
    g_pre = _mean(c("loans_growth_yoy"), m - 12, m)
    g_min = float(np.nanmin(c("loans_growth_yoy")[m:m + post])) if m + 1 < len(df) else float("nan")
    checks.append(Check(
        "Кредитование замедляется", bool(np.isfinite(g_pre) and g_min <= g_pre - 2.0),
        f"рост г/г: до {g_pre:.1f} % → минимум {g_min:.1f} %" if np.isfinite(g_pre) else "мало истории до скачка",
        "падение темпа роста ≥ 2 п.п."))
    npl_pre = _mean(c("npl_share"), m - 12, m)
    npl_post = float(np.nanmax(c("npl_share")[m:m + post]))
    checks.append(Check("Дефолты заёмщиков растут", bool(npl_post >= 1.25 * npl_pre),
                        f"доля дефолтов за мес.: {npl_pre:.2f} % → пик {npl_post:.2f} %", "рост ≥ 25 %"))
    car_pre = _mean(c("car"), m - 3, m)
    car_min = float(np.nanmin(c("car")[m:m + post]))
    checks.append(Check("Капитал банков снижается", bool(car_min <= car_pre - 0.2),
                        f"CAR сектора: {car_pre:.1f} % → минимум {car_min:.1f} %", "снижение ≥ 0.2 п.п."))
    panic = float(np.nanmax(c("panic")[m:m + 3]))
    if panic >= 0.3:
        mbk_pre = _mean(c("mbk_volume"), m - 6, m)
        mbk_post = _mean(c("mbk_volume"), m, m + 2)
        checks.append(Check("При панике рынок МБК замирает", bool(mbk_post < 0.7 * mbk_pre),
                            f"объём МБК: {mbk_pre:,.0f} → {mbk_post:,.0f} млрд", "падение ≥ 30 %"))
        cb_pre = _mean(c("cb_debt"), m - 6, m)
        cb_post = float(np.nanmax(c("cb_debt")[m:m + 6]))
        checks.append(Check("Банки обращаются к ЦБ", bool(cb_post > cb_pre + 1e-6 and cb_post > 1.5 * cb_pre),
                            f"долг перед ЦБ: {cb_pre:,.0f} → пик {cb_post:,.0f} млрд", "рост долга ≥ 50 %"))
        out = float(np.nanmax(c("deposit_outflow")[m:m + 3]) / max(c("deposits_total")[max(m - 1, 0)], 1e-9) * 100)
        checks.append(Check("Часть вкладов уходит", bool(out >= 1.0),
                            f"снято за месяц {out:.1f} % вкладов", "≥ 1 %"))
    else:
        checks.append(Check("Паника вкладчиков", None, f"паника {panic:.2f} < 0.3",
                            "проверки МБК/ЦБ/оттока применимы при панике"))
    neg = bool((df[["loans_total", "deposits_total", "capital"]] < 0).any().any())
    checks.append(Check("Объёмы неотрицательны", not neg, "ок" if not neg else "найдены отрицательные", "всегда"))
    return checks


def checks_table(checks: list[Check]) -> pd.DataFrame:
    return pd.DataFrame([c.as_dict() for c in checks])


# --------------------------------------------------------------------------
# Распределение банков по размеру (Малахов, Пильник, Радионов, 2015)
# --------------------------------------------------------------------------
def zipf_slope(sizes: np.ndarray, top: int | None = None) -> float:
    s = np.sort(np.asarray(sizes, dtype=float))[::-1]
    s = s[s > 0]
    if top:
        s = s[:top]
    if len(s) < 4:
        return float("nan")
    r = np.arange(1, len(s) + 1)
    return float(np.polyfit(np.log(r), np.log(s), 1)[0])


def size_distribution_stability(banks: pd.DataFrame) -> pd.DataFrame:
    """Динамика формы распределения банков по активам: наклон Ципфа, Gini, HHI, топ-1.

    Если распределение стабильно (в терминах нормированных размеров), показатели
    меняются слабо, а ранги банков сохраняются (корреляция Спирмена к начальному месяцу).
    """
    rows = []
    first = None
    for month, g in banks[banks["active"]].groupby("month"):
        a = g["assets"].to_numpy()
        share = a / a.sum()
        x = np.sort(a)
        n = len(x)
        gini = (2 * np.sum(np.arange(1, n + 1) * x) / (n * x.sum())) - (n + 1) / n
        if first is None:
            first = g.set_index("slot")["assets"]
        both = pd.concat([first, g.set_index("slot")["assets"]], axis=1, join="inner").dropna()
        rho = both.corr(method="spearman").iloc[0, 1] if len(both) > 3 else np.nan
        rows.append({"month": month, "n_banks": n, "zipf_slope": zipf_slope(a), "gini": gini,
                     "hhi": float((share ** 2).sum() * 1e4), "top1": float(share.max()),
                     "rank_corr_to_start": float(rho)})
    return pd.DataFrame(rows).set_index("month")


# --------------------------------------------------------------------------
# Топология сети МБК (Леонидов, Нечитайло, Серебрянникова, 2018)
# --------------------------------------------------------------------------
def mbk_network_stats(lender: np.ndarray, borrower: np.ndarray, amount: np.ndarray, n: int) -> dict:
    """Показатели сети по списку ребер (кредитор → заёмщик, объём)."""
    if len(amount) == 0:
        return {"edges": 0, "density": 0.0, "reciprocity": np.nan, "mean_degree": 0.0,
                "max_in_degree": 0, "max_out_degree": 0, "assortativity": np.nan,
                "top3_volume_share": np.nan, "volume": 0.0}
    out_deg = np.bincount(lender, minlength=n)
    in_deg = np.bincount(borrower, minlength=n)
    edges = set(zip(lender.tolist(), borrower.tolist()))
    recip = sum((b, a) in edges for a, b in edges) / len(edges)
    deg = in_deg + out_deg
    if len(lender) > 2 and deg[lender].std() > 0 and deg[borrower].std() > 0:
        assort = float(np.corrcoef(deg[lender], deg[borrower])[0, 1])
    else:
        assort = np.nan
    vol_by_bank = np.bincount(lender, weights=amount, minlength=n) + np.bincount(borrower, weights=amount, minlength=n)
    top3 = np.sort(vol_by_bank)[::-1][:3].sum() / max(vol_by_bank.sum(), 1e-12)
    return {"edges": len(edges), "density": len(edges) / (n * (n - 1)), "reciprocity": float(recip),
            "mean_degree": float(deg.mean()), "max_in_degree": int(in_deg.max()),
            "max_out_degree": int(out_deg.max()), "assortativity": assort,
            "top3_volume_share": float(top3), "volume": float(amount.sum())}


def mbk_topology_series(network: pd.DataFrame, n_banks: int) -> pd.DataFrame:
    """Показатели сети по месяцам (нужен ``record_network=True``)."""
    rows = []
    for m, g in network.groupby("month"):
        st = mbk_network_stats(g["lender"].to_numpy(), g["borrower"].to_numpy(),
                               g["amount"].to_numpy(), n_banks)
        rows.append({"month": m, **st})
    return pd.DataFrame(rows).set_index("month")
