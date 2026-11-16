"""Таблицы показателей, производные ряды и сводки по прогонам."""
from __future__ import annotations

import numpy as np
import pandas as pd

# Подписи для интерфейса и графиков
LABELS = {
    "key_rate": "Ключевая ставка, %",
    "loan_rate": "Ставка по кредитам, %",
    "loan_rate_corp": "Ставка по кредитам организациям, %",
    "loan_rate_retail": "Ставка по кредитам населению, %",
    "deposit_rate": "Ставка по вкладам, %",
    "deposit_rate_corp": "Ставка по вкладам организаций, %",
    "deposit_rate_retail": "Ставка по вкладам населения, %",
    "loans_total": "Кредиты, млрд руб.",
    "deposits_total": "Вклады, млрд руб.",
    "loans_growth_yoy": "Рост кредитов г/г, %",
    "deposits_growth_yoy": "Рост вкладов г/г, %",
    "loans_growth_mom_ann": "Рост кредитов, % годовых (мес.)",
    "deposits_growth_mom_ann": "Рост вкладов, % годовых (мес.)",
    "cb_debt": "Долг банков перед ЦБ, млрд руб.",
    "cb_deposits": "Депозиты банков в ЦБ, млрд руб.",
    "liquidity_surplus": "Профицит ликвидности, млрд руб.",
    "n_banks": "Число действующих банков",
    "failures": "Отзывы лицензий за месяц",
    "car": "Достаточность капитала сектора, %",
    "npl_share": "Доля дефолтов за месяц, %",
    "roe": "Рентабельность капитала (год.), %",
    "mbk_volume": "Объём МБК, млрд руб.",
    "mbk_rate": "Ставка МБК, %",
    "hhi": "Индекс Херфиндаля–Хиршмана (по активам)",
    "stress": "Стресс банковской системы (0–1)",
    "panic": "Паника вкладчиков (0–1)",
}


def build_sector_frame(records: list[dict], start: str = "2010-01") -> pd.DataFrame:
    df = pd.DataFrame(records)
    if df.empty:
        return df
    df = df.set_index("month")
    df.index = pd.PeriodIndex(pd.period_range(start, periods=len(df), freq="M"), name="date")
    df.insert(0, "t", np.arange(len(df)))
    return add_derived(df)


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Темпы роста, спреды и профицит ликвидности в обозначениях ЦБ."""
    out = df.copy()
    for col, name in (("loans_total", "loans"), ("deposits_total", "deposits"),
                      ("loans_corp", "loans_corp"), ("loans_retail", "loans_retail"),
                      ("deposits_corp", "deposits_corp"), ("deposits_retail", "deposits_retail")):
        x = out[col]
        out[f"{name}_growth_mom"] = x.pct_change() * 100
        out[f"{name}_growth_mom_ann"] = (x / x.shift(1)) ** 12 * 100 - 100
        out[f"{name}_growth_yoy"] = x.pct_change(12) * 100
    out["loan_spread"] = out["loan_rate"] - out["key_rate"]
    out["deposit_spread"] = out["key_rate"] - out["deposit_rate"]
    out["structural_liquidity"] = -out["liquidity_surplus"]       # знак как в статистике ЦБ: дефицит > 0
    return out


def build_bank_frame(records: list[dict], start: str = "2010-01") -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    cols = records[0].keys()
    data = {c: np.concatenate([r[c] for r in records]) for c in cols}
    df = pd.DataFrame(data)
    df["date"] = pd.PeriodIndex(pd.period_range(start, periods=df["month"].max() + 1, freq="M"))[df["month"].to_numpy()]
    return df


# --------------------------------------------------------------------------
# Сводки по многим прогонам
# --------------------------------------------------------------------------
def quantile_bands(runs: pd.DataFrame, column: str, qs=(0.05, 0.25, 0.5, 0.75, 0.95),
                   index: str = "t") -> pd.DataFrame:
    """Квантили показателя по прогонам для «веерных» графиков.

    ``runs`` — длинная таблица (столбцы ``run`` и ``index``)."""
    piv = runs.pivot(index=index, columns="run", values=column)
    return piv.quantile(list(qs), axis=1).T.rename(columns=lambda q: f"q{int(round(q * 100)):02d}")


def inequality(shares: np.ndarray) -> dict[str, float]:
    """Gini, HHI, доля топ-1 и топ-5 для вектора размеров."""
    x = np.sort(np.asarray(shares, dtype=float))
    x = x[x > 0]
    if len(x) == 0:
        return {"gini": np.nan, "hhi": np.nan, "top1": np.nan, "top5": np.nan}
    n = len(x)
    gini = (2 * np.sum(np.arange(1, n + 1) * x) / (n * x.sum())) - (n + 1) / n
    s = x / x.sum()
    return {"gini": float(gini), "hhi": float((s ** 2).sum() * 1e4),
            "top1": float(s[-1]), "top5": float(s[-5:].sum())}


def event_study(df: pd.DataFrame, column: str, month: int, pre: int = 6, post: int = 18) -> pd.Series:
    """Значение показателя относительно уровня за ``pre`` месяцев до события."""
    t = df["t"].to_numpy()
    base = df.loc[(t >= month - pre) & (t < month), column].mean()
    w = df[(t >= month - pre) & (t <= month + post)]
    return pd.Series(w[column].to_numpy() - base, index=w["t"].to_numpy() - month, name=column)
