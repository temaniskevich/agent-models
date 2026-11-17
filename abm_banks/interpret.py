"""Текстовая интерпретация результатов прогона (на русском) для интерфейса и отчётов."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .validation import detect_shock


@dataclass
class Finding:
    level: str              # info | good | warn | bad
    title: str
    text: str


def _fmt(x: float, nd: int = 1) -> str:
    return "н/д" if x is None or not np.isfinite(x) else f"{x:,.{nd}f}".replace(",", " ")


def _lag_to_half(series: np.ndarray, m: int, base: float, final: float, horizon: int = 24) -> int | None:
    """Через сколько месяцев после ``m`` показатель прошёл половину пути до итогового уровня."""
    if not np.isfinite(final - base) or abs(final - base) < 1e-9:
        return None
    seg = series[m:m + horizon + 1]
    prog = (seg - base) / (final - base)
    hit = np.flatnonzero(prog >= 0.5)
    return int(hit[0]) if len(hit) else None


def interpret(df: pd.DataFrame, events: pd.DataFrame | None = None) -> list[Finding]:
    """Выводы по одному прогону. ``df`` — ``result.sector``."""
    out: list[Finding] = []
    n = len(df)
    k = df["key_rate"].to_numpy()
    shock = detect_shock(df)
    out.append(Finding(
        "info", "Сценарий",
        f"{n} мес. Ключевая ставка: старт {_fmt(k[0])} %, конец {_fmt(k[-1])} %, "
        f"максимум {_fmt(k.max())} % (месяц {int(np.argmax(k))}). "
        + (f"Самый резкий рост — на {_fmt(shock[1])} п.п. с месяца {shock[0]}." if shock
           else "Резких скачков ставки (≥ 2 п.п. за 3 месяца) нет — это стационарный режим.")))

    if shock:
        m, size = shock
        c = lambda col: df[col].to_numpy()                      # noqa: E731
        pre = lambda col, w=6: float(np.nanmean(c(col)[max(m - w, 0):m]))     # noqa: E731
        # --- передача ставки
        lr0, dr0 = pre("loan_rate"), pre("deposit_rate")
        lr_f = float(np.nanmean(c("loan_rate")[m + 9:m + 15])) if m + 9 < n else float("nan")
        dr_f = float(np.nanmean(c("deposit_rate")[m + 9:m + 15])) if m + 9 < n else float("nan")
        lag_l = _lag_to_half(c("loan_rate"), m, lr0, lr_f)
        lag_d = _lag_to_half(c("deposit_rate"), m, dr0, dr_f)
        if np.isfinite(lr_f):
            out.append(Finding(
                "info", "Передача ставки банкам",
                f"Кредитная ставка выросла на {_fmt(lr_f - lr0)} п.п. "
                f"({_fmt((lr_f - lr0) / size * 100, 0)} % скачка), ставка по вкладам — на {_fmt(dr_f - dr0)} п.п. "
                f"({_fmt((dr_f - dr0) / size * 100, 0)} %). Половина эффекта достигнута через "
                f"{lag_l if lag_l is not None else 'н/д'} мес. (кредиты) и {lag_d if lag_d is not None else 'н/д'} мес. (вклады): "
                "банки подстраивают ставки постепенно из-за доли фиксированных ставок в портфелях и скорости пересмотра цен."))
        # --- кредит и вклады
        g_pre = pre("loans_growth_yoy", 12)
        g = c("loans_growth_yoy")[m:m + 24]
        if len(g) and np.isfinite(g).any():
            i = int(np.nanargmin(g))
            out.append(Finding(
                "warn" if np.nanmin(g) < g_pre - 2 else "info", "Кредитование",
                f"Рост кредитов г/г до скачка {_fmt(g_pre)} %, минимум {_fmt(np.nanmin(g))} % через {i} мес. "
                f"Причины в модели: спрос падает с ростом кредитных ставок, а ужесточение по капиталу "
                f"и ликвидности ограничивает предложение."))
        d_pre = pre("deposits_growth_yoy", 12)
        d = c("deposits_growth_yoy")[m:m + 24]
        if len(d) and np.isfinite(d).any():
            out.append(Finding(
                "info", "Вклады",
                f"Рост вкладов г/г до скачка {_fmt(d_pre)} %, после — от {_fmt(np.nanmin(d))} до {_fmt(np.nanmax(d))} %. "
                "Высокие ставки привлекают вклады; при панике часть средств уходит из слабых банков в надёжные "
                "и частично за пределы сектора."))
        # --- риск и капитал
        npl_pre, npl_pk = pre("npl_share", 12), float(np.nanmax(c("npl_share")[m:m + 24]))
        car_pre, car_min = pre("car", 3), float(np.nanmin(c("car")[m:m + 24]))
        lvl = "bad" if car_min < car_pre - 1.5 else ("warn" if car_min < car_pre - 0.3 else "info")
        out.append(Finding(
            lvl, "Кредитный риск и капитал",
            f"Доля новых дефолтов выросла с {_fmt(npl_pre, 2)} % до {_fmt(npl_pk, 2)} % в месяц "
            f"(×{_fmt(npl_pk / max(npl_pre, 1e-9))}). Достаточность капитала сектора: {_fmt(car_pre)} % → минимум {_fmt(car_min)} %. "
            f"Банков ниже норматива в худший месяц: {int(np.nanmax(c('n_below_car_min')[m:m + 24]))}."))
        # --- ликвидность и ЦБ
        cb_pre, cb_pk = pre("cb_debt"), float(np.nanmax(c("cb_debt")[m:m + 12]))
        mbk_pre, mbk_min = pre("mbk_volume"), float(np.nanmin(c("mbk_volume")[m:m + 6]))
        panic = float(np.nanmax(c("panic")[m:m + 3]))
        out.append(Finding(
            "warn" if cb_pk > 2 * max(cb_pre, 1e-9) + 1 else "info", "Ликвидность и ЦБ",
            f"Паника вкладчиков в пике {_fmt(panic, 2)} (0–1). Долг банков перед ЦБ: {_fmt(cb_pre, 0)} → пик {_fmt(cb_pk, 0)} млрд руб.; "
            f"объём МБК: {_fmt(mbk_pre, 0)} → минимум {_fmt(mbk_min, 0)} млрд руб. "
            "Кредиторы повышают порог доверия и копят ликвидность, слабые банки идут за рефинансированием к ЦБ."))
    else:
        out.append(Finding(
            "info", "Стационарный режим",
            f"Средний рост кредитов {_fmt(df['loans_growth_yoy'].mean())} % г/г, вкладов {_fmt(df['deposits_growth_yoy'].mean())} % г/г; "
            f"ставка по кредитам {_fmt(df['loan_rate'].mean())} %, по вкладам {_fmt(df['deposit_rate'].mean())} %; "
            f"ROE {_fmt(df['roe'].mean())} %, CAR {_fmt(df['car'].mean())} %. Добавьте скачок ставки, чтобы увидеть реакцию."))

    # --- прибыльность
    out.append(Finding(
        "info", "Прибыльность",
        f"Средняя рентабельность капитала {_fmt(df['roe'].median())} % годовых (медиана по месяцам); "
        f"худший месяц {_fmt(df['roe'].min(), 0)} % (переоценка бумаг и потери по кредитам при скачке ставки)."))

    # --- выход банков
    fails = int(df["failures"].sum())
    if fails:
        cause = ""
        if events is not None and len(events):
            vc = events["cause"].value_counts()
            names = {"capital": "потеря капитала", "liquidity": "неплатёжеспособность по ликвидности",
                     "contagion": "заражение через МБК"}
            cause = " Причины: " + ", ".join(f"{names.get(i, i)} — {v}" for i, v in vc.items()) + "."
        cost = float(df["resolution_cost"].sum())
        out.append(Finding(
            "bad" if fails >= 3 else "warn", "Выбытие банков",
            f"Отозвано лицензий: {fails}; вошло новых банков: {int(df['entries'].sum())}.{cause} "
            f"Потери АСВ/ЦБ при ликвидации — {_fmt(cost, 0)} млрд руб.; "
            f"максимальный каскад — {int(df['cascade_rounds'].max())} раунд(ов) заражения."))
    else:
        out.append(Finding("good", "Выбытие банков", "Ни один банк не потерял лицензию."))
    if float(df["bailout_cost"].sum()) > 0:
        out.append(Finding("warn", "Спасение системно значимых банков",
                           f"ЦБ докапитализировал крупные банки на {_fmt(df['bailout_cost'].sum(), 0)} млрд руб."))

    # --- концентрация
    out.append(Finding(
        "info", "Концентрация",
        f"HHI по активам: {_fmt(df['hhi'].iloc[0], 0)} → {_fmt(df['hhi'].iloc[-1], 0)}; "
        f"доля крупнейшего банка {_fmt(df['top1_share'].iloc[0])} % → {_fmt(df['top1_share'].iloc[-1])} %."))
    out.append(Finding(
        "info", "Как читать результат",
        "Параметры по умолчанию иллюстративные и не оценены по данным. Для выводов в статье используйте "
        "откалиброванный конфиг (вкладка «Калибровка») и ансамбль прогонов с разными зёрнами."))
    return out


def findings_markdown(findings: list[Finding]) -> str:
    icon = {"info": "ℹ️", "good": "✅", "warn": "⚠️", "bad": "🔴"}
    return "\n\n".join(f"**{icon[f.level]} {f.title}.** {f.text}" for f in findings)


# --------------------------------------------------------------------------
# Ансамбль прогонов
# --------------------------------------------------------------------------
def ensemble_summary(runs: pd.DataFrame) -> pd.DataFrame:
    """Итоговые показатели по каждому прогону (строка = прогон)."""
    rows = []
    for run, g in runs.groupby("run"):
        rows.append({
            "run": run,
            "max_key_rate": g["key_rate"].max(),
            "min_loans_growth_yoy": g["loans_growth_yoy"].min(),
            "max_npl_share": g["npl_share"].max(),
            "min_car": g["car"].min(),
            "max_cb_debt": g["cb_debt"].max(),
            "min_mbk": g["mbk_volume"].min(),
            "failures": g["failures"].sum(),
            "bailout_cost": g["bailout_cost"].sum(),
            "resolution_cost": g["resolution_cost"].sum(),
            "mean_roe": g["roe"].mean(),
            "final_hhi": g["hhi"].iloc[-1],
        })
    return pd.DataFrame(rows).set_index("run")


def interpret_ensemble(runs: pd.DataFrame) -> list[Finding]:
    s = ensemble_summary(runs)
    n = len(s)
    q = lambda col, p: float(np.nanpercentile(s[col], p))              # noqa: E731
    p_fail = float((s["failures"] > 0).mean() * 100)
    p_bail = float((s["bailout_cost"] > 0).mean() * 100)
    return [
        Finding("info", "Ансамбль", f"{n} прогонов с разными зёрнами и (если задано) разными сценариями."),
        Finding("info", "Кредитование",
                f"Минимальный рост кредитов г/г: медиана {_fmt(q('min_loans_growth_yoy', 50))} % "
                f"(5–95 %: {_fmt(q('min_loans_growth_yoy', 5))} … {_fmt(q('min_loans_growth_yoy', 95))} %)."),
        Finding("warn" if p_fail > 30 else "info", "Выбытие банков",
                f"Хотя бы один отзыв лицензии — в {_fmt(p_fail, 0)} % прогонов; в среднем {_fmt(s['failures'].mean())} за прогон; "
                f"системно значимых банков докапитализировали в {_fmt(p_bail, 0)} % прогонов."),
        Finding("info", "Капитал и ЦБ",
                f"Минимальный CAR сектора: медиана {_fmt(q('min_car', 50))} % (5 %-квантиль {_fmt(q('min_car', 5))} %); "
                f"пик долга перед ЦБ: медиана {_fmt(q('max_cb_debt', 50), 0)} млрд руб. (95 %: {_fmt(q('max_cb_debt', 95), 0)})."),
    ]
