"""Сопоставление выдач АБМ и ЦБ и пересчёт ставки при другом составе выдач."""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pdfplumber
from openpyxl import load_workbook

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "sources"
BULLETIN = "cbr_mortgage_bulletin_2026-09"
WINDOW = pd.period_range("2024-01", "2026-09", freq="M", name="month")
RU_MONTHS = dict(zip("январь февраль март апрель май июнь июль август сентябрь октябрь ноябрь декабрь".split(),
                     range(1, 13)))


def month_label(value):
    match = re.fullmatch(r"([а-яё]+)\s+(\d{4})", str(value).strip().lower())
    if match and match[1] in RU_MONTHS:
        return pd.Period(year=int(match[2]), month=RU_MONTHS[match[1]], freq="M")
    return pd.NaT


def by_month(values, dates, name):
    index = pd.PeriodIndex(dates, freq="M", name="month")
    series = pd.Series(np.asarray(values, dtype=float), index=index, name=name)
    series = series.loc[series.index.notna()].sort_index()
    assert series.index.is_unique
    return series


def household_issuance():
    book = load_workbook(SOURCE / "cbr_households_issuance.xlsx", read_only=True, data_only=True)
    ws = book["в рублях"]
    labels, values = list(ws.iter_rows(min_row=3, max_row=4, values_only=True))
    assert values[0] == "РОССИЙСКАЯ ФЕДЕРАЦИЯ"
    book.close()
    return by_month(values[1:], list(map(month_label, labels[1:])), "all_issuance_bn") / 1000


def rates_workbook():
    book = load_workbook(SOURCE / "cbr_loan_rates_households.xlsx", read_only=True, data_only=True)
    rates = pd.DataFrame(book["ставки_руб"].values)
    structure = pd.DataFrame(book["cтруктура_руб"].values)
    assert rates.iat[4, 9] == "все виды кредитов" and "автокредиты" in structure.iat[3, 9]
    book.close()
    return pd.concat([by_month(pd.to_numeric(rates[9], errors="coerce"), rates[0].map(month_label), "cbr_rate_pct"),
                      by_month(pd.to_numeric(structure[9], errors="coerce") / 100, structure[0].map(month_label),
                               "cbr_auto_share")], axis=1)


def bulletin_tables():
    book = load_workbook(SOURCE / f"{BULLETIN}.xlsx", read_only=True, data_only=True)
    government = pd.DataFrame(book["Т_7а"].values)
    assert government.iat[3, 1] == "Всего" and "господдержки" in government.iat[2, 1]
    govt_months = government[0].map(month_label)
    subsidy = by_month(pd.to_numeric(government[1], errors="coerce") / 1000, govt_months, "subsidized_bn")
    family = by_month(pd.to_numeric(government[2], errors="coerce") / 1000, govt_months, "family_bn")
    households = pd.DataFrame(book["Т_2"].values)
    mortgage = pd.DataFrame(book["Т_3"].values)
    assert households.iat[3, 1] == "в рублях" and "Объем предоставленных кредитов" in mortgage.iat[3, 2]
    book.close()
    result = [subsidy, family]
    for frame, i, name in [(households, 1, "bulletin_all_bn"), (mortgage, 2, "mortgage_bn")]:
        dates = pd.to_datetime(frame[0], format="%Y-%m-%d", errors="coerce").dt.to_period("M") - 1
        result.append(by_month(pd.to_numeric(frame[i], errors="coerce") / 1000, dates, name))
    return pd.concat(result, axis=1)


def mortgage_rates(path):
    html = path.read_text(encoding="utf-8")
    labels = {"Рыночные ИЖК": "market_mortgage_pct", "Льготные ИЖК": "subsidized_mortgage_pct",
              "ИЖК, в целом": "mortgage_pct"}
    for match in re.finditer(r"var settings\s*=\s*", html):
        chart, _ = json.JSONDecoder().raw_decode(html[match.end():])
        series = {s.get("name"): s.get("data") for s in chart.get("series", [])}
        if not labels.keys() <= series.keys():
            continue
        return pd.concat([by_month([p["y"] for p in series[label]],
                                  pd.to_datetime([p["x"] for p in series[label]], unit="ms").to_period("M"), name)
                          for label, name in labels.items()], axis=1)
    raise ValueError(f"Нет графика ставок по рыночной и льготной ипотеке в {path.name}")


def chart_issuance():
    """Рис. 6 бюллетеня. Линейка от 0 до 1500 млрд и 33 группы столбцов."""
    colors = {"cards_bn": (0.733, 0, 0.009, 0), "unsecured_without_cards_bn": (0.33, 0.055, 0.322, 0),
              "chart_mortgage_bn": (0, 0.237, 0.484, 0), "auto_bn": (0, 0.759, 0.517, 0),
              "other_bn": (0.327, 0.233, 0.231, 0)}
    with pdfplumber.open(SOURCE / f"{BULLETIN}.pdf") as pdf:
        page = pdf.pages[6]
        caption = "ОБЪЕМ КРЕДИТОВ, ПРЕДОСТАВЛЕННЫХ ФИЗИЧЕСКИМ ЛИЦАМ Рис. 6"
        assert caption in page.extract_text()
        ticks = sorted(line["top"] for line in page.lines if 565 < line["top"] < 705 and line["width"] > 430)
        assert len(ticks) == 11 and np.ptp(np.diff(ticks)) < 0.07
        scale = 1500 / (ticks[-1] - ticks[0])
        bars = [r for r in page.rects if 1.9 < r["width"] < 2.1 and abs(r["bottom"] - ticks[-1]) < 0.06]
        values = {}
        for name, color in colors.items():
            series = sorted([x for x in bars if x["non_stroking_color"] == color], key=lambda r: r["x0"])
            assert len(series) == len(WINDOW) and np.ptp(np.diff([x["x0"] for x in series])) < 0.12, (name, len(series))
            values[name] = [r["height"] * scale for r in series]
    return pd.DataFrame(values, index=WINDOW)


def annual_context(data):
    # Страница 5 обзора банковского сектора. Все годы, кроме 2026 (январь–сентябрь).
    book = load_workbook(SOURCE / "cbr_banking_sector_2026q3.xlsx", read_only=True, data_only=True)
    sheet = pd.DataFrame(book["Стр.5"].values)
    book.close()
    share = sheet.loc[sheet[1].eq("YoY")].iloc[0, 2:7].to_numpy(dtype=float)
    mortgage = pd.to_numeric(sheet.iloc[23, 2:7]).to_numpy() * 1000
    totals = data.all_issuance_bn.groupby(data.index.year).sum().loc[2022:2026]
    return pd.DataFrame({"all_issuance_bn": totals, "mortgage_bn": mortgage,
                         "subsidized_bn": mortgage * share, "subsidized_of_mortgage": share,
                         "subsidized_of_all": mortgage * share / totals}).rename_axis("year")


def score(data):
    groups = {"pooled": data, **{str(y): g for y, g in data.groupby(data.index.year)}}
    rows = []
    for period, group in groups.items():
        row = {"period": period, "months": len(group),
               "mortgage_share_in_noncard": group.mortgage_bn.sum() / group.noncard_bn.sum(),
               "subsidized_share_in_noncard": group.subsidized_bn.sum() / group.noncard_bn.sum(),
               "subsidized_rate_pct": np.average(group.subsidized_mortgage_pct, weights=group.subsidized_bn),
               "mechanical_gap_pp": group.removal_uplift_pp.mean(),
               "mechanical_gap_all_issuance_pp": group.removal_uplift_all_pp.mean(),
               "market_repricing_uplift_pp": group.market_repricing_uplift_pp.mean()}
        for variant in ("original", "retail_speed"):
            raw = group[f"{variant}_rate_pct"] - group.cbr_rate_pct
            errors = {"raw": raw, "ex_subsidy": group[f"{variant}_rate_pct"] - group.implied_ex_subsidy_pct,
                      "mixed": group[f"{variant}_mixed_pct"] - group.cbr_rate_pct}
            for name, err in errors.items():
                row[f"{variant}_{name}_bias_pp"] = err.mean()
                row[f"{variant}_{name}_rmse_pp"] = np.sqrt((err ** 2).mean())
        rows.append(row)
    return pd.DataFrame(rows).set_index("period")


def main():
    replay = pd.read_csv(HERE / "model_replay.csv", index_col="month")
    replay.index = pd.PeriodIndex(replay.index, freq="M", name="month")
    tables = bulletin_tables()
    rates = mortgage_rates(SOURCE / "cbr_rates_review_2026-09.html")
    corroboration = mortgage_rates(SOURCE / "cbr_mortgage_review_2026-09.html")
    np.testing.assert_allclose(rates.loc[WINDOW], corroboration.loc[WINDOW], atol=0.03, rtol=0)
    raw = pd.concat([replay, household_issuance(), rates_workbook(), tables, rates, chart_issuance()], axis=1)
    annual_context(raw.loc["2022-01":"2026-09"]).to_csv(HERE / "annual_context.csv")
    d = raw.loc[WINDOW].copy()
    assert d.notna().all().all()
    np.testing.assert_allclose(d.all_issuance_bn, d.bulletin_all_bn, atol=0.1, rtol=0)
    np.testing.assert_allclose(d.chart_mortgage_bn, d.mortgage_bn, atol=0.85, rtol=0)
    d["chart_total_bn"] = d[["cards_bn", "unsecured_without_cards_bn", "chart_mortgage_bn", "auto_bn", "other_bn"]].sum(axis=1)
    np.testing.assert_allclose(d.chart_total_bn, d.all_issuance_bn, atol=5.0, rtol=0)
    d["noncard_bn"] = d.all_issuance_bn - d.cards_bn
    d["auto_inferred_issuance_bn"] = d.auto_bn / d.cbr_auto_share
    d["subsidized_share"] = w = d.subsidized_bn / d.noncard_bn
    d["subsidized_share_all"] = w_all = d.subsidized_bn / d.all_issuance_bn
    assert ((d.subsidized_bn < d.mortgage_bn) & (d.mortgage_bn < d.noncard_bn)).all()
    r_all, r_sub = d.cbr_rate_pct, d.subsidized_mortgage_pct
    d["implied_ex_subsidy_pct"] = (r_all - w * r_sub) / (1 - w)
    d["removal_uplift_pp"] = d.implied_ex_subsidy_pct - r_all
    d["removal_uplift_all_pp"] = w_all * (r_all - r_sub) / (1 - w_all)
    d["market_repricing_uplift_pp"] = w * (d.market_mortgage_pct - r_sub)
    for variant in ("original", "retail_speed"):
        d[f"{variant}_mixed_pct"] = (1 - w) * d[f"{variant}_rate_pct"] + w * r_sub
    d.to_csv(HERE / "monthly_comparison.csv")
    score(d).to_csv(HERE / "summary.csv")


if __name__ == "__main__":
    main()
