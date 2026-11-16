"""Графики живой вкладки на Altair (Vega-Lite).

Зачем отдельно от ``charts.py`` (Plotly): Streamlit считает идентификатор элемента
``st.plotly_chart`` от его содержимого, поэтому при каждом обновлении данных браузер
удаляет график и создаёт заново — экран мигает. Vega-Lite-график без выделения
идентификатора не имеет, и его данные обновляются на месте.

Все функции — чистые: принимают таблицы, возвращают ``alt.Chart``. Выделения
(``selection``/``params``) намеренно не используются.
"""
from __future__ import annotations

from contextlib import nullcontext

import altair as alt
import numpy as np
import pandas as pd

alt.data_transformers.disable_max_rows()

COL = {"key": "#1f2937", "loan": "#2563eb", "deposit": "#0d9488", "mbk": "#f59e0b",
       "cb": "#7c3aed", "bad": "#dc2626", "good": "#16a34a", "grey": "#9ca3af"}
H = 300


class _DS:
    """Именованные наборы данных. Имя постоянно, поэтому спецификация графика между обновлениями
    не меняется и Vega обновляет только данные — без повторной отрисовки (нет мигания)."""

    def __init__(self) -> None:
        self.sets: dict[str, list[dict]] = {}

    def add(self, name: str, df: pd.DataFrame) -> alt.NamedData:
        clean = df.astype(object).where(pd.notna(df), None)
        self.sets[name] = clean.to_dict("records")
        return alt.NamedData(name)

    def done(self, chart, title: str, height: int = 0, width: int | None = None):
        props = {"title": _title(title), "datasets": self.sets, "height": height or H}
        if width:
            props["width"] = width
        return chart.properties(**props)


def _title(text: str) -> alt.TitleParams:
    return alt.TitleParams(text, anchor="start", fontSize=14, fontWeight="bold", offset=14, limit=360)


def _style(chart):
    return (chart.configure_view(strokeWidth=0)
            .configure_axis(labelFontSize=11, titleFontSize=11, gridColor="#eceff3", domainColor="#c7ccd4",
                            tickColor="#c7ccd4")
            .configure_legend(orient="top", title=None, labelFontSize=11, symbolStrokeWidth=3, padding=0,
                              offset=4, columns=5)
            .configure_title(font="sans-serif"))


def _long(df: pd.DataFrame, cols: dict[str, str]) -> pd.DataFrame:
    out = df[["t", *cols]].rename(columns=cols).melt("t", var_name="series", value_name="value")
    return out.dropna(subset=["value"])


def _color(names: list[str], colors: list[str]):
    return alt.Color("series:N", scale=alt.Scale(domain=names, range=colors), legend=alt.Legend(title=None))


def _hover_points(data, color, y_title: str = ""):
    """Невидимые точки поверх линий — чтобы работала подсказка при наведении."""
    return (alt.Chart(data).mark_circle(size=70, opacity=0)
            .encode(x="t:Q", y=alt.Y("value:Q", title=y_title), color=color,
                    tooltip=[alt.Tooltip("t:Q", title="Месяц"), alt.Tooltip("series:N", title=""),
                             alt.Tooltip("value:Q", title="значение", format=",.2f")]))


def _shocks(interventions: list[dict] | None, ds: "_DS"):
    if not interventions:
        return None
    d = pd.DataFrame({"t": sorted({e["month"] for e in interventions})})
    return alt.Chart(ds.add("shocks", d)).mark_rule(strokeDash=[3, 3], color=COL["grey"], opacity=0.8).encode(x="t:Q")


def _layer(*charts):
    layers = [c for c in charts if c is not None]
    return layers[0] if len(layers) == 1 else alt.layer(*layers)


XAXIS = alt.Axis(title="Месяц", tickMinStep=1)


# --------------------------------------------------------------------------
def rates(df: pd.DataFrame, interventions=None) -> alt.Chart:
    ds = _DS()
    names = ["Ключевая", "Кредиты", "Вклады", "МБК"]
    data = ds.add("rates", _long(df, {"key_rate": names[0], "loan_rate": names[1], "deposit_rate": names[2],
                                      "mbk_rate": names[3]}))
    color = _color(names, [COL["key"], COL["loan"], COL["deposit"], COL["mbk"]])
    base = alt.Chart(data).encode(
        x=alt.X("t:Q", axis=XAXIS), y=alt.Y("value:Q", title="% годовых", scale=alt.Scale(zero=False)),
        color=color,
        strokeDash=alt.StrokeDash("series:N", scale=alt.Scale(domain=names, range=[[1, 0], [1, 0], [1, 0], [5, 4]]), legend=None),
        size=alt.Size("series:N", scale=alt.Scale(domain=names, range=[3.2, 2.2, 2.2, 1.5]), legend=None))
    lines = (base.mark_line(interpolate="step-after", clip=True).transform_filter(alt.datum.series == names[0])
             + base.mark_line(clip=True).transform_filter(alt.datum.series != names[0]))
    ch = _layer(lines, _hover_points(data, color, "% годовых"), _shocks(interventions, ds))
    return ds.done(ch, "Ставки: ключевая и банковские, %")


def credit_deposits(df: pd.DataFrame, mode: str = "growth", interventions=None) -> alt.Chart:
    ds = _DS()
    names = ["Кредиты", "Вклады"]
    if mode == "growth":
        data = ds.add("cd", _long(df, {"loans_growth_yoy": names[0], "deposits_growth_yoy": names[1]}))
        title, ytitle = "Рост кредитов и вкладов, % г/г", "% г/г"
    else:
        data = ds.add("cd", _long(df, {"loans_total": names[0], "deposits_total": names[1]}))
        title, ytitle = "Объёмы кредитов и вкладов, млрд руб.", "млрд руб."
    color = _color(names, [COL["loan"], COL["deposit"]])
    lines = alt.Chart(data).mark_line(strokeWidth=2.6, clip=True).encode(
        x=alt.X("t:Q", axis=XAXIS), y=alt.Y("value:Q", title=ytitle, scale=alt.Scale(zero=mode != "growth")), color=color)
    zero = (alt.Chart(ds.add("zero", pd.DataFrame({"y": [0.0]}))).mark_rule(color=COL["grey"]).encode(y="y:Q")
            if mode == "growth" else None)
    ch = _layer(zero, lines, _hover_points(data, color, ytitle), _shocks(interventions, ds))
    return ds.done(ch, title)


def liquidity(df: pd.DataFrame, interventions=None) -> alt.Chart:
    """Слева — депозиты банков в ЦБ (профицит), справа — долг перед ЦБ и МБК (на порядок меньше)."""
    ds = _DS()
    d = ds.add("liq", df[["t", "cb_deposits", "cb_debt", "mbk_volume"]])
    x = alt.X("t:Q", axis=XAXIS)
    left = alt.Chart(d).mark_line(strokeDash=[2, 3], strokeWidth=2.2, color=COL["cb"], clip=True).encode(
        x=x, y=alt.Y("cb_deposits:Q", title="депозиты в ЦБ (пунктир), млрд", axis=alt.Axis(titleColor=COL["cb"])),
        tooltip=[alt.Tooltip("t:Q", title="Месяц"), alt.Tooltip("cb_deposits:Q", title="депозиты в ЦБ", format=",.0f")])
    area = alt.Chart(d).mark_area(color=COL["cb"], opacity=0.25, line={"color": COL["cb"]}, clip=True).encode(
        x=x, y=alt.Y("cb_debt:Q", title="долг ЦБ (заливка) и МБК, млрд", axis=alt.Axis(orient="right", titleColor=COL["mbk"])),
        tooltip=[alt.Tooltip("t:Q", title="Месяц"), alt.Tooltip("cb_debt:Q", title="долг перед ЦБ", format=",.1f")])
    mbk = alt.Chart(d).mark_line(strokeWidth=2.2, color=COL["mbk"], clip=True).encode(
        x=x, y=alt.Y("mbk_volume:Q", axis=alt.Axis(orient="right")),
        tooltip=[alt.Tooltip("t:Q", title="Месяц"), alt.Tooltip("mbk_volume:Q", title="МБК", format=",.0f")])
    ch = alt.layer(left, alt.layer(area, mbk)).resolve_scale(y="independent")
    sh = _shocks(interventions, ds)
    ch = alt.layer(ch, sh) if sh is not None else ch
    return ds.done(ch, "Ликвидность: ЦБ и МБК, млрд руб.")


def risk(df: pd.DataFrame, car_min: float = 8.0, interventions=None) -> alt.Chart:
    ds = _DS()
    d = ds.add("risk", df[["t", "car", "car_min_bank", "npl_share"]])
    x = alt.X("t:Q", axis=XAXIS)
    ysc = alt.Scale(zero=False, nice=True)             # домен считает Vega: спецификация не зависит от данных
    car = alt.Chart(d).mark_line(strokeWidth=2.6, color=COL["loan"], clip=True).encode(
        x=x, y=alt.Y("car:Q", title=f"CAR, % (норматив {car_min:.0f} %)", scale=ysc, axis=alt.Axis(titleColor=COL["loan"])),
        tooltip=[alt.Tooltip("t:Q", title="Месяц"), alt.Tooltip("car:Q", title="CAR сектора, %", format=".2f")])
    weak = alt.Chart(d).mark_line(strokeWidth=1.5, strokeDash=[2, 3], color=COL["grey"], clip=True).encode(
        x=x, y=alt.Y("car_min_bank:Q", scale=ysc),
        tooltip=[alt.Tooltip("t:Q", title="Месяц"), alt.Tooltip("car_min_bank:Q", title="CAR слабейшего, %", format=".2f")])
    norm = alt.Chart(ds.add("norm", pd.DataFrame({"y": [car_min]}))).mark_rule(
        color=COL["bad"], strokeDash=[6, 4], opacity=0.6).encode(y=alt.Y("y:Q", scale=ysc))
    npl = alt.Chart(d).mark_line(strokeWidth=2.2, color=COL["bad"], clip=True).encode(
        x=x, y=alt.Y("npl_share:Q", title="новые дефолты, % в мес.", scale=alt.Scale(zero=True),
                     axis=alt.Axis(orient="right", titleColor=COL["bad"])),
        tooltip=[alt.Tooltip("t:Q", title="Месяц"), alt.Tooltip("npl_share:Q", title="дефолты, % в мес.", format=".2f")])
    ch = alt.layer(alt.layer(norm, weak, car), npl).resolve_scale(y="independent")
    sh = _shocks(interventions, ds)
    ch = alt.layer(ch, sh) if sh is not None else ch
    return ds.done(ch, "Капитал и дефолты")


def exit_panic(df: pd.DataFrame, interventions=None, n_slots: int | None = None) -> alt.Chart:
    ds = _DS()
    d = ds.add("exit", df[["t", "n_banks", "failures", "entries", "panic"]])
    x = alt.X("t:Q", axis=XAXIS)
    # диапазон по числу «слотов» банков (постоянен в прогоне → спецификация не меняется между тиками)
    n = n_slots or int(df["n_banks"].max())
    ysc = alt.Scale(domain=[max(0.0, n * 0.5), n * 1.1 + 1], clamp=True)
    panic = alt.Chart(d).mark_area(color=COL["mbk"], opacity=0.35, line={"color": COL["mbk"]}, clip=True).encode(
        x=x, y=alt.Y("panic:Q", title="паника (0–1)", scale=alt.Scale(domain=[0, 1.05]),
                     axis=alt.Axis(orient="right", titleColor=COL["mbk"])),
        tooltip=[alt.Tooltip("t:Q", title="Месяц"), alt.Tooltip("panic:Q", title="паника", format=".2f")])
    banks = alt.Chart(d).mark_line(interpolate="step-after", strokeWidth=2.6, color=COL["key"], clip=True).encode(
        x=x, y=alt.Y("n_banks:Q", title="банки (✕ отзыв, ▲ вход)", scale=ysc),
        tooltip=[alt.Tooltip("t:Q", title="Месяц"), alt.Tooltip("n_banks:Q", title="банков"), alt.Tooltip("failures:Q", title="отозвано")])
    # события — те же данные с фильтром: спецификация постоянна, даже когда событий ещё нет
    fail = alt.Chart(d).mark_point(shape="cross", size=140, strokeWidth=3, color=COL["bad"], clip=True).encode(
        x="t:Q", y=alt.Y("n_banks:Q", scale=ysc),
        tooltip=[alt.Tooltip("t:Q", title="Месяц"), alt.Tooltip("failures:Q", title="отозвано лицензий")]
    ).transform_filter(alt.datum.failures > 0)
    ent = alt.Chart(d).mark_point(shape="triangle-up", size=90, filled=True, color=COL["good"], clip=True).encode(
        x="t:Q", y=alt.Y("n_banks:Q", scale=ysc),
        tooltip=[alt.Tooltip("t:Q", title="Месяц"), alt.Tooltip("entries:Q", title="вошло банков")]
    ).transform_filter(alt.datum.entries > 0)
    ch = alt.layer(alt.layer(banks, fail, ent), panic).resolve_scale(y="independent")
    sh = _shocks(interventions, ds)
    ch = alt.layer(ch, sh) if sh is not None else ch
    return ds.done(ch, "Банки и паника вкладчиков")


def profit(df: pd.DataFrame, interventions=None) -> alt.Chart:
    ds = _DS()
    x = alt.X("t:Q", axis=XAXIS)
    bars = alt.Chart(ds.add("profit", df[["t", "net_income"]])).mark_bar(clip=True).encode(
        x=x, y=alt.Y("net_income:Q", title="млрд руб. в месяц"),
        color=alt.condition(alt.datum.net_income >= 0, alt.value(COL["good"]), alt.value(COL["bad"])),
        tooltip=[alt.Tooltip("t:Q", title="Месяц"), alt.Tooltip("net_income:Q", title="чистая прибыль", format=",.0f")])
    d = df.assign(loss=-df["loan_losses"], state=-(df["resolution_cost"] + df["bailout_cost"]))
    names = ["Потери по кредитам (−)", "Расходы АСВ/ЦБ (−)"]
    data = ds.add("losses", _long(d, {"loss": names[0], "state": names[1]}))
    color = alt.Color("series:N", scale=alt.Scale(domain=names, range=[COL["bad"], COL["cb"]]), legend=alt.Legend(title=None))
    lines = alt.Chart(data).mark_line(strokeWidth=1.8, clip=True).encode(x="t:Q", y="value:Q", color=color)
    ch = _layer(bars, lines, _shocks(interventions, ds))
    return ds.done(ch, "Прибыль сектора (столбцы) и потери")


def structure(snap: dict, car: np.ndarray, car_min: float = 8.0) -> alt.Chart:
    """Размер банков (активы) с цветом по достаточности капитала."""
    ds = _DS()
    idx = np.flatnonzero(snap["active"])
    d = pd.DataFrame({"bank": [f"Банк {int(snap['bank_id'][i]) + 1}" for i in idx],
                      "assets": snap["assets"][idx], "car": car[idx] * 100})
    scale = alt.Scale(domain=[car_min - 2, car_min + 2, car_min + 8], range=[COL["bad"], COL["mbk"], COL["good"]], clamp=True)
    ch = alt.Chart(ds.add("struct", d)).mark_bar().encode(
        x=alt.X("assets:Q", title="Активы, млрд руб."), y=alt.Y("bank:N", sort="-x", title=None),
        color=alt.Color("car:Q", scale=scale, legend=alt.Legend(title="CAR, %", orient="right")),
        tooltip=[alt.Tooltip("bank:N", title=""), alt.Tooltip("assets:Q", title="активы, млрд", format=",.0f"),
                 alt.Tooltip("car:Q", title="CAR, %", format=".1f")])
    return ds.done(ch, "Банки: размер и достаточность капитала", height=max(H, 15 * len(d) + 50))


def network(snap: dict, max_edges: int = 300) -> alt.Chart:
    """Сеть МБК на окружности: узлы — банки (размер — активы, цвет — надёжность), линии — кредиты."""
    ds = _DS()
    n = len(snap["active"])
    order = np.argsort(-snap["assets"])
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pos = np.empty((n, 2))
    pos[order] = np.column_stack([np.cos(ang), np.sin(ang)])
    act = np.flatnonzero(snap["active"])
    nodes = ds.add("nodes", pd.DataFrame({
        "x": pos[act, 0], "y": pos[act, 1], "label": [str(int(snap["bank_id"][i]) + 1) for i in act],
        "assets": snap["assets"][act], "reliability": snap["reliability"][act], "cb": snap["cb_debt"][act]}))
    dom = alt.Scale(domain=[-1.3, 1.3])
    la, bo, am = snap["lender"], snap["borrower"], snap["amount"]
    keep = np.argsort(-am)[:max_edges] if len(am) else np.array([], int)
    # рёбра — всегда, даже пустой набор: спецификация постоянна
    edges = ds.add("edges", pd.DataFrame({"x": pos[la[keep], 0], "y": pos[la[keep], 1], "x2": pos[bo[keep], 0],
                                          "y2": pos[bo[keep], 1], "amount": am[keep]}))
    edge_layer = alt.Chart(edges).mark_rule(color=COL["mbk"], opacity=0.4).encode(
        x=alt.X("x:Q", scale=dom, axis=None), y=alt.Y("y:Q", scale=dom, axis=None), x2="x2:Q", y2="y2:Q",
        strokeWidth=alt.StrokeWidth("amount:Q", scale=alt.Scale(range=[0.6, 4.5]), legend=None))
    node = alt.Chart(nodes).mark_circle(opacity=0.95, stroke="white", strokeWidth=1.5).encode(
        x=alt.X("x:Q", scale=dom, axis=None), y=alt.Y("y:Q", scale=dom, axis=None),
        size=alt.Size("assets:Q", scale=alt.Scale(range=[120, 2200]), legend=None),
        color=alt.Color("reliability:Q", scale=alt.Scale(domain=[0.2, 0.55, 0.9], range=[COL["bad"], "#fde68a", COL["good"]], clamp=True),
                        legend=alt.Legend(title="надёжность", orient="right")),
        tooltip=[alt.Tooltip("label:N", title="Банк"), alt.Tooltip("assets:Q", title="активы, млрд", format=",.0f"),
                 alt.Tooltip("reliability:Q", title="надёжность", format=".2f"), alt.Tooltip("cb:Q", title="долг ЦБ", format=",.1f")])
    text = alt.Chart(nodes).mark_text(fontSize=9, color="white", fontWeight="bold").encode(
        x=alt.X("x:Q", scale=dom, axis=None), y=alt.Y("y:Q", scale=dom, axis=None), text="label:N")
    return ds.done(alt.layer(edge_layer, node, text), "Сеть МБК (толщина — объём кредита)", height=360)


def finish(chart: alt.Chart) -> alt.Chart:
    """Единый стиль (применять к итоговому графику перед выводом)."""
    return _style(chart)


def to_spec(chart: alt.Chart) -> dict:
    """Спецификация Vega-Lite для ``st.vega_lite_chart``.

    ``st.altair_chart`` подменяет имена наборов данных хэшем содержимого, из-за чего спецификация
    меняется на каждом обновлении и Vega перерисовывает график целиком (мигание). Прямая передача
    спецификации сохраняет постоянные имена — обновляются только данные.
    """
    ctx = alt.themes.enable("none") if alt.themes.active == "default" else nullcontext()
    with ctx:
        return finish(chart).to_dict()
