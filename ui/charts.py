"""Графики Plotly для интерфейса: функции без побочных эффектов (удобно тестировать).

Единый стиль: светлый фон, одна палитра на весь интерфейс, подписи на русском.
Палитра подобрана различимой при дальтонизме (синий/оранжевый/бирюзовый/пурпурный).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

COL = {
    "key": "#1f2937", "loan": "#2563eb", "deposit": "#0d9488", "mbk": "#f59e0b",
    "cb": "#7c3aed", "bad": "#dc2626", "good": "#16a34a", "grey": "#9ca3af",
    "loan2": "#93c5fd", "deposit2": "#99f6e4",
}
LAYOUT = dict(template="plotly_white", margin=dict(l=50, r=20, t=85, b=40), height=350,
              legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, font=dict(size=11)),
              hovermode="x unified", font=dict(size=12))


def _x(df: pd.DataFrame) -> np.ndarray:
    return df["t"].to_numpy() if "t" in df else np.arange(len(df))


def _base(title: str, ytitle: str = "", height: int | None = None) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(**LAYOUT, title=dict(text=title, x=0, y=0.97, yanchor="top", font=dict(size=14)),
                      yaxis_title=ytitle, xaxis_title="Месяц")
    if height:
        fig.update_layout(height=height)
    return fig


def _shock_lines(fig: go.Figure, interventions: list[dict] | None) -> None:
    for e in interventions or []:
        fig.add_vline(x=e["month"], line_dash="dot", line_color=COL["grey"], opacity=0.7)


def fig_rates(df: pd.DataFrame, interventions=None) -> go.Figure:
    fig = _base("Ставки: ключевая и банковские, %", "% годовых")
    x = _x(df)
    fig.add_trace(go.Scatter(x=x, y=df["key_rate"], name="Ключевая", line=dict(color=COL["key"], width=3, shape="hv")))
    fig.add_trace(go.Scatter(x=x, y=df["loan_rate"], name="Кредиты", line=dict(color=COL["loan"], width=2)))
    fig.add_trace(go.Scatter(x=x, y=df["deposit_rate"], name="Вклады", line=dict(color=COL["deposit"], width=2)))
    if df["mbk_rate"].notna().any():
        fig.add_trace(go.Scatter(x=x, y=df["mbk_rate"], name="МБК", line=dict(color=COL["mbk"], width=1.5, dash="dash"),
                                 connectgaps=False))
    _shock_lines(fig, interventions)
    return fig


def fig_credit_deposits(df: pd.DataFrame, mode: str = "growth", interventions=None) -> go.Figure:
    x = _x(df)
    if mode == "growth":
        fig = _base("Рост кредитов и вкладов, % г/г", "% г/г")
        fig.add_trace(go.Scatter(x=x, y=df["loans_growth_yoy"], name="Кредиты", line=dict(color=COL["loan"], width=2.5)))
        fig.add_trace(go.Scatter(x=x, y=df["deposits_growth_yoy"], name="Вклады", line=dict(color=COL["deposit"], width=2.5)))
        fig.add_hline(y=0, line_color=COL["grey"], line_width=1)
    else:
        fig = _base("Объёмы кредитов и вкладов, млрд руб.", "млрд руб.")
        fig.add_trace(go.Scatter(x=x, y=df["loans_corp"], name="Кредиты орг.", stackgroup="l", line=dict(color=COL["loan"])))
        fig.add_trace(go.Scatter(x=x, y=df["loans_retail"], name="Кредиты нас.", stackgroup="l", line=dict(color=COL["loan2"])))
        fig.add_trace(go.Scatter(x=x, y=-df["deposits_corp"], name="Вклады орг.", stackgroup="d", line=dict(color=COL["deposit"]),
                                 hovertemplate="%{customdata:,.0f}", customdata=df["deposits_corp"]))
        fig.add_trace(go.Scatter(x=x, y=-df["deposits_retail"], name="Вклады нас.", stackgroup="d", line=dict(color=COL["deposit2"]),
                                 hovertemplate="%{customdata:,.0f}", customdata=df["deposits_retail"]))
        fig.update_layout(yaxis_title="кредиты (+) / вклады (−), млрд руб.")
    _shock_lines(fig, interventions)
    return fig


def fig_liquidity(df: pd.DataFrame, interventions=None) -> go.Figure:
    """Слева — депозиты банков в ЦБ (профицит), справа — долг перед ЦБ и МБК (они на порядок меньше)."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.update_layout(**LAYOUT, title=dict(text="Ликвидность: депозиты в ЦБ (лев.), долг перед ЦБ и МБК (прав.), млрд руб.",
                                          x=0, y=0.97, yanchor="top", font=dict(size=14)), xaxis_title="Месяц")
    x = _x(df)
    fig.add_trace(go.Scatter(x=x, y=df["cb_deposits"], name="Депозиты в ЦБ", line=dict(color=COL["cb"], width=2, dash="dot")),
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=x, y=df["cb_debt"], name="Долг перед ЦБ", fill="tozeroy", fillcolor="rgba(124,58,237,0.25)",
                             line=dict(color=COL["cb"], width=1.5)), secondary_y=True)
    fig.add_trace(go.Scatter(x=x, y=df["mbk_volume"], name="Объём МБК", line=dict(color=COL["mbk"], width=2)),
                  secondary_y=True)
    fig.update_yaxes(title_text="депозиты в ЦБ", secondary_y=False)
    fig.update_yaxes(title_text="долг ЦБ и МБК", secondary_y=True, showgrid=False)
    _shock_lines(fig, interventions)
    return fig


def fig_risk(df: pd.DataFrame, car_min: float = 8.0, interventions=None) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.update_layout(**LAYOUT, title=dict(text="Капитал (лев.) и новые дефолты (прав.)", x=0, y=0.97, yanchor="top",
                                          font=dict(size=14)), xaxis_title="Месяц")
    x = _x(df)
    fig.add_trace(go.Scatter(x=x, y=df["car"], name="CAR сектора, %", line=dict(color=COL["loan"], width=2.5)), secondary_y=False)
    fig.add_trace(go.Scatter(x=x, y=df["car_min_bank"], name="CAR слабейшего банка, %",
                             line=dict(color=COL["grey"], width=1.5, dash="dot")), secondary_y=False)
    fig.add_trace(go.Scatter(x=x, y=df["npl_share"], name="Новые дефолты, % в мес.", line=dict(color=COL["bad"], width=2)),
                  secondary_y=True)
    fig.add_hline(y=car_min, line_color=COL["bad"], line_dash="dash", line_width=1, opacity=0.6,
                  annotation_text=f"норматив {car_min:.0f} %", annotation_position="top left")
    lo = min(0.0, float(np.nanmin(df["car_min_bank"])) - 1) if len(df) else 0.0
    hi = max(float(np.nanmax(df["car"])) + 3, car_min + 6) if len(df) else 20.0
    fig.update_yaxes(title_text="CAR, %", secondary_y=False, range=[lo, hi])
    fig.update_yaxes(title_text="дефолты, %", secondary_y=True, showgrid=False, rangemode="tozero")
    _shock_lines(fig, interventions)
    return fig


def fig_exit(df: pd.DataFrame, interventions=None) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.update_layout(**LAYOUT, title=dict(text="Банки и доверие вкладчиков", x=0, y=0.97, yanchor="top", font=dict(size=14)), xaxis_title="Месяц")
    x = _x(df)
    fig.add_trace(go.Scatter(x=x, y=df["panic"], name="Паника (0–1, пр. шкала)", fill="tozeroy",
                             line=dict(color=COL["mbk"], width=1), opacity=0.5), secondary_y=True)
    fig.add_trace(go.Scatter(x=x, y=df["n_banks"], name="Действующие банки", line=dict(color=COL["key"], width=2.5, shape="hv")),
                  secondary_y=False)
    fail = df[df["failures"] > 0]
    if len(fail):
        fig.add_trace(go.Scatter(x=_x(fail), y=fail["n_banks"], mode="markers+text", name="Отзыв лицензии",
                                 text=[f"−{int(v)}" for v in fail["failures"]], textposition="top center",
                                 marker=dict(color=COL["bad"], size=11, symbol="x")), secondary_y=False)
    ent = df[df["entries"] > 0]
    if len(ent):
        fig.add_trace(go.Scatter(x=_x(ent), y=ent["n_banks"], mode="markers", name="Вход нового банка",
                                 marker=dict(color=COL["good"], size=9, symbol="triangle-up")), secondary_y=False)
    n_max = float(df["n_banks"].max()) if len(df) else 20
    fig.update_yaxes(title_text="Число банков", secondary_y=False, range=[max(0, n_max * 0.5), n_max * 1.1 + 1])
    fig.update_yaxes(title_text="Паника", secondary_y=True, range=[0, 1.05], showgrid=False)
    _shock_lines(fig, interventions)
    return fig


def fig_profit(df: pd.DataFrame, interventions=None) -> go.Figure:
    fig = _base("Прибыльность сектора", "млрд руб. в месяц")
    x = _x(df)
    fig.add_trace(go.Bar(x=x, y=df["net_income"], name="Чистая прибыль", marker_color=np.where(df["net_income"] >= 0, COL["good"], COL["bad"])))
    fig.add_trace(go.Scatter(x=x, y=-df["loan_losses"], name="Потери по кредитам (−)", line=dict(color=COL["bad"], width=1.5)))
    fig.add_trace(go.Scatter(x=x, y=-df["resolution_cost"] - df["bailout_cost"], name="Расходы АСВ/ЦБ (−)",
                             line=dict(color=COL["cb"], width=1.5, dash="dot")))
    _shock_lines(fig, interventions)
    return fig


def fig_structure(snap: dict, car: np.ndarray, car_min: float = 8.0) -> go.Figure:
    """Размер банков (активы) с цветом по достаточности капитала."""
    act = snap["active"]
    idx = np.flatnonzero(act)
    idx = idx[np.argsort(-snap["assets"][idx])]
    labels = [f"Банк {int(snap['bank_id'][i]) + 1}" for i in idx]
    fig = go.Figure(go.Bar(
        x=snap["assets"][idx], y=labels, orientation="h",
        marker=dict(color=car[idx] * 100, colorscale=[[0, COL["bad"]], [0.35, COL["mbk"]], [1, COL["good"]]],
                    cmin=car_min - 2, cmax=car_min + 8, showscale=True,
                    colorbar=dict(title="CAR, %", thickness=12)),
        hovertemplate="%{y}<br>активы %{x:,.0f} млрд<br>CAR %{marker.color:.1f} %<extra></extra>"))
    fig.update_layout(**{**LAYOUT, "height": max(330, 22 * len(idx) + 90), "hovermode": "closest"},
                      title=dict(text="Банки: размер и достаточность капитала", x=0, y=0.97, yanchor="top", font=dict(size=14)),
                      xaxis_title="Активы, млрд руб.", yaxis=dict(autorange="reversed"))
    return fig


def fig_network(snap: dict, max_edges: int = 400) -> go.Figure:
    """Сеть МБК: узлы на окружности (размер — активы, цвет — надёжность), ребра — кредиты."""
    n = len(snap["active"])
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    order = np.argsort(-snap["assets"])               # крупные банки рядом
    pos = np.empty((n, 2))
    pos[order] = np.column_stack([np.cos(ang), np.sin(ang)])
    fig = go.Figure()
    la, bo, am = snap["lender"], snap["borrower"], snap["amount"]
    if len(am):
        keep = np.argsort(-am)[:max_edges]
        la, bo, am = la[keep], bo[keep], am[keep]
        bins = np.quantile(am, [0.0, 0.5, 0.85, 1.0]) if len(am) > 3 else np.array([0, am.max() / 2, am.max(), am.max()])
        widths = [0.8, 1.8, 3.5]
        for b in range(3):
            sel = (am >= bins[b]) & (am <= bins[b + 1] + 1e-12) if b == 2 else (am >= bins[b]) & (am < bins[b + 1])
            if not sel.any():
                continue
            xs, ys = [], []
            for i, j in zip(la[sel], bo[sel]):
                xs += [pos[i, 0], pos[j, 0], None]
                ys += [pos[i, 1], pos[j, 1], None]
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(width=widths[b], color="rgba(245,158,11,0.45)"),
                                     hoverinfo="skip", showlegend=False))
    act = snap["active"]
    size = 10 + 40 * np.sqrt(np.maximum(snap["assets"], 0) / max(snap["assets"].max(), 1e-9))
    fig.add_trace(go.Scatter(
        x=pos[act, 0], y=pos[act, 1], mode="markers+text", text=[str(int(snap["bank_id"][i]) + 1) for i in np.flatnonzero(act)],
        textposition="middle center", textfont=dict(size=9, color="white"),
        marker=dict(size=size[act], color=snap["reliability"][act], colorscale="RdYlGn", cmin=0.2, cmax=0.9,
                    line=dict(width=1.5, color=np.where(snap["cb_debt"][act] > 1e-6, COL["cb"], "white")),
                    showscale=True, colorbar=dict(title="надёжность", thickness=10)),
        customdata=np.column_stack([snap["assets"][act], snap["cb_debt"][act]]),
        hovertemplate="Банк %{text}<br>активы %{customdata[0]:,.0f} млрд<br>долг ЦБ %{customdata[1]:,.1f}<extra></extra>",
        showlegend=False))
    fig.update_layout(**{**LAYOUT, "height": 420, "hovermode": "closest"},
                      title=dict(text="Сеть межбанковского кредитования (фиолетовый контур — заём у ЦБ)", x=0, y=0.97, yanchor="top", font=dict(size=14)),
                      xaxis=dict(visible=False, range=[-1.35, 1.35]), yaxis=dict(visible=False, range=[-1.35, 1.35], scaleanchor="x"))
    return fig


def fig_bank_heatmap(banks: pd.DataFrame, value: str = "car", title: str = "CAR по банкам, %") -> go.Figure:
    piv = banks.pivot(index="slot", columns="month", values=value).where(banks.pivot(index="slot", columns="month", values="active"))
    fig = go.Figure(go.Heatmap(z=piv.values, x=piv.columns, y=[f"Банк {i + 1}" for i in piv.index],
                               colorscale="RdYlGn", zmin=4, zmax=20 if value == "car" else None,
                               colorbar=dict(title="%"), hoverongaps=False))
    fig.update_layout(**{**LAYOUT, "height": max(330, 18 * len(piv) + 100), "hovermode": "closest"},
                      title=dict(text=title, x=0, y=0.97, yanchor="top", font=dict(size=14)), xaxis_title="Месяц", yaxis=dict(autorange="reversed"))
    return fig


def fig_fan(runs: pd.DataFrame, column: str, title: str, ytitle: str = "", color: str = COL["loan"],
            show_paths: int = 0) -> go.Figure:
    """Веерный график по ансамблю прогонов: медиана и интервалы 50 % / 90 %."""
    piv = runs.pivot(index="t", columns="run", values=column)
    q = piv.quantile([0.05, 0.25, 0.5, 0.75, 0.95], axis=1).T
    x = q.index.to_numpy()
    fig = _base(title, ytitle)
    rgb = tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))
    fig.add_trace(go.Scatter(x=np.r_[x, x[::-1]], y=np.r_[q[0.95], q[0.05][::-1]], fill="toself",
                             fillcolor=f"rgba{rgb + (0.15,)}", line=dict(width=0), name="90 % прогонов", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=np.r_[x, x[::-1]], y=np.r_[q[0.75], q[0.25][::-1]], fill="toself",
                             fillcolor=f"rgba{rgb + (0.30,)}", line=dict(width=0), name="50 % прогонов", hoverinfo="skip"))
    for r in list(piv.columns[:show_paths]):
        fig.add_trace(go.Scatter(x=x, y=piv[r], line=dict(width=0.6, color="rgba(100,100,100,0.35)"), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=q[0.5], name="Медиана", line=dict(color=color, width=2.5)))
    return fig


def fig_validation(paths: pd.DataFrame, title: str, ytitle: str = "") -> go.Figure:
    """Факт против прогноза модели (полоса 5–95 % по зёрнам)."""
    fig = _base(title, ytitle)
    x = [str(p) for p in paths.index]
    fig.add_trace(go.Scatter(x=x + x[::-1], y=np.r_[paths["model_p95"], paths["model_p05"][::-1]], fill="toself",
                             fillcolor="rgba(37,99,235,0.18)", line=dict(width=0), name="модель 5–95 %", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=paths["model_mean"], name="модель (среднее)", line=dict(color=COL["loan"], width=2.5)))
    fig.add_trace(go.Scatter(x=x, y=paths["actual"], name="данные ЦБ", line=dict(color=COL["key"], width=2.5, dash="dot"),
                             mode="lines+markers"))
    fig.update_layout(xaxis_title="Месяц")
    return fig


def fig_interventions_note() -> str:
    return "Пунктирные вертикали — ваши вмешательства в ходе симуляции."
