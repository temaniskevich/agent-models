"""Интерактивный стенд агентной модели банковского сектора России.

Запуск:  streamlit run app.py

Вкладки: живая симуляция (графики обновляются по ходу прогона, шоки можно
подавать прямо во время расчёта), банки и сеть МБК, ансамбль прогонов,
интерпретация, калибровка по данным ЦБ, генератор синтетики, описание модели.
"""
from __future__ import annotations

import io
import json
import os
import time

import numpy as np
import pandas as pd
import streamlit as st

from abm_banks import BankingModel, ModelConfig, PRESET_LABELS, preset
from abm_banks.model import SectorCollapse
from abm_banks.batch import DEFAULT_PARAM_RANGES, generate_synthetic, run_many, sample_param_sets
from abm_banks.calibration import (DATA_DIR, DEFAULT_PARAMS, WINDOWS, CalibParam, CalibrationResult,
                                   SMMCalibrator, default_params, load_cbr_monthly)
from abm_banks.interpret import (ensemble_summary, findings_markdown, interpret,
                                 interpret_ensemble)
from abm_banks.metrics import LABELS, build_sector_frame
from abm_banks.synthetic import ARJumpControl
from abm_banks.validation import (checks_table, mbk_network_stats, mbk_topology_series,
                                  size_distribution_stability, stylized_facts)
from ui import charts, live_charts
from ui.params import (GROUPS, KIND_LABELS, P, all_specs, apply_widgets, fill_widgets,
                       widget_key)

st.set_page_config(page_title="АМ банковского сектора РФ", page_icon="🏦", layout="wide")

# скорость: (период обновления графиков, сек.; месяцев модели за одно обновление)
SPEEDS = {"Медленно (≈ 2 мес./с)": (0.5, 1), "Нормально (≈ 8 мес./с)": (0.5, 4),
          "Быстро (≈ 24 мес./с)": (0.5, 12), "Мгновенно (только итог)": (0.5, 10 ** 6)}


# ======================================================================
# Состояние
# ======================================================================
def init_state() -> None:
    ss = st.session_state
    if "initialized" in ss:
        return
    base = preset("rate_shock_panic").replace(record_network=True, n_months=96)
    fill_widgets(base, ss)
    ss.update(initialized=True, preset_name="rate_shock_panic", model=None, records=[], running=False,
              finished=False, run_cfg=None, step_once=False, ensemble=None, ensemble_cfg=None,
              calib=None, synth=None, speed="Нормально (≈ 8 мес./с)", layout_mode="growth")


def current_config() -> ModelConfig:
    cfg = apply_widgets(ModelConfig(), st.session_state)
    cfg.record_banks = True
    cfg.record_network = True
    return cfg


def load_preset_cb() -> None:
    name = st.session_state.preset_name
    fill_widgets(preset(name).replace(record_network=True), st.session_state)
    reset_cb()


def apply_config_cb(cfg: ModelConfig) -> None:
    fill_widgets(cfg, st.session_state)


def new_model(cfg: ModelConfig) -> None:
    ss = st.session_state
    ss.model = BankingModel(cfg, seed=cfg.seed)
    ss.records, ss.finished, ss.run_cfg = [], False, cfg.to_dict()


def start_cb() -> None:
    ss = st.session_state
    if ss.model is None or ss.finished:
        new_model(current_config())
    ss.running = True


def restart_cb() -> None:
    new_model(current_config())
    st.session_state.running = True


def pause_cb() -> None:
    st.session_state.running = False


def step_cb() -> None:
    ss = st.session_state
    if ss.model is None or ss.finished:
        new_model(current_config())
    ss.running, ss.step_once = False, True


def reset_cb() -> None:
    ss = st.session_state
    ss.model, ss.records, ss.running, ss.finished, ss.run_cfg = None, [], False, False, None


def shock_cb(kind: str, value: float = 0.0) -> None:
    ss = st.session_state
    m = ss.model
    if m is None or m.t >= m.total_steps:
        st.toast("Сначала запустите симуляцию — шоки подаются в ходе прогона.", icon="⚠️")
        return
    if kind == "rate":
        m.shift_key_rate(value / 100.0)
        st.toast(f"Ключевая ставка {value:+.0f} п.п. со следующего месяца")
    elif kind == "panic":
        m.inject_panic(value)
        st.toast("Паника вкладчиков в следующем месяце")
    elif kind == "credit":
        m.inject_credit_stress(value)
        st.toast("Рост дефолтов заёмщиков")


# ======================================================================
# Боковая панель
# ======================================================================
def visible(spec: P, state) -> bool:
    kind = state.get(widget_key("scenario.kind"), "constant")
    p = spec.path
    if p in ("scenario.shock_size", "scenario.shock_month", "scenario.shock_decay_months"):
        return kind in ("step", "ramp")
    if p == "scenario.ramp_months":
        return kind == "ramp"
    if p.startswith("scenario.jump") or p == "scenario.drift_sd":
        return kind == "random_jumps"
    if p == "scenario.historical_start":
        return kind == "historical"
    if p == "scenario.key_rate_start":
        return kind != "historical"
    if p.startswith("scenario.") and p.endswith("_peak"):
        return kind in ("step", "ramp", "random_jumps")
    return True


def render_param(spec: P) -> None:
    key = widget_key(spec.path)
    if spec.kind == "bool":
        st.checkbox(spec.label, key=key, help=spec.help or None)
    elif spec.kind == "select":
        names = dict(zip(spec.options, spec.labels))
        st.selectbox(spec.label, spec.options, format_func=lambda o: names.get(o, o), key=key,
                     help=spec.help or None)
    elif spec.kind == "int":
        st.number_input(spec.label, min_value=int(spec.lo), max_value=int(spec.hi),
                        step=int(max(spec.step, 1)), key=key, help=spec.help or None)
    else:
        st.slider(spec.label, float(spec.lo), float(spec.hi), step=float(spec.step), format=spec.fmt,
                  key=key, help=spec.help or None)


def sidebar() -> ModelConfig:
    sb = st.sidebar
    sb.title("⚙️ Конфигурация")
    sb.selectbox("Пресет", list(PRESET_LABELS), format_func=PRESET_LABELS.get, key="preset_name",
                 on_change=load_preset_cb, help="Готовые настройки; после выбора их можно менять ниже.")
    up = sb.file_uploader("Загрузить конфиг (JSON/YAML)", type=["json", "yaml", "yml"], key="cfg_upload")
    if up is not None and st.session_state.get("_last_upload") != (up.name, up.size):
        try:
            text = up.getvalue().decode("utf-8")
            cfg_up = ModelConfig.from_json(text) if up.name.endswith(".json") else ModelConfig.from_yaml(text)
            fill_widgets(cfg_up, st.session_state)
            st.session_state["_last_upload"] = (up.name, up.size)
            sb.success("Конфиг загружен.")
            st.rerun()
        except Exception as e:                                   # noqa: BLE001
            sb.error(f"Не удалось прочитать конфиг: {e}")
    for gname, specs in GROUPS.items():
        with sb.expander(gname, expanded=gname in ("Сценарий ключевой ставки", "Запуск")):
            for spec in specs:
                if visible(spec, st.session_state):
                    render_param(spec)
    cfg = current_config()
    sb.download_button("⬇️ Скачать конфиг (JSON)", cfg.to_json(), file_name="abm_config.json",
                       mime="application/json", use_container_width=True)
    return cfg


# ======================================================================
# Рендер дашборда
# ======================================================================
def show(fig, key: str) -> None:
    st.plotly_chart(fig, use_container_width=True, key=key, config={"displaylayoutbar": False, "displaylogo": False})


def frame_from_records(records: list[dict], cfg: ModelConfig) -> pd.DataFrame:
    return build_sector_frame(records, cfg.start_date)


def snapshot_arrays(model: BankingModel) -> tuple[dict, np.ndarray]:
    snap = model.network_snapshot()
    return snap, model.car_now


def render_kpis(df: pd.DataFrame, model: BankingModel) -> None:
    c = st.columns(8)
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last
    c[0].metric("Месяц", f"{int(last['t']) + 1} / {model.cfg.n_months}")
    c[1].metric("Ключевая, %", f"{last['key_rate']:.2f}", f"{last['key_rate'] - prev['key_rate']:+.2f}")
    c[2].metric("Кредитная, %", f"{last['loan_rate']:.2f}", f"{last['loan_rate'] - prev['loan_rate']:+.2f}")
    c[3].metric("По вкладам, %", f"{last['deposit_rate']:.2f}", f"{last['deposit_rate'] - prev['deposit_rate']:+.2f}")
    g = last["loans_growth_yoy"]
    c[4].metric("Кредиты г/г, %", "—" if pd.isna(g) else f"{g:.1f}")
    g = last["deposits_growth_yoy"]
    c[5].metric("Вклады г/г, %", "—" if pd.isna(g) else f"{g:.1f}")
    c[6].metric("CAR сектора, %", f"{last['car']:.1f}", f"{last['car'] - prev['car']:+.2f}")
    c[7].metric("Банков", f"{int(last['n_banks'])}", f"−{int(df['failures'].sum())} всего" if df["failures"].sum() else None,
                delta_color="inverse")


def render_live(model: BankingModel, records: list[dict], cfg: ModelConfig, mode: str) -> None:
    """Рисует дашборд на Vega-Lite с постоянными именами наборов данных: у таких графиков нет
    идентификатора по содержимому, и при обновлении браузер меняет только данные, не перерисовывая
    график. (Plotly в Streamlit 1.45 пересоздаёт график при каждом обновлении — экран мигает.)"""
    df = frame_from_records(records, cfg)
    iv = model.interventions
    render_kpis(df, model)
    lc = live_charts

    def chart(ch, col, fixed_width: int | None = None) -> None:
        if fixed_width:
            ch = ch.properties(width=fixed_width)
        col.vega_lite_chart(spec=lc.to_spec(ch), use_container_width=fixed_width is None, theme=None)

    r1, r2, r3, r4 = st.columns(2), st.columns(2), st.columns(2), st.columns(2)
    chart(lc.rates(df, iv), r1[0])
    chart(lc.credit_deposits(df, mode, iv), r1[1])
    chart(lc.liquidity(df, iv), r2[0])
    chart(lc.risk(df, cfg.banks.car_min * 100, iv), r2[1])
    chart(lc.exit_panic(df, iv, cfg.sector.n_banks), r3[0])
    chart(lc.profit(df, iv), r3[1])
    snap, car = snapshot_arrays(model)
    chart(lc.structure(snap, car, cfg.banks.car_min * 100), r4[0])
    chart(lc.network(snap), r4[1], fixed_width=420)


# ======================================================================
# Вкладка «Симуляция»
# ======================================================================
def tab_live(cfg: ModelConfig) -> None:
    ss = st.session_state
    model: BankingModel | None = ss.model

    c = st.columns([1.1, 1, 1, 1, 1.3, 2.3])
    c[0].button("▶️ Запустить / продолжить", on_click=start_cb, type="primary", use_container_width=True,
                disabled=ss.running)
    c[1].button("⏸ Пауза", on_click=pause_cb, use_container_width=True, disabled=not ss.running)
    c[2].button("⏭ Шаг", on_click=step_cb, use_container_width=True, disabled=ss.running)
    c[3].button("⏹ Сброс", on_click=reset_cb, use_container_width=True)
    c[4].button("🔄 Новый запуск с этими параметрами", on_click=restart_cb, use_container_width=True)
    c[5].select_slider("Скорость", options=list(SPEEDS), key="speed", label_visibility="collapsed")

    if model is not None and ss.run_cfg is not None and cfg.to_dict() != ss.run_cfg:
        st.info("Параметры в боковой панели изменены. Текущий прогон идёт со старыми — нажмите "
                "«🔄 Новый запуск с этими параметрами», чтобы применить.")

    with st.expander("⚡ Шоки в ходе симуляции (применяются со следующего месяца)", expanded=True):
        b = st.columns(8)
        for col, (lab, v) in zip(b[:5], [("Ставка +1 п.п.", 1), ("Ставка +3 п.п.", 3), ("Ставка +5 п.п.", 5),
                                         ("Ставка −1 п.п.", -1), ("Ставка −3 п.п.", -3)]):
            col.button(lab, on_click=shock_cb, args=("rate", v), use_container_width=True, key=f"sh_rate_{v}")
        b[5].button("😱 Паника вкладчиков", on_click=shock_cb, args=("panic", 0.8), use_container_width=True)
        b[6].button("📉 Рост дефолтов", on_click=shock_cb, args=("credit", 1.0), use_container_width=True)
        b[7].radio("График кредитов", ["growth", "levels"], key="layout_mode", horizontal=True,
                   format_func=lambda x: "темпы" if x == "growth" else "объёмы", label_visibility="collapsed")

    if model is None:
        st.info("Выберите пресет и параметры слева и нажмите «▶️ Запустить». Графики будут строиться по ходу расчёта; "
                "шоки можно подавать кнопками выше прямо во время симуляции.")
        return

    # Живая часть — фрагмент с таймером. Каждый тик продвигает модель на несколько месяцев и обновляет
    # графики на месте. Пока идёт расчёт, остальной экран не перерисовывается (нет мигания).
    interval, per_tick = SPEEDS[ss.speed]
    panel = st.fragment(run_every=interval if ss.running else None)(live_panel)
    panel(per_tick)


def live_panel(per_tick: int) -> None:
    ss = st.session_state
    model: BankingModel = ss.model
    run_cfg = model.cfg
    collapsed = None

    try:
        if ss.step_once and model.t < model.total_steps:
            ss.step_once = False
            _advance(model, 1)
        elif ss.running:
            _advance(model, per_tick)
    except SectorCollapse as exc:
        collapsed = str(exc)

    done = model.t >= model.total_steps
    total = model.total_steps
    if model.t < run_cfg.burn_in:
        st.progress(model.t / total, text=f"Прогрев модели: {model.t}/{run_cfg.burn_in}")
    else:
        st.progress(min(model.t / total, 1.0), text=f"Месяц {max(model.t - run_cfg.burn_in, 0)} из {run_cfg.n_months}")
    if ss.records:
        render_live(model, ss.records, run_cfg, ss.layout_mode)

    if collapsed:
        ss.running, ss.finished = False, True
        st.error(f"Банковская система разрушена: {collapsed}. Результат до этого момента сохранён на вкладках.")
    elif done and not ss.finished:
        ss.running, ss.finished = False, True
        st.rerun(scope="app")                       # остановить таймер и обновить остальные вкладки
    elif ss.finished:
        st.success("Симуляция завершена. Подробности — на вкладках «Банки», «Интерпретация», «Ансамбль».")


def _advance(model: BankingModel, months: int) -> None:
    """Продвигает модель на ``months`` записанных месяцев (прогрев не считается)."""
    ss = st.session_state
    got = 0
    while model.t < model.total_steps and got < months:
        rec = model.step()
        if rec is not None:
            ss.records.append(rec)
            got += 1


# ======================================================================
# Вкладка «Банки»
# ======================================================================
def snap_from_panel(panel: pd.DataFrame, net: pd.DataFrame | None, month: int, n: int) -> tuple[dict, np.ndarray]:
    g = panel[panel["month"] == month].sort_values("slot")
    edges = net[net["month"] == month] if net is not None else pd.DataFrame(columns=["lender", "borrower", "amount"])
    snap = {"lender": edges["lender"].to_numpy(dtype=int), "borrower": edges["borrower"].to_numpy(dtype=int),
            "amount": edges["amount"].to_numpy(), "assets": g["assets"].to_numpy(), "active": g["active"].to_numpy(),
            "reliability": g["reliability"].to_numpy(), "bank_id": g["bank_id"].to_numpy(),
            "cb_debt": g["cb_debt"].to_numpy()}
    return snap, g["car"].to_numpy() / 100.0


def tab_banks(cfg: ModelConfig) -> None:
    ss = st.session_state
    if ss.model is None or not ss.records:
        st.info("Запустите симуляцию — здесь появятся данные по каждому банку.")
        return
    if ss.running:
        st.info("Идёт расчёт. Вкладка обновится по завершении.")
        return
    res = ss.model.result()
    panel = res.banks
    last = int(panel["month"].max())
    if last > 0:
        month = st.slider("Месяц", 0, last, last, key=f"bank_month_{id(ss.model)}")
    else:
        month = 0
    snap, car = snap_from_panel(panel, res.network, month, cfg.sector.n_banks)
    c = st.columns(2)
    with c[0]:
        show(charts.fig_structure(snap, car, cfg.banks.car_min * 100), "bk_struct")
    with c[1]:
        show(charts.fig_network(snap), "bk_net")
    st.subheader("Достаточность капитала по банкам во времени")
    show(charts.fig_bank_heatmap(panel, "car", "CAR по банкам, % (серые клетки — банк не действует)"), "bk_heat")
    g = panel[(panel["month"] == month) & panel["active"]].copy()
    g["Банк"] = "Банк " + (g["bank_id"] + 1).astype(str)
    tbl = g[["Банк", "age", "assets", "loans_corp", "loans_retail", "deposits_corp", "deposits_retail", "capital",
             "car", "liquid", "mbk_lent", "mbk_borrowed", "cb_debt", "loan_rate_corp", "deposit_rate_retail",
             "reliability", "stress"]].rename(columns={
        "age": "Возраст, мес.", "assets": "Активы", "loans_corp": "Кредиты орг.", "loans_retail": "Кредиты нас.",
        "deposits_corp": "Вклады орг.", "deposits_retail": "Вклады нас.", "capital": "Капитал", "car": "CAR, %",
        "liquid": "Ликвидные", "mbk_lent": "МБК выдано", "mbk_borrowed": "МБК получено", "cb_debt": "Долг ЦБ",
        "loan_rate_corp": "Ставка кред. орг., %", "deposit_rate_retail": "Ставка вкл. нас., %",
        "reliability": "Надёжность", "stress": "Стресс"})
    st.dataframe(tbl.sort_values("Активы", ascending=False).round(2), use_container_width=True, hide_index=True)
    if len(res.events):
        st.subheader("Выбывшие банки")
        ev = res.events.copy()
        ev["bank"] = ev["bank"] + 1
        st.dataframe(ev.rename(columns={"month": "Месяц", "bank": "Банк", "cause": "Причина", "assets_bn": "Активы, млрд",
                                        "round": "Раунд каскада", "cost_bn": "Потери АСВ/ЦБ, млрд"}).drop(columns="slot"),
                     use_container_width=True, hide_index=True)


# ======================================================================
# Вкладка «Ансамбль»
# ======================================================================
def tab_ensemble(cfg: ModelConfig) -> None:
    ss = st.session_state
    st.markdown("Много прогонов с разными зёрнами (и, по желанию, разными параметрами и случайными путями ставки): "
                "так отделяют закономерности модели от случайности одного прогона.")
    c = st.columns(4)
    n_runs = c[0].number_input("Число прогонов", 5, 1000, 40, 5)
    seed0 = c[1].number_input("Начальное зерно", 0, 10 ** 6, 1000)
    vary = c[2].checkbox("Разброс параметров (по диапазонам синтетики)", value=False,
                         help="Если включено, параметры спроса/рисков берутся из диапазонов (латинский гиперкуб).")
    jobs = c[3].number_input("Процессов", 1, max(os.cpu_count() or 1, 1), 1)
    if st.button("🧪 Запустить ансамбль", type="primary"):
        bar = st.progress(0.0, text="Ансамбль…")
        seeds = list(range(int(seed0), int(seed0) + int(n_runs)))
        psets = sample_param_sets(len(seeds), DEFAULT_PARAM_RANGES, int(seed0)) if vary else None
        t0 = time.time()
        runs = run_many(cfg.replace(record_banks=False, record_network=False), seeds, n_jobs=int(jobs), param_sets=psets,
                        progress=lambda i, n: bar.progress(i / n, text=f"Прогон {i} из {n}"))
        ss.ensemble, ss.ensemble_cfg = runs, cfg.to_dict()
        bar.progress(1.0, text=f"Готово за {time.time() - t0:.1f} с")
    runs = ss.ensemble
    if runs is None:
        return
    keys = [k for k in LABELS if k in runs.columns]
    sel = st.multiselect("Показатели", keys, default=["loan_rate", "loans_growth_yoy", "cb_debt", "car"],
                         format_func=LABELS.get)
    cols = st.columns(2)
    palette = [charts.COL["loan"], charts.COL["deposit"], charts.COL["cb"], charts.COL["mbk"], charts.COL["bad"]]
    for i, k in enumerate(sel):
        with cols[i % 2]:
            show(charts.fig_fan(runs, k, LABELS[k], color=palette[i % len(palette)], show_paths=8), f"fan_{k}")
    st.markdown(findings_markdown(interpret_ensemble(runs)))
    summ = ensemble_summary(runs)
    st.dataframe(summ.describe(percentiles=[0.05, 0.5, 0.95]).T.round(2), use_container_width=True)
    st.download_button("⬇️ Скачать все прогоны (CSV)", runs.to_csv(index=False).encode("utf-8"),
                       file_name="ensemble.csv", mime="text/csv")


# ======================================================================
# Вкладка «Интерпретация»
# ======================================================================
def tab_interpret(cfg: ModelConfig) -> None:
    ss = st.session_state
    if ss.model is None or len(ss.records) < 6:
        st.info("Запустите симуляцию (не менее 6 месяцев), чтобы получить интерпретацию.")
        return
    if ss.running:
        st.info("Идёт расчёт. Интерпретация появится по завершении или на паузе.")
        return
    res = ss.model.result()
    df = res.sector
    st.subheader("Что произошло")
    st.markdown(findings_markdown(interpret(df, res.events)))
    st.subheader("Критерии качественного соответствия")
    st.caption("Ожидаемая реакция банковского сектора на скачок ключевой ставки (критерий готовности шага 2).")
    st.dataframe(checks_table(stylized_facts(df)), use_container_width=True, hide_index=True)
    if ss.ensemble is not None:
        st.subheader("Ансамбль прогонов")
        st.markdown(findings_markdown(interpret_ensemble(ss.ensemble)))
    st.subheader("Проверки по российским работам")
    with st.expander("Стабильность распределения банков по размеру (Малахов, Пильник, Радионов, 2015)"):
        stab = size_distribution_stability(res.banks)
        st.dataframe(stab.round(3).iloc[::max(len(stab) // 12, 1)], use_container_width=True)
        st.caption("Если распределение стабильно, наклон Ципфа, Gini и HHI меняются слабо, а ранги банков сохраняются "
                   "(корреляция Спирмена к начальному месяцу близка к 1). Сопоставьте с таблицами статьи.")
    with st.expander("Топология сети МБК (Леонидов, Нечитайло, Серебрянникова, 2018)"):
        if res.network is not None and len(res.network):
            ser = mbk_topology_series(res.network, cfg.sector.n_banks)
            st.dataframe(ser.describe().T.round(3), use_container_width=True)
            st.caption("Плотность, взаимность связей, ассортативность (отрицательная — «ядро–периферия»), доля трёх "
                       "крупнейших узлов в обороте. Сопоставьте с эмпирической топологией из статьи.")
        else:
            st.write("Сеть не записывалась.")
    ts = st.session_state.model.interventions
    if ts:
        st.subheader("Ваши вмешательства")
        st.dataframe(pd.DataFrame(ts), use_container_width=True, hide_index=True)


# ======================================================================
# Вкладка «Калибровка»
# ======================================================================
def tab_calibration(cfg: ModelConfig) -> None:
    ss = st.session_state
    st.markdown("**Метод симулированных моментов.** Параметры подбираются так, чтобы модель, подчинённая реальному пути "
                "ключевой ставки, воспроизводила моменты данных ЦБ **только докризисного окна**; затем модель проверяется "
                "на окне, которого в калибровке не было.")
    try:
        data = load_cbr_monthly()
    except FileNotFoundError:
        st.error("Нет данных ЦБ. Выполните в терминале: `python -I scripts/fetch_cbr_data.py`")
        return
    wname = st.radio("Окно", list(WINDOWS), format_func=lambda k: f"{k}: {WINDOWS[k].description}", horizontal=False)
    w = WINDOWS[wname]
    st.caption(f"Калибровка: {w.calib[0]} … {w.calib[1]}; проверка: {w.test[0]} … {w.test[1]}. "
               f"{'Ставок по кредитам и вкладам до 2014 г. в данных нет — подгоняются только темпы роста.' if wname == 'A' else ''}")
    pre_file = DATA_DIR / f"calibrated_window_{wname}.json"
    if pre_file.exists():
        st.button("📥 Загрузить предварительные параметры этого окна (малый бюджет, не для публикации)",
                  on_click=apply_config_cb, args=(cfg.replace(**CalibrationResult.load_theta(pre_file)),))
    c = st.columns(4)
    n_seeds = c[0].slider("Зёрен на вычисление", 2, 8, 4)
    budget = c[1].slider("Бюджет (вычислений цели)", 40, 800, 150, 10)
    method = c[2].selectbox("Метод", ["lhs_nm", "de", "optuna"],
                            format_func={"lhs_nm": "Случайный поиск + Нелдер–Мид", "de": "Дифф. эволюция",
                                         "optuna": "Байесовская (TPE, нужен optuna)"}.get)
    names = [p.name for p in DEFAULT_PARAMS]
    chosen = st.multiselect("Калибруемые параметры", names, default=[p.name for p in default_params(wname)],
                            key=f"calib_params_{wname}",
                            help="В окне A нет ставок банков, поэтому по умолчанию подгоняются только тренды роста.")
    st.caption(f"Оценка времени: ≈ {budget * 0.045 * n_seeds * (len(slice_len(w)) / 60):.0f} с.")
    cal_kwargs = dict(base_config=cfg.replace(record_banks=False, record_network=False), data=data, window=wname,
                      params=[p for p in DEFAULT_PARAMS if p.name in chosen], n_seeds=n_seeds)
    b = st.columns(3)
    if b[0].button("📊 Моменты при текущем конфиге", use_container_width=True):
        cal = SMMCalibrator(**cal_kwargs)
        ss.calib = {"moments": cal.moments_table({}), "theta": None, "window": wname, "result": None}
    if b[1].button("🎯 Откалибровать", type="primary", use_container_width=True):
        cal = SMMCalibrator(**cal_kwargs)
        bar = st.progress(0.0, text="Калибровка…")
        res = cal.fit(method, budget=budget, callback=lambda i, q: bar.progress(min(i / (budget + 40), 1.0),
                                                                                  text=f"Вычисление {i}, цель {q:.1f}"))
        bar.progress(1.0, text=f"Готово: цель {res.objective:.2f}, вычислений {res.n_evals}")
        ss.calib = {"moments": res.moments, "theta": res.theta, "window": wname, "result": res}
    cal_res = ss.calib
    if not cal_res or cal_res["window"] != wname:
        return
    st.subheader("Моменты: данные ЦБ и модель")
    mt = cal_res["moments"].copy()
    st.dataframe(mt.round(3), use_container_width=True)
    if cal_res["theta"]:
        res = cal_res["result"]
        st.subheader("Найденные параметры")
        st.dataframe(pd.Series(cal_res["theta"], name="значение").to_frame().round(4), use_container_width=True)
        cc = st.columns(3)
        new_cfg = cfg.replace(**cal_res["theta"])
        cc[0].button("✅ Применить к конфигу", on_click=apply_config_cb, args=(new_cfg,), use_container_width=True)
        cc[1].download_button("⬇️ Параметры (JSON)", json.dumps(cal_res["theta"], ensure_ascii=False, indent=2),
                              file_name=f"theta_window_{wname}.json", use_container_width=True)
        if len(res.history):
            hist = res.history["objective"].cummin().to_numpy()
            fig = charts._base("Сходимость: лучшее значение цели", "цель (лог. шкала)")
            fig.add_scatter(x=np.arange(1, len(hist) + 1), y=hist, line=dict(color=charts.COL["loan"], width=2.5))
            fig.update_yaxes(type="log")
            show(fig, "calib_hist")
        st.subheader("Проверка вне выборки")
        cal = SMMCalibrator(**cal_kwargs)
        rep = cal.validate(cal_res["theta"])
        ser = st.selectbox("Ряд", [c for c in ["loan_rate_corp", "loan_rate_retail", "deposit_rate_corp", "deposit_rate_retail",
                                                 "loans_corp", "loans_retail", "deposits_corp", "deposits_retail"]
                                    if rep.actual[c].notna().any()])
        show(charts.fig_validation(rep.paths(ser), f"{ser}: данные ЦБ и модель, окно проверки"), "calib_val")
        st.dataframe(rep.summary().round(3), use_container_width=True)
        st.caption("RMSE и смещение — в процентных пунктах (для объёмов — в п.п. годового роста месячных изменений; "
                   "в данных они включают валютную переоценку, которой в модели нет).")
    st.warning("Результат — предварительный: малый бюджет и упрощения модели. Для публикации увеличьте бюджет, "
               "число зёрен и проверьте устойчивость к стартовым точкам.")


def slice_len(w) -> pd.PeriodIndex:
    return pd.period_range(w.calib[0], w.calib[1], freq="M")


# ======================================================================
# Вкладка «Синтетика»
# ======================================================================
def tab_synthetic(cfg: ModelConfig) -> None:
    ss = st.session_state
    st.markdown("Набор синтетических историй для обучения прогнозных моделей: параметры берутся из диапазонов, пути ключевой "
                "ставки — из генератора случайных скачков на 5–10 п.п. (без копирования 2014–15 и 2022 гг.). "
                "Контрольная выборка «без экономики» повторяет те же моменты простой авторегрессией со скачками.")
    c = st.columns(4)
    n_runs = c[0].number_input("Прогонов", 10, 5000, 200, 10, key="syn_n")
    base_seed = c[1].number_input("Базовое зерно", 0, 10 ** 6, 12345, key="syn_seed")
    lo = c[2].number_input("Стартовая ставка от, %", 2.0, 20.0, 5.0, 0.5)
    hi = c[3].number_input("до, %", 2.0, 25.0, 12.0, 0.5)
    ranges = pd.DataFrame([{"параметр": k, "от": v[0], "до": v[1]} for k, v in DEFAULT_PARAM_RANGES.items()])
    ranges = st.data_editor(ranges, hide_index=True, use_container_width=True, key="syn_ranges",
                            disabled=["параметр"])
    jobs = st.number_input("Процессов", 1, max(os.cpu_count() or 1, 1), 1, key="syn_jobs")
    if st.button("🧬 Сгенерировать синтетику", type="primary"):
        bar = st.progress(0.0, text="Генерация…")
        rng = {r["параметр"]: (float(r["от"]), float(r["до"])) for _, r in ranges.iterrows()}
        scfg = cfg.replace(**{"record_banks": False, "record_network": False})
        t0 = time.time()
        data, meta = generate_synthetic(scfg, int(n_runs), int(base_seed), rng, (lo / 100, hi / 100), int(jobs),
                                        progress=lambda i, n: bar.progress(i / n, text=f"Прогон {i} из {n}"))
        ss.synth = {"data": data, "meta": meta, "cfg": scfg.to_dict(), "ranges": rng, "seconds": time.time() - t0}
        bar.progress(1.0, text=f"Готово за {time.time() - t0:.1f} с")
    syn = ss.synth
    if not syn:
        return
    data, meta = syn["data"], syn["meta"]
    st.success(f"{data['run'].nunique()} прогонов × {data['t'].nunique()} мес. = {len(data):,} строк за {syn['seconds']:.1f} с.")
    cc = st.columns(2)
    with cc[0]:
        show(charts.fig_fan(data, "key_rate", "Ключевая ставка в синтетике, %", "%", charts.COL["key"], 12), "syn_key")
    with cc[1]:
        show(charts.fig_fan(data, "loans_growth_yoy", "Рост кредитов г/г, %", "%", charts.COL["loan"]), "syn_loans")
    summ = ensemble_summary(data)
    st.caption("Распределение итоговых показателей по историям")
    st.dataframe(summ.describe(percentiles=[0.05, 0.5, 0.95]).T.round(2), use_container_width=True)
    buf = io.BytesIO()
    data.to_parquet(buf, index=False)
    d = st.columns(4)
    d[0].download_button("⬇️ Данные (Parquet)", buf.getvalue(), file_name="synthetic.parquet", use_container_width=True)
    d[1].download_button("⬇️ Данные (CSV)", data.to_csv(index=False).encode("utf-8"), file_name="synthetic.csv", use_container_width=True)
    d[2].download_button("⬇️ Метаданные (CSV)", meta.to_csv(index=False).encode("utf-8"), file_name="synthetic_meta.csv",
                         use_container_width=True)
    d[3].download_button("⬇️ Манифест (JSON)", json.dumps({"config": syn["cfg"], "ranges": syn["ranges"]}, ensure_ascii=False, indent=2),
                         file_name="manifest.json", use_container_width=True)
    st.subheader("Контроль без экономики")
    if st.button("Подогнать контрольную модель и сравнить моменты"):
        ctl = ARJumpControl().fit(data)
        st.dataframe(ctl.moments_report(data).round(3), use_container_width=True, hide_index=True)
        ctrl = ctl.simulate(int(data["t"].nunique()), int(min(data["run"].nunique(), 200)), seed=7)
        st.download_button("⬇️ Контрольная синтетика (CSV)", ctrl.to_csv(index=False).encode("utf-8"),
                           file_name="synthetic_control.csv")


# ======================================================================
# Вкладка «О модели»
# ======================================================================
ABOUT = """
### Что моделируется
Месячная агентная модель банковского сектора России. **Банки** — неоднородные агенты с полным балансом
(кредиты организациям и населению, ценные бумаги, ликвидность, обязательные резервы, МБК, депозиты в ЦБ; вклады, МБК, долг перед ЦБ, капитал).
**Клиенты** выбирают банк по ставке, размеру и надёжности, реагируют на ставки объёмом спроса и бегут из слабых банков при панике.
**Банк России** задаёт путь ключевой ставки, коридор ставок, резервирование, рефинансирование под залог, отзывает лицензии и спасает системно значимые банки.

### Порядок месяца
погашение МБК и долга перед ЦБ → ставки банков → проценты → дефолты и амортизация → переоценка бумаг → новые кредиты
(ограничены капиталом: CAR) → вклады (бегство, выбор банка, спрос) → резервы, издержки, налоги, дивиденды →
ликвидность (МБК → рефинансирование ЦБ → продажа бумаг; избыток → бумаги или депозит в ЦБ) → платёжеспособность, выбытие
банков, каскады заражения, вход новых банков.

### Ключевые механизмы
- **Ставки банков** = ключевая ± спред; спред зависит от надёжности (капитал, размер), давления на ликвидность, нехватки капитала и ожидаемых потерь; подстройка с лагом.
- **Спрос**: желаемый запас кредитов падает с ростом кредитных ставок, желаемый запас вкладов растёт со ставкой по вкладам; подстройка постепенная.
- **Кредитный риск**: вероятность дефолта растёт со ставкой, списания уменьшают капитал; редкие крупные убытки отдельных банков.
- **Выход**: отзыв лицензии при исчерпании капитала или неплатёжеспособности по ликвидности; ликвидация с возмещением, страхование вкладов,
  заражение кредиторов по МБК; вход нового небольшого банка.
- **Паника**: большой скачок ставки и выбытие банков запускают бегство вкладов в надёжные банки и за пределы сектора, заморозку МБК.

### Ограничения
Параметры по умолчанию иллюстративные (не оценены). Нет валюты, бюджета, реального сектора; ВВП и инфляция входят только через тренды.
Данные ЦБ о ставках начинаются с 2014 г., поэтому окно калибровки A использует только объёмы.
"""


def tab_about(cfg: ModelConfig) -> None:
    st.markdown(ABOUT)
    c = st.columns(2)
    c[0].download_button("⬇️ Текущий конфиг (YAML)", cfg.to_yaml(), file_name="abm_config.yaml", use_container_width=True)
    c[1].download_button("⬇️ Текущий конфиг (JSON)", cfg.to_json(), file_name="abm_config.json", use_container_width=True)
    with st.expander("Показать конфиг"):
        st.code(cfg.to_yaml(), language="yaml")


# ======================================================================
# Streamlit затемняет «устаревшие» элементы (data-stale) на время перерисовки — при обновлении фрагмента
# каждые полсекунды это выглядит как мигание. Отключаем затемнение для графиков, метрик и прогресса.
NO_STALE_DIM = """
<style>
[data-testid="stElementContainer"][data-stale="true"]:has([data-testid="stVegaLiteChart"]),
[data-testid="stElementContainer"][data-stale="true"]:has([data-testid="stMetric"]),
[data-testid="stElementContainer"][data-stale="true"]:has([data-testid="stProgress"]) {
    opacity: 1 !important; transition: none !important;
}
</style>
"""


def main() -> None:
    st.markdown(NO_STALE_DIM, unsafe_allow_html=True)
    init_state()
    cfg = sidebar()
    st.title("🏦 Агентная модель банковского сектора России")
    st.caption("Синтетические истории банковского сектора: балансы банков, МБК, ЦБ, спрос клиентов, выход и вход банков.")
    tabs = st.tabs(["📈 Симуляция", "🏦 Банки и сеть", "🧪 Ансамбль", "🔍 Интерпретация", "🎯 Калибровка",
                    "🧬 Синтетика", "ℹ️ О модели"])
    with tabs[1]:
        tab_banks(cfg)
    with tabs[2]:
        tab_ensemble(cfg)
    with tabs[3]:
        tab_interpret(cfg)
    with tabs[4]:
        tab_calibration(cfg)
    with tabs[5]:
        tab_synthetic(cfg)
    with tabs[6]:
        tab_about(cfg)
    with tabs[0]:                      # живой цикл — последним: он блокирует выполнение скрипта
        tab_live(cfg)


main()
