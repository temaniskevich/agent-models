"""Реестр параметров для боковой панели: подписи, диапазоны, группы.

Виджет хранит значение в «удобных» единицах (проценты, п.п.), конфиг — в долях;
``scale`` переводит одно в другое: ``widget = config * scale``.
"""
from __future__ import annotations

from dataclasses import dataclass

from abm_banks.config import ModelConfig, get_by_path, set_param


@dataclass(frozen=True)
class P:
    path: str
    label: str
    lo: float = 0.0
    hi: float = 1.0
    step: float = 0.01
    scale: float = 1.0
    kind: str = "float"            # float | int | bool | select
    options: tuple = ()
    labels: tuple = ()
    fmt: str = "%.2f"
    help: str = ""


KIND_LABELS = {"constant": "Постоянная ставка", "step": "Скачок (и возврат)",
               "ramp": "Постепенное повышение", "random_jumps": "Случайные скачки",
               "historical": "История ЦБ (с 2013 г.)"}
SIZE_LABELS = {"top20": "Топ-20 из ВКР", "uniform": "Равные банки", "zipf": "Закон Ципфа",
               "lognormal": "Лог-нормальное"}

GROUPS: dict[str, list[P]] = {
    "Сценарий ключевой ставки": [
        P("scenario.kind", "Тип сценария", kind="select", options=tuple(KIND_LABELS),
          labels=tuple(KIND_LABELS.values())),
        P("scenario.key_rate_start", "Начальная ставка, %", 2, 25, 0.25, 100, fmt="%.2f"),
        P("scenario.shock_size", "Размер скачка, п.п.", -10, 15, 0.5, 100, fmt="%.1f",
          help="Для типов «Скачок» и «Постепенное повышение»."),
        P("scenario.shock_month", "Месяц скачка", 1, 240, 1, kind="int", fmt="%d"),
        P("scenario.shock_decay_months", "Полупериод возврата, мес. (0 = без возврата)", 0, 48, 1, fmt="%.0f"),
        P("scenario.ramp_months", "Длительность повышения, мес.", 1, 24, 1, kind="int", fmt="%d"),
        P("scenario.jump_rate_per_year", "Скачков в год (случайный режим)", 0.0, 2.0, 0.05, fmt="%.2f"),
        P("scenario.jump_min", "Мин. размер скачка, п.п.", 1, 15, 0.5, 100, fmt="%.1f"),
        P("scenario.jump_max", "Макс. размер скачка, п.п.", 1, 20, 0.5, 100, fmt="%.1f"),
        P("scenario.jump_up_prob", "Доля скачков вверх", 0.0, 1.0, 0.05),
        P("scenario.jump_half_life_months", "Полупериод затухания скачка, мес.", 1, 36, 1, fmt="%.0f"),
        P("scenario.historical_start", "Начало (YYYY-MM), история ЦБ", kind="select",
          options=tuple(f"{y}-{m:02d}" for y in range(2013, 2022) for m in (1, 7)),
          labels=tuple(f"{y}-{m:02d}" for y in range(2013, 2022) for m in (1, 7))),
        P("scenario.confidence_shock_peak", "Доп. паника в месяц скачка (0–1)", 0.0, 1.0, 0.05,
          help="Внешний импульс паники вкладчиков; без него паника возникает сама из величины скачка."),
        P("scenario.credit_stress_peak", "Доп. рост дефолтов в пик", 0.0, 3.0, 0.1,
          help="Рецессионный шок: множитель вероятности дефолтов = 1 + значение."),
        P("scenario.deposit_shock_peak", "Шок желаемых вкладов (лог)", -0.3, 0.1, 0.01),
        P("scenario.demand_shock_peak", "Шок спроса на кредит (лог)", -0.3, 0.1, 0.01),
    ],
    "Сектор и банки": [
        P("sector.n_banks", "Число банков", 3, 150, 1, kind="int", fmt="%d"),
        P("sector.size_distribution", "Распределение банков по размеру", kind="select",
          options=tuple(SIZE_LABELS), labels=tuple(SIZE_LABELS.values())),
        P("sector.total_assets_bn", "Активы сектора, млрд руб.", 5_000, 200_000, 5_000, fmt="%.0f"),
        P("sector.init_loans_corp", "Кредиты организациям / активы", 0.1, 0.6, 0.01),
        P("sector.init_loans_retail", "Кредиты населению / активы", 0.05, 0.4, 0.01),
        P("banks.car_target_mean", "Целевой CAR, %", 9, 20, 0.25, 100, fmt="%.2f"),
        P("banks.car_min", "Минимальный CAR (норматив), %", 6, 12, 0.5, 100, fmt="%.1f"),
        P("banks.liq_target_mean", "Целевая ликвидность / вклады, %", 2, 15, 0.5, 100, fmt="%.1f"),
        P("banks.cost_ratio_mean", "Опер. расходы / активы, % год", 1, 6, 0.1, 100, fmt="%.1f"),
        P("banks.tax_rate", "Налог на прибыль, %", 0, 40, 1, 100, fmt="%.0f"),
        P("banks.payout_ratio", "Доля дивидендов, %", 0, 80, 5, 100, fmt="%.0f"),
        P("banks.quality_sigma", "Разброс качества заёмщиков", 0.0, 1.0, 0.05),
        P("banks.appetite_sd", "Разброс аппетита к росту кредитов", 0.0, 0.8, 0.05),
    ],
    "Ставки банков": [
        P("banks.loan_spread.0", "Надбавка по кредитам организациям, п.п.", 0, 8, 0.25, 100, fmt="%.2f"),
        P("banks.loan_spread.1", "Надбавка по кредитам населению, п.п.", 0, 15, 0.25, 100, fmt="%.2f"),
        P("banks.deposit_spread.0", "Скидка по вкладам организаций, п.п.", -2, 10, 0.25, 100, fmt="%.2f"),
        P("banks.deposit_spread.1", "Скидка по вкладам населения, п.п.", -2, 8, 0.25, 100, fmt="%.2f"),
        P("banks.rate_speed_loan", "Скорость подстройки кредитных ставок (мес.⁻¹)", 0.05, 1.0, 0.05),
        P("banks.rate_speed_deposit", "Скорость подстройки ставок по вкладам", 0.05, 1.0, 0.05),
        P("banks.market_power_loan", "Рыночная сила: влияние надёжности на кредитные ставки", 0.0, 0.05, 0.002, fmt="%.3f"),
        P("banks.market_power_deposit", "Рыночная сила: влияние надёжности на ставки по вкладам", 0.0, 0.05, 0.002, fmt="%.3f"),
        P("banks.risk_pricing", "Доля ожидаемых потерь в ставке", 0.0, 1.5, 0.1),
        P("banks.capital_scarcity_loan", "Надбавка при нехватке капитала", 0.0, 2.0, 0.1),
        P("banks.float_share_loans.0", "Плавающая ставка: кредиты организациям", 0.0, 1.0, 0.05),
        P("banks.float_share_loans.1", "Плавающая ставка: кредиты населению", 0.0, 1.0, 0.05),
    ],
    "Спрос клиентов": [
        P("demand.loan_trend_growth.0", "Тренд роста кредитов орг., % год", 0, 30, 0.5, 100, fmt="%.1f"),
        P("demand.loan_trend_growth.1", "Тренд роста кредитов нас., % год", 0, 35, 0.5, 100, fmt="%.1f"),
        P("demand.deposit_trend_growth.0", "Тренд роста вкладов орг., % год", 0, 30, 0.5, 100, fmt="%.1f"),
        P("demand.deposit_trend_growth.1", "Тренд роста вкладов нас., % год", 0, 30, 0.5, 100, fmt="%.1f"),
        P("demand.loan_rate_elasticity.0", "Эластичность спроса на кредит орг. к ставке", 0, 8, 0.1, fmt="%.1f"),
        P("demand.loan_rate_elasticity.1", "Эластичность спроса на кредит нас. к ставке", 0, 8, 0.1, fmt="%.1f"),
        P("demand.deposit_rate_semielasticity.0", "Чувствительность вкладов орг. к ставке", 0, 4, 0.1, fmt="%.1f"),
        P("demand.deposit_rate_semielasticity.1", "Чувствительность вкладов нас. к ставке", 0, 4, 0.1, fmt="%.1f"),
        P("demand.loan_adjust_speed", "Скорость подстройки кредитов к желаемому уровню", 0.02, 0.6, 0.02),
        P("demand.deposit_adjust_speed", "Скорость подстройки вкладов", 0.02, 0.6, 0.02),
        P("demand.credit_deposit_link", "Кредит создаёт вклады (мультипликатор)", 0.0, 1.0, 0.05),
        P("demand.deposit_mobility", "Доля вкладов, ежемесячно меняющая банк", 0.0, 0.2, 0.005, fmt="%.3f"),
        P("demand.flight_to_quality", "Бегство в качество при панике", 0.0, 5.0, 0.25),
        P("demand.demand_noise_sd", "Шум спроса", 0.0, 0.03, 0.001, fmt="%.3f"),
    ],
    "Кредитный риск": [
        P("credit.pd_monthly.0", "Дефолты орг., % в мес.", 0.0, 1.5, 0.01, 100, fmt="%.2f"),
        P("credit.pd_monthly.1", "Дефолты нас., % в мес.", 0.0, 2.5, 0.01, 100, fmt="%.2f"),
        P("credit.lgd.0", "Потери при дефолте: орг.", 0.1, 1.0, 0.05),
        P("credit.lgd.1", "Потери при дефолте: нас.", 0.1, 1.0, 0.05),
        P("credit.rate_sensitivity.0", "Рост дефолтов со ставкой: орг.", 0, 40, 1, fmt="%.0f"),
        P("credit.rate_sensitivity.1", "Рост дефолтов со ставкой: нас.", 0, 40, 1, fmt="%.0f"),
        P("credit.idio_event_prob", "Вероятность крупного убытка банка в мес.", 0.0, 0.03, 0.001, fmt="%.3f"),
        P("credit.idio_loss_median", "Медианный размер такого убытка", 0.0, 0.3, 0.01),
    ],
    "МБК и Банк России": [
        P("interbank.structural_share", "Структурная доля МБК в активах", 0.0, 0.15, 0.005, fmt="%.3f"),
        P("interbank.borrower_fraction", "Доля структурных заёмщиков МБК", 0.0, 0.7, 0.05),
        P("interbank.lender_fraction", "Доля структурных кредиторов МБК", 0.0, 0.7, 0.05),
        P("interbank.spread_base", "Базовый спред МБК к ключевой, п.п.", 0, 2, 0.05, 100, fmt="%.2f"),
        P("interbank.spread_risk", "Надбавка за риск заёмщика МБК, п.п.", 0, 6, 0.1, 100, fmt="%.1f"),
        P("interbank.exposure_limit", "Лимит МБК в капитале кредитора", 0.0, 1.0, 0.05),
        P("interbank.counterparty_limit", "Лимит на одного заёмщика (доля капитала)", 0.0, 0.5, 0.01),
        P("interbank.panic_freeze", "Заморозка МБК при панике", 0.0, 1.0, 0.05),
        P("cb.refi_spread", "Ставка рефинансирования = ключевая + , п.п.", 0, 5, 0.25, 100, fmt="%.2f"),
        P("cb.deposit_spread", "Ставка депозитов в ЦБ = ключевая − , п.п.", 0, 5, 0.25, 100, fmt="%.2f"),
        P("cb.reserve_ratio", "Норматив обязательных резервов, %", 0, 15, 0.5, 100, fmt="%.1f"),
        P("cb.pledgeable_share", "Доля активов, годная в залог ЦБ", 0.0, 1.0, 0.05),
        P("cb.tbtf_threshold", "Порог «слишком большой, чтобы упасть» (доля активов)", 0.01, 0.5, 0.01),
        P("cb.bailout_enabled", "ЦБ спасает системно значимые банки", kind="bool"),
        P("cb.revocation_car", "Отзыв лицензии при CAR ниже, %", 0, 8, 0.5, 100, fmt="%.1f",
          help="0 = только при отрицательном капитале (как в плане); 2 % = практика ЦБ по Н1.0."),
    ],
    "Выход, вход и доверие вкладчиков": [
        P("resolution.entry_enabled", "Вход новых банков на место выбывших", kind="bool"),
        P("resolution.entry_delay_months", "Задержка входа, мес.", 0, 24, 1, kind="int", fmt="%d"),
        P("resolution.entry_size_share", "Размер нового банка (доля активов сектора)", 0.001, 0.05, 0.001, fmt="%.3f"),
        P("resolution.insured_share.1", "Застрахованная доля вкладов населения", 0.0, 1.0, 0.05),
        P("resolution.recovery_loans.0", "Возврат по кредитам орг. при ликвидации", 0.0, 1.0, 0.05),
        P("resolution.recovery_loans.1", "Возврат по кредитам нас. при ликвидации", 0.0, 1.0, 0.05),
        P("resolution.deposit_leak", "Утечка вкладов из сектора (обычная)", 0.0, 0.6, 0.02),
        P("resolution.deposit_leak_panic", "Дополнительная утечка при панике", 0.0, 0.8, 0.05),
        P("confidence.run_base", "Плановый отток вкладов, % в мес.", 0.0, 1.0, 0.01, 100, fmt="%.2f"),
        P("confidence.run_stress", "Отток из проблемного банка", 0.0, 0.3, 0.005, fmt="%.3f"),
        P("confidence.run_panic", "Отток при панике", 0.0, 0.5, 0.01),
        P("confidence.rate_jump_threshold", "Скачок ставки, вызывающий панику, п.п.", 0, 10, 0.25, 100, fmt="%.2f"),
        P("confidence.rate_jump_sensitivity", "Сила паники от скачка", 0, 20, 0.5, fmt="%.1f"),
        P("confidence.failure_contagion", "Паника от выбытия банков", 0, 10, 0.5, fmt="%.1f"),
        P("confidence.panic_decay", "Затухание паники (мес.⁻¹)", 0.0, 0.98, 0.02),
    ],
    "Запуск": [
        P("n_months", "Длительность, мес.", 12, 360, 6, kind="int", fmt="%d"),
        P("burn_in", "Прогрев, мес.", 0, 60, 1, kind="int", fmt="%d"),
        P("seed", "Зерно генератора", 0, 999_999, 1, kind="int", fmt="%d"),
        P("policy", "Поведение банков", kind="select", options=("rule", "cautious"),
          labels=("Встроенные правила", "Осторожные банки (эвристика)")),
        P("check_identities", "Проверять балансы на каждом шаге", kind="bool"),
    ],
}


def all_specs() -> list[P]:
    return [p for g in GROUPS.values() for p in g]


def widget_key(path: str) -> str:
    return "w::" + path


def to_widget(spec: P, cfg: ModelConfig):
    v = get_by_path(cfg, spec.path)
    if spec.kind in ("bool", "select"):
        return v
    if spec.kind == "int":
        return int(round(v * spec.scale))
    return float(v) * spec.scale


def from_widget(spec: P, value):
    if spec.kind in ("bool", "select"):
        return value
    if spec.kind == "int":
        return int(round(value / spec.scale))
    return float(value) / spec.scale


def apply_widgets(cfg: ModelConfig, state) -> ModelConfig:
    """Конфиг с значениями из виджетов (session_state)."""
    out = cfg.copy()
    for spec in all_specs():
        key = widget_key(spec.path)
        if key in state:
            set_param(out, spec.path, from_widget(spec, state[key]))
    return out


def fill_widgets(cfg: ModelConfig, state) -> None:
    """Записывает значения конфига в виджеты (вызывать до их создания, например в callback)."""
    for spec in all_specs():
        v = to_widget(spec, cfg)
        if spec.kind == "select" and v not in spec.options:
            v = spec.options[0]
        state[widget_key(spec.path)] = min(max(v, spec.lo), spec.hi) if spec.kind in ("float", "int") else v
