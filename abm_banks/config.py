"""Конфигурация агентной модели банковского сектора.

Вся модель получает параметры ТОЛЬКО через ``ModelConfig`` (никаких глобальных
``settings``). Конфиг сериализуется в dict/JSON/YAML, поэтому любой прогон
воспроизводится по паре (конфиг, зерно).

Соглашения:
* шаг модели = 1 месяц;
* денежные величины — млрд руб.;
* ставки в конфиге и внутри модели — годовые доли (0.08 = 8 %), в выходных
  таблицах — проценты;
* сегменты везде идут в порядке (организации, население).
"""
from __future__ import annotations

import copy
import dataclasses
import json
import typing
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

SEGMENTS = ("corp", "retail")
SEGMENT_LABELS = {"corp": "Организации", "retail": "Население"}

# Доли ликвидности топ-20 банков из ВКР (сумма = 1)
TOP20_SHARES = (0.45, 0.14, 0.08, 0.06, 0.05, 0.03, 0.02, 0.02, 0.02, 0.02,
                0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01)


# --------------------------------------------------------------------------
# Блоки конфигурации
# --------------------------------------------------------------------------
@dataclass
class SectorConfig:
    """Размер сектора и стартовая структура балансов."""
    n_banks: int = 20
    size_distribution: str = "top20"        # top20 | uniform | zipf | lognormal
    zipf_exponent: float = 1.0
    lognormal_sigma: float = 1.2
    top20_tail_share: float = 0.15          # доля «хвоста» при n_banks > 20
    total_assets_bn: float = 50_000.0       # активы сектора на старте, млрд руб.
    init_loans_corp: float = 0.40           # кредиты организациям / активы
    init_loans_retail: float = 0.17         # кредиты населению / активы
    init_retail_deposit_share: float = 0.52  # доля вкладов населения во вкладах
    composition_noise: float = 0.10         # разброс стартовых долей между банками
    init_car_excess_mean: float = 0.015     # на сколько стартовый CAR выше цели
    init_car_excess_sd: float = 0.008


@dataclass
class BankConfig:
    """Поведение и неоднородность банков."""
    # регулирование и капитал
    car_min: float = 0.08                   # норматив достаточности капитала
    car_target_mean: float = 0.125          # управленческая цель по CAR
    car_target_sd: float = 0.010
    rw_corp: float = 1.0                    # веса риска
    rw_retail: float = 1.2
    rw_securities: float = 0.4
    rw_interbank: float = 0.5
    # ликвидность
    liq_target_mean: float = 0.06           # целевая ликвидность / вклады
    liq_target_sd: float = 0.010
    liq_floor_share: float = 0.5            # минимум = доля от цели
    # издержки и прибыль
    cost_ratio_mean: float = 0.024          # опер. расходы / активы, год
    cost_ratio_sd: float = 0.004
    tax_rate: float = 0.20
    payout_ratio: float = 0.25              # дивиденды из чистой прибыли
    # ценообразование
    loan_spread: tuple[float, float] = (0.010, 0.050)    # базовая надбавка к ключевой
    deposit_spread: tuple[float, float] = (0.035, 0.010)  # базовая скидка к ключевой
    risk_pricing: float = 0.7               # доля ожидаемых потерь, закладываемая в ставку
    risk_premium_cap: float = 0.05          # потолок премии за кредитный риск в ставке (годовых)
    market_power_loan: float = 0.012        # влияние надёжности на ставки
    market_power_deposit: float = 0.012
    liq_pressure_deposit: float = 0.30      # ставка по вкладам растёт при дефиците ликвидности
    liq_pressure_loan: float = 0.10
    capital_scarcity_loan: float = 0.50     # надбавка к кредитной ставке у границы по капиталу
    stress_deposit_premium: float = 0.020   # премия за риск по вкладам у проблемного банка
    rate_speed_loan: float = 0.30           # скорость подстройки ставок (мес.)
    rate_speed_deposit: float = 0.40
    rate_speed_sd: float = 0.06
    pass_through_loan: tuple[float, float] = (1.0, 1.0)      # долгосрочный перенос ключевой ставки в кредитные (1 = полный)
    pass_through_deposit: tuple[float, float] = (1.0, 1.0)   # то же для ставок по вкладам
    loan_rate_cap: float = 0.60
    deposit_rate_floor: float = 0.001
    float_share_loans: tuple[float, float] = (0.35, 0.10)    # доля кредитов с плавающей ставкой
    float_share_deposits: tuple[float, float] = (0.40, 0.15)
    # надёжность
    reliability_scale: float = 0.03
    reliability_car_ref: float = 0.10
    reliability_size_weight: float = 0.35
    # прочее
    securities_yield_spread: float = 0.010  # доходность бумаг = ключевая - spread
    securities_mtm_share: float = 0.20      # доля бумаг по рыночной оценке
    securities_sale_haircut: float = 0.002  # дисконт при вынужденной продаже бумаг
    securities_duration: float = 2.0
    quality_sigma: float = 0.40             # разброс качества заёмщиков между банками
    appetite_sd: float = 0.30               # разброс аппетита к росту кредитов
    deposit_appeal_sd: float = 0.15         # разброс привлекательности по вкладам (бренд, сеть)


@dataclass
class DemandConfig:
    """Спрос на кредиты и предложение вкладов (агрегированные домохозяйства и фирмы)."""
    loan_trend_growth: tuple[float, float] = (0.10, 0.13)     # номинальный тренд, в год
    deposit_trend_growth: tuple[float, float] = (0.100, 0.115)
    loan_rate_elasticity: tuple[float, float] = (2.5, 1.8)    # d ln(кредит*) / d ставка (доли)
    deposit_rate_semielasticity: tuple[float, float] = (1.0, 0.8)
    loan_adjust_speed: float = 0.12         # скорость сближения с желаемым запасом (мес.)
    deposit_adjust_speed: float = 0.20
    loan_maturity_months: tuple[float, float] = (30.0, 36.0)
    credit_deposit_link: float = 0.30       # влияние кредитного разрыва на вклады (денежный мультипликатор)
    demand_noise_sd: float = 0.004
    # выбор банка клиентами
    deposit_mobility: float = 0.04          # доля вкладов, ежемесячно меняющая банк
    deposit_rate_sensitivity: float = 0.35  # логит-чувствительность к ставке (на 1 п.п.)
    deposit_quality_sensitivity: float = 1.2
    deposit_size_bias: float = 0.80         # предпочтение крупных банков (степень)
    loan_rate_sensitivity: float = 0.25     # заёмщики ищут дешёвый кредит (на 1 п.п.)
    loan_size_bias: float = 0.80
    flight_to_quality: float = 2.0          # усиление выбора надёжных при панике
    liq_rationing: float = 0.50             # банк с дефицитом ликвидности урезает выдачу


@dataclass
class CreditRiskConfig:
    pd_monthly: tuple[float, float] = (0.0020, 0.0045)    # базовая доля дефолтов в месяц
    lgd: tuple[float, float] = (0.55, 0.65)
    rate_sensitivity: tuple[float, float] = (10.0, 8.0)  # рост PD на единицу ставки сверх нейтральной
    rate_multiplier_cap: float = 3.0        # потолок роста PD из-за высоких ставок (множитель)
    idio_event_prob: float = 0.004          # вероятность крупного идиосинкратического убытка в месяц
    idio_loss_median: float = 0.05          # медианная потеря, доля кредитного портфеля
    idio_loss_sigma: float = 0.90
    noise_sd: float = 0.15
    pd_smoothing: float = 0.10              # сглаживание наблюдаемой доли дефолтов (EMA)
    max_pd: float = 0.15


@dataclass
class InterbankConfig:
    spread_base: float = 0.002              # МБК = ключевая + base + risk*(1-надёжность)
    spread_risk: float = 0.015
    rate_cap_over_refi: float = 0.03        # верхняя граница МБК относительно ставки рефинансирования
    lend_share: float = 0.80                # какую часть избытка ликвидности банк готов дать
    exposure_limit: float = 0.30            # лимит МБК-кредитов в капитале кредитора
    counterparty_limit: float = 0.10        # лимит на одного заёмщика
    risk_aversion_mean: float = 0.45        # порог надёжности контрагента
    risk_aversion_sd: float = 0.10
    link_persistence: float = 1.5           # устойчивость связей (вес прошлого партнёра)
    max_lenders: int = 8
    hoarding_sensitivity: float = 0.6       # накопление ликвидности при стрессе
    hoarding_liq_multiplier: float = 0.5
    structural_share: float = 0.045         # средняя доля МБК в активах у структурных заёмщиков/кредиторов
    borrower_fraction: float = 0.40         # доля банков — структурных заёмщиков МБК
    lender_fraction: float = 0.40           # доля банков — структурных кредиторов МБК
    panic_freeze: float = 0.40              # при панике кредиторы повышают порог доверия
    panic_liquidity_demand: float = 0.60    # при панике банки хотят больше ликвидности


@dataclass
class CentralBankConfig:
    refi_spread: float = 0.010              # ставка рефинансирования = ключевая + spread
    deposit_spread: float = 0.010           # ставка по депозитам ЦБ = ключевая - spread
    reserve_ratio: float = 0.040            # норматив обязательных резервов
    pledgeable_share: float = 0.35          # доля кредитов и бумаг, годная в залог
    collateral_haircut: float = 0.20
    tbtf_threshold: float = 0.05            # банки крупнее спасаются капиталом ЦБ
    bailout_enabled: bool = True
    bailout_target_car: float = 0.10
    revocation_car: float = 0.0             # отзыв лицензии при CAR ниже порога (0 = отрицательный капитал)


@dataclass
class ResolutionConfig:
    """Выход банков: ликвидация, страхование вкладов, вход новых банков."""
    insured_share: tuple[float, float] = (0.0, 0.70)   # застрахованная доля вкладов
    recovery_loans: tuple[float, float] = (0.55, 0.45)
    securities_haircut: float = 0.05
    deposit_leak: float = 0.10              # доля снятых/выплаченных вкладов, уходящая из сектора
    deposit_leak_panic: float = 0.40        # дополнительная утечка при панике (наличные, валюта)
    entry_enabled: bool = True
    entry_delay_months: int = 3
    entry_size_share: float = 0.005         # размер нового банка / активы сектора на старте
    entry_capital_buffer: float = 0.04


@dataclass
class ConfidenceConfig:
    """Доверие вкладчиков: бегство вкладов и паника."""
    run_base: float = 0.0010                # плановый отток из банка, мес.
    run_stress: float = 0.060               # дополнительный отток при стрессе банка
    run_panic: float = 0.180                # отток при панике (на единицу паники)
    insured_run_dampening: float = 0.70     # застрахованные вклады менее подвержены бегству
    rate_jump_threshold: float = 0.020      # скачок ставки за месяц, после которого растёт паника
    rate_jump_sensitivity: float = 8.0
    failure_contagion: float = 3.0          # паника от доли активов выбывших банков
    panic_decay: float = 0.80
    stress_smoothing: float = 0.30


@dataclass
class ScenarioConfig:
    """Как построить сценарий (путь ключевой ставки и макрошоки)."""
    kind: str = "constant"                  # constant | step | ramp | path | random_jumps | historical
    key_rate_start: float = 0.08
    shock_size: float = 0.07                # величина скачка ставки (доли)
    shock_month: int = 12
    shock_decay_months: float = 0.0         # 0 = скачок остаётся; >0 = полураспад возврата
    ramp_months: int = 6
    path: list[float] | None = None         # явный путь (в долях), если kind == "path"
    historical_start: str = "2010-01"
    # random_jumps
    jump_rate_per_year: float = 0.25
    jump_min: float = 0.05
    jump_max: float = 0.10
    jump_up_prob: float = 0.85
    jump_half_life_months: float = 9.0
    drift_sd: float = 0.0015                # шум в месячных изменениях ставки
    mean_reversion: float = 0.03
    key_rate_floor: float = 0.025
    key_rate_cap: float = 0.30
    # дополнительные макрошоки (по умолчанию выключены: источник кризиса — только ставка)
    credit_stress_peak: float = 0.0         # +доля к вероятности дефолтов в пик скачка
    confidence_shock_peak: float = 0.0      # паника вкладчиков в месяц скачка (0..1)
    deposit_shock_peak: float = 0.0         # лог-шок желаемых вкладов в месяц скачка
    demand_shock_peak: float = 0.0          # лог-шок спроса на кредит в месяц скачка


@dataclass
class ModelConfig:
    n_months: int = 120
    burn_in: int = 12
    seed: int = 0
    start_date: str = "2010-01"
    record_banks: bool = True
    record_network: bool = False            # сохранять межбанковскую сеть каждый месяц
    check_identities: bool = False          # проверка балансовых тождеств на каждом шаге (медленнее)
    policy: str = "rule"                    # rule | см. abm_banks.policies

    sector: SectorConfig = field(default_factory=SectorConfig)
    banks: BankConfig = field(default_factory=BankConfig)
    demand: DemandConfig = field(default_factory=DemandConfig)
    credit: CreditRiskConfig = field(default_factory=CreditRiskConfig)
    interbank: InterbankConfig = field(default_factory=InterbankConfig)
    cb: CentralBankConfig = field(default_factory=CentralBankConfig)
    resolution: ResolutionConfig = field(default_factory=ResolutionConfig)
    confidence: ConfidenceConfig = field(default_factory=ConfidenceConfig)
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)

    # ---- сервис -----------------------------------------------------------
    def validate(self) -> "ModelConfig":
        s, b, c = self.sector, self.banks, self.cb
        errors = []
        if s.n_banks < 3:
            errors.append("n_banks должно быть ≥ 3")
        if s.size_distribution not in ("top20", "uniform", "zipf", "lognormal"):
            errors.append("size_distribution: top20 | uniform | zipf | lognormal")
        if s.init_loans_corp + s.init_loans_retail > 0.85:
            errors.append("доля кредитов в активах слишком велика (> 0.85)")
        if not 0.0 <= s.init_retail_deposit_share <= 1.0:
            errors.append("init_retail_deposit_share должно быть в [0, 1]")
        if self.n_months < 2:
            errors.append("n_months должно быть ≥ 2")
        if self.burn_in < 0:
            errors.append("burn_in не может быть отрицательным")
        if b.car_min <= 0 or b.car_target_mean <= b.car_min:
            errors.append("car_target_mean должна быть выше car_min")
        if not 0.0 <= c.reserve_ratio < 0.5:
            errors.append("reserve_ratio должна быть в [0, 0.5)")
        if self.scenario.kind not in ("constant", "step", "ramp", "path",
                                      "random_jumps", "historical"):
            errors.append(f"неизвестный тип сценария: {self.scenario.kind}")
        if self.scenario.kind == "path" and not self.scenario.path:
            errors.append("для kind='path' нужен scenario.path")
        sc = self.scenario
        if sc.jump_min > sc.jump_max:
            errors.append("scenario.jump_min не может быть больше jump_max")
        if sc.key_rate_floor >= sc.key_rate_cap:
            errors.append("key_rate_floor должна быть меньше key_rate_cap")
        if c.reserve_ratio + b.liq_target_mean + 0.10 > 0.90:
            errors.append("резервы + целевая ликвидность оставляют слишком мало места для активов")
        if errors:
            raise ValueError("Некорректный конфиг: " + "; ".join(errors))
        return self

    def copy(self) -> "ModelConfig":
        return copy.deepcopy(self)

    def replace(self, **updates: Any) -> "ModelConfig":
        """Копия с изменениями. Ключи вида ``"banks.car_min"`` задают вложенные поля."""
        new = self.copy()
        for key, value in updates.items():
            set_param(new, key, value)
        return new

    def to_dict(self) -> dict[str, Any]:
        return _to_plain(dataclasses.asdict(self))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelConfig":
        return _from_dict(cls, data)

    def to_json(self, path: str | Path | None = None, indent: int = 2) -> str:
        text = json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text

    @classmethod
    def from_json(cls, source: str | Path) -> "ModelConfig":
        return cls.from_dict(json.loads(_read_source(source)))

    def to_yaml(self, path: str | Path | None = None) -> str:
        import yaml
        text = yaml.safe_dump(self.to_dict(), allow_unicode=True, sort_keys=False)
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text

    @classmethod
    def from_yaml(cls, source: str | Path) -> "ModelConfig":
        import yaml
        return cls.from_dict(yaml.safe_load(_read_source(source)))


def _read_source(source: str | Path) -> str:
    """Путь к файлу или сам текст (JSON/YAML)."""
    if isinstance(source, Path):
        return source.read_text(encoding="utf-8")
    text = str(source)
    if "\n" in text or text.lstrip().startswith("{"):
        return text
    return Path(text).read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# Вспомогательные функции
# --------------------------------------------------------------------------
def set_by_path(cfg: Any, dotted: str, value: Any) -> None:
    """``set_by_path(cfg, "banks.car_min", 0.1)``; для кортежей по сегментам
    допускается индекс: ``"demand.loan_rate_elasticity.0"``."""
    parts = dotted.split(".")
    obj = cfg
    for p in parts[:-1]:
        obj = obj[int(p)] if isinstance(obj, (list, tuple)) else getattr(obj, p)
    last = parts[-1]
    if isinstance(obj, (list, tuple)):
        raise ValueError("индекс допустим только после имени поля: поле.0")
    if last.isdigit():
        raise ValueError(f"плохой путь {dotted}")
    if not hasattr(obj, last):
        raise AttributeError(f"нет параметра {dotted}")
    setattr(obj, last, value)


def get_by_path(cfg: Any, dotted: str) -> Any:
    obj = cfg
    for p in dotted.split("."):
        obj = obj[int(p)] if isinstance(obj, (list, tuple)) else getattr(obj, p)
    return obj


def set_param(cfg: ModelConfig, dotted: str, value: Any) -> None:
    """Как set_by_path, но понимает ``segment``-индексы у кортежей."""
    parts = dotted.split(".")
    if parts[-1].isdigit():
        field_path = ".".join(parts[:-1])
        idx = int(parts[-1])
        cur = list(get_by_path(cfg, field_path))
        cur[idx] = value
        set_by_path(cfg, field_path, tuple(cur))
    else:
        set_by_path(cfg, dotted, value)


def _to_plain(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    return obj


def _from_dict(cls: type, data: dict[str, Any]) -> Any:
    hints = typing.get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"{cls.__name__}: неизвестные поля {sorted(unknown)}")
    for f in fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        hint = hints[f.name]
        if is_dataclass(hint) and isinstance(value, dict):
            kwargs[f.name] = _from_dict(hint, value)
        elif typing.get_origin(hint) is tuple and isinstance(value, list):
            kwargs[f.name] = tuple(value)
        else:
            kwargs[f.name] = value
    return cls(**kwargs)


# --------------------------------------------------------------------------
# Пресеты
# --------------------------------------------------------------------------
def preset(name: str) -> ModelConfig:
    """Готовые конфигурации для интерфейса и экспериментов."""
    base = ModelConfig()
    if name == "baseline":
        return base
    if name == "rate_shock":
        return base.replace(**{"scenario.kind": "step", "scenario.shock_size": 0.07,
                               "scenario.shock_month": 24, "n_months": 96})
    if name == "rate_shock_panic":
        return base.replace(**{"scenario.kind": "step", "scenario.shock_size": 0.09,
                               "scenario.shock_month": 24, "n_months": 96,
                               "scenario.confidence_shock_peak": 0.6,
                               "scenario.credit_stress_peak": 1.0,
                               "scenario.deposit_shock_peak": -0.04})
    if name == "spike_and_retreat":
        return base.replace(**{"scenario.kind": "step", "scenario.shock_size": 0.09,
                               "scenario.shock_month": 24, "n_months": 96,
                               "scenario.shock_decay_months": 10.0})
    if name == "random_jumps":
        return base.replace(**{"scenario.kind": "random_jumps", "n_months": 180})
    if name == "uniform_banks":
        return base.replace(**{"sector.size_distribution": "uniform"})
    if name == "no_exit":
        return base.replace(**{"resolution.entry_enabled": False,
                               "cb.bailout_enabled": True})
    raise KeyError(f"неизвестный пресет {name}")


PRESET_LABELS = {
    "baseline": "Базовый (спокойный фон)",
    "rate_shock": "Скачок ключевой ставки +7 п.п.",
    "rate_shock_panic": "Скачок +9 п.п. с паникой вкладчиков",
    "spike_and_retreat": "Скачок +9 п.п. и постепенный возврат",
    "random_jumps": "Случайные скачки ставки (генератор синтетики)",
    "uniform_banks": "Равные банки (контрольный)",
    "no_exit": "Без входа новых банков",
}
