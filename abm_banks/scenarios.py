"""Сценарии: путь ключевой ставки и (необязательные) макрошоки.

Сценарий — это набор массивов длиной ``n_months`` (месяцы ПОСЛЕ прогрева).
Модель сама «растягивает» его на период прогрева, повторяя первое значение.

* ``key_rate``     — ключевая ставка, годовая доля (0.08 = 8 %);
* ``credit_stress``— добавка к вероятности дефолтов (множитель = 1 + значение);
* ``confidence``   — паника вкладчиков, 0..1 (разовый импульс);
* ``deposit_shock``— лог-шок желаемого запаса вкладов (уровень);
* ``demand_shock`` — лог-шок желаемого запаса кредитов (уровень);
* ``rw_retail_mult`` — множитель к весу риска розничных кредитов (макропруденциальные надбавки ЦБ:
  1.0 — без надбавки, 1.5 — вес 1.2 → 1.8); по умолчанию 1.0.

Реальные макрошоки по умолчанию выключены: по плану исследования кризис задаёт
только путь ставки, а остальное должна породить сама модель.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ScenarioConfig

DATA_DIR = Path(__file__).resolve().parent / "data"
MIN_KEY_RATE = 0.005                    # нижняя граница ключевой ставки в моделируемых путях (0,5 %)


@dataclass
class Scenario:
    name: str
    key_rate: np.ndarray
    credit_stress: np.ndarray = field(default=None)    # type: ignore[assignment]
    confidence: np.ndarray = field(default=None)       # type: ignore[assignment]
    deposit_shock: np.ndarray = field(default=None)    # type: ignore[assignment]
    demand_shock: np.ndarray = field(default=None)     # type: ignore[assignment]
    rw_retail_mult: np.ndarray = field(default=None)   # type: ignore[assignment]
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.key_rate)
        self.key_rate = np.asarray(self.key_rate, dtype=float)
        for attr in ("credit_stress", "confidence", "deposit_shock", "demand_shock"):
            val = getattr(self, attr)
            setattr(self, attr, np.zeros(n) if val is None else np.asarray(val, dtype=float))
            if len(getattr(self, attr)) != n:
                raise ValueError(f"{attr}: длина должна совпадать с key_rate ({n})")
        self.rw_retail_mult = (np.ones(n) if self.rw_retail_mult is None
                               else np.asarray(self.rw_retail_mult, dtype=float))
        if len(self.rw_retail_mult) != n:
            raise ValueError(f"rw_retail_mult: длина должна совпадать с key_rate ({n})")
        if np.any(self.rw_retail_mult <= 0):
            raise ValueError("rw_retail_mult должен быть положительным")
        if np.any(self.key_rate <= 0):
            raise ValueError("ключевая ставка должна быть положительной")

    def __len__(self) -> int:
        return len(self.key_rate)

    def padded(self, burn_in: int) -> "Scenario":
        """Добавляет ``burn_in`` месяцев прогрева (ставка постоянна, шоков нет)."""
        if burn_in <= 0:
            return self
        pad = lambda a, v: np.concatenate([np.full(burn_in, v), a])   # noqa: E731
        return Scenario(
            name=self.name,
            key_rate=pad(self.key_rate, self.key_rate[0]),
            credit_stress=pad(self.credit_stress, 0.0),
            confidence=pad(self.confidence, 0.0),
            deposit_shock=pad(self.deposit_shock, 0.0),
            demand_shock=pad(self.demand_shock, 0.0),
            rw_retail_mult=pad(self.rw_retail_mult, 1.0),
            meta=dict(self.meta, burn_in=burn_in),
        )

    def to_frame(self, start: str = "2010-01") -> pd.DataFrame:
        idx = pd.period_range(start, periods=len(self), freq="M")
        return pd.DataFrame({
            "key_rate": self.key_rate * 100, "credit_stress": self.credit_stress,
            "confidence": self.confidence, "deposit_shock": self.deposit_shock,
            "demand_shock": self.demand_shock}, index=idx)


# --------------------------------------------------------------------------
# Базовые формы пути ставки
# --------------------------------------------------------------------------
def constant_path(rate: float, n: int) -> np.ndarray:
    return np.full(n, float(rate))


def step_path(rate0: float, delta: float, month: int, n: int,
              decay_months: float = 0.0) -> np.ndarray:
    """Скачок на ``delta`` в месяце ``month``; при ``decay_months`` > 0 — возврат
    с полураспадом ``decay_months`` месяцев."""
    t = np.arange(n)
    after = (t >= month).astype(float)
    if decay_months and decay_months > 0:
        shape = after * 0.5 ** (np.maximum(t - month, 0) / decay_months)
    else:
        shape = after
    return rate0 + delta * shape


def ramp_path(rate0: float, delta: float, month: int, ramp: int, n: int) -> np.ndarray:
    t = np.arange(n)
    frac = np.clip((t - month + 1) / max(ramp, 1), 0.0, 1.0)
    return rate0 + delta * frac


def random_jump_path(rng: np.random.Generator, n: int, cfg: ScenarioConfig,
                     k0: float | None = None) -> tuple[np.ndarray, list[dict]]:
    """Случайный путь: медленный дрейф с возвратом к ``k0`` плюс редкие скачки
    на 5–10 п.п., затухающие с заданным полупериодом.

    Правило не копирует 2014–15 и 2022 годы: моменты скачков, их знак и размер
    случайны. Возвращает путь и список событий (для метаданных синтетики).
    """
    k0 = cfg.key_rate_start if k0 is None else k0
    decay = 0.5 ** (1.0 / max(cfg.jump_half_life_months, 1e-6))
    base, jump = k0, 0.0
    path = np.empty(n)
    events: list[dict] = []
    p_jump = cfg.jump_rate_per_year / 12.0
    for t in range(n):
        base += cfg.mean_reversion * (k0 - base) + cfg.drift_sd * rng.standard_normal()
        jump *= decay
        if rng.random() < p_jump:
            size = rng.uniform(cfg.jump_min, cfg.jump_max)
            sign = 1.0 if rng.random() < cfg.jump_up_prob else -1.0
            jump += sign * size
            events.append({"month": t, "size": sign * size})
        path[t] = np.clip(base + jump, cfg.key_rate_floor, cfg.key_rate_cap)
    return path, events


# --------------------------------------------------------------------------
# История ключевой ставки (месячные данные ЦБ)
# --------------------------------------------------------------------------
def load_key_rate_history(path: str | Path | None = None) -> pd.Series:
    """Месячная ключевая ставка (в процентах), индекс — PeriodIndex."""
    p = Path(path) if path else DATA_DIR / "key_rate_monthly.csv"
    if not p.exists():
        raise FileNotFoundError(
            f"Нет файла {p}. Выполните: python scripts/fetch_cbr_data.py")
    df = pd.read_csv(p, parse_dates=["date"])
    s = pd.Series(df["key_rate"].values, index=pd.PeriodIndex(df["date"], freq="M"))
    return s.sort_index()


def historical_path(start: str, n: int, path: str | Path | None = None) -> np.ndarray:
    hist = load_key_rate_history(path)
    idx = pd.period_range(start, periods=n, freq="M")
    missing = [str(i) for i in idx if i not in hist.index or pd.isna(hist.loc[i])]
    if missing:
        raise ValueError(f"нет данных по ключевой ставке за {missing[:3]}…")
    return hist.loc[idx].to_numpy() / 100.0


# --------------------------------------------------------------------------
# Сборка сценария из конфига
# --------------------------------------------------------------------------
def _pulse(n: int, month: int, tau: float) -> np.ndarray:
    t = np.arange(n)
    return np.where(t >= month, np.exp(-np.maximum(t - month, 0) / tau), 0.0)


def build_scenario(cfg: ScenarioConfig, n_months: int,
                   rng: np.random.Generator | None = None) -> Scenario:
    rng = rng if rng is not None else np.random.default_rng(0)
    kind = cfg.kind
    meta: dict = {"kind": kind}
    events: list[dict] = []
    if kind == "constant":
        key = constant_path(cfg.key_rate_start, n_months)
    elif kind == "step":
        key = step_path(cfg.key_rate_start, cfg.shock_size, cfg.shock_month,
                        n_months, cfg.shock_decay_months)
        events = [{"month": cfg.shock_month, "size": cfg.shock_size}]
    elif kind == "ramp":
        key = ramp_path(cfg.key_rate_start, cfg.shock_size, cfg.shock_month,
                        cfg.ramp_months, n_months)
        events = [{"month": cfg.shock_month, "size": cfg.shock_size}]
    elif kind == "path":
        arr = np.asarray(cfg.path, dtype=float)
        if len(arr) < n_months:
            arr = np.concatenate([arr, np.full(n_months - len(arr), arr[-1])])
        key = arr[:n_months]
    elif kind == "random_jumps":
        key, events = random_jump_path(rng, n_months, cfg)
    elif kind == "historical":
        key = historical_path(cfg.historical_start, n_months)
        meta["historical_start"] = cfg.historical_start
    else:
        raise ValueError(f"неизвестный тип сценария {kind}")

    key = np.maximum(key, MIN_KEY_RATE)               # ставка не бывает отрицательной

    # необязательные макрошоки, привязанные к самому крупному скачку вверх
    credit = np.zeros(n_months)
    conf = np.zeros(n_months)
    dep = np.zeros(n_months)
    dem = np.zeros(n_months)
    ups = [e for e in events if e["size"] > 0]
    if ups and any([cfg.credit_stress_peak, cfg.confidence_shock_peak,
                    cfg.deposit_shock_peak, cfg.demand_shock_peak]):
        for e in ups:
            m = int(e["month"])
            scale = min(e["size"] / 0.07, 2.0)
            credit += cfg.credit_stress_peak * scale * _pulse(n_months, m, 6.0)
            conf += cfg.confidence_shock_peak * scale * _pulse(n_months, m, 1.5)
            dep += cfg.deposit_shock_peak * scale * _pulse(n_months, m, 9.0)
            dem += cfg.demand_shock_peak * scale * _pulse(n_months, m, 9.0)
        conf = np.clip(conf, 0.0, 1.0)
    meta["events"] = events
    return Scenario(name=kind, key_rate=key, credit_stress=credit, confidence=conf,
                    deposit_shock=dep, demand_shock=dem, meta=meta)
