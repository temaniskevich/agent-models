"""Калибровка методом симулированных моментов (SMM) и проверка вне выборки (шаг 3 плана).

Схема
-----
1. ``load_cbr_monthly`` — реальные месячные ряды ЦБ (см. ``scripts/fetch_cbr_data.py``).
2. ``compute_moments`` считает одни и те же моменты по данным и по выходу модели:
   рост кредитов/вкладов (среднее, разброс, автокорреляция), уровни ставок и спреды,
   долгосрочный перенос ключевой ставки в ставки банков, доля долга перед ЦБ.
3. ``SMMCalibrator`` подбирает параметры, минимизируя взвешенное расстояние между
   моментами модели и данных (веса — обратные дисперсии из блочного бутстрэпа).
   Зёрна фиксированы между вычислениями (общие случайные числа) → гладкая цель.
4. ``validate_window`` прогоняет модель с откалиброванными параметрами на окне
   проверки, которого не было в калибровке (2014–15 и 2022), и сравнивает с данными.

Два независимых набора окон (как в плане): A — калибровка 2010–2013 / проверка
2014–15; B — калибровка 2017–2021 / проверка 2022. До 2014 г. ставок по кредитам и
вкладам в данных нет, поэтому в окне A используются только моменты объёмов.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .config import ModelConfig, get_by_path, set_param
from .model import BankingModel
from .scenarios import Scenario

DATA_DIR = Path(__file__).resolve().parent / "data"
RATE_COLS = ["loan_rate_corp", "loan_rate_retail", "deposit_rate_corp", "deposit_rate_retail"]
VOLUME_COLS = ["loans_corp", "loans_retail", "deposits_corp", "deposits_retail"]


@dataclass(frozen=True)
class Window:
    name: str
    calib: tuple[str, str]
    test: tuple[str, str]
    description: str = ""
    lp: bool = False                # добавить в цель моменты реакций (локальные проекции)


WINDOWS = {
    "A": Window("A", ("2010-06", "2013-12"), ("2014-01", "2015-12"),
                "Калибровка до 2014 г. (только объёмы) → проверка на кризисе 2014–15"),
    "B": Window("B", ("2017-01", "2021-12"), ("2022-01", "2022-12"),
                "Калибровка 2017–2021 → проверка на 2022 г."),
    "C": Window("C", ("2014-01", "2019-12"), ("2020-01", "2026-09"),
                "Калибровка по реакциям на ключевую ставку 2014–2019 → проверка реакций 2020–2026", lp=True),
}
LP_HORIZONS = (3, 6, 12)


# --------------------------------------------------------------------------
# Данные
# --------------------------------------------------------------------------
def load_cbr_monthly(path: str | Path | None = None, currency: str = "rub") -> pd.DataFrame:
    """Месячные ряды ЦБ. ``currency="rub"`` (по умолчанию): объёмы кредитов и вкладов только в рублях —
    в модели нет валюты, а валютная переоценка (особенно 2014–15 и 2022) искажает измеряемые реакции на ставку.
    ``"all"`` — рубли и валюта вместе. Суммарные ряды при ``"rub"`` сохраняются в колонках ``*_all``."""
    p = Path(path) if path else DATA_DIR / "cbr_monthly.csv"
    if not p.exists():
        raise FileNotFoundError(f"Нет файла {p}. Выполните: python -I scripts/fetch_cbr_data.py")
    df = pd.read_csv(p, parse_dates=["date"])
    df.index = pd.PeriodIndex(df["date"], freq="M", name="date")
    df = df.drop(columns=["date"])
    for c in ("loans_corp", "loans_retail", "deposits_corp", "deposits_retail"):
        df[c + "_all"] = df[c]
        if currency == "rub" and c + "_rub" in df:
            df[c] = df[c + "_rub"]
    df["loans_total"] = df["loans_corp"] + df["loans_retail"]
    df["deposits_total"] = df["deposits_corp"] + df["deposits_retail"]
    df["cb_debt"] = df["cb_debt_stock"]
    return df


def slice_window(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    return df.loc[pd.Period(start, "M"):pd.Period(end, "M")]


# --------------------------------------------------------------------------
# Моменты
# --------------------------------------------------------------------------
def _ac1(x: np.ndarray) -> float:
    x = x[~np.isnan(x)]
    if len(x) < 8 or x.std() < 1e-12:
        return np.nan
    return float(np.corrcoef(x[1:], x[:-1])[0, 1])


def _pass_through(rate: pd.Series, key: pd.Series, lags: int = 6) -> float:
    """Долгосрочный перенос: сумма коэффициентов при ΔключевойСтавки(t..t-lags) в
    регрессии Δставки банков."""
    d = pd.concat([rate.diff(), key.diff()], axis=1, keys=["r", "k"]).dropna()
    if len(d) < lags + 12 or d["k"].std() < 0.05:
        return np.nan
    X = np.column_stack([d["k"].shift(i).to_numpy() for i in range(lags + 1)])
    y = d["r"].to_numpy()
    ok = ~np.isnan(X).any(axis=1)
    if ok.sum() < lags + 10:
        return np.nan
    coef, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
    return float(np.clip(coef.sum(), -1.0, 3.0))


def compute_moments(df: pd.DataFrame) -> dict[str, float]:
    """Моменты по таблице с колонками данных ЦБ / выходом модели (одинаковые имена)."""
    m: dict[str, float] = {}
    for c in VOLUME_COLS:
        if c not in df:
            continue
        g = 1200.0 * np.log(df[c]).diff().to_numpy()
        m[f"{c}.growth_mean"] = float(np.nanmean(g)) if np.isfinite(g).sum() > 6 else np.nan
        m[f"{c}.growth_std"] = float(np.nanstd(g)) if np.isfinite(g).sum() > 6 else np.nan
        m[f"{c}.growth_ac1"] = _ac1(g)
    key = df["key_rate"] if "key_rate" in df else None
    for c in RATE_COLS:
        if c not in df or df[c].notna().sum() < 12:
            m[f"{c}.mean"] = m[f"{c}.spread"] = m[f"{c}.pass_through"] = np.nan
            continue
        m[f"{c}.mean"] = float(df[c].mean())
        if key is not None:
            m[f"{c}.spread"] = float((df[c] - key).mean())
            m[f"{c}.pass_through"] = _pass_through(df[c], key)
    if "cb_debt" in df and "deposits_total" in df:
        m["cb_debt_to_deposits"] = float((df["cb_debt"] / df["deposits_total"]).mean() * 100)
    return m


def lp_moments(df: pd.DataFrame, hs: tuple[int, ...] = LP_HORIZONS) -> tuple[dict[str, float], dict[str, float]]:
    """Моменты реакций: β локальных проекций (на 1 п.п. ключевой ставки) и их дисперсии (SE²)."""
    from .irf import OUTCOMES, local_projection
    mom, var = {}, {}
    for c in OUTCOMES:
        if c not in df or df[c].notna().sum() < 40:
            continue
        lp = local_projection(df.dropna(subset=["key_rate", c]), c, hs=hs).set_index("h")
        for h in hs:
            if np.isfinite(lp.loc[h, "beta"]):
                mom[f"lp.{c}.h{h}"], var[f"lp.{c}.h{h}"] = float(lp.loc[h, "beta"]), float(lp.loc[h, "se"]) ** 2
    return mom, var


def block_bootstrap_variance(df: pd.DataFrame, n_boot: int = 200, block: int = 12,
                             seed: int = 0) -> dict[str, float]:
    """Дисперсия моментов по скользящему блочному бутстрэпу окна данных."""
    rng = np.random.default_rng(seed)
    n = len(df)
    if n < 2 * block:
        block = max(3, n // 4)
    keys = list(compute_moments(df).keys())
    draws = {k: [] for k in keys}
    for _ in range(n_boot):
        starts = rng.integers(0, n - block + 1, size=int(np.ceil(n / block)))
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        mm = compute_moments(df.iloc[idx].reset_index(drop=True))
        for k in keys:
            draws[k].append(mm.get(k, np.nan))
    return {k: float(np.nanvar(v)) if np.isfinite(v).sum() > 10 else np.nan for k, v in draws.items()}


# Моменты, по которым идёт подгонка по умолчанию. Разброс и автокорреляция месячного роста
# в данных ЦБ определяются валютной переоценкой и сезонностью (в модели валюты нет), а доля
# долга перед ЦБ — льготными программами рефинансирования; их выводим в отчёт, но не подгоняем.
DEFAULT_MOMENT_SUFFIXES = (".growth_mean", ".spread", ".pass_through")


def default_moment_filter(name: str) -> bool:
    return name.endswith(DEFAULT_MOMENT_SUFFIXES) or name.startswith("lp.")


# --------------------------------------------------------------------------
# Параметры калибровки
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class CalibParam:
    name: str
    low: float
    high: float


DEFAULT_PARAMS: list[CalibParam] = [
    CalibParam("demand.loan_trend_growth.0", 0.02, 0.25),
    CalibParam("demand.loan_trend_growth.1", 0.02, 0.30),
    CalibParam("demand.deposit_trend_growth.0", 0.02, 0.25),
    CalibParam("demand.deposit_trend_growth.1", 0.02, 0.25),
    CalibParam("banks.loan_spread.0", 0.0, 0.06),
    CalibParam("banks.loan_spread.1", 0.02, 0.14),
    CalibParam("banks.deposit_spread.0", -0.01, 0.08),
    CalibParam("banks.deposit_spread.1", -0.02, 0.06),
    CalibParam("banks.rate_speed_loan", 0.05, 0.80),
    CalibParam("banks.rate_speed_deposit", 0.05, 0.80),
    CalibParam("demand.loan_rate_elasticity.0", 0.5, 6.0),
    CalibParam("demand.deposit_rate_semielasticity.0", 0.2, 2.5),
]


# Параметры, идентифицируемые по данным окна. В окне A (до 2014 г.) нет ставок банков, поэтому
# подгоняются только тренды роста: остальные параметры не идентифицированы и остались бы произвольными.
IRF_PARAMS: list[CalibParam] = DEFAULT_PARAMS + [
    CalibParam("demand.loan_rate_elasticity.1", 0.2, 8.0),
    CalibParam("demand.deposit_rate_semielasticity.1", 0.1, 3.0),
    CalibParam("demand.loan_adjust_speed", 0.03, 0.5),
    CalibParam("demand.deposit_adjust_speed", 0.03, 0.5),
    CalibParam("banks.pass_through_loan.0", 0.2, 1.3),
    CalibParam("banks.pass_through_loan.1", 0.2, 1.3),
    CalibParam("banks.pass_through_deposit.0", 0.2, 1.3),
    CalibParam("banks.pass_through_deposit.1", 0.2, 1.3),
]

VOLUME_ONLY_PARAMS = [p for p in DEFAULT_PARAMS if "_trend_growth" in p.name]


def default_params(window: str) -> list[CalibParam]:
    """Набор параметров по умолчанию для окна калибровки."""
    if window == "C":
        return list(IRF_PARAMS)
    return list(VOLUME_ONLY_PARAMS if window == "A" else DEFAULT_PARAMS)


@dataclass
class CalibrationResult:
    window: str
    theta: dict[str, float]
    objective: float
    history: pd.DataFrame
    moments: pd.DataFrame           # данные vs модель по моментам
    n_evals: int
    notes: str = ""

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({
            "window": self.window, "theta": self.theta, "objective": self.objective,
            "n_evals": self.n_evals, "notes": self.notes,
            "moments": self.moments.reset_index().to_dict(orient="records")},
            ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load_theta(cls, path: str | Path) -> dict[str, float]:
        return json.loads(Path(path).read_text(encoding="utf-8"))["theta"]


class SMMCalibrator:
    """Метод симулированных моментов с общими случайными числами."""

    def __init__(self, base_config: ModelConfig, data: pd.DataFrame, window: Window | str = "B",
                 params: list[CalibParam] | None = None, n_seeds: int = 4, seed0: int = 1000,
                 moment_filter: Callable[[str], bool] | None = default_moment_filter):
        self.base = base_config
        self.window = WINDOWS[window] if isinstance(window, str) else window
        self.params = params or default_params(self.window.name)
        self.n_seeds = n_seeds
        self.seeds = [seed0 + i for i in range(n_seeds)]
        self.data = data
        self.calib_df = slice_window(data, *self.window.calib)
        self.target = compute_moments(self.calib_df)
        var = block_bootstrap_variance(self.calib_df)
        floors = {k: (0.05 * abs(v) + 0.5) ** 2 for k, v in self.target.items() if np.isfinite(v)}
        self.keys = [k for k, v in self.target.items()
                     if np.isfinite(v) and (moment_filter is None or moment_filter(k))]
        self.report_keys = [k for k, v in self.target.items() if np.isfinite(v)]
        self.lp_hs = LP_HORIZONS if self.window.lp else None
        if self.window.lp:                                    # моменты реакций: вес 1/SE² (не менее 0.02)
            lp_m, lp_v = lp_moments(self.calib_df)
            self.target.update(lp_m)
            self.keys += [k for k in lp_m if (moment_filter is None or moment_filter(k))]
            self.report_keys += list(lp_m)
            floors.update({k: 0.02 for k in lp_m})
            var.update(lp_v)
        self.weights = {k: 1.0 / max(var.get(k, np.nan) if np.isfinite(var.get(k, np.nan)) else 0.0,
                                     floors[k]) for k in self.keys}
        self.n_calib = len(self.calib_df)
        self.n_evals = 0
        self.log: list[dict] = []

    # ---- симуляция --------------------------------------------------------
    def _scenario(self, df: pd.DataFrame) -> Scenario:
        return Scenario("historical", key_rate=df["key_rate"].to_numpy() / 100.0,
                        meta={"kind": "historical"})

    def _config(self, theta: dict[str, float], n_months: int) -> ModelConfig:
        cfg = self.base.replace(n_months=n_months, record_banks=False)
        for k, v in theta.items():
            set_param(cfg, k, float(v))
        # стартовый уровень ставки — по данным окна
        cfg.scenario.key_rate_start = float(self.calib_df["key_rate"].iloc[0] / 100.0)
        return cfg

    def simulate_runs(self, theta: dict[str, float], df: pd.DataFrame | None = None,
                      pre: pd.DataFrame | None = None) -> list[pd.DataFrame]:
        """Прогоны на путях ставки окна ``df`` (при ``pre`` — с «разогревом» реальной историей)."""
        df = self.calib_df if df is None else df
        full = df if pre is None else pd.concat([pre, df])
        scn = self._scenario(full)
        cfg = self._config(theta, len(full))
        cfg.scenario.key_rate_start = float(full["key_rate"].iloc[0] / 100.0)
        outs = []
        for s in self.seeds:
            res = BankingModel(cfg, scenario=scn, seed=s).run()
            outs.append(res.sector)
        return outs

    def simulate_moments(self, theta: dict[str, float]) -> dict[str, float]:
        runs = self.simulate_runs(theta)
        acc: dict[str, list[float]] = {}
        for r in runs:
            mm = compute_moments(r)
            if self.window.lp:
                mm.update(lp_moments(r)[0])
            for k, v in mm.items():
                acc.setdefault(k, []).append(v)
        return {k: float(np.nanmean(v)) if np.isfinite(v).any() else np.nan for k, v in acc.items()}

    # ---- цель -------------------------------------------------------------
    def _theta(self, x: np.ndarray) -> dict[str, float]:
        return {p.name: float(np.clip(v, p.low, p.high)) for p, v in zip(self.params, x)}

    def objective(self, x: np.ndarray) -> float:
        theta = self._theta(x)
        try:
            sim = self.simulate_moments(theta)
        except Exception:                                    # неудачные параметры → большой штраф
            return 1e9
        q = 0.0
        for k in self.keys:
            v = sim.get(k, np.nan)
            if not np.isfinite(v):
                q += 1e3
                continue
            q += self.weights[k] * (v - self.target[k]) ** 2
        self.n_evals += 1
        self.log.append({"eval": self.n_evals, "objective": q, **theta})
        return float(q)

    def moments_table(self, theta: dict[str, float]) -> pd.DataFrame:
        sim = self.simulate_moments(theta)
        rows = [{"moment": k, "data": self.target[k], "model": sim.get(k, np.nan),
                 "weight": self.weights.get(k, np.nan), "in_objective": k in self.keys}
                for k in self.report_keys]
        df = pd.DataFrame(rows).set_index("moment")
        df["z"] = (df["model"] - df["data"]) * np.sqrt(df["weight"])
        return df

    # ---- оптимизация ------------------------------------------------------
    def fit(self, method: str = "lhs_nm", budget: int = 120, seed: int = 0,
            x0: dict[str, float] | None = None,
            callback: Callable[[int, float], None] | None = None) -> CalibrationResult:
        """``method``: ``lhs_nm`` (случайный поиск + Нелдер–Мид), ``de`` (дифф. эволюция),
        ``optuna`` (байесовская оптимизация TPE; нужен пакет optuna)."""
        from scipy import optimize
        lo = np.array([p.low for p in self.params])
        hi = np.array([p.high for p in self.params])
        rng = np.random.default_rng(seed)
        self.n_evals, self.log = 0, []

        def wrapped(x):
            q = self.objective(x)
            if callback:
                callback(self.n_evals, q)
            return q

        if method == "de":
            r = optimize.differential_evolution(
                wrapped, bounds=list(zip(lo, hi)), maxiter=max(2, budget // (len(lo) * 5)), popsize=5,
                seed=seed, tol=1e-3, polish=False, init="sobol")
            best = r.x
        elif method == "optuna":
            try:
                import optuna
            except ImportError as e:
                raise ImportError("для method='optuna' установите: pip install optuna") from e
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            study = optuna.create_study(direction="minimize",
                                        sampler=optuna.samplers.TPESampler(seed=seed))
            if x0:
                study.enqueue_trial({p.name: x0[p.name] for p in self.params if p.name in x0})

            def obj(trial):
                x = np.array([trial.suggest_float(p.name, p.low, p.high) for p in self.params])
                return wrapped(x)
            study.optimize(obj, n_trials=budget)
            best = np.array([study.best_params[p.name] for p in self.params])
        else:                                                     # lhs_nm
            n0 = max(len(lo) * 4, budget // 2)
            u = (np.argsort(rng.random((n0, len(lo))), axis=0) + rng.random((n0, len(lo)))) / n0
            pts = lo + u * (hi - lo)
            if x0:
                pts[0] = [x0.get(p.name, (p.low + p.high) / 2) for p in self.params]
            vals = np.array([wrapped(x) for x in pts])
            best, best_q = pts[np.argmin(vals)], vals.min()
            remaining = max(budget - n0, 20)
            r = optimize.minimize(lambda z: wrapped(np.clip(z, lo, hi)), best, method="Nelder-Mead",
                                  options={"maxfev": remaining, "xatol": 1e-4, "fatol": 1e-6,
                                           "initial_simplex": _simplex(best, lo, hi)})
            best = np.clip(r.x, lo, hi) if r.fun < best_q else best
        theta = self._theta(best)
        q = self.objective(best)
        return CalibrationResult(
            window=self.window.name, theta=theta, objective=q, history=pd.DataFrame(self.log),
            moments=self.moments_table(theta), n_evals=self.n_evals,
            notes=f"метод {method}, бюджет {budget}, зёрен {self.n_seeds}, окно {self.window.calib}")

    # ---- проверка вне выборки ----------------------------------------------
    def validate(self, theta: dict[str, float], pre_months: int = 24) -> "ValidationReport":
        test = slice_window(self.data, *self.window.test)
        start = test.index[0]
        pre = self.data.loc[start - pre_months:start - 1]
        runs = self.simulate_runs(theta, df=test, pre=pre)
        runs = [r.iloc[pre_months:].copy() for r in runs]
        return ValidationReport(self.window.name, test, runs, theta)


def _simplex(x: np.ndarray, lo: np.ndarray, hi: np.ndarray, frac: float = 0.08) -> np.ndarray:
    pts = [x]
    for i in range(len(x)):
        y = x.copy()
        step = frac * (hi[i] - lo[i])
        y[i] = x[i] + step if x[i] + step <= hi[i] else x[i] - step
        pts.append(y)
    return np.array(pts)


@dataclass
class ValidationReport:
    window: str
    actual: pd.DataFrame
    runs: list[pd.DataFrame]
    theta: dict[str, float]

    def paths(self, column: str) -> pd.DataFrame:
        """Факт и распределение модели по прогонам для графика."""
        sim = pd.concat([r[column].reset_index(drop=True) for r in self.runs], axis=1)
        out = pd.DataFrame({"actual": self.actual[column].to_numpy()}, index=self.actual.index)
        out["model_mean"] = sim.mean(axis=1).to_numpy()
        out["model_p05"] = sim.quantile(0.05, axis=1).to_numpy()
        out["model_p95"] = sim.quantile(0.95, axis=1).to_numpy()
        return out

    def summary(self) -> pd.DataFrame:
        """Ошибки вне выборки: ставки (п.п.), рост объёмов (п.п. годовых), корреляции путей."""
        rows = []
        for c in RATE_COLS + VOLUME_COLS:
            if c not in self.actual or self.actual[c].isna().all():
                continue
            if c in VOLUME_COLS:
                act = 1200 * np.log(self.actual[c]).diff()
                sim = pd.concat([1200 * np.log(r[c]).diff().reset_index(drop=True) for r in self.runs], axis=1).mean(axis=1)
            else:
                act = self.actual[c]
                sim = pd.concat([r[c].reset_index(drop=True) for r in self.runs], axis=1).mean(axis=1)
            a, s = act.to_numpy(), sim.to_numpy()
            ok = np.isfinite(a) & np.isfinite(s)
            if ok.sum() < 4:
                continue
            rows.append({"series": c, "rmse": float(np.sqrt(np.mean((a[ok] - s[ok]) ** 2))),
                         "bias": float(np.mean(s[ok] - a[ok])),
                         "corr": float(np.corrcoef(a[ok], s[ok])[0, 1]) if a[ok].std() > 0 and s[ok].std() > 0 else np.nan})
        return pd.DataFrame(rows).set_index("series")
