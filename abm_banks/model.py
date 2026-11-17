"""Агентная модель банковского сектора России (месячный шаг, векторизованная).

Агенты
------
* **Банки** (неоднородные: размер, цель по капиталу, буфер ликвидности, издержки,
  качество заёмщиков, терпимость к риску) с полным балансом
  «активы = кредиты (орг., нас.) + ценные бумаги + ликвидные средства +
  обязательные резервы + МБК выданный + депозиты в ЦБ;
  пассивы = вклады (орг., нас.) + МБК полученный + долг перед ЦБ + капитал».
* **Клиенты** (вкладчики и заёмщики) — агрегированные популяции, которые
  выбирают банк по ставке, размеру и надёжности (логит), реагируют на ставки
  объёмом спроса и бегут из слабых банков при панике.
* **Банк России** — задаёт ключевую ставку (путь на входе), коридор ставок,
  норматив резервирования, рефинансирование под залог, приём депозитов,
  отзыв лицензий и спасение системно значимых банков.
* **АСВ/ликвидатор** — страхует вклады и несёт потери при выбытии банков.

Порядок месяца (см. ``BankingModel._step``)
-------------------------------------------
погашение МБК/ЦБ → ставки → проценты → дефолты и амортизация → переоценка бумаг →
новые кредиты (ограничены капиталом) → вклады (бегство, выбор банка, спрос) →
резервы, издержки, налоги, дивиденды → ликвидность (МБК → ЦБ → продажа бумаг,
избыток → бумаги/депозит в ЦБ) → проверка платёжеспособности, выход банков,
каскады, вход новых → запись показателей.

Бухгалтерское тождество ``активы − обязательства = капитал`` выполняется для
каждого банка на каждом шаге (``check_identities``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
import pandas as pd

from .config import TOP20_SHARES, ModelConfig
from .interbank import match_interbank, network_edges, sigmoid
from .policies import BehaviorPolicy, PolicyAdjustments, get_policy
from .scenarios import Scenario, build_scenario

C, R = 0, 1          # индексы сегментов: организации, население


class SectorCollapse(RuntimeError):
    """Все банки выбыли: дальнейший расчёт невозможен (это допустимый исход при экстремальных параметрах)."""
EPS = 1e-9


# --------------------------------------------------------------------------
def size_shares(cfg, rng: np.random.Generator) -> np.ndarray:
    """Стартовое распределение размеров банков (доли, сумма = 1)."""
    n = cfg.n_banks
    kind = cfg.size_distribution
    if kind == "uniform":
        s = np.ones(n)
    elif kind == "zipf":
        s = 1.0 / np.arange(1, n + 1) ** cfg.zipf_exponent
    elif kind == "lognormal":
        s = np.sort(rng.lognormal(0.0, cfg.lognormal_sigma, n))[::-1]
    elif kind == "top20":
        top = np.array(TOP20_SHARES)
        if n == 20:
            s = top
        elif n < 20:
            s = top[:n]
        else:
            tail = 1.0 / np.arange(1, n - 20 + 1) ** 1.0
            tail = tail / tail.sum() * cfg.top20_tail_share
            s = np.concatenate([top * (1 - cfg.top20_tail_share), tail])
    else:
        raise ValueError(kind)
    return s / s.sum()


@dataclass
class SimulationResult:
    """Итог прогона: таблицы показателей + метаданные для воспроизводимости."""
    sector: pd.DataFrame
    banks: pd.DataFrame | None
    events: pd.DataFrame
    config: ModelConfig
    scenario: Scenario
    seed: int
    network: pd.DataFrame | None = None
    meta: dict = field(default_factory=dict)


class BankingModel:
    """Агентная модель. ``BankingModel(cfg, seed=1).run()`` → ``SimulationResult``."""

    def __init__(self, config: ModelConfig, scenario: Scenario | None = None,
                 seed: int | None = None, policy: BehaviorPolicy | str | None = None):
        config.validate()
        self.cfg = config
        self.seed = int(config.seed if seed is None else seed)
        children = np.random.SeedSequence(self.seed).spawn(6)
        (self.rng_init, self.rng_scen, self.rng_demand,
         self.rng_risk, self.rng_mbk, self.rng_entry) = (np.random.default_rng(c) for c in children)

        if scenario is None:
            scenario = build_scenario(config.scenario, config.n_months, self.rng_scen)
        if len(scenario) != config.n_months:
            raise ValueError(f"длина сценария {len(scenario)} ≠ n_months {config.n_months}")
        self.scenario = scenario
        self._scn = scenario.padded(config.burn_in)
        self.total_steps = config.burn_in + config.n_months
        self.policy = get_policy(policy if policy is not None else config.policy)

        self.t = 0                       # глобальный шаг (с учётом прогрева)
        self._init_banks()
        self._records: list[dict] = []
        self._bank_records: list[dict] = []
        self._events: list[dict] = []
        self._network_records: list[tuple] = []
        self.interventions: list[dict] = []
        self.last_network = np.zeros((self.n, self.n))
        self.last_adjustments = PolicyAdjustments.zeros(self.n)

    # ======================================================================
    # Инициализация
    # ======================================================================
    @property
    def n(self) -> int:
        return self.cfg.sector.n_banks

    def _draw_heterogeneity(self, rng: np.random.Generator, size: int) -> dict[str, np.ndarray]:
        b, ib = self.cfg.banks, self.cfg.interbank
        return {
            "car_target": np.clip(rng.normal(b.car_target_mean, b.car_target_sd, size),
                                  b.car_min + 0.015, 0.22),
            "liq_ratio": np.clip(rng.normal(b.liq_target_mean, b.liq_target_sd, size), 0.02, 0.15),
            "cost_ratio": np.clip(rng.normal(b.cost_ratio_mean, b.cost_ratio_sd, size), 0.008, 0.06),
            "quality": np.exp(rng.normal(0.0, b.quality_sigma, size)),
            "risk_aversion": np.clip(rng.normal(ib.risk_aversion_mean, ib.risk_aversion_sd, size),
                                     0.1, 0.9),
            "speed_loan": np.clip(rng.normal(b.rate_speed_loan, b.rate_speed_sd, size), 0.05, 0.9),
            "speed_dep": np.clip(rng.normal(b.rate_speed_deposit, b.rate_speed_sd, size), 0.05, 0.9),
            "appetite": np.exp(rng.normal(0.0, b.appetite_sd, size)),
            "appeal": np.exp(rng.normal(0.0, b.deposit_appeal_sd, size)),
        }

    def _balance_sheet(self, assets, loans_c, loans_r, car0, liq_ratio, retail_share,
                       mbk_b=0.0, mbk_l=0.0):
        """Решает стартовый баланс при заданных кредитах, МБК и цели по CAR.

        Если при заданных кредитах не хватает места для бумаг (или вклады < 0), кредиты
        банка уменьшаются на 10 % до выполнения условия. Возвращает также итоговые кредиты."""
        b, cb = self.cfg.banks, self.cfg.cb
        rho = cb.reserve_ratio
        a = rho + liq_ratio
        ws, wi = b.rw_securities, b.rw_interbank
        loans_c, loans_r = np.array(loans_c, float), np.array(loans_r, float)
        for _ in range(60):
            w_l = b.rw_corp * loans_c + b.rw_retail * loans_r
            base = (w_l + ws * (assets - loans_c - loans_r - mbk_l - a * (assets - mbk_b)) + wi * mbk_l)
            capital = car0 * base / (1 - car0 * ws * a)
            dep = assets - capital - mbk_b
            reserves = rho * dep
            liq = liq_ratio * dep
            sec = assets - loans_c - loans_r - reserves - liq - mbk_l
            bad = (sec < 0.01 * assets) | (dep <= 0.2 * assets) | (capital <= 0)
            if not np.any(bad):
                d_ret = retail_share * dep
                return (capital, np.stack([dep - d_ret, d_ret], axis=1), reserves, liq, sec,
                        loans_c, loans_r)
            loans_c = np.where(bad, loans_c * 0.9, loans_c)
            loans_r = np.where(bad, loans_r * 0.9, loans_r)
        raise ValueError("не удалось построить стартовый баланс: проверьте резервы, ликвидность и МБК")

    def _seed_interbank(self, rng, assets):
        """Стартовые роли МБК и сеть: часть банков — структурные заёмщики, часть — кредиторы."""
        ib, n = self.cfg.interbank, self.cfg.sector.n_banks
        u = rng.random(n)
        role = np.where(u < ib.borrower_fraction, 1,
                        np.where(u < ib.borrower_fraction + ib.lender_fraction, -1, 0))
        if ib.structural_share <= 0 or (role == 1).sum() == 0 or (role == -1).sum() == 0:
            role[:] = 0
        b_share = np.where(role == 1, rng.uniform(0.5, 1.5, n) * ib.structural_share, 0.0)
        l_share = np.where(role == -1, rng.uniform(0.5, 1.5, n) * ib.structural_share, 0.0)
        mb, ml = b_share * assets, l_share * assets
        if mb.sum() > 0 and ml.sum() > 0:
            total = min(mb.sum(), ml.sum())          # сеть замкнута: выдано = получено; доли не раздуваем
            mb, ml = mb * total / mb.sum(), ml * total / ml.sum()
            b_share, l_share = mb / assets, ml / assets
        # матрица связей: метод итеративной пропорциональной подгонки (RAS)
        M = np.outer(ml, mb) * np.exp(0.5 * rng.standard_normal((n, n)))
        np.fill_diagonal(M, 0.0)
        for _ in range(50):
            rs = M.sum(axis=1)
            M *= np.where(rs > EPS, ml / np.maximum(rs, EPS), 0.0)[:, None]
            cs = M.sum(axis=0)
            M *= np.where(cs > EPS, mb / np.maximum(cs, EPS), 0.0)[None, :]
        return role, b_share, l_share, mb, M.sum(axis=1), M

    def _init_banks(self) -> None:
        cfg, s = self.cfg, self.cfg.sector
        rng, n = self.rng_init, s.n_banks
        shares = size_shares(s, rng)
        self.assets0 = shares * s.total_assets_bn
        het = self._draw_heterogeneity(rng, n)
        self.car_target, self.liq_ratio = het["car_target"], het["liq_ratio"]
        self.cost_ratio, self.quality = het["cost_ratio"], het["quality"]
        self.risk_aversion, self.speed_loan = het["risk_aversion"], het["speed_loan"]
        self.speed_dep, self.appetite = het["speed_dep"], het["appetite"]
        self.appeal = het["appeal"]

        self.mbk_role, self.mbk_b_share, self.mbk_l_share, mb, ml, M0 = self._seed_interbank(rng, self.assets0)
        noise = lambda: np.exp(rng.normal(0.0, s.composition_noise, n))    # noqa: E731
        loans_c = np.clip(s.init_loans_corp * noise(), 0.05, 0.70) * self.assets0
        loans_r = np.clip(s.init_loans_retail * noise(), 0.02, 0.50) * self.assets0
        # доля кредитов ограничена, чтобы оставались бумаги и ликвидность
        cap = np.maximum(np.minimum(0.78 * self.assets0 - ml,
                                    self.assets0 * (0.97 - cfg.cb.reserve_ratio - self.liq_ratio) - ml),
                         0.1 * self.assets0)
        scale = np.minimum(1.0, cap / (loans_c + loans_r))
        loans_c, loans_r = loans_c * scale, loans_r * scale
        retail_share = np.clip(s.init_retail_deposit_share * noise(), 0.1, 0.9)
        car0 = self.car_target + np.abs(rng.normal(s.init_car_excess_mean,
                                                   s.init_car_excess_sd, n))
        K, D, res, liq, sec, loans_c, loans_r = self._balance_sheet(
            self.assets0, loans_c, loans_r, car0, self.liq_ratio, retail_share, mb, ml)

        self.active = np.ones(n, dtype=bool)
        self.bank_id = np.arange(n)
        self.next_bank_id = n
        self.age = np.zeros(n, dtype=int)
        self.free_since = np.full(n, -1)
        self.K, self.D, self.res, self.liq, self.sec = K, D, res, liq, sec
        self.L = np.stack([loans_c, loans_r], axis=1)
        self.cb_dep = np.zeros(n)
        self.cb_debt = np.zeros(n)
        self.M = M0                          # M[i, j]: банк i одолжил банку j
        self.I = M0 * (self._scn.key_rate[0] + cfg.interbank.spread_base + 0.5 * cfg.interbank.spread_risk) / 12.0
        self.ra_claims = np.zeros(n)         # долги заёмщиков перед ликвидатором (после выбытия кредитора)
        self.ra_interest = np.zeros(n)
        self.sec_target_ratio = sec / self.assets0

        self.stress = np.zeros(n)
        self.pressure = np.zeros(n)
        self.panic_endo = 0.0
        self.pending_dep = np.zeros(2)       # вклады выбывших банков, ждущие перераспределения
        self.recent_failure_assets = 0.0
        self.recent_fail_counts: list[int] = []
        self.cb_use_prev = np.zeros(n, dtype=bool)
        self.npl_prev = np.zeros(n)

        self.trend_loans = self.L.sum(axis=0)
        self.trend_dep = self.D.sum(axis=0)
        self.trend_loans0 = self.trend_loans.copy()
        self.prev_links = np.zeros((n, n), dtype=bool)
        self.pd_ema = np.array([self.cfg.credit.pd_monthly]) * self.quality[:, None]

        k0 = self._scn.key_rate[0]
        self.k_anchor = k0                   # нейтральный уровень, от которого считается неполный перенос
        self.prev_k = k0
        self.last_refi = k0 + cfg.cb.refi_spread
        self.last_cbd = max(k0 - cfg.cb.deposit_spread, 0.0)
        self.prev_y_sec = max(k0 - cfg.banks.securities_yield_spread, 0.001)

        # стартовые ставки = цели при ключевой k0; «нейтральные» уровни фиксируем здесь
        self.rate_loan = np.zeros((n, 2))
        self.rate_dep = np.zeros((n, 2))
        self.reliability = self._reliability()
        self._set_rates(k0, PolicyAdjustments.zeros(n), speed=1.0)
        self.vint_loan = self.rate_loan.copy()
        self.vint_dep = self.rate_dep.copy()
        self.ref_loan_rate = self._wavg(self.rate_loan, self.L)
        self.ref_dep_rate = self._wavg(self.rate_dep, self.D)
        self.car_now = self._car()
        self.total_assets_start = float(self._assets().sum())

    # ======================================================================
    # Вспомогательные расчёты
    # ======================================================================
    def _wavg(self, x: np.ndarray, w: np.ndarray) -> np.ndarray:
        """Средневзвешенное по банкам для каждого сегмента (учёт только активных)."""
        w = w * self.active[:, None]
        tot = w.sum(axis=0)
        return np.where(tot > EPS, (x * w).sum(axis=0) / np.maximum(tot, EPS), x[self.active].mean(axis=0))

    def _mbk_lent(self) -> np.ndarray:
        return self.M.sum(axis=1)

    def _mbk_borrowed(self) -> np.ndarray:
        return self.M.sum(axis=0) + self.ra_claims

    def _assets(self) -> np.ndarray:
        return (self.liq + self.res + self.sec + self.L.sum(axis=1)
                + self._mbk_lent() + self.cb_dep)

    def _liabilities(self) -> np.ndarray:
        return self.D.sum(axis=1) + self._mbk_borrowed() + self.cb_debt

    def _rwa(self) -> np.ndarray:
        b = self.cfg.banks
        return (b.rw_corp * self.L[:, C] + b.rw_retail * self.L[:, R]
                + b.rw_securities * self.sec + b.rw_interbank * self._mbk_lent())

    def _car(self) -> np.ndarray:
        rwa = self._rwa()
        return np.where(rwa > EPS, self.K / np.maximum(rwa, EPS), 1.0)

    def _reliability(self) -> np.ndarray:
        """Надёжность банка в глазах клиентов и контрагентов (0..1): капитал, размер, стресс."""
        b = self.cfg.banks
        assets = self._assets()
        act = self.active & (assets > EPS)
        med = np.median(assets[act]) if act.any() else 1.0
        car = self._car()
        z = ((car - b.reliability_car_ref) / b.reliability_scale
             + b.reliability_size_weight * np.log(np.maximum(assets, EPS) / med)
             - 1.5 * self.stress)
        return np.where(self.active, sigmoid(z), 0.0)

    def _environment(self, t: int) -> dict:
        cb, scn = self.cfg.cb, self._scn
        k = float(scn.key_rate[t])
        dk = k - self.prev_k
        c = self.cfg.confidence
        self.panic_endo = (c.panic_decay * self.panic_endo
                           + c.rate_jump_sensitivity * max(0.0, dk - c.rate_jump_threshold)
                           + c.failure_contagion * self.recent_failure_assets)
        self.recent_failure_assets = 0.0
        panic = float(np.clip(max(scn.confidence[t], self.panic_endo), 0.0, 1.0))
        return {
            "t": t, "k": k, "dk": dk,
            "refi": k + cb.refi_spread,
            "cbd": max(k - cb.deposit_spread, 0.0),
            "y_sec": max(k - self.cfg.banks.securities_yield_spread, 0.001),
            "credit_stress": float(scn.credit_stress[t]),
            "deposit_shock": float(scn.deposit_shock[t]),
            "demand_shock": float(scn.demand_shock[t]),
            "panic": panic,
        }

    # ======================================================================
    # Ставки
    # ======================================================================
    def _set_rates(self, k: float, adj: PolicyAdjustments, speed: float | None = None) -> None:
        b, cr = self.cfg.banks, self.cfg.credit
        trust, car = self.reliability, self._car()
        car_tgt = self.car_target + adj.car_target
        scarcity = np.maximum(car_tgt + 0.01 - car, 0.0)
        for s in (C, R):
            k_l = self.k_anchor + b.pass_through_loan[s] * (k - self.k_anchor)
            k_d = self.k_anchor + b.pass_through_deposit[s] * (k - self.k_anchor)
            prem = np.minimum(b.risk_pricing * 12.0 * self.pd_ema[:, s] * cr.lgd[s], b.risk_premium_cap)
            tgt_l = (k_l + b.loan_spread[s] + prem + b.market_power_loan * (trust - 0.5)
                     + b.liq_pressure_loan * np.maximum(self.pressure, 0.0)
                     + b.capital_scarcity_loan * scarcity + adj.loan_spread)
            tgt_d = (k_d - b.deposit_spread[s] - b.market_power_deposit * (trust - 0.5)
                     + b.liq_pressure_deposit * self.pressure
                     + b.stress_deposit_premium * self.stress + adj.deposit_spread)
            tgt_l = np.clip(tgt_l, k * 0.5, b.loan_rate_cap)
            tgt_d = np.clip(tgt_d, b.deposit_rate_floor, tgt_l - 0.005)
            sl = self.speed_loan if speed is None else speed
            sd = self.speed_dep if speed is None else speed
            new_l = self.rate_loan[:, s] + sl * (tgt_l - self.rate_loan[:, s])
            new_d = self.rate_dep[:, s] + sd * (tgt_d - self.rate_dep[:, s])
            self.rate_loan[:, s] = np.where(self.active, new_l, 0.0)
            self.rate_dep[:, s] = np.where(self.active, new_d, 0.0)

    def _yield_loans(self) -> np.ndarray:
        phi = np.array(self.cfg.banks.float_share_loans)
        return (1 - phi) * self.vint_loan + phi * self.rate_loan

    def _cost_deposits(self) -> np.ndarray:
        phi = np.array(self.cfg.banks.float_share_deposits)
        return (1 - phi) * self.vint_dep + phi * self.rate_dep

    # ======================================================================
    # Шаг модели
    # ======================================================================
    def _snapshot(self, env: dict) -> dict:
        assets = self._assets()
        d = np.maximum(self.D.sum(axis=1), EPS)
        return {
            "n_banks": self.n, "month": env["t"] - self.cfg.burn_in, "key_rate": env["k"],
            "active": self.active.copy(), "car": self._car(), "car_target": self.car_target.copy(),
            "car_min": self.cfg.banks.car_min, "liq_ratio": self.liq / d,
            "liq_target": self.liq_ratio.copy(), "cb_use": self.cb_use_prev.copy(),
            "npl": self.npl_prev.copy(), "assets": assets,
            "recent_failures": int(sum(self.recent_fail_counts[-3:])),
        }

    def step(self) -> dict | None:
        """Один месяц. Возвращает запись сектора (None на прогреве)."""
        if self.t >= self.total_steps:
            raise StopIteration("симуляция завершена")
        rec = self._step(self.t)
        self.t += 1
        return rec

    def _step(self, t: int) -> dict | None:
        cfg = self.cfg
        if not self.active.any():
            raise SectorCollapse(f"все банки выбыли к месяцу {t - cfg.burn_in}")
        env = self._environment(t)
        k = env["k"]
        flows: dict[str, float] = {}
        n = self.n

        adj = self.policy.decide(self._snapshot(env)).clipped()
        self.last_adjustments = adj
        k_start = self.K.copy()
        sys_assets_start = float(self._assets().sum())
        L_start = self.L.sum(axis=0)
        D_start = self.D.sum(axis=0)
        self._L_before = L_start.copy()

        # 1. погашение МБК / ЦБ, проценты по ним
        self._mature_positions(flows)
        # 2. ставки месяца
        self.reliability = self._reliability()
        self._set_rates(k, adj)
        # 3. проценты по кредитам, вкладам, бумагам
        self._accrue_interest(env, flows)
        # 4. дефолты, амортизация, переоценка бумаг
        npl = self._credit_losses(env, flows)
        self._securities_mtm(env, flows)
        # 5. новые кредиты
        new_loans = self._lending(env, flows, adj)
        # 6. вклады
        self._deposits(env, flows)
        # 7. резервы, издержки, налоги, дивиденды
        self._costs_tax_dividends(k_start, flows)
        # 8. ликвидность: МБК, ЦБ
        self._liquidity(env, flows, adj)
        # 9. платёжеспособность, выход, вход
        n_fail = self._solvency_and_exit(env, flows)
        self._entry(env, flows)
        # 10. итоговые состояния и запись
        self._update_states(env, new_loans, npl)

        self.prev_k = k
        self.prev_y_sec = env["y_sec"]
        self.last_refi, self.last_cbd = env["refi"], env["cbd"]
        self.age[self.active] += 1
        if cfg.check_identities:
            err = self.identity_error()
            if err > 1e-6 * max(1.0, sys_assets_start):
                raise AssertionError(f"нарушено балансовое тождество на шаге {t}: {err:.3e}")

        if t < cfg.burn_in:
            return None
        return self._record(env, flows, new_loans, npl, n_fail, L_start, D_start)

    # ---- 1. погашение позиций ------------------------------------------
    def _mature_positions(self, flows: dict) -> None:
        # депозиты в ЦБ и долг перед ЦБ (договоры на месяц; ставки прошлого месяца)
        gain = self.cb_dep * self.last_cbd / 12.0
        cost = self.cb_debt * self.last_refi / 12.0
        self.liq += self.cb_dep + gain - self.cb_debt - cost
        self.K += gain - cost
        flows["int_cb"] = float((gain - cost).sum())
        self.cb_dep = np.zeros(self.n)
        self.cb_debt = np.zeros(self.n)
        # МБК
        inc = self.I.sum(axis=1)
        exp = self.I.sum(axis=0) + self.ra_interest
        self.liq += self.M.sum(axis=1) + inc - self._mbk_borrowed() - exp
        self.K += inc - exp
        flows["int_mbk"] = float((inc - exp).sum())
        self.prev_links = self.M > 0
        self.M = np.zeros((self.n, self.n))
        self.I = np.zeros((self.n, self.n))
        self.ra_claims = np.zeros(self.n)
        self.ra_interest = np.zeros(self.n)

    # ---- 3. проценты ---------------------------------------------------
    def _accrue_interest(self, env: dict, flows: dict) -> None:
        inc_loans = (self.L * self._yield_loans()).sum(axis=1) / 12.0
        inc_sec = self.sec * env["y_sec"] / 12.0
        exp_dep = (self.D * self._cost_deposits()).sum(axis=1) / 12.0
        net = inc_loans + inc_sec - exp_dep
        self.liq += net
        self.K += net
        flows["int_loans"] = float(inc_loans.sum())
        flows["int_sec"] = float(inc_sec.sum())
        flows["int_dep"] = float(exp_dep.sum())

    # ---- 4. кредитный риск ----------------------------------------------
    def _credit_losses(self, env: dict, flows: dict) -> np.ndarray:
        cr, dm = self.cfg.credit, self.cfg.demand
        yl = self._yield_loans()
        ref_y = self.ref_loan_rate
        loss_total = 0.0
        default_total = 0.0
        amort_total = 0.0
        npl = np.zeros(self.n)
        for s in (C, R):
            noise = np.exp(cr.noise_sd * self.rng_risk.standard_normal(self.n) - 0.5 * cr.noise_sd ** 2)
            rate_mult = np.minimum(np.exp(cr.rate_sensitivity[s] * (yl[:, s] - ref_y[s])), cr.rate_multiplier_cap)
            d = (cr.pd_monthly[s] * self.quality * rate_mult
                 * (1.0 + env["credit_stress"]) * noise)
            d = np.where(self.active, np.clip(d, 0.0, cr.max_pd), 0.0)
            default = d * self.L[:, s]
            loss = cr.lgd[s] * default
            recovered = default - loss
            amort = (self.L[:, s] - default) / dm.loan_maturity_months[s]
            self.L[:, s] -= default + amort
            self.liq += recovered + amort
            self.K -= loss
            self.pd_ema[:, s] = (1 - cr.pd_smoothing) * self.pd_ema[:, s] + cr.pd_smoothing * d
            loss_total += loss.sum()
            default_total += default.sum()
            amort_total += amort.sum()
            npl += default
        # крупные идиосинкратические убытки (мошенничество, концентрация): списание без возмещения
        hit = self.active & (self.rng_risk.random(self.n) < cr.idio_event_prob)
        if hit.any():
            frac = np.clip(cr.idio_loss_median * np.exp(cr.idio_loss_sigma * self.rng_risk.standard_normal(self.n)),
                           0.0, 0.6)
            frac = np.where(hit, frac, 0.0)
            for s in (C, R):
                w_off = frac * self.L[:, s]
                self.L[:, s] -= w_off
                self.K -= w_off
                loss_total += w_off.sum()
                npl += w_off
        flows["loan_losses"] = float(loss_total)
        flows["defaults"] = float(default_total)
        flows["repayments"] = float(amort_total)
        base = np.maximum(self.L.sum(axis=1) + npl, EPS)
        return npl / base

    def _securities_mtm(self, env: dict, flows: dict) -> None:
        b = self.cfg.banks
        dy = env["y_sec"] - self.prev_y_sec
        pct = np.clip(b.securities_mtm_share * b.securities_duration * dy, -0.25, 0.25)
        loss = self.sec * pct
        self.sec -= loss
        self.K -= loss
        flows["mtm"] = float(loss.sum())

    # ---- 5. новые кредиты -------------------------------------------------
    def _lending(self, env: dict, flows: dict, adj: PolicyAdjustments) -> np.ndarray:
        cfg, dm, b = self.cfg, self.cfg.demand, self.cfg.banks
        n = self.n
        rw = np.array([b.rw_corp, b.rw_retail])
        L_open = self.L.copy()
        L_sys_after = self.L.sum(axis=0)
        # желаемый запас кредитов: тренд × реакция на ставку (частичная подстройка)
        r_bar = self._wavg(self.rate_loan, np.maximum(L_open, 1e-6))
        want = np.zeros(2)
        for s in (C, R):
            x_star = (-dm.loan_rate_elasticity[s] * (r_bar[s] - self.ref_loan_rate[s])
                      + env["demand_shock"] + dm.demand_noise_sd * self.rng_demand.standard_normal())
            prev_trend = self.trend_loans[s]
            self.trend_loans[s] = prev_trend * np.exp(dm.loan_trend_growth[s] / 12.0)
            # отклонение фактического запаса (с учётом списаний и амортизации) от тренда
            x_prev = np.log(max(self._L_before[s], EPS)) - np.log(prev_trend)
            x_new = x_prev + dm.loan_adjust_speed * (x_star - x_prev)
            target = self.trend_loans[s] * np.exp(x_new)
            want[s] = max(0.0, target - L_sys_after[s])

        # распределение спроса по банкам
        floor = 0.05 * np.maximum(L_open.mean(axis=0), 1e-6)
        pressure_mult = np.clip(1.0 - dm.liq_rationing * np.maximum(self.pressure, 0.0) / 0.05, 0.2, 1.0)
        weights = np.zeros((n, 2))
        for s in (C, R):
            w = ((L_open[:, s] + floor[s]) ** dm.loan_size_bias
                 * np.exp(-dm.loan_rate_sensitivity * 100.0 * (self.rate_loan[:, s] - r_bar[s]))
                 * self.appetite)
            w = np.where(self.active, w, 0.0)
            weights[:, s] = w / max(w.sum(), EPS)
        desired = weights * want[None, :]
        rwa_now = self._rwa()
        car_tgt = self.car_target + adj.car_target
        room = np.maximum(self.K / car_tgt - rwa_now, 0.0)
        room = np.where(self.active, room, 0.0)
        need_rwa = (desired * rw).sum(axis=1)
        scale = np.where(need_rwa > EPS, np.minimum(1.0, room / np.maximum(need_rwa, EPS)), 1.0)
        scale = np.minimum(scale, pressure_mult)
        new = desired * scale[:, None]
        # второй проход: нераспределённый спрос уходит в банки со свободным капиталом
        spare = np.maximum(room - (new * rw).sum(axis=1), 0.0) * pressure_mult
        for s in (C, R):
            unmet = want[s] - new[:, s].sum()
            if unmet > EPS and spare.sum() > EPS:
                extra = np.minimum(unmet * spare / spare.sum(), spare / rw[s])
                new[:, s] += extra
                spare = np.maximum(spare - extra * rw[s], 0.0)
        new = np.where(self.active[:, None], new, 0.0)
        # учёт: кредиты растут, ликвидность уходит, средневзвешенная ставка «когорт»
        L_new = self.L + new
        w_new = np.where(L_new > EPS, new / np.maximum(L_new, EPS), 0.0)
        self.vint_loan = (1 - w_new) * self.vint_loan + w_new * self.rate_loan
        self.L = L_new
        self.liq -= new.sum(axis=1)
        flows["new_loans"] = float(new.sum())
        flows["loan_demand_unmet"] = float(max(want.sum() - new.sum(), 0.0))
        self._r_bar_loan = r_bar
        return new

    # ---- 6. вклады ---------------------------------------------------------
    def _deposits(self, env: dict, flows: dict) -> None:
        cfg, dm, ccfg = self.cfg, self.cfg.demand, self.cfg.confidence
        res_cfg = self.cfg.resolution
        D_open = self.D.copy()
        panic = env["panic"]
        trust = self.reliability
        insured = np.array(res_cfg.insured_share)
        damp = 1.0 - ccfg.insured_run_dampening * insured          # (2,)
        r_bar = self._wavg(self.rate_dep, np.maximum(D_open, 1e-6))
        L_sys_now = self.L.sum(axis=0)
        withdrawn_total = 0.0
        leak_total = 0.0
        new_D = np.zeros_like(self.D)
        for s in (C, R):
            psi = (ccfg.run_base
                   + (ccfg.run_stress * self.stress + ccfg.run_panic * panic * (1.3 - trust)) * damp[s])
            psi = np.where(self.active, np.clip(psi, 0.0, 0.5), 0.0)
            W = psi * D_open[:, s]
            leak_rate = min(res_cfg.deposit_leak + res_cfg.deposit_leak_panic * panic, 0.9)
            leak_run = leak_rate * W.sum()
            pool_runs = W.sum() - leak_run
            retained = D_open[:, s] - W
            churn = dm.deposit_mobility * retained
            kept = retained - churn
            pending = self.pending_dep[s]
            d_ex = retained.sum() + pool_runs + pending           # запас после утечек, до макро-притока
            # желаемый запас вкладов
            gap_credit = np.log(max(L_sys_now[s], EPS)) - np.log(self.trend_loans[s])
            x_star = (dm.deposit_rate_semielasticity[s] * (r_bar[s] - self.ref_dep_rate[s])
                      + dm.credit_deposit_link * gap_credit + env["deposit_shock"]
                      + dm.demand_noise_sd * self.rng_demand.standard_normal())
            prev_trend = self.trend_dep[s]
            self.trend_dep[s] = prev_trend * np.exp(dm.deposit_trend_growth[s] / 12.0)
            x_prev = np.log(max(d_ex, EPS)) - np.log(prev_trend)
            x_new = x_prev + dm.deposit_adjust_speed * (x_star - x_prev)
            target = self.trend_dep[s] * np.exp(x_new)
            F = target - d_ex
            pool = churn.sum() + pool_runs + pending + max(F, 0.0)
            kept = kept * (1.0 + min(F, 0.0) / max(kept.sum(), EPS))
            flight = 1.0 + dm.flight_to_quality * panic
            floor = 0.02 * max(D_open[:, s].mean(), 1e-6)
            w = ((retained + floor) ** dm.deposit_size_bias
                 * np.exp(dm.deposit_rate_sensitivity * 100.0 * (self.rate_dep[:, s] - r_bar[s]))
                 * np.exp(dm.deposit_quality_sensitivity * flight * (trust - 0.5))
                 * self.appeal)
            w = np.where(self.active, w, 0.0)
            w = w / max(w.sum(), EPS)
            new_D[:, s] = np.where(self.active, kept + pool * w, 0.0)
            withdrawn_total += W.sum()
            leak_total += leak_run
        self.pending_dep = np.zeros(2)
        # денежный эффект: ликвидность меняется на прирост вкладов
        self.liq += (new_D - D_open).sum(axis=1)
        # «когортная» ставка по вкладам: притекающие деньги несут текущую ставку
        turnover = np.clip(dm.deposit_mobility + 0.02, 0.0, 1.0)
        self.vint_dep = (1 - turnover) * self.vint_dep + turnover * self.rate_dep
        self.D = new_D
        flows["withdrawn"] = float(withdrawn_total)
        flows["leak"] = float(leak_total)
        self._r_bar_dep = r_bar

    # ---- 7. резервы, издержки, налоги, дивиденды ------------------------
    def _costs_tax_dividends(self, k_start: np.ndarray, flows: dict) -> None:
        cb, bk = self.cfg.cb, self.cfg.banks
        res_new = cb.reserve_ratio * self.D.sum(axis=1)
        self.liq -= res_new - self.res
        self.res = res_new
        opex = self.cost_ratio * self._assets() / 12.0
        self.liq -= opex
        self.K -= opex
        pnl = self.K - k_start
        tax = bk.tax_rate * np.maximum(pnl, 0.0)
        car = self._car()
        payout = np.where(car >= self.car_target, bk.payout_ratio, 0.0)
        div = payout * np.maximum(pnl - tax, 0.0)
        self.liq -= tax + div
        self.K -= tax + div
        flows["opex"] = float(opex.sum())
        flows["tax"] = float(tax.sum())
        flows["dividends"] = float(div.sum())
        self._pnl = self.K - k_start + tax + div    # прибыль до налогов и дивидендов
        self._net_income = pnl - tax

    # ---- 8. ликвидность ----------------------------------------------------
    def _liquidity(self, env: dict, flows: dict, adj: PolicyAdjustments) -> None:
        cfg, ib, cb, b = self.cfg, self.cfg.interbank, self.cfg.cb, self.cfg.banks
        n = self.n
        dep_tot = np.maximum(self.D.sum(axis=1), EPS)
        panic = env["panic"]
        sys_stress = self._sys_stress()
        liq_tgt = self.liq_ratio * dep_tot * (1.0 + ib.panic_liquidity_demand * panic)
        liq_floor = b.liq_floor_share * liq_tgt
        hoard = 1.0 + ib.hoarding_liq_multiplier * (sys_stress + panic)
        # потребность в МБК — восстановить целевую ликвидность; у ЦБ берут только до минимума
        assets_now = self._assets()
        struct_need = self.mbk_b_share * assets_now
        need = np.where(self.active, np.maximum(np.maximum(liq_tgt - self.liq, 0.0), struct_need), 0.0)
        self._pre_need = np.where(self.active, np.maximum(liq_floor - self.liq, 0.0), 0.0)
        # «разрыв финансирования» до операций на МБК/у ЦБ — сигнал для ставок и кредитования
        self.pressure = np.where(self.active, np.clip((liq_tgt - self.liq) / dep_tot, -0.10, 0.10), 0.0)
        k = env["k"]
        trust = self.reliability

        # предложение МБК
        excess = np.maximum(self.liq - liq_tgt * hoard, 0.0)
        struct_sup = self.mbk_l_share * assets_now * (1.0 - min(1.0, 2.0 * panic))
        supply = np.where(self.active, np.maximum(ib.lend_share * excess, struct_sup), 0.0)
        room = np.maximum(ib.exposure_limit * self.K - self._mbk_lent(), 0.0)
        cp_limit = np.maximum(ib.counterparty_limit * self.K, 0.0)
        thr = np.clip(self.risk_aversion + adj.mbk_threshold
                      + ib.hoarding_sensitivity * sys_stress * 0.5 + ib.panic_freeze * panic,
                      0.05, 0.95)
        rate_b = k + ib.spread_base + ib.spread_risk * (1.0 - trust)
        collateral = (1 - cb.collateral_haircut) * cb.pledgeable_share * (self.L.sum(axis=1) + self.sec)
        max_rate = np.where(collateral > EPS, env["refi"], env["refi"] + ib.rate_cap_over_refi)
        M = match_interbank(self.rng_mbk, need, supply, room, cp_limit, trust, thr,
                            rate_b, max_rate, self.prev_links,
                            ib.link_persistence, ib.max_lenders)
        borrowed = M.sum(axis=0)
        lent = M.sum(axis=1)
        self.M = M
        self.I = M * (rate_b[None, :] / 12.0)
        # кредитор, давший больше избытка, продаёт бумаги по номиналу (замена бумаг на МБК)
        fund = np.maximum(lent - np.maximum(self.liq - liq_tgt, 0.0), 0.0)
        swap = np.minimum(fund, self.sec)
        self.sec -= swap
        self.liq += swap
        self.liq += borrowed - lent
        flows["mbk_volume"] = float(M.sum())
        flows["mbk_rate_num"] = float((M * rate_b[None, :]).sum())

        # ЦБ: рефинансирование под залог
        need2 = np.where(self.active, np.maximum(liq_floor - self.liq, 0.0), 0.0)
        take = np.minimum(need2, collateral)
        self.cb_debt = take.copy()
        self.liq += take
        flows["cb_loans"] = float(take.sum())

        # крайняя мера: продажа бумаг
        need3 = np.where(self.active, np.maximum(liq_floor - self.liq, 0.0), 0.0)
        hair = self.cfg.banks.securities_sale_haircut
        sold = np.minimum(self.sec, need3 / (1.0 - hair))
        self.sec -= sold
        self.liq += sold * (1.0 - hair)
        self.K -= sold * hair
        flows["fire_sales"] = float((sold).sum())
        flows["fire_loss"] = float((sold * hair).sum())
        self.liquidity_shortfall = np.where(self.active, np.maximum(-self.liq, 0.0), 0.0)

        # избыток: ценные бумаги до целевой доли, остаток — в депозит ЦБ
        free = np.where(self.active, np.maximum(self.liq - liq_tgt, 0.0), 0.0)
        assets = self._assets()
        room_sec = np.maximum(self.sec_target_ratio * assets - self.sec, 0.0)
        buy = np.minimum(free, room_sec)
        self.sec += buy
        self.liq -= buy
        free -= buy
        self.cb_dep = free.copy()
        self.liq -= free
        flows["cb_deposits_new"] = float(free.sum())
        self.cb_use_prev = take > 1e-6

    def _sys_stress(self) -> float:
        a = self._assets() * self.active
        return float((self.stress * a).sum() / max(a.sum(), EPS))

    # ---- 9. выход и вход ---------------------------------------------------
    def _solvency_and_exit(self, env: dict, flows: dict) -> int:
        cb, res = self.cfg.cb, self.cfg.resolution
        total_failed = 0
        rounds = 0
        failed_assets = 0.0
        bailout_cost = 0.0
        resolution_cost = 0.0
        t_rel = env["t"] - self.cfg.burn_in
        while True:
            assets = self._assets()
            sys_assets = max(float((assets * self.active).sum()), EPS)
            rwa = self._rwa()
            car = np.where(rwa > EPS, self.K / np.maximum(rwa, EPS), 1.0)
            thr = cb.revocation_car
            cap_fail = self.active & ((self.K <= 0.0) | ((car < thr) & (thr > 0)))
            liq_fail = self.active & (self.liq < -1e-6)
            fail = cap_fail | liq_fail
            if not fail.any():
                break
            shares = assets / sys_assets
            for j in np.flatnonzero(fail):
                big = shares[j] >= cb.tbtf_threshold
                if cb.bailout_enabled and big:
                    bailout_cost += self._bailout(j)
                    continue
                cause = "capital" if cap_fail[j] else "liquidity"
                if rounds > 0:
                    cause = "contagion"
                a_j = float(assets[j])
                cost = self._resolve(j)
                resolution_cost += cost
                failed_assets += a_j
                total_failed += 1
                self._events.append({
                    "month": t_rel, "bank": int(self.bank_id[j]), "slot": int(j), "cause": cause,
                    "assets_bn": a_j, "round": rounds, "cost_bn": cost})
            rounds += 1
            if rounds > self.n + 2:
                break
        self.recent_failure_assets = failed_assets / max(sys_assets, EPS) if total_failed else 0.0
        self.recent_fail_counts.append(total_failed)
        flows["failures"] = total_failed
        flows["failed_assets"] = failed_assets
        flows["bailout_cost"] = bailout_cost
        flows["resolution_cost"] = resolution_cost
        flows["cascade_rounds"] = max(rounds - 1, 0) if total_failed else 0
        return total_failed

    def _bailout(self, j: int) -> float:
        """ЦБ докапитализирует системно значимый банк и закрывает дефицит ликвидности."""
        cb = self.cfg.cb
        rwa = max(self._rwa()[j], EPS)
        inject = max(cb.bailout_target_car * rwa - self.K[j], 0.0)
        self.K[j] += inject
        self.liq[j] += inject
        if self.liq[j] < 0:
            self.cb_debt[j] += -self.liq[j]
            self.liq[j] = 0.0
        self.stress[j] = max(self.stress[j], 0.5)
        return float(inject)

    def _resolve(self, j: int) -> float:
        """Отзыв лицензии: ликвидация, выплаты кредиторам, страхование вкладов.

        Возвращает чистые потери АСВ/ЦБ (в млрд руб.). Состояние банка обнуляется.
        """
        res, b = self.cfg.resolution, self.cfg.banks
        L, D = self.L[j], self.D[j]
        recovery = float(L[C] * res.recovery_loans[C] + L[R] * res.recovery_loans[R])
        claims_out = float(self.M[j, :].sum())
        estate = (max(self.liq[j], 0.0) + self.res[j] + self.cb_dep[j]
                  + (1 - res.securities_haircut) * self.sec[j] + recovery + claims_out)
        pay_cb = min(self.cb_debt[j], estate)
        rest = estate - pay_cb
        ins = np.array(res.insured_share)
        insured_dep = float((D * ins).sum())
        uninsured_dep = float((D * (1 - ins)).sum())
        mbk_claims = self.M[:, j].copy()
        ra_c = self.ra_claims[j]
        unpaid = max(-self.liq[j], 0.0)
        junior = uninsured_dep + insured_dep + mbk_claims.sum() + ra_c + unpaid
        rho = 1.0 if junior <= EPS else float(min(1.0, rest / junior))
        # кредиторы-банки: получают rho от долга, остальное — убыток капитала
        paid = rho * mbk_claims
        self.liq += paid
        self.K -= mbk_claims - paid
        self.I[:, j] = 0.0
        self.M[:, j] = 0.0
        self.ra_claims[j] = 0.0
        self.ra_interest[j] = 0.0
        # требования самого банка к другим переходят ликвидатору
        self.ra_claims += self.M[j, :]
        self.ra_interest += self.I[j, :]
        self.M[j, :] = 0.0
        self.I[j, :] = 0.0
        # вкладчики: застрахованные получают всё, прочие — rho
        payout_seg = D * ins + rho * D * (1 - ins)
        self.pending_dep += payout_seg * (1.0 - res.deposit_leak)
        cost = insured_dep * (1.0 - rho) + max(self.cb_debt[j] - estate, 0.0)
        # обнуление банка
        self.L[j] = 0.0
        self.D[j] = 0.0
        for arr in (self.liq, self.res, self.sec, self.cb_dep, self.cb_debt, self.K):
            arr[j] = 0.0
        self.stress[j] = 0.0
        self.pressure[j] = 0.0
        self.active[j] = False
        self.free_since[j] = self.t
        self.age[j] = 0
        self.reliability[j] = 0.0
        self.rate_loan[j] = 0.0
        self.rate_dep[j] = 0.0
        return float(cost)

    def _entry(self, env: dict, flows: dict) -> None:
        res = self.cfg.resolution
        flows["entries"] = 0
        flows["entry_capital"] = 0.0
        if not res.entry_enabled:
            return
        for j in np.flatnonzero(~self.active):
            if self.free_since[j] >= 0 and self.t - self.free_since[j] >= res.entry_delay_months:
                flows["entry_capital"] += self._spawn_bank(j)
                flows["entries"] += 1

    def _spawn_bank(self, j: int) -> float:
        """Новый небольшой банк на месте выбывшего. Возвращает вложенный капитал."""
        s, b = self.cfg.sector, self.cfg.banks
        rng = self.rng_entry
        het = self._draw_heterogeneity(rng, 1)
        self.car_target[j], self.liq_ratio[j] = het["car_target"][0], het["liq_ratio"][0]
        self.cost_ratio[j], self.quality[j] = het["cost_ratio"][0], het["quality"][0]
        self.risk_aversion[j], self.speed_loan[j] = het["risk_aversion"][0], het["speed_loan"][0]
        self.speed_dep[j], self.appetite[j] = het["speed_dep"][0], het["appetite"][0]
        self.appeal[j] = het["appeal"][0]
        grow = float(np.mean(self.trend_loans / self.trend_loans0))
        assets = self.cfg.resolution.entry_size_share * s.total_assets_bn * max(grow, 1.0)
        a = np.array([assets])
        lc = np.array([s.init_loans_corp * assets])
        lr = np.array([s.init_loans_retail * assets])
        car0 = np.array([self.car_target[j] + self.cfg.resolution.entry_capital_buffer])
        K, D, res_, liq, sec, lc, lr = self._balance_sheet(
            a, lc, lr, car0, np.array([self.liq_ratio[j]]), np.array([s.init_retail_deposit_share]))
        self.K[j], self.D[j], self.res[j], self.liq[j], self.sec[j] = K[0], D[0], res_[0], liq[0], sec[0]
        self.L[j] = [lc[0], lr[0]]
        self.cb_dep[j] = self.cb_debt[j] = 0.0
        self.sec_target_ratio[j] = sec[0] / assets
        self.pd_ema[j] = np.array(self.cfg.credit.pd_monthly) * self.quality[j]
        self.mbk_b_share[j] = self.mbk_l_share[j] = 0.0
        self.prev_links[j, :] = False
        self.prev_links[:, j] = False
        self.active[j] = True
        self.free_since[j] = -1
        self.age[j] = 0
        self.bank_id[j] = self.next_bank_id
        self.next_bank_id += 1
        self.stress[j] = 0.0
        self.reliability = self._reliability()
        # ставки нового банка = средние по сектору
        act = self.active.copy()
        act[j] = False
        for sgm in (C, R):
            if act.any():
                self.rate_loan[j, sgm] = np.average(self.rate_loan[act, sgm], weights=self.L[act, sgm] + EPS)
                self.rate_dep[j, sgm] = np.average(self.rate_dep[act, sgm], weights=self.D[act, sgm] + EPS)
            else:
                self.rate_loan[j, sgm] = self.ref_loan_rate[sgm]
                self.rate_dep[j, sgm] = self.ref_dep_rate[sgm]
        self.vint_loan[j] = self.rate_loan[j]
        self.vint_dep[j] = self.rate_dep[j]
        return float(K[0])

    # ---- 10. состояния -----------------------------------------------------
    def _update_states(self, env: dict, new_loans: np.ndarray, npl: np.ndarray) -> None:
        cmf = self.cfg.confidence
        car = self._car()
        raw = np.clip((self.car_target - car) / 0.06, 0.0, 1.0)
        raw = raw + 0.2 * (self.cb_debt > 1e-6) + 0.3 * (self._pre_need > 1e-6) * (self.cb_debt > 1e-6)
        raw = np.clip(raw, 0.0, 1.0)
        s = cmf.stress_smoothing
        self.stress = np.where(self.active, (1 - s) * self.stress + s * raw, 0.0)
        self.reliability = self._reliability()
        self.car_now = self._car()
        self.npl_prev = npl
        self._panic_now = env["panic"]
        self.last_network = self.M.copy()
        if self.cfg.record_network and env["t"] >= self.cfg.burn_in:
            lenders, borrowers, amounts = network_edges(self.M)
            self._network_records.append((env["t"] - self.cfg.burn_in, lenders, borrowers, amounts))

    # ======================================================================
    # Тождества и запись
    # ======================================================================
    def identity_error(self) -> float:
        """Максимум |активы − обязательства − капитал| по активным банкам."""
        err = np.abs(self._assets() - self._liabilities() - self.K)
        return float(err[self.active].max()) if self.active.any() else 0.0

    def _record(self, env, flows, new_loans, npl, n_fail, L_start, D_start) -> dict:
        cfg = self.cfg
        act = self.active
        assets = self._assets()
        rwa = self._rwa()
        a = np.where(act, assets, 0.0)
        tot_a = max(a.sum(), EPS)
        sh = np.sort(a / tot_a)[::-1]
        loans = self.L.sum(axis=0)
        deps = self.D.sum(axis=0)
        new_by_seg = new_loans.sum(axis=0)

        def wmean(x, w):
            w = np.where(act, w, 0.0)
            return float((x * w).sum() / max(w.sum(), EPS))

        mbk_vol = float(self.M.sum())
        mbk_rate = flows["mbk_rate_num"] / mbk_vol if mbk_vol > EPS else np.nan
        cb_debt, cb_dep = float(self.cb_debt.sum()), float(self.cb_dep.sum())
        free_liq = float((self.liq * act).sum())
        res_sum = float((self.res * act).sum())
        lr_c = wmean(self.rate_loan[:, C], new_loans[:, C] + 1e-9)
        lr_r = wmean(self.rate_loan[:, R], new_loans[:, R] + 1e-9)
        dr_c = wmean(self.rate_dep[:, C], self.D[:, C])
        dr_r = wmean(self.rate_dep[:, R], self.D[:, R])
        loan_rate_avg = ((lr_c * new_by_seg[C] + lr_r * new_by_seg[R]) / new_by_seg.sum()
                         if new_by_seg.sum() > EPS else 0.5 * (lr_c + lr_r))
        profit = float(self._pnl[act].sum())
        net_income = float(self._net_income[act].sum())
        k_sum = float(self.K[act].sum())
        rec = {
            "month": env["t"] - cfg.burn_in,
            "key_rate": env["k"] * 100,
            "refi_rate": env["refi"] * 100,
            "loans_corp": loans[C], "loans_retail": loans[R], "loans_total": loans.sum(),
            "deposits_corp": deps[C], "deposits_retail": deps[R], "deposits_total": deps.sum(),
            "new_loans_corp": new_by_seg[C], "new_loans_retail": new_by_seg[R],
            "loan_rate_corp": lr_c * 100, "loan_rate_retail": lr_r * 100,
            "loan_rate": loan_rate_avg * 100,
            "deposit_rate_corp": dr_c * 100, "deposit_rate_retail": dr_r * 100,
            "deposit_rate": (dr_c * deps[C] + dr_r * deps[R]) / max(deps.sum(), EPS) * 100,
            "loan_yield": float((self._yield_loans() * self.L * act[:, None]).sum()
                                / max(self.L[act].sum(), EPS) * 100),
            "cb_debt": cb_debt, "cb_deposits": cb_dep,
            "liquidity_surplus": cb_dep + free_liq + res_sum - cb_debt,
            "mbk_volume": mbk_vol, "mbk_rate": mbk_rate * 100 if mbk_vol > EPS else np.nan,
            "mbk_links": int((self.M > 1e-6).sum()),
            "mbk_borrowers": int((self.M.sum(axis=0) > 1e-6).sum()),
            "n_banks": int(act.sum()),
            "failures": int(n_fail), "failed_assets_share": flows["failed_assets"] / tot_a,
            "entries": int(flows["entries"]),
            "cascade_rounds": int(flows["cascade_rounds"]),
            "bailout_cost": flows["bailout_cost"], "resolution_cost": flows["resolution_cost"],
            "capital": k_sum, "assets": float(a.sum()),
            "car": k_sum / max(float(rwa[act].sum()), EPS) * 100,
            "car_min_bank": float(self.car_now[act].min() * 100) if act.any() else np.nan,
            "n_below_car_min": int((self.car_now[act] < cfg.banks.car_min).sum()),
            "loan_losses": flows["loan_losses"], "npl_share": flows["defaults"] / max(float(self.L[act].sum()) + flows["defaults"], EPS) * 100,
            "profit": profit, "net_income": net_income,
            "roe": net_income / max(k_sum, EPS) * 12 * 100,
            "securities": float((self.sec * act).sum()), "liquid_assets": free_liq,
            "reserves": res_sum,
            "hhi": float((sh ** 2).sum() * 10000), "top1_share": float(sh[0] * 100),
            "top5_share": float(sh[:5].sum() * 100),
            "stress": self._sys_stress(), "panic": getattr(self, "_panic_now", 0.0),
            "deposit_outflow": flows["withdrawn"], "credit_demand_unmet": flows["loan_demand_unmet"],
            "reliability_mean": float(self.reliability[act].mean()) if act.any() else np.nan,
            "int_income_loans": flows["int_loans"], "int_expense_dep": flows["int_dep"],
            "opex": flows["opex"],
            "entry_capital": flows["entry_capital"],
        }
        rec = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in rec.items()}
        self._records.append(rec)
        if cfg.record_banks:
            m = rec["month"]
            self._bank_records.append({
                "month": np.full(self.n, m), "slot": np.arange(self.n), "bank_id": self.bank_id.copy(),
                "active": act.copy(), "age": self.age.copy(),
                "assets": assets * act, "loans_corp": self.L[:, C].copy(), "loans_retail": self.L[:, R].copy(),
                "deposits_corp": self.D[:, C].copy(), "deposits_retail": self.D[:, R].copy(),
                "capital": self.K.copy(), "car": self.car_now * 100 * act, "liquid": self.liq.copy(),
                "securities": self.sec.copy(), "mbk_lent": self._mbk_lent(),
                "mbk_borrowed": self._mbk_borrowed(), "cb_debt": self.cb_debt.copy(),
                "cb_deposit": self.cb_dep.copy(),
                "loan_rate_corp": self.rate_loan[:, C] * 100, "loan_rate_retail": self.rate_loan[:, R] * 100,
                "deposit_rate_corp": self.rate_dep[:, C] * 100, "deposit_rate_retail": self.rate_dep[:, R] * 100,
                "reliability": self.reliability.copy(), "stress": self.stress.copy(),
                "npl_share": npl * 100, "car_target": self.car_target * 100,
            })
        return rec

    # ======================================================================
    # Запуск
    # ======================================================================
    def iter_months(self) -> Iterator[dict]:
        """Генератор месячных записей (для «живого» интерфейса). Прогрев пропускается."""
        while self.t < self.total_steps:
            rec = self.step()
            if rec is not None:
                yield rec

    def run(self) -> SimulationResult:
        try:
            for _ in self.iter_months():
                pass
        except SectorCollapse as exc:
            self.collapsed = str(exc)
        return self.result()

    def result(self) -> SimulationResult:
        from .metrics import build_sector_frame, build_bank_frame
        sector = build_sector_frame(self._records, self.cfg.start_date)
        banks = build_bank_frame(self._bank_records, self.cfg.start_date) if self.cfg.record_banks else None
        all_events = pd.DataFrame(self._events, columns=["month", "bank", "slot", "cause",
                                                         "assets_bn", "round", "cost_bn"])
        events = all_events[all_events["month"] >= 0].reset_index(drop=True)
        network = None
        if self.cfg.record_network and self._network_records:
            network = pd.DataFrame({
                "month": np.concatenate([np.full(len(l), m) for m, l, _, _ in self._network_records]),
                "lender": np.concatenate([l for _, l, _, _ in self._network_records]),
                "borrower": np.concatenate([b for _, _, b, _ in self._network_records]),
                "amount": np.concatenate([a for _, _, _, a in self._network_records])})
        return SimulationResult(sector=sector, banks=banks, events=events, network=network,
                                config=self.cfg, scenario=self.effective_scenario(), seed=self.seed,
                                meta={"policy": self.policy.name, "steps": self.t,
                                      "collapsed": getattr(self, "collapsed", None),
                                      "burn_in_failures": int((all_events["month"] < 0).sum())})

    # ---- вмешательства «на лету» (интерфейс, эксперименты) ----------------
    def _check_running(self) -> int:
        if self.t >= self.total_steps:
            raise RuntimeError("симуляция завершена: вмешательство невозможно")
        return self.t

    def shift_key_rate(self, delta: float, months: int | None = None) -> None:
        """Сдвигает ключевую ставку на ``delta`` (доли) на все будущие месяцы (или на ``months``)."""
        t = self._check_running()
        end = self.total_steps if months is None else min(self.total_steps, t + months)
        self._scn.key_rate[t:end] = np.maximum(self._scn.key_rate[t:end] + delta, 0.001)
        self.interventions.append({"month": t - self.cfg.burn_in, "type": "key_rate", "value": delta})

    def inject_panic(self, level: float) -> None:
        """Импульс паники вкладчиков (0..1) в текущем месяце с затуханием."""
        t = self._check_running()
        self._scn.confidence[t] = max(self._scn.confidence[t], float(np.clip(level, 0, 1)))
        self.interventions.append({"month": t - self.cfg.burn_in, "type": "panic", "value": level})

    def inject_credit_stress(self, peak: float, half_life: float = 6.0) -> None:
        """Рост вероятности дефолтов заёмщиков на ``peak`` (доля) с затуханием."""
        t = self._check_running()
        k = np.arange(self.total_steps - t)
        self._scn.credit_stress[t:] += peak * 0.5 ** (k / half_life)
        self.interventions.append({"month": t - self.cfg.burn_in, "type": "credit_stress", "value": peak})

    def effective_scenario(self) -> Scenario:
        """Сценарий с учётом вмешательств (без периода прогрева)."""
        b = self.cfg.burn_in
        s = self._scn
        return Scenario(name=self.scenario.name, key_rate=s.key_rate[b:].copy(),
                        credit_stress=s.credit_stress[b:].copy(), confidence=s.confidence[b:].copy(),
                        deposit_shock=s.deposit_shock[b:].copy(), demand_shock=s.demand_shock[b:].copy(),
                        meta=dict(self.scenario.meta, interventions=list(self.interventions)))

    # ---- сеть для визуализации ------------------------------------------
    def network_snapshot(self) -> dict:
        lenders, borrowers, amounts = network_edges(self.last_network)
        return {
            "lender": lenders, "borrower": borrowers, "amount": amounts,
            "assets": self._assets() * self.active, "active": self.active.copy(),
            "reliability": self.reliability.copy(), "bank_id": self.bank_id.copy(),
            "cb_debt": self.cb_debt.copy(),
        }
