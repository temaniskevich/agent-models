"""Контрольная синтетика «без экономики» (план, шаг 4).

Нужна, чтобы отделить эффект *структуры* агентной модели от эффекта «просто
больше данных»: те же средние, разбросы, автокорреляции, корреляции между
рядами и такая же частота/размер скачков, но без балансов, капитала и причинных
связей — только векторная авторегрессия первого порядка с независимыми скачками.

Ограничение: для сильно скошенных рядов (долг перед ЦБ) совпадают среднее,
автокорреляция и корреляции, а тяжёлые хвосты воспроизводятся приближённо.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

DEFAULT_COLUMNS = ["key_rate", "loan_rate", "deposit_rate", "loans_growth_mom_ann",
                   "deposits_growth_mom_ann", "cb_debt"]


@dataclass
class ARJumpControl:
    """VAR(1) по компонентам с диагональной динамикой + скачки.

    ``fit`` подгоняет параметры по таблице прогонов агентной модели (или по
    реальным данным), ``simulate`` генерирует независимые истории.
    """
    columns: list[str] = field(default_factory=lambda: list(DEFAULT_COLUMNS))
    # сильно скошенные неотрицательные ряды моделируем в логарифмах: log(1 + x)
    log_columns: tuple[str, ...] = ("cb_debt",)
    jump_z: float = 3.0
    mean_: np.ndarray | None = None
    phi_: np.ndarray | None = None
    chol_: np.ndarray | None = None
    jump_prob_: np.ndarray | None = None
    jump_sizes_: list[np.ndarray] | None = None
    start_: np.ndarray | None = None

    def _forward(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df[self.columns].copy()
        for c in self.log_columns:
            if c in out:
                out[c] = np.log1p(np.maximum(out[c], 0.0))
        return out

    def _inverse(self, x: np.ndarray) -> np.ndarray:
        x = x.copy()
        for c in self.log_columns:
            if c in self.columns:
                j = self.columns.index(c)
                x[:, j] = np.expm1(x[:, j])
        return x

    def fit(self, runs: pd.DataFrame, run_col: str = "run") -> "ARJumpControl":
        cols = self.columns
        runs = runs.dropna(subset=cols)
        clean = self._forward(runs)
        clean[run_col] = runs[run_col].to_numpy()
        self.mean_ = clean[cols].mean().to_numpy()
        phis, innov, starts = [], [], []
        for _, g in clean.groupby(run_col):
            x = g[cols].to_numpy()
            if len(x) < 5:
                continue
            starts.append(x[0])
            d = x - self.mean_
            num = (d[1:] * d[:-1]).sum(axis=0)
            den = (d[:-1] ** 2).sum(axis=0)
            phis.append(num / np.maximum(den, 1e-12))
        self.phi_ = np.clip(np.mean(phis, axis=0), -0.99, 0.995)
        e_all = []
        for _, g in clean.groupby(run_col):
            d = g[cols].to_numpy() - self.mean_
            e_all.append(d[1:] - self.phi_ * d[:-1])
        e = np.vstack(e_all)
        med = np.median(e, axis=0)
        mad = np.median(np.abs(e - med), axis=0) * 1.4826 + 1e-12
        is_jump = np.abs(e - med) > self.jump_z * mad
        self.jump_prob_ = is_jump.mean(axis=0)
        self.jump_sizes_ = [e[is_jump[:, j], j] for j in range(len(cols))]
        base = np.where(is_jump, np.nan, e)
        cov = pd.DataFrame(base).cov().to_numpy()
        cov = np.nan_to_num(cov) + 1e-12 * np.eye(len(cols))
        self.chol_ = np.linalg.cholesky(cov)
        self.start_ = np.mean(starts, axis=0)
        self._match_log_means(runs)
        return self

    def _match_log_means(self, runs: pd.DataFrame, iterations: int = 8) -> None:
        """Подстраивает уровень логарифмируемых рядов, чтобы среднее в исходных единицах совпало
        (поправка на неравенство Йенсена)."""
        idx = [self.columns.index(c) for c in self.log_columns if c in self.columns]
        if not idx:
            return
        n = int(runs.groupby("run").size().median())
        target = runs[self.columns].mean().to_numpy()
        for it in range(iterations):
            pilot = self.simulate(n, 24, seed=1000 + it)
            got = pilot[self.columns].mean().to_numpy()
            for j in idx:
                self.mean_[j] += np.log1p(max(target[j], 0.0)) - np.log1p(max(got[j], 0.0))

    def simulate(self, n_months: int, n_runs: int, seed: int = 0) -> pd.DataFrame:
        if self.mean_ is None:
            raise RuntimeError("сначала вызовите fit()")
        rng = np.random.default_rng(seed)
        k = len(self.columns)
        frames = []
        for r in range(n_runs):
            x = np.empty((n_months, k))
            d = self.start_ - self.mean_
            for t in range(n_months):
                eps = self.chol_ @ rng.standard_normal(k)
                for j in range(k):
                    if rng.random() < self.jump_prob_[j] and len(self.jump_sizes_[j]):
                        eps[j] += rng.choice(self.jump_sizes_[j])
                d = self.phi_ * d + eps
                x[t] = self.mean_ + d
            x = self._inverse(x)
            df = pd.DataFrame(x, columns=self.columns)
            df.insert(0, "t", np.arange(n_months))
            df.insert(0, "run", r)
            frames.append(df)
        return pd.concat(frames, ignore_index=True)

    def moments_report(self, runs: pd.DataFrame) -> pd.DataFrame:
        """Сравнение моментов: исходные данные vs контроль (для проверки подгонки)."""
        ctrl = self.simulate(int(runs.groupby("run").size().median()),
                             max(5, runs["run"].nunique()), seed=1)
        rows = []
        for c in self.columns:
            rows.append({"variable": c, "mean_src": runs[c].mean(), "mean_ctrl": ctrl[c].mean(),
                         "std_src": runs[c].std(), "std_ctrl": ctrl[c].std(),
                         "ac1_src": _ac1(runs, c), "ac1_ctrl": _ac1(ctrl, c)})
        return pd.DataFrame(rows)


def _ac1(df: pd.DataFrame, col: str) -> float:
    vals = []
    for _, g in df.groupby("run"):
        x = g[col].to_numpy()
        if len(x) > 3 and x.std() > 0:
            vals.append(np.corrcoef(x[1:], x[:-1])[0, 1])
    return float(np.nanmean(vals)) if vals else np.nan


# --------------------------------------------------------------------------
# Линейный двойник агентной модели (поправка 18.11.2026 к docs/CAPACITY_PROTOCOL.md, раздел 12)
# --------------------------------------------------------------------------
TWIN_STATE = ["lr_c", "lr_r", "dr_c", "dr_r", "g_lc", "g_lr", "g_dc", "g_dr"]


@dataclass
class LinearTwin:
    """VARX(p, q) с фиксированными эффектами историй: линейная проекция агентной панели.

    ``w_t = a_i + Σ_{j=1..p} A_j w_{t−j} + Σ_{l=0..q} b_l k_{t−l} + u_t``, коэффициенты общие для всех историй,
    ``a_i`` — константа истории i. Ключевая ставка ``k`` — экзогенный вход: при генерации берётся путь из исходной панели,
    поэтому двойник и агентная модель отличаются только банковским блоком (нелинейности, режимы, разброс наклонов).
    Остатки при генерации — бутстрэп векторов остатков подгонки.
    """
    state: list[str] = field(default_factory=lambda: list(TWIN_STATE))
    exog: str = "key_rate"
    q: int = 12
    p_max: int = 6
    p: int | None = None                       # None — выбор по BIC из 1..p_max
    p_bic_: int | None = None                  # p по BIC (до проверки устойчивости)
    bic_: dict | None = None
    A_: np.ndarray | None = None               # (p, k, k): w_t = Σ A_j w_{t−j}
    B_: np.ndarray | None = None               # (q+1, k): вклад k_{t−l}
    fe_: dict | None = None                    # run -> константа (k,)
    resid_: np.ndarray | None = None
    max_root_: float | None = None
    n_obs_: int | None = None

    # -- выборка ----------------------------------------------------------------
    @property
    def skip(self) -> int:
        """Первые месяцы каждой истории, исключаемые при подгонке (общая выборка для всех p)."""
        return max(self.p_max, self.q)

    def _runs(self, panel: pd.DataFrame, run_col: str = "run"):
        for run, g in panel.sort_values([run_col, "t"]).groupby(run_col, sort=True):
            yield run, g[self.state].to_numpy(float), g[self.exog].to_numpy(float)

    def _design(self, W: np.ndarray, K: np.ndarray, p: int) -> tuple[np.ndarray, np.ndarray]:
        T = len(W)
        idx = np.arange(self.skip, T)
        X = [W[idx - j] for j in range(1, p + 1)] + [K[idx - l][:, None] for l in range(self.q + 1)]
        return np.hstack(X), W[idx]

    def _fit_p(self, panel: pd.DataFrame, p: int):
        Xs, Ys, runs = [], [], []
        for run, W, K in self._runs(panel):
            if len(W) <= self.skip + 1:
                continue
            X, Y = self._design(W, K, p)
            Xs.append(X)
            Ys.append(Y)
            runs.append(run)
        # внутригрупповое преобразование: МНК по отклонениям от средних истории
        Xd = np.vstack([x - x.mean(axis=0) for x in Xs])
        Yd = np.vstack([y - y.mean(axis=0) for y in Ys])
        beta, *_ = np.linalg.lstsq(Xd, Yd, rcond=None)
        resid = Yd - Xd @ beta
        fe = {r: y.mean(axis=0) - x.mean(axis=0) @ beta for r, x, y in zip(runs, Xs, Ys)}
        return beta, resid, fe

    def _bic(self, resid: np.ndarray, n_coef: int) -> float:
        n, k = resid.shape
        sign, logdet = np.linalg.slogdet(resid.T @ resid / n)
        return float(n * logdet + np.log(n) * n_coef * k)

    def _unpack(self, beta: np.ndarray, p: int) -> tuple[np.ndarray, np.ndarray]:
        k = len(self.state)
        A = np.stack([beta[j * k:(j + 1) * k].T for j in range(p)])        # A_j[i_out, i_in]
        B = beta[p * k:]                                                    # (q+1, k)
        return A, B

    @staticmethod
    def companion_max_root(A: np.ndarray) -> float:
        p, k, _ = A.shape
        C = np.zeros((k * p, k * p))
        C[:k, :] = np.hstack(list(A))
        if p > 1:
            C[k:, :-k] = np.eye(k * (p - 1))
        return float(np.max(np.abs(np.linalg.eigvals(C))))

    # -- подгонка и генерация -----------------------------------------------------
    def fit(self, panel: pd.DataFrame) -> "LinearTwin":
        panel = panel.dropna(subset=self.state + [self.exog])
        k = len(self.state)
        if self.p is None:
            self.bic_ = {}
            for p in range(1, self.p_max + 1):
                _, resid, _ = self._fit_p(panel, p)
                self.bic_[p] = self._bic(resid, p * k + self.q + 1)
            p = min(self.bic_, key=self.bic_.get)
        else:
            p = self.p
        self.p_bic_ = p
        while True:
            beta, resid, fe = self._fit_p(panel, p)
            A, B = self._unpack(beta, p)
            root = self.companion_max_root(A)
            if root < 1 or p == 1:
                break
            p -= 1                                                          # правило протокола: уменьшать p до устойчивости
        self.p, self.A_, self.B_, self.fe_, self.resid_, self.max_root_ = p, A, B, fe, resid, root
        self.n_obs_ = len(resid)
        return self

    def simulate(self, panel: pd.DataFrame, seed: int, rate_floor: float = 0.1,
                 rate_cols: tuple[str, ...] = ("lr_c", "lr_r", "dr_c", "dr_r")) -> pd.DataFrame:
        """Двойник каждой истории ``panel``: тот же путь ключевой ставки и те же первые p месяцев состояния.

        Лаги ключевой ставки до начала истории — первое значение пути (ставка до старта считается постоянной)."""
        if self.A_ is None:
            raise RuntimeError("сначала вызовите fit()")
        rng = np.random.default_rng(seed)
        p, q = self.p, self.q
        frames = []
        for run, W0, K in self._runs(panel):
            T = len(W0)
            Kp = np.concatenate([np.full(q, K[0]), K])                      # Kp[t + q] = K[t]
            W = np.empty_like(W0)
            W[:p] = W0[:p]
            draws = self.resid_[rng.integers(0, len(self.resid_), size=T)]
            a = self.fe_.get(run, np.zeros(len(self.state)))
            for t in range(p, T):
                x = a + draws[t]
                for j in range(p):
                    x = x + self.A_[j] @ W[t - j - 1]
                x = x + Kp[t + q - np.arange(q + 1)] @ self.B_
                W[t] = x
            g = pd.DataFrame(W, columns=self.state)
            g.insert(0, self.exog, K)
            g.insert(0, "t", panel.loc[panel["run"] == run, "t"].sort_values().to_numpy())
            g.insert(0, "run", run)
            frames.append(g)
        out = pd.concat(frames, ignore_index=True)
        for c in rate_cols:
            if c in out:
                out[c] = out[c].clip(lower=rate_floor)
        return out


def panel_lp(panel: pd.DataFrame, columns, horizons=(0, 3, 6, 12), exog: str = "key_rate",
             growth_cols=("g_lc", "g_lr", "g_dc", "g_dr")) -> pd.DataFrame:
    """Описательная панельная локальная проекция: реакция на Δk_t (контроли Δk_{t−1..t−3}).

    Ставки: ``y_{t+h} − y_{t−1}`` (п.п.); рост: ``100·(ln V_{t+h} − ln V_{t−1}) = Σ_{s=t..t+h} g_s / 12`` (%)."""
    rows = []
    for c in columns:
        for h in horizons:
            Xs, ys = [], []
            for _, g in panel.sort_values(["run", "t"]).groupby("run"):
                k = g[exog].to_numpy(float)
                v = g[c].to_numpy(float)
                dk = np.diff(k, prepend=np.nan)
                T = len(v)
                idx = np.arange(4, T - h)
                if c in growth_cols:
                    cs = np.concatenate([[0.0], np.cumsum(v / 12.0)])       # cs[s] = Σ_{u<s} v_u/12
                    y = cs[idx + h + 1] - cs[idx]
                else:
                    y = v[idx + h] - v[idx - 1]
                X = np.column_stack([np.ones(len(idx)), dk[idx], dk[idx - 1], dk[idx - 2], dk[idx - 3]])
                Xs.append(X)
                ys.append(y)
            X, y = np.vstack(Xs), np.concatenate(ys)
            ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
            beta, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
            rows.append({"series": c, "h": h, "beta": float(beta[1])})
    return pd.DataFrame(rows)
