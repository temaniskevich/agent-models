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
