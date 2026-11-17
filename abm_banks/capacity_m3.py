"""M3: небольшой трансформер (протокол — docs/CAPACITY_PROTOCOL.md, раздел 4 и «Уточнения реализации»).

Импортирует torch: не импортировать в одном процессе с lightgbm (конфликт OpenMP на macOS), см. ``capacity.py``.

Одна модель на окно: вход — контекст CTX месяцев × каналы (изменение ключевой ставки; для каждого ряда цели — первая разность
для ставок и значение для роста), выход — стандартизованные цели «ряд × горизонт». Стандартизация — по реальной обучающей выборке.
Обучение смешанными батчами (доля синтетики = w), потеря Хьюбера с маской пропусков, ранняя остановка по последним 12
месяцам реальной обучающей выборки, среднее по зёрнам.
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import torch
from numpy.lib.stride_tricks import sliding_window_view
from torch import nn

from . import forecasting as fc
from .capacity import CTX, HORIZONS, SHARES, CWindow, _row

D_MODEL, HEADS, LAYERS, FF, DROPOUT = 32, 2, 2, 128, 0.1
LR, BATCH, MAX_STEPS, EVAL_EVERY = 1e-3, 256, 3000, 100
NO_VAL_STEPS = 500                  # если валидационных пар меньше MIN_VAL, ранней остановки нет: фиксированное число шагов
MIN_VAL = 5
CLIP = 8.0                          # обрезка стандартизованных входов (защита от масштаба синтетики)
SD_FLOOR = {"rate": 0.25, "growth": 2.0}


def _kind(name: str) -> str:
    return "rate" if name in fc.RATE_T else "growth"


def channel_names(series) -> list[str]:
    return ["dk"] + list(series)


def channels(df: pd.DataFrame, series) -> np.ndarray:
    cols = [df["key_rate"].diff()] + [df[s].diff() if s in fc.RATE_T else df[s] for s in series]
    return np.column_stack([c.to_numpy(float) for c in cols])


def output_names(series) -> list[tuple[str, int]]:
    return [(s, h) for s in series for h in HORIZONS]


def outputs(df: pd.DataFrame, series) -> np.ndarray:
    return np.column_stack([fc.outcome(df, s, h).to_numpy(float) for s, h in output_names(series)])


def contexts(ch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Все контексты длины CTX: массив [N, CTX, C] и позиции t0 (последний месяц контекста)."""
    if len(ch) < CTX:
        return np.empty((0, CTX, ch.shape[1]), np.float32), np.empty(0, int)
    win = sliding_window_view(ch, (CTX, ch.shape[1]))[:, 0]                # [T-CTX+1, CTX, C]
    pos = np.arange(CTX - 1, len(ch))
    ok = ~np.isnan(win).any(axis=(1, 2))
    return win[ok].astype(np.float32), pos[ok]


def synthetic_samples(panel: pd.DataFrame, series) -> tuple[np.ndarray, np.ndarray]:
    Xs, Ys = [], []
    for _, g in panel.groupby("run"):
        g = g.reset_index(drop=True)
        X, pos = contexts(channels(g, series))
        Y = outputs(g, series)[pos]
        keep = ~np.isnan(Y).all(axis=1)
        Xs.append(X[keep])
        Ys.append(Y[keep].astype(np.float32))
    return np.concatenate(Xs), np.concatenate(Ys)


class Stats:
    """Средние и разброс по реальной обучающей выборке (с нижними границами по типу ряда)."""

    def __init__(self, real: pd.DataFrame, series, start: pd.Period, Ytrain: np.ndarray):
        ch = channels(real.loc[: start - 1], series)
        self.c_mu = np.nanmean(ch, axis=0)
        floors = [SD_FLOOR["rate"]] + [SD_FLOOR[_kind(s)] for s in series]
        self.c_sd = np.maximum(np.nan_to_num(np.nanstd(ch, axis=0), nan=0.0), floors)
        self.c_mu = np.nan_to_num(self.c_mu)
        yfloor = [SD_FLOOR[_kind(s)] for s, _ in output_names(series)]
        if len(Ytrain):
            self.y_mu = np.nan_to_num(np.nanmean(Ytrain, axis=0))
            self.y_sd = np.maximum(np.nan_to_num(np.nanstd(Ytrain, axis=0), nan=0.0), yfloor)
        else:
            self.y_mu, self.y_sd = np.zeros(len(yfloor)), np.array(yfloor, float)

    def x(self, X: np.ndarray) -> np.ndarray:
        return np.clip((X - self.c_mu) / self.c_sd, -CLIP, CLIP).astype(np.float32)

    def y(self, Y: np.ndarray) -> np.ndarray:
        return ((Y - self.y_mu) / self.y_sd).astype(np.float32)

    def y_inv(self, Z: np.ndarray) -> np.ndarray:
        return Z * self.y_sd + self.y_mu


class Net(nn.Module):
    def __init__(self, c_in: int, n_out: int):
        super().__init__()
        self.proj = nn.Linear(c_in, D_MODEL)
        self.pos = nn.Parameter(torch.zeros(CTX, D_MODEL))
        layer = nn.TransformerEncoderLayer(D_MODEL, HEADS, FF, DROPOUT, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, LAYERS, enable_nested_tensor=False)
        self.head = nn.Linear(D_MODEL, n_out)

    def forward(self, x):
        return self.head(self.enc(self.proj(x) + self.pos)[:, -1])


def n_params(c_in: int, n_out: int) -> int:
    return sum(p.numel() for p in Net(c_in, n_out).parameters())


def _loss(pred, y):
    mask = ~torch.isnan(y)
    if not mask.any():
        return pred.sum() * 0.0
    return nn.functional.huber_loss(pred[mask], y[mask], delta=1.0)


def real_samples(window: CWindow, real: pd.DataFrame, series):
    """Реальные пары: (контексты, цели с маской: месяц цели раньше начала окна), метки t0, признак валидации."""
    ch, Y = channels(real, series), outputs(real, series)
    X, pos = contexts(ch)
    Yp = Y[pos].copy()
    t0 = real.index[pos]
    for j, (_, h) in enumerate(output_names(series)):
        Yp[:, j] = np.where(np.array([(p + h) < window.start for p in t0]), Yp[:, j], np.nan)
    keep = ~np.isnan(Yp).all(axis=1)
    X, Yp, t0 = X[keep], Yp[keep].astype(np.float32), t0[keep]
    is_val = np.array([p >= window.start - 12 for p in t0])
    return X, Yp, t0, is_val


def train_one(window: CWindow, real: pd.DataFrame, syn, w: float, seed: int, series) -> pd.DataFrame:
    """Обучает одну модель (вариант: доля синтетики ``w``, 0 — только реальные) и возвращает прогнозы окна."""
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    rng = np.random.default_rng([seed, int(w * 1000)])
    X, Y, t0, is_val = real_samples(window, real, series)
    Xtr, Ytr, Xv, Yv = X[~is_val], Y[~is_val], X[is_val], Y[is_val]
    if len(Xtr) == 0:
        raise RuntimeError(f"окно {window.name}: нет реальных обучающих пар")
    st = Stats(real, series, window.start, Ytr)
    Xtr_t, Ytr_t = torch.tensor(st.x(Xtr)), torch.tensor(st.y(Ytr))
    Xv_t, Yv_t = torch.tensor(st.x(Xv)), torch.tensor(st.y(Yv))
    if w > 0:
        Xs, Ys = syn
        Xs_t, Ys_t = torch.tensor(st.x(Xs)), torch.tensor(st.y(Ys))
    c_in, n_out = Xtr.shape[2], Ytr.shape[1]
    net = Net(c_in, n_out)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    n_syn = int(round(BATCH * w))
    n_real = BATCH - n_syn
    use_val = len(Xv) >= MIN_VAL
    steps = MAX_STEPS if use_val else NO_VAL_STEPS
    best, best_state = np.inf, None
    for step in range(1, steps + 1):
        net.train()
        ir = torch.as_tensor(rng.integers(0, len(Xtr), n_real)) if n_real else None
        parts_x = [Xtr_t[ir]] if n_real else []
        parts_y = [Ytr_t[ir]] if n_real else []
        if n_syn:
            isyn = torch.as_tensor(rng.integers(0, len(Xs_t), n_syn))
            parts_x.append(Xs_t[isyn]); parts_y.append(Ys_t[isyn])
        xb, yb = torch.cat(parts_x), torch.cat(parts_y)
        opt.zero_grad()
        _loss(net(xb), yb).backward()
        opt.step()
        if use_val and step % EVAL_EVERY == 0:
            net.eval()
            with torch.no_grad():
                v = float(_loss(net(Xv_t), Yv_t))
            if v < best:
                best, best_state = v, copy.deepcopy(net.state_dict())
    if use_val and best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    return _predict(window, real, series, net, st)


def _predict(window: CWindow, real: pd.DataFrame, series, net: Net, st: Stats) -> pd.DataFrame:
    ch = channels(real, series)
    names = output_names(series)
    idx = {p: i for i, p in enumerate(real.index)}
    rows = []
    for t in series:
        for h in HORIZONS:
            j = names.index((t, h))
            y_true = fc.outcome(real, t, h)
            for m in window.months:
                t0 = m - h
                if t0 not in idx or m not in idx or not np.isfinite(y_true.loc[t0]):
                    continue
                i = idx[t0]
                if i < CTX - 1:
                    continue
                ctx = ch[i - CTX + 1: i + 1]
                if np.isnan(ctx).any():
                    continue
                with torch.no_grad():
                    z = net(torch.tensor(st.x(ctx[None].astype(np.float32)))).numpy()[0]
                p = st.y_inv(z)[j]
                rows.append({"target": t, "h": h, "month": str(m), "forecast": float(p), "actual": float(y_true.loc[t0])})
    return pd.DataFrame(rows)


def forecast_m3(window: CWindow, real: pd.DataFrame, abm: pd.DataFrame, control: pd.DataFrame,
                shares=SHARES, seeds=(0, 1, 2), only_src: str | None = None) -> pd.DataFrame:
    """Прогнозы M3 по всем вариантам; среднее по зёрнам. ``only_src`` (real|abm|control) — для запуска по частям."""
    series = list(window.targets)
    syn = {"abm": synthetic_samples(abm, series), "control": synthetic_samples(control, series)}
    jobs = []
    if only_src in (None, "real"):
        jobs.append(("real", "real", 0.0))
    for src in ("abm", "control"):
        if only_src in (None, src):
            jobs += [(f"{src}@{w}", src, w) for w in shares]
    out = []
    for variant, src, w in jobs:
        per_seed = [train_one(window, real, syn.get(src), w, s, series) for s in seeds]
        d = per_seed[0][["target", "h", "month", "actual"]].copy()
        d["forecast"] = np.mean([p["forecast"].to_numpy() for p in per_seed], axis=0)
        for _, r in d.iterrows():
            out.append({"model": "M3", "window": window.name, "variant": variant, "target": r["target"], "h": int(r["h"]),
                        "month": r["month"], "forecast": r["forecast"], "actual": r["actual"],
                        "error": r["forecast"] - r["actual"]})
    return pd.DataFrame(out)
