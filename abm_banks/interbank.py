"""Рынок межбанковского кредитования (МБК): поиск партнёров и формирование сети.

Механизм (развитие случайного выбора кредитора из ВКР):
* заёмщики идут по очереди в случайном порядке;
* кредитор соглашается с вероятностью, растущей с надёжностью заёмщика
  (порог терпимости к риску у каждого банка свой и растёт при стрессе — «накопление
  ликвидности»);
* из согласных кредиторов выбирают в случайном порядке с весом, который выше у
  прежних партнёров (устойчивость связей);
* объём ограничен свободной ликвидностью кредитора, лимитом МБК в его капитале и
  лимитом на одного контрагента;
* ставка зависит от надёжности заёмщика; если она выше ставки рефинансирования
  ЦБ, заёмщик идёт к ЦБ.
"""
from __future__ import annotations

import numpy as np


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def match_interbank(
    rng: np.random.Generator,
    need: np.ndarray,
    supply: np.ndarray,
    exposure_room: np.ndarray,
    counterparty_limit: np.ndarray,
    trust: np.ndarray,
    threshold: np.ndarray,
    borrower_rate: np.ndarray,
    max_rate: np.ndarray,
    prev_links: np.ndarray,
    persistence: float,
    max_lenders: int,
    eps: float = 1e-9,
) -> np.ndarray:
    """Возвращает матрицу ``M[i, j]`` — сколько банк ``i`` одолжил банку ``j``.

    Параметры (все по банкам, длина N):
    ``need`` — потребность заёмщика; ``supply`` — готовность кредитора дать;
    ``exposure_room`` — остаток лимита МБК у кредитора;
    ``counterparty_limit`` — лимит кредитора на одного заёмщика;
    ``trust`` — надёжность; ``threshold`` — порог надёжности для кредитора;
    ``borrower_rate``/``max_rate`` — ставка МБК для заёмщика и предел, который он
    готов платить; ``prev_links[i, j]`` — была ли связь в прошлом месяце.
    """
    n = len(need)
    M = np.zeros((n, n))
    rem_supply = supply.astype(float).copy()
    rem_room = exposure_room.astype(float).copy()
    borrowers = np.flatnonzero(need > eps)
    if len(borrowers) == 0 or rem_supply.sum() <= eps:
        return M
    rng.shuffle(borrowers)
    idx = np.arange(n)
    for b in borrowers:
        if borrower_rate[b] > max_rate[b]:
            continue                       # дешевле взять у ЦБ
        accept_p = sigmoid((trust[b] - threshold) / 0.08)
        elig = (rem_supply > eps) & (rem_room > eps) & (idx != b)
        elig &= rng.random(n) < accept_p
        cand = np.flatnonzero(elig)
        if len(cand) == 0:
            continue
        weights = 1.0 + persistence * prev_links[cand, b]
        keys = rng.exponential(size=len(cand)) / weights
        order = cand[np.argsort(keys)][:max_lenders]
        remaining = float(need[b])
        for i in order:
            amt = min(remaining, rem_supply[i], rem_room[i], counterparty_limit[i] - M[i, b])
            if amt <= eps:
                continue
            M[i, b] += amt
            rem_supply[i] -= amt
            rem_room[i] -= amt
            remaining -= amt
            if remaining <= eps:
                break
    return M


def network_edges(M: np.ndarray, min_amount: float = 1e-6):
    """Список ребер сети (кредитор, заёмщик, объём) из матрицы МБК."""
    lenders, borrowers = np.nonzero(M > min_amount)
    return lenders, borrowers, M[lenders, borrowers]
