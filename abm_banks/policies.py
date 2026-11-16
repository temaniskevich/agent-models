"""Поведенческие политики банков: точка расширения для LLM-агентов (план, шаг 5).

Раз в месяц модель вызывает ``policy.decide(snapshot)`` и получает поправки к
правилам поведения каждого банка. Базовая политика (``RuleBasedPolicy``) не
меняет ничего — банки действуют по встроенным правилам. Любая другая политика
(включая языковую модель) должна вернуть ``PolicyAdjustments``.

Поправки намеренно ограничены (клиппинг в модели), чтобы ошибка внешней модели не
ломала балансовые тождества: политика влияет на *решения* (спреды, цель по
капиталу, порог доверия на МБК), но не на бухгалтерию.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np


@dataclass
class PolicyAdjustments:
    """Поправки по банкам (массивы длины N)."""
    loan_spread: np.ndarray            # добавка к кредитной ставке, доли годовых
    deposit_spread: np.ndarray         # добавка к ставке по вкладам, доли годовых
    car_target: np.ndarray             # сдвиг целевого CAR
    mbk_threshold: np.ndarray          # сдвиг порога надёжности контрагента на МБК

    @classmethod
    def zeros(cls, n: int) -> "PolicyAdjustments":
        z = lambda: np.zeros(n)    # noqa: E731
        return cls(z(), z(), z(), z())

    def clipped(self) -> "PolicyAdjustments":
        return PolicyAdjustments(
            np.clip(np.nan_to_num(self.loan_spread), -0.03, 0.05),
            np.clip(np.nan_to_num(self.deposit_spread), -0.03, 0.05),
            np.clip(np.nan_to_num(self.car_target), -0.03, 0.05),
            np.clip(np.nan_to_num(self.mbk_threshold), -0.2, 0.3),
        )


class BehaviorPolicy(Protocol):
    name: str

    def decide(self, snapshot: dict) -> PolicyAdjustments: ...


class RuleBasedPolicy:
    """Базовая политика: никаких поправок (чистые правила модели)."""
    name = "rule"

    def decide(self, snapshot: dict) -> PolicyAdjustments:
        return PolicyAdjustments.zeros(snapshot["n_banks"])


class CautiousPolicy:
    """Пример простой эвристики: при падении CAR ниже цели банк ужесточает условия.

    Нужна как образец и как тест точки расширения без обращения к LLM.
    """
    name = "cautious"

    def __init__(self, strength: float = 1.0):
        self.strength = strength

    def decide(self, snapshot: dict) -> PolicyAdjustments:
        n = snapshot["n_banks"]
        adj = PolicyAdjustments.zeros(n)
        gap = np.clip(snapshot["car_target"] - snapshot["car"], 0, 0.05)
        adj.loan_spread = self.strength * 0.4 * gap
        adj.deposit_spread = self.strength * 0.2 * gap
        adj.mbk_threshold = self.strength * 2.0 * gap
        return adj.clipped()


class LLMPolicy:
    """Решения банков принимает языковая/decision-модель.

    ``complete`` — любая функция ``prompt:str -> str`` (OpenAI-совместимый клиент,
    локальная модель и т. д.). Модель получает текстовую сводку о положении банка
    и отвечает JSON-ом с четырьмя числами. Ответы кэшируются по тексту промпта,
    при ошибке разбора берётся нулевая поправка. Температуру 0 и версию модели
    фиксируйте на стороне ``complete`` — модель сохраняет промпты в ``self.log``.
    """
    name = "llm"

    PROMPT = (
        "Ты — риск-менеджер банка. Положение банка:\n"
        "- ключевая ставка ЦБ: {key_rate:.1f}%\n"
        "- достаточность капитала: {car:.1f}% (цель {car_target:.1f}%, минимум {car_min:.1f}%)\n"
        "- ликвидные активы к вкладам: {liq:.1f}% (цель {liq_target:.1f}%)\n"
        "- обращение к ЦБ в прошлом месяце: {cb_use}\n"
        "- доля проблемных кредитов за месяц: {npl:.2f}%\n"
        "- число банков, лишившихся лицензии за 3 месяца: {failures}\n"
        "Ответь ТОЛЬКО JSON вида {{\"loan_spread_pp\": число, \"deposit_spread_pp\": число, "
        "\"car_target_pp\": число, \"mbk_threshold\": число}}: поправки к кредитной ставке, "
        "ставке по вкладам (в п.п.), к цели по CAR (в п.п.) и к порогу надёжности "
        "контрагента на МБК (от -0.2 до 0.3)."
    )

    def __init__(self, complete: Callable[[str], str], max_banks_per_call: int | None = None):
        self.complete = complete
        self.cache: dict[str, dict] = {}
        self.log: list[dict] = []
        self.max_banks_per_call = max_banks_per_call

    def _parse(self, text: str) -> dict:
        m = re.search(r"\{.*?\}", text, flags=re.S)
        if not m:
            return {}
        try:
            data = json.loads(m.group(0))
            return {k: float(v) for k, v in data.items()
                    if isinstance(v, (int, float))}
        except (ValueError, TypeError):
            return {}

    def decide(self, snapshot: dict) -> PolicyAdjustments:
        n = snapshot["n_banks"]
        adj = PolicyAdjustments.zeros(n)
        limit = self.max_banks_per_call or n
        for i in range(min(n, limit)):
            if not snapshot["active"][i]:
                continue
            prompt = self.PROMPT.format(
                key_rate=snapshot["key_rate"] * 100,
                car=snapshot["car"][i] * 100, car_target=snapshot["car_target"][i] * 100,
                car_min=snapshot["car_min"] * 100,
                liq=snapshot["liq_ratio"][i] * 100, liq_target=snapshot["liq_target"][i] * 100,
                cb_use="да" if snapshot["cb_use"][i] else "нет",
                npl=snapshot["npl"][i] * 100, failures=snapshot["recent_failures"])
            if prompt not in self.cache:
                try:
                    reply = self.complete(prompt)
                except Exception as exc:               # сетевые ошибки не должны ронять прогон
                    reply = ""
                    self.log.append({"month": snapshot["month"], "bank": i, "error": str(exc)})
                self.cache[prompt] = self._parse(reply)
                self.log.append({"month": snapshot["month"], "bank": i,
                                 "prompt": prompt, "reply": reply})
            d = self.cache[prompt]
            adj.loan_spread[i] = d.get("loan_spread_pp", 0.0) / 100.0
            adj.deposit_spread[i] = d.get("deposit_spread_pp", 0.0) / 100.0
            adj.car_target[i] = d.get("car_target_pp", 0.0) / 100.0
            adj.mbk_threshold[i] = d.get("mbk_threshold", 0.0)
        return adj.clipped()


def get_policy(name: str | BehaviorPolicy | None) -> BehaviorPolicy:
    if name is None or name == "rule":
        return RuleBasedPolicy()
    if isinstance(name, str):
        if name == "cautious":
            return CautiousPolicy()
        raise KeyError(f"неизвестная политика {name!r}; передайте объект LLMPolicy(...)")
    return name
