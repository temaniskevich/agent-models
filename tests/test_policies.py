import json

import numpy as np

from abm_banks import BankingModel, ModelConfig, preset
from abm_banks.policies import CautiousPolicy, LLMPolicy, PolicyAdjustments, RuleBasedPolicy


def test_rule_policy_is_identity():
    cfg = ModelConfig(n_months=24)
    a = BankingModel(cfg, seed=1, policy=RuleBasedPolicy()).run().sector
    b = BankingModel(cfg, seed=1).run().sector
    assert a.equals(b)


def test_cautious_policy_changes_behaviour_and_keeps_identity():
    cfg = preset("rate_shock").replace(n_months=48, check_identities=True)
    base = BankingModel(cfg, seed=2).run().sector
    caut = BankingModel(cfg, seed=2, policy="cautious").run().sector
    assert not np.allclose(base["loan_rate"], caut["loan_rate"])


def test_llm_policy_parses_clips_and_survives_errors():
    calls = []

    def fake(prompt):
        calls.append(prompt)
        if len(calls) % 5 == 0:
            raise RuntimeError("сеть недоступна")
        if len(calls) % 3 == 0:
            return "не JSON"
        return 'Ответ: {"loan_spread_pp": 100, "deposit_spread_pp": -1, "car_target_pp": 0.5, "mbk_threshold": 0.1}'

    pol = LLMPolicy(fake)
    cfg = ModelConfig(n_months=6, check_identities=True)
    res = BankingModel(cfg, seed=1, policy=pol).run()
    assert len(res.sector) == 6 and len(pol.log) > 0
    adj = pol.decide({"n_banks": 2, "month": 0, "key_rate": 0.1, "active": np.array([True, True]),
                      "car": np.array([0.12, 0.1]), "car_target": np.array([0.125, 0.125]),
                      "car_min": 0.08, "liq_ratio": np.array([0.06, 0.05]),
                      "liq_target": np.array([0.06, 0.06]), "cb_use": np.array([False, True]),
                      "npl": np.array([0.003, 0.01]), "recent_failures": 0})
    assert adj.loan_spread.max() <= 0.05            # 100 п.п. обрезаны до допустимого
    assert isinstance(adj, PolicyAdjustments)


def test_llm_policy_caches_identical_prompts():
    seen = []

    def fake(prompt):
        seen.append(prompt)
        return json.dumps({"loan_spread_pp": 0.1})

    pol = LLMPolicy(fake)
    snap = {"n_banks": 3, "month": 0, "key_rate": 0.1, "active": np.ones(3, bool),
            "car": np.full(3, 0.12), "car_target": np.full(3, 0.125), "car_min": 0.08,
            "liq_ratio": np.full(3, 0.06), "liq_target": np.full(3, 0.06),
            "cb_use": np.zeros(3, bool), "npl": np.full(3, 0.003), "recent_failures": 0}
    pol.decide(snap)
    assert len(seen) == 1                                 # одинаковые банки → один вызов
