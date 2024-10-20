real = [0.45, 0.14, 0.08, 0.06, 0.05,
        0.03, 0.02, 0.02, 0.02, 0.02,
        0.02, 0.01, 0.01, 0.01, 0.01,
        0.01, 0.01, 0.01, 0.01, 0.01]

uniform = [0.05] * 20

settings = {"liquid_distribution": real,
            "deposit_amount_bound": (10, 1000),
            "credit_amount_bound": (10, 1000),
            "deposit_volume_bound": (10000, 10000000),
            "credit_volume_bound": (10000, 10000000),
            "deposit_maturity": [180, 360, 720, 1080],
            "credit_maturity": [180, 360, 720, 1080],
            "mbk_maturity": [90, 180],
            "cb_maturity": [90,180],
            "cb_rate": 0.1,
            "cb_reserve_rate": 0.2,
            "flow_distribution": 'uniform',
            "num_steps": 10,
            "payment_period": [30, 90],
            "bank_start_cash": 1e11,
            "bank_fixed_costs": 0,
            "bank_operating_costs": 0.02

            }

