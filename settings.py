real = [0.45, 0.14, 0.08, 0.06, 0.05,
        0.03, 0.02, 0.02, 0.02, 0.02,
        0.02, 0.01, 0.01, 0.01, 0.01,
        0.01, 0.01, 0.01, 0.01, 0.01]

uniform = [0.05] * 20
print(uniform)

settings = {"liquid_distribution": real,
            "deposit_amount_bound": (20, 41),
            "credit_amount_bound": (20, 41),
            "deposit_volume_bound": (10000, 500000),
            "credit_volume_bound": (10000, 500000),
            "deposit_maturity": [90, 180, 360, 720, 1800],
            "credit_maturity": [90, 180, 360, 720, 1800],
            "cb_rate": 0.1,
            "flow_distribution": 'uniform',
            "reserves_rate": 0.2,
            "num_steps": 10,
            "payment_period": [30, 90]

            }

