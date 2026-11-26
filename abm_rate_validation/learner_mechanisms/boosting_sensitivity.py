"""Проверка чувствительности LightGBM к весу синтетики и остановке по валидации."""

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from forecast_data import HERE, RULES, RATES, real_case, synthetic, tabular


def check(job):
    import lightgbm as lgb
    from abm_banks.capacity import LGB_PARAMS, LGB_ROUNDS

    window, seed = job
    data = real_case(window)
    xr, yr = tabular(data["train"][0]), data["train"][1]
    xv, yv = tabular(data["val"][0]), data["val"][1]
    xs, ys = synthetic(seed, "abm", data["lag"])
    xs, new = tabular(xs), tabular(data["new"])
    rows = []
    for share in (0.0, 0.1, 0.25, 0.5):
        x, y = (xr, yr) if share == 0 else (np.vstack((xr, xs)), np.vstack((yr, ys)))
        weights = np.ones(len(yr)) if share == 0 else np.repeat(
            [1.0, share * len(yr) / ((1 - share) * len(ys))], [len(yr), len(ys)])
        for j, rate in enumerate(RATES):
            train = lgb.Dataset(x, label=np.ascontiguousarray(y[:, j]), weight=weights)
            val = lgb.Dataset(xv, label=np.ascontiguousarray(yv[:, j]), reference=train)
            model = lgb.train({**LGB_PARAMS, "seed": seed}, train, num_boost_round=LGB_ROUNDS, valid_sets=[val],
                              callbacks=[lgb.early_stopping(40, verbose=False)])
            pred = data["anchor"][:, j] + model.predict(new, num_threads=1)
            rows.extend({"window": window, "seed": seed, "share": share, "target": str(month), "rate": rate,
                         "prediction": prediction, "actual": data["actual"][i, j], "trees": model.best_iteration}
                        for i, (month, prediction) in enumerate(zip(data["targets"], pred)))
    return rows


def main():
    jobs = [(w, seed) for w in RULES["target_windows"] for seed in RULES["seeds"]]
    with ProcessPoolExecutor(max_workers=4) as pool:
        d = pd.DataFrame([row for batch in pool.map(check, jobs) for row in batch])
    d.to_csv(HERE / "boosting_sensitivity_predictions.csv", index=False)
    d = d.assign(se=(d.prediction - d.actual) ** 2)
    rmse = pd.concat([d, d.assign(window="all")]).groupby(["window", "share"]).se.mean().pow(0.5).unstack("share")
    rmse.to_csv(HERE / "boosting_sensitivity.csv")
    print(rmse.round(4).to_string(), flush=True)


if __name__ == "__main__":
    main()
