"""Парные прогнозные оценки и два теста чувствительности трансформера."""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
INDEX = ["model", "train_size", "source"]
ABLATIONS = ("twin", "no_interbank", "no_liquidity_pricing", "equal_banks", "weak_transfer")


def block_indices(windows):
    rng = np.random.default_rng(231121)
    draws, offset = [], 0
    for n in windows:
        blocks = rng.integers(n, size=(2000, (n + 2) // 3, 1))
        idx = (blocks + np.arange(3)[None, None, :]) % n
        draws.append(idx.reshape(2000, -1)[:, :n] + offset)
        offset += n
    return np.concatenate(draws, axis=1)


def descriptives(data):
    group = data.groupby(["scope", *INDEX]).sq_error.mean().pow(0.5)
    group.rename("rmse_pp").reset_index().to_csv(HERE / "rmse.csv", index=False)
    per_rate = data.groupby(["scope", "rate", *INDEX]).sq_error.mean().pow(0.5)
    per_rate.rename("rmse_pp").reset_index().to_csv(HERE / "rmse_by_rate.csv", index=False)
    by_seed = data.groupby(["scope", "seed", *INDEX]).sq_error.mean().pow(0.5)
    by_seed.rename("rmse_pp").reset_index().to_csv(HERE / "rmse_by_seed.csv", index=False)


def comparisons(data):
    augmentation, removed, interactions = [], [], []
    for scope, group in data.groupby("scope", sort=False):
        means = group.groupby(["window", "target", *INDEX]).sq_error.mean().unstack(INDEX)
        draws = block_indices(means.groupby(level=0).size().to_numpy())
        rmse = means.mean(axis=0).pow(0.5)
        sampled = pd.DataFrame(np.sqrt(means.to_numpy()[draws].mean(axis=1)), columns=means.columns)
        seed_rmse = group.groupby(["seed", *INDEX]).sq_error.mean().pow(0.5).unstack(INDEX)
        for model, size, source in means.columns:
            if source == "abm":
                a, b = (model, size, "abm"), (model, size, "real")
                frac = 1 - rmse[a] / rmse[b]
                dist = 1 - sampled[a] / sampled[b]
                across = 1 - seed_rmse[a] / seed_rmse[b]
                augmentation.append({"scope": scope, "model": model, "size": size, "rmse_real": rmse[b],
                                     "rmse_abm": rmse[a], "gain": frac, "lo90": dist.quantile(.05),
                                     "hi90": dist.quantile(.95), "seed_min": across.min(), "seed_max": across.max()})
            elif source in ABLATIONS:
                a, b = (model, size, source), (model, size, "abm")
                frac, dist = rmse[a] / rmse[b] - 1, sampled[a] / sampled[b] - 1
                across = seed_rmse[a] / seed_rmse[b] - 1
                removed.append({"scope": scope, "model": model, "source": source, "rmse_full_abm": rmse[b],
                                "rmse_variant": rmse[a], "loss_vs_full": frac, "lo90": dist.quantile(.05),
                                "hi90": dist.quantile(.95), "seed_min": across.min(), "seed_max": across.max()})

        def gain(x, model, size):
            return 1 - x[(model, size, "abm")] / x[(model, size, "real")]

        cases = [("width", s, "M3_64", s, "M3_32", s) for s in ("all", "few")]
        cases += [("less_real_data", m, m, "few", m, "all") for m in ("M3_32", "M3_64")]
        for name, level, m1, s1, m0, s0 in cases:
            point = gain(rmse, m1, s1) - gain(rmse, m0, s0)
            dist = gain(sampled, m1, s1) - gain(sampled, m0, s0)
            interactions.append({"scope": scope, "contrast": name, "level": level, "effect": point,
                                 "lo90": dist.quantile(.05), "hi90": dist.quantile(.95)})
    pd.DataFrame(augmentation).to_csv(HERE / "augmentation_effects.csv", index=False)
    pd.DataFrame(removed).to_csv(HERE / "ablation_effects.csv", index=False)
    pd.DataFrame(interactions).to_csv(HERE / "capacity_and_sample_effects.csv", index=False)


def main():
    predictions = pd.concat([pd.read_csv(HERE / f"predictions_{family}.csv")
                             for family in ("tabular", "neural")], ignore_index=True)
    keys = ["seed", "target", "rate", *INDEX]
    assert not predictions.duplicated(keys).any()
    assert predictions.groupby(INDEX).size().eq(33 * 4 * 3).all()
    assert predictions.groupby(["target", "rate"]).actual.nunique().eq(1).all()
    assert predictions.groupby(["target", "rate"]).naive.nunique().eq(1).all()
    data = predictions.assign(sq_error=(predictions.forecast - predictions.actual) ** 2)
    data = pd.concat([data.assign(scope=data.window), data.assign(scope="all")], ignore_index=True)
    descriptives(data)
    comparisons(data)
    baseline = data.drop_duplicates(["scope", "target", "rate"])
    mse = ((baseline.naive - baseline.actual) ** 2).groupby(baseline.scope).mean()
    mse.pow(.5).rename("rmse_pp").to_csv(HERE / "persistence_rmse.csv")


if __name__ == "__main__":
    main()
