"""Сравнение прогнозистов и синтетических панелей с общими выпусками."""

import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from forecast_data import HERE, RULES, RATES, SOURCES, real_case, synthetic, tabular


def tabular_predict(model, train, syn, new, seed):
    X, y = train
    n = len(y)
    if syn is None:
        combined, target, weights = X, y, np.ones(n)
    else:
        xs, ys = syn
        p = RULES["share_synthetic"]
        combined = np.vstack([X, xs])
        target = np.vstack([y, ys])
        weights = np.repeat([1.0, (p / (1 - p)) * n / len(ys)], [n, len(ys)])
    if model == "ARX":
        center, scale = X.mean(axis=0), np.maximum(X.std(axis=0), 0.25)
        z = np.column_stack([np.ones(len(combined)), (combined - center) / scale])
        z_new = np.column_stack([np.ones(len(new)), (new - center) / scale])
        penalty = 0.01 * np.eye(z.shape[1])
        penalty[0, 0] = 0.0
        cross = z.T * (weights / weights.sum())
        return z_new @ np.linalg.solve(cross @ z + penalty, cross @ target)

    import lightgbm as lgb
    from abm_banks.capacity import LGB_PARAMS, LGB_ROUNDS

    columns = []
    for outcome in range(target.shape[1]):
        pool = lgb.Dataset(combined, label=target[:, outcome], weight=weights)
        learner = lgb.train({**LGB_PARAMS, "seed": seed}, pool, num_boost_round=LGB_ROUNDS)
        columns.append(learner.predict(new, num_threads=1))
    return np.column_stack(columns)


def make_net(model, inputs):
    import torch
    from torch import nn

    p = RULES["neural_training"]
    if model == "MLP":
        return nn.Sequential(nn.Flatten(), nn.Linear(RULES["context"] * inputs, p["neurons_mlp"]), nn.ReLU(),
                             nn.Dropout(p["dropout"]), nn.Linear(p["neurons_mlp"], len(RATES)))

    width = int(model.split("_")[1])

    class Transformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(inputs, width)
            self.pos = nn.Parameter(torch.zeros(RULES["context"], width))
            layer = nn.TransformerEncoderLayer(width, 2, 4 * width, p["dropout"], batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
            self.head = nn.Linear(width, len(RATES))

        def forward(self, x):
            return self.head(self.encoder(self.proj(x) + self.pos)[:, -1])

    return Transformer()


def neural_predict(model, train, val, syn, new, seed):
    import torch

    torch.set_num_threads(1)
    torch.manual_seed(seed)
    real_rng, syn_rng = np.random.default_rng(seed), np.random.default_rng(seed + 5000)
    p = RULES["neural_training"]
    Xr, Yr = train
    mu_x, mu_y = Xr.mean(axis=(0, 1)), Yr.mean(axis=0)
    sd_x, sd_y = np.maximum(Xr.std(axis=(0, 1)), 0.25), np.maximum(Yr.std(axis=0), 0.25)

    def scale_x(x):
        return torch.from_numpy(np.clip((x - mu_x) / sd_x, -8.0, 8.0))

    def scale_y(y):
        return torch.from_numpy((y - mu_y) / sd_y)

    xr, yr = scale_x(Xr), scale_y(Yr)
    xv, yv = scale_x(val[0]), scale_y(val[1])
    batch = p["batch"]
    n_syn = 0 if syn is None else round(batch * RULES["share_synthetic"])
    if n_syn:
        xs, ys = scale_x(syn[0]), scale_y(syn[1])
    net = make_net(model, Xr.shape[-1])
    params = sum(a.numel() for a in net.parameters())
    optim = torch.optim.Adam(net.parameters(), lr=p["lr"])
    best_loss, best_step, best = np.inf, 0, None
    for step in range(1, p["steps"] + 1):
        net.train()
        ri = real_rng.integers(0, len(xr), batch)[:batch - n_syn]
        x, y = xr[ri], yr[ri]
        if n_syn:
            si = syn_rng.integers(0, len(xs), n_syn)
            x, y = torch.cat((x, xs[si])), torch.cat((y, ys[si]))
        optim.zero_grad()
        loss = torch.nn.functional.huber_loss(net(x), y)
        assert torch.isfinite(loss), (model, step)
        loss.backward()
        optim.step()
        if step % p["eval_every"] == 0:
            net.eval()
            with torch.no_grad():
                val_loss = float(torch.nn.functional.huber_loss(net(xv), yv))
                if val_loss < best_loss:
                    best_loss, best_step = val_loss, step
                    best = {k: v.clone() for k, v in net.state_dict().items()}
    assert best is not None
    net.load_state_dict(best)
    net.eval()
    with torch.no_grad():
        pred = net(scale_x(new)).numpy() * sd_y + mu_y
    return pred, params, best_step


def cases(family):
    if family == "tabular":
        return [(m, "all", src) for m in ("ARX", "LightGBM") for src in SOURCES]
    base = [("M3_32", "all", src) for src in SOURCES]
    wide = [(m, "all", src) for m in ("M3_64", "MLP") for src in ("real", "abm")]
    few = [(m, "few", src) for m in ("M3_32", "M3_64") for src in ("real", "abm")]
    return base + wide + few


def run(job):
    window, seed, family = job
    d = real_case(window)
    train, val, x_new = d["train"], d["val"], d["new"]
    test_cases = cases(family)
    pools = {src: synthetic(seed, src, d["lag"]) for src in set(x[2] for x in test_cases) - {"real"}}
    if family == "tabular":
        train, x_new = (tabular(train[0]), train[1]), tabular(x_new)
        pools = {src: (tabular(x), y) for src, (x, y) in pools.items()}
    few = np.rint(np.linspace(0, len(train[1]) - 1, RULES["few_real_pairs"])).astype(int)
    assert len(np.unique(few)) == len(few)
    rows = []
    for model, size, src in test_cases:
        picked = train if size == "all" else (train[0][few], train[1][few])
        if family == "tabular":
            delta = tabular_predict(model, picked, pools.get(src), x_new, seed)
            params, step = None, None
        else:
            delta, params, step = neural_predict(model, picked, val, pools.get(src), x_new, seed)
        prediction = d["anchor"] + delta
        assert np.isfinite(prediction).all()
        for i, month in enumerate(d["targets"]):
            for j, rate in enumerate(RATES):
                rows.append({"window": window, "seed": seed, "model": model, "train_size": size, "source": src,
                             "target": str(month), "rate": rate, "actual": d["actual"][i, j],
                             "forecast": prediction[i, j], "naive": d["anchor"][i, j], "lag": d["lag"],
                             "train_pairs": len(picked[1]), "last_train_label": d["last_train_label"],
                             "first_release_through": d["first_release_through"], "params": params, "best_step": step})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("tabular", "neural"), required=True)
    parser.add_argument("--jobs", type=int, default=6)
    args = parser.parse_args()
    jobs = [(w, seed, args.family) for w in RULES["target_windows"] for seed in RULES["seeds"]]
    predictions = []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for job, rows in zip(jobs, pool.map(run, jobs), strict=True):
            predictions.extend(rows)
            print(f"Завершены {job[2]}, окно {job[0]}, зерно {job[1]}", flush=True)
    pd.DataFrame(predictions).to_csv(HERE / f"predictions_{args.family}.csv", index=False)


if __name__ == "__main__":
    main()
