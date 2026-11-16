"""Пакетные запуски: много прогонов, параллельность, наборы синтетических данных.

``run_many(config, seeds)`` → длинная таблица (``run``, ``seed``, ``t``, показатели).
Одинаковые конфиг и зёрна всегда дают идентичную таблицу.
"""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from .config import ModelConfig, get_by_path, set_param
from .model import BankingModel
from .scenarios import Scenario, build_scenario


def _scenario_to_dict(s: Scenario | None) -> dict | None:
    if s is None:
        return None
    return {"name": s.name, "key_rate": s.key_rate.tolist(), "credit_stress": s.credit_stress.tolist(),
            "confidence": s.confidence.tolist(), "deposit_shock": s.deposit_shock.tolist(),
            "demand_shock": s.demand_shock.tolist(), "meta": s.meta}


def _scenario_from_dict(d: dict | None) -> Scenario | None:
    if d is None:
        return None
    return Scenario(name=d["name"], key_rate=np.array(d["key_rate"]),
                    credit_stress=np.array(d["credit_stress"]), confidence=np.array(d["confidence"]),
                    deposit_shock=np.array(d["deposit_shock"]), demand_shock=np.array(d["demand_shock"]),
                    meta=d.get("meta", {}))


def run_one(config: ModelConfig | dict, seed: int, scenario: Scenario | dict | None = None,
            run_id: int = 0, keep_banks: bool = False) -> pd.DataFrame:
    """Один прогон → таблица сектора с колонками ``run``/``seed``."""
    cfg = ModelConfig.from_dict(config) if isinstance(config, dict) else config
    if not keep_banks:
        cfg = cfg.replace(record_banks=False)
    scn = _scenario_from_dict(scenario) if isinstance(scenario, dict) else scenario
    res = BankingModel(cfg, scenario=scn, seed=seed).run()
    df = res.sector.reset_index()
    df["date"] = df["date"].astype(str)
    df.insert(0, "seed", seed)
    df.insert(0, "run", run_id)
    return df


def _worker(args) -> pd.DataFrame:
    cfg_dict, seed, scn_dict, run_id = args
    return run_one(cfg_dict, seed, scn_dict, run_id)


def run_many(config: ModelConfig, seeds: Sequence[int] | int, n_jobs: int = 1,
             scenarios: Sequence[Scenario | None] | None = None,
             param_sets: Sequence[dict] | None = None,
             progress: Callable[[int, int], None] | None = None) -> pd.DataFrame:
    """Много прогонов.

    * ``seeds`` — список зёрен или число прогонов (зёрна 0..n-1);
    * ``scenarios`` — свой сценарий на каждый прогон (иначе строится из конфига
      по зерну, то есть случайные сценарии различаются между прогонами);
    * ``param_sets`` — словари вида ``{"banks.car_min": 0.09}`` для каждого прогона;
    * ``n_jobs`` — число процессов (1 = последовательно).
    """
    seeds = list(range(seeds)) if isinstance(seeds, int) else list(seeds)
    n = len(seeds)
    if scenarios is not None and len(scenarios) != n:
        raise ValueError("число сценариев должно совпадать с числом зёрен")
    if param_sets is not None and len(param_sets) != n:
        raise ValueError("число наборов параметров должно совпадать с числом зёрен")
    jobs = []
    for i, seed in enumerate(seeds):
        cfg = config.replace(**param_sets[i]) if param_sets else config
        jobs.append((cfg.to_dict(), seed,
                     _scenario_to_dict(scenarios[i]) if scenarios else None, i))
    frames: list[pd.DataFrame] = []
    if n_jobs == 1 or n < 4:
        for k, job in enumerate(jobs):
            frames.append(_worker(job))
            if progress:
                progress(k + 1, n)
    else:
        workers = min(n_jobs if n_jobs > 0 else (os.cpu_count() or 2), n)
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for k, df in enumerate(ex.map(_worker, jobs, chunksize=max(1, n // (workers * 4)))):
                frames.append(df)
                if progress:
                    progress(k + 1, n)
    out = pd.concat(frames, ignore_index=True)
    out.attrs["config"] = config.to_dict()
    return out


# --------------------------------------------------------------------------
# Синтетика (шаг 4 плана)
# --------------------------------------------------------------------------
DEFAULT_PARAM_RANGES: dict[str, tuple[float, float]] = {
    # границы выбраны вокруг базовых значений; после калибровки замените на доверительные
    "demand.loan_rate_elasticity.0": (1.5, 4.0),
    "demand.loan_rate_elasticity.1": (1.0, 3.0),
    "demand.deposit_rate_semielasticity.0": (0.5, 1.5),
    "demand.deposit_rate_semielasticity.1": (0.4, 1.2),
    "banks.rate_speed_loan": (0.15, 0.50),
    "banks.rate_speed_deposit": (0.20, 0.60),
    "credit.rate_sensitivity.0": (8.0, 20.0),
    "credit.rate_sensitivity.1": (6.0, 18.0),
    "confidence.run_panic": (0.10, 0.30),
    "banks.car_target_mean": (0.115, 0.14),
}


def sample_param_sets(n: int, ranges: dict[str, tuple[float, float]], seed: int) -> list[dict]:
    """Равномерная выборка параметров из заданных диапазонов (Latin hypercube)."""
    rng = np.random.default_rng(seed)
    keys = list(ranges)
    u = np.empty((n, len(keys)))
    for j in range(len(keys)):
        u[:, j] = (rng.permutation(n) + rng.random(n)) / n
    sets = []
    for i in range(n):
        sets.append({k: float(ranges[k][0] + u[i, j] * (ranges[k][1] - ranges[k][0]))
                     for j, k in enumerate(keys)})
    return sets


def generate_synthetic(config: ModelConfig, n_runs: int, base_seed: int = 12345,
                       param_ranges: dict[str, tuple[float, float]] | None = None,
                       key_rate_start_range: tuple[float, float] = (0.05, 0.12),
                       n_jobs: int = 1, outdir: str | Path | None = None,
                       progress: Callable[[int, int], None] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Набор синтетических историй: случайные параметры + случайные пути ставки.

    Каждый прогон получает собственное зерно, набор параметров и сценарий
    ``random_jumps``. Возвращает ``(данные, метаданные)``; при ``outdir`` сохраняет
    ``synthetic.parquet`` и ``meta.parquet`` + ``manifest.json``.
    """
    ss = np.random.SeedSequence(base_seed)
    seeds = [int(s.generate_state(1)[0]) for s in ss.spawn(n_runs)]
    param_sets = sample_param_sets(n_runs, param_ranges or DEFAULT_PARAM_RANGES, base_seed + 1)
    rng = np.random.default_rng(base_seed + 2)
    scenarios, meta_rows = [], []
    cfg = config.replace(**{"scenario.kind": "random_jumps"})
    for i, seed in enumerate(seeds):
        k0 = float(rng.uniform(*key_rate_start_range))
        scn_cfg = cfg.scenario
        scn_cfg.key_rate_start = k0
        scn = build_scenario(scn_cfg, cfg.n_months, np.random.default_rng(seed + 7))
        scenarios.append(scn)
        meta_rows.append({"run": i, "seed": seed, "key_rate_start": k0,
                          "n_jumps": len(scn.meta.get("events", [])),
                          "max_jump": max([abs(e["size"]) for e in scn.meta.get("events", [])] or [0.0]),
                          **param_sets[i]})
    t0 = time.time()
    data = run_many(cfg, seeds, n_jobs=n_jobs, scenarios=scenarios, param_sets=param_sets,
                    progress=progress)
    meta = pd.DataFrame(meta_rows)
    if outdir is not None:
        out = Path(outdir)
        out.mkdir(parents=True, exist_ok=True)
        data.to_parquet(out / "synthetic.parquet", index=False)
        meta.to_parquet(out / "meta.parquet", index=False)
        manifest = {"n_runs": n_runs, "base_seed": base_seed, "elapsed_sec": round(time.time() - t0, 1),
                    "param_ranges": param_ranges or DEFAULT_PARAM_RANGES, "config": cfg.to_dict(),
                    "key_rate_start_range": key_rate_start_range}
        (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
    return data, meta
