"""Генерация набора синтетических историй (шаг 4 плана) без интерфейса.

Пример:
    python scripts/generate_synthetic.py --runs 2000 --months 180 --jobs 8 --out outputs/synthetic_v1
    python scripts/generate_synthetic.py --runs 500 --theta theta_window_B.json --control

Сохраняет synthetic.parquet, meta.parquet и manifest.json (конфиг, зёрна, диапазоны параметров).
С ``--control`` дополнительно строится контрольная выборка «без экономики» (AR + скачки).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from abm_banks import ModelConfig                                   # noqa: E402
from abm_banks.batch import DEFAULT_PARAM_RANGES, generate_synthetic  # noqa: E402
from abm_banks.synthetic import ARJumpControl                       # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--months", type=int, default=180)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out", default="outputs/synthetic")
    ap.add_argument("--config", help="JSON/YAML конфиг модели (иначе значения по умолчанию)")
    ap.add_argument("--theta", help="JSON с откалиброванными параметрами (путь→значение)")
    ap.add_argument("--ranges", help="JSON с диапазонами параметров {путь: [от, до]}")
    ap.add_argument("--control", action="store_true", help="сгенерировать контрольную выборку без экономики")
    a = ap.parse_args()

    cfg = ModelConfig()
    if a.config:
        cfg = ModelConfig.from_json(a.config) if a.config.endswith(".json") else ModelConfig.from_yaml(a.config)
    if a.theta:
        cfg = cfg.replace(**json.loads(Path(a.theta).read_text(encoding="utf-8")).get("theta", {}))
    cfg = cfg.replace(n_months=a.months, record_banks=False, record_network=False)
    ranges = {k: tuple(v) for k, v in json.loads(Path(a.ranges).read_text()).items()} if a.ranges else DEFAULT_PARAM_RANGES

    out = Path(a.out)
    data, meta = generate_synthetic(cfg, a.runs, a.seed, ranges, n_jobs=a.jobs, outdir=out,
                                    progress=lambda i, n: print(f"\rпрогон {i}/{n}", end="", flush=True))
    print(f"\nготово: {len(data):,} строк → {out}")
    if a.control:
        ctl = ARJumpControl().fit(data)
        ctrl = ctl.simulate(a.months, a.runs, seed=a.seed + 99)
        ctrl.to_parquet(out / "control.parquet", index=False)
        ctl.moments_report(data).to_csv(out / "control_moments.csv", index=False)
        print("контрольная выборка:", out / "control.parquet")


if __name__ == "__main__":
    main()
