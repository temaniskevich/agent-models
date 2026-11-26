"""ПОИСКОВАЯ проверка (не входит в протокол; docs/CAPACITY_RESULTS.md, раздел 9): даёт ли линейная синтетика БЕЗ агентной модели
тот же выигрыш, что агентная синтетика и её линейный двойник.

Источник `realvar`: VARX(p, 6), подогнанный к РЕАЛЬНЫМ данным до начала окна (для B — 2014-01…2021-12, для E — 2014-01…2023-06; p по BIC из 1…3),
генерация на тех же 300 путях ключевой ставки, что у агентной панели θ = B (запуск 0), с бутстрэпом реальных остатков.
Обучение M1/M2/M3 — тем же кодом, что `twin@w` (источник передаётся как twin, затем вариант переименовывается). Только окна B и E.

    python scripts/capacity_realvar_explore.py --part make
    python scripts/capacity_realvar_explore.py --part m12      # без torch
    python scripts/capacity_realvar_explore.py --part m3       # без lightgbm
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np                                                     # noqa: E402
import pandas as pd                                                    # noqa: E402

from abm_banks import capacity as cp                                   # noqa: E402
from abm_banks import forecasting as fc                                # noqa: E402
from abm_banks.synthetic import TWIN_STATE, LinearTwin, panel_lp       # noqa: E402

OUT = Path("outputs/capacity")
SEED = 12345 + 3


def make() -> None:
    real = fc.load_real()
    abm, _ = cp.load_panels("B", OUT, 0)
    info = {}
    for name in ("B", "E"):
        w = cp.WINDOWS[name]
        r = real.loc[pd.Period("2014-01", "M"): w.start - 1, ["key_rate"] + TWIN_STATE].dropna().copy()
        r["run"], r["t"] = 0, np.arange(len(r))
        best = None
        for p in (1, 2, 3):                       # p_max = 3 → skip = max(3, 6) = 6 месяцев
            tw = LinearTwin(q=6, p_max=3, p=p).fit(r)
            n, k = tw.resid_.shape
            bic = tw._bic(tw.resid_, p * k + 7)
            if best is None or bic < best[0]:
                best = (bic, tw)
        tw = best[1]
        tw.fe_ = {run: tw.fe_[0] for run in abm["run"].unique()}     # одна реальная история → общая константа
        sim = tw.simulate(abm, seed=SEED)
        sim.to_parquet(OUT / f"panel_realvar_{name}.parquet")
        lp = panel_lp(sim, ["lr_c", "lr_r", "dr_c", "dr_r"], horizons=(3, 12))
        info[name] = {"p": tw.p, "max_root": tw.max_root_, "n_obs": tw.n_obs_, "lp": lp.to_dict(orient="records")}
        print(f"{name}: p {tw.p}, корень {tw.max_root_:.3f}, наблюдений {tw.n_obs_}; перенос h=3:",
              lp[lp.h == 3].set_index("series")["beta"].round(2).to_dict())
    (OUT / "realvar_check.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")


def _rename(df: pd.DataFrame) -> pd.DataFrame:
    df["variant"] = df["variant"].str.replace("twin@", "realvar@", regex=False)
    return df


def m12() -> None:
    real = fc.load_real()
    for name in ("B", "E"):
        w, p = cp.WINDOWS[name], pd.read_parquet(OUT / f"panel_realvar_{name}.parquet")
        for model, fn in (("M1", cp.forecast_m1), ("M2", cp.forecast_m2)):
            _rename(fn(w, real, None, None, twin=p, only_src="twin")).to_csv(OUT / f"{name}_{model}_realvar.csv", index=False)
            print(name, model, flush=True)


def m3() -> None:
    from abm_banks import capacity_m3 as m3mod
    real = fc.load_real()
    for name in ("B", "E"):
        w, p = cp.WINDOWS[name], pd.read_parquet(OUT / f"panel_realvar_{name}.parquet")
        df = m3mod.forecast_m3(w, real, None, None, seeds=(0, 1, 2), only_src="twin", twin=p)
        _rename(df).to_csv(OUT / f"{name}_M3_realvar.csv", index=False)
        print(name, "M3", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["make", "m12", "m3"], required=True)
    {"make": make, "m12": m12, "m3": m3}[ap.parse_args().part]()
