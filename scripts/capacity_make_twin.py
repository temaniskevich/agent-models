"""Линейный двойник агентной панели (поправка 18.11.2026, docs/CAPACITY_PROTOCOL.md раздел 12).

    python scripts/capacity_make_twin.py --theta B            # запуск 0
    python scripts/capacity_make_twin.py --theta B --rep 3    # повтор проверки шума

Читает outputs/capacity/panel_abm_{θ}{_rk}.parquet, подгоняет VARX(p, 12) с фиксированными эффектами историй и пишет
panel_twin_{θ}{_rk}.parquet и описательную проверку двойника twin_check_{θ}{_rk}.json (p, корень, моменты, реакции на Δk).
Зерно двойника — 12345 + 1000·rep + 2.
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
from abm_banks.synthetic import TWIN_STATE, panel_lp                   # noqa: E402


def _ac1(panel: pd.DataFrame, c: str) -> float:
    v = []
    for _, g in panel.groupby("run"):
        x = g[c].to_numpy(float)
        if x.std() > 0:
            v.append(np.corrcoef(x[1:], x[:-1])[0, 1])
    return float(np.nanmean(v))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--theta", choices=["A", "B"], required=True)
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--dir", default="outputs/capacity")
    a = ap.parse_args()
    d, sfx = Path(a.dir), cp.rep_suffix(a.rep)
    abm, ctl = cp.load_panels(a.theta, d, a.rep)
    twin_panel, twin = fc.make_twin_panel(abm, seed=a.seed + 1000 * a.rep + 2)
    twin_panel.to_parquet(d / f"panel_twin_{a.theta}{sfx}.parquet")
    panels = {"abm": abm, "twin": twin_panel, "control": ctl}
    moments = [{"series": c, "source": s, "mean": float(p[c].mean()), "sd": float(p[c].std()), "ac1": _ac1(p, c)}
               for c in TWIN_STATE for s, p in panels.items()]
    lp = pd.concat([panel_lp(p, TWIN_STATE).assign(source=s) for s, p in panels.items()], ignore_index=True)
    check = {"theta": a.theta, "rep": a.rep, "p_bic": twin.p_bic_, "p": twin.p, "q": twin.q, "max_root": twin.max_root_,
             "n_obs": twin.n_obs_, "bic": {str(k): v for k, v in (twin.bic_ or {}).items()},
             "moments": moments, "lp": lp.to_dict(orient="records")}
    (d / f"twin_check_{a.theta}{sfx}.json").write_text(json.dumps(check, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"θ={a.theta} rep={a.rep}: p по BIC {twin.p_bic_}, p {twin.p}, корень {twin.max_root_:.4f}, панель {twin_panel.shape}")


if __name__ == "__main__":
    main()
