"""Simulation experiments E1-E5 (run from the project root):

    python -m research.sim_experiments e1 e2 e3 e4 e5 [--seeds 8] [--T 3600]

E1 information speed & liquidity     arms vary market-maker depth and informed-trader speed
E2 calibration                       24 independent binary markets per run; longshot-biased noise flow on/off,
                                     our MM on/off, informed share
E3 microstructure predictability     imbalance / order-flow features -> next 30 s mid change (out of sample)
E4 related-market arbitrage          incidence and persistence of executable arbitrage with/without arbitrageurs
                                     and fees (read-only LP monitor)
E5 market-maker design               our MM variants vs naive fixed-spread MM and LMSR AMM

Every arm x seed is an independent run; results land in runs/<stamp>_<experiment>/ with a manifest.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"   # before numpy loads: one BLAS thread per worker process

from pmkt.backtest.runner import run_experiment  # noqa: E402

KX_ONLY_FEES = {"KX": {"fee": {"kind": "quadratic", "taker_rate": 0.07, "maker_rate": 0.0}, "msg_rate": 10,
                       "conversion": False}}


def e1_arms(T):
    base = {"T": T}
    return {
        "baseline": base,
        "no_our_mm": {**base, "strategies": {"mm": {"enabled": False}}},
        "thin": {**base, "agents": {"naive_mm": {"size": 5}, "amm": {"venues": []}}, "strategies": {"mm": {"enabled": False}}},
        "deep": {**base, "agents": {"naive_mm": {"size": 40}},
                 "strategies": {"mm": {"params": {"size": 60, "levels": 3}}}},
        "fast_informed": {**base, "agents": {"informed": {"n": 10, "react_delay": [0.2, 1.0]}}},
        "slow_informed": {**base, "agents": {"informed": {"n": 3, "react_delay": [5.0, 20.0]}}},
    }


def e2_arms(T):
    fams = [{"type": "binary", "id": f"B{i:02d}", "p0_range": [0.03, 0.97], "total_var": 1.0, "jump_share": 0.25,
             "n_jumps": 4, "title": f"Independent event {i}"} for i in range(24)]
    base = {"T": T, "families": fams, "terminal_share": 0.6, "listings": {"KX": [f["id"] for f in fams], "PM": []},
            "agents": {"amm": {"venues": []}, "noise": {"n": 12}},
            "strategies": {"arb": {"enabled": False}, "mispricing": {"enabled": False}}}
    return {
        "unbiased_mm": {**base, "agents": {**base["agents"], "noise": {"n": 12, "longshot_bias": 0.0}}},
        "biased_mm": {**base, "agents": {**base["agents"], "noise": {"n": 12, "longshot_bias": 0.35}}},
        "biased_nomm": {**base, "agents": {**base["agents"], "noise": {"n": 12, "longshot_bias": 0.35}},
                        "strategies": {**base["strategies"], "mm": {"enabled": False}}},
        "biased_fewinformed": {**base, "agents": {**base["agents"], "noise": {"n": 12, "longshot_bias": 0.35},
                                                  "informed": {"n": 1}}},
    }


def e3_arms(T):
    return {"baseline": {"T": T}, "fast_informed": {"T": T, "agents": {"informed": {"n": 10, "react_delay": [0.2, 1.0]}}}}


def e4_arms(T):
    mon = {"monitor": {"enabled": True, "period": 5.0, "min_profit": 1.0}}
    no_arbs = {"agents": {"arbitrageur": {"n": 0}}, "strategies": {"arb": {"enabled": False}}}
    return {
        "no_arbitrageurs": {"T": T, **mon, **no_arbs},
        "simple_arbitrageur": {"T": T, **mon, "strategies": {"arb": {"enabled": False}}},
        "lp_arbitrageur": {"T": T, **mon, "agents": {"arbitrageur": {"n": 0}}},
        "both": {"T": T, **mon},
        "no_fees_no_arbs": {"T": T, **mon, **no_arbs, "venues": {"KX": {"fee": {"kind": "zero"}}}},
    }


def e5_arms(T):
    """Every arm spells out the settings it changes relative to the shipped market maker, which has portfolio skew
    OFF (it changed after the first 8-seed run, whose reference arm had it ON; that run is kept in e5_mm_design_main)."""
    def mm(**kw):
        return {"T": T, "strategies": {"mm": {"portfolio": False, **kw}}}
    return {
        "mm_default": mm(),
        "mm_portfolio": mm(portfolio=True),
        "mm_mid_fair": mm(fair_mode="mid"),
        "mm_oracle_fair": mm(fair_mode="oracle"),
        "mm_no_event_logic": mm(pull_s=0.0, params={"c_evt": 0.0}),
        "mm_no_inventory_skew": mm(params={"gamma": 0.01}),
        "mm_tight": mm(params={"kappa": 25.0}),
    }


EXPS = {"e1": ("e1_speed_liquidity", e1_arms, "research.sim_metrics.e1_metrics"),
        "e2": ("e2_calibration", e2_arms, "research.sim_metrics.e2_metrics"),
        "e3": ("e3_microstructure", e3_arms, "research.sim_metrics.e3_metrics"),
        "e4": ("e4_arbitrage", e4_arms, "research.sim_metrics.e4_metrics"),
        "e5": ("e5_mm_design", e5_arms, "research.sim_metrics.e1_metrics")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exps", nargs="+", choices=list(EXPS))
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--seed0", type=int, default=100)
    ap.add_argument("--T", type=float, default=3600.0)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--tag", default="main", help="fixed run-folder suffix so an interrupted suite resumes")
    a = ap.parse_args()
    for e in a.exps:
        name, arms_fn, metric = EXPS[e]
        seeds = list(range(a.seed0, a.seed0 + (a.seeds * 2 if e == "e2" else a.seeds)))
        t0 = time.time()
        out_dir = os.path.join(ROOT, "runs", f"{name}_{a.tag}")
        df, out = run_experiment(name, arms_fn(a.T), seeds, metric_fn=metric, workers=a.workers, out_dir=out_dir)
        print(f"{name}: {len(df)} runs in {time.time() - t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
