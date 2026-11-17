"""Paired (common-random-numbers) contrasts between simulation arms: every arm uses the same seeds, so the
difference arm - reference is computed seed by seed and tested with a paired t interval."""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from research import plotstyle as ps  # noqa: E402

CONTRASTS = [
    # (run folder, reference arm, arms, metrics); E5 uses the 32-seed rerun when it exists
    ("e5_mm_design_seeds32", "mm_default", ["mm_portfolio", "mm_mid_fair", "mm_no_event_logic", "mm_no_inventory_skew",
                                            "mm_tight", "mm_oracle_fair"], ["pnl_mm", "mo5_mm", "sc_mm", "dd_mm"]),
    ("e5_mm_design_main", "mm_default", ["mm_no_portfolio", "mm_mid_fair", "mm_no_event_logic", "mm_no_inventory_skew",
                                         "mm_tight", "mm_oracle_fair"], ["pnl_mm", "mo5_mm", "dd_mm"]),
    ("e1_speed_liquidity_main", "baseline", ["thin", "no_our_mm", "deep", "slow_informed", "fast_informed"],
     ["mae", "t50_med", "t90_med", "spread_c", "pnl_informed", "pnl_noise"]),
    ("e4_arbitrage_main", "no_arbitrageurs", ["simple_arbitrageur", "lp_arbitrageur", "both", "no_fees_no_arbs"],
     ["opp_share", "opp_profit_mean_c", "persist_scans_mean", "mae"]),
]


def main():
    rows = []
    for exp, ref, arms, metrics in CONTRASTS:
        path = os.path.join(ROOT, "runs", exp, "results.csv")
        if not os.path.exists(path):
            continue
        d = pd.read_csv(path)
        base = d[d.arm == ref].set_index("seed")
        for a in arms:
            g = d[d.arm == a].set_index("seed")
            common = base.index.intersection(g.index)
            for m in metrics:
                if m not in d:
                    continue
                diff = (g.loc[common, m] - base.loc[common, m]).dropna()
                if len(diff) < 3:
                    continue
                h = stats.t.ppf(0.975, len(diff) - 1) * diff.std(ddof=1) / np.sqrt(len(diff))
                t = diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff))) if diff.std(ddof=1) > 0 else np.nan
                rows.append({"experiment": exp, "arm": a, "vs": ref, "metric": m, "n_pairs": len(diff),
                             "mean_diff": diff.mean(), "ci95": h, "t": t,
                             "p": 2 * stats.t.sf(abs(t), len(diff) - 1) if np.isfinite(t) else np.nan,
                             "wins": int((diff > 0).sum())})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(ps.TAB, "sim_paired_contrasts.csv"), index=False)
    pd.set_option("display.width", 220)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
