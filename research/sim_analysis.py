"""Tables and figures for simulation experiments E1-E5 (reads runs/<experiment>_main/; E5 prefers the 32-seed
runs/e5_mm_design_seeds32/ when it exists)."""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pmkt.pricing.calibration import calibration_regression  # noqa: E402
from pmkt.pricing.normal import logit  # noqa: E402
from research import plotstyle as ps  # noqa: E402

RUNS = os.path.join(ROOT, "runs")


def load(name, tag="main"):
    return pd.read_csv(os.path.join(RUNS, f"{name}_{tag}", "results.csv"))


def mean_ci(x):
    x = pd.Series(x).dropna()
    if len(x) < 2:
        return float(x.mean()) if len(x) else np.nan, np.nan
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean()), float(h)


def table(df, arms, cols, name):
    rows = []
    for a in arms:
        g = df[df.arm == a]
        r = {"arm": a, "seeds": len(g)}
        for c in cols:
            if c in g:
                m, h = mean_ci(g[c])
                r[c] = m
                r[c + "_ci"] = h
        rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(ps.TAB, name), index=False)
    return t


def dotplot(ax, t, col, arms, labels, color, scale=1.0):
    y = np.arange(len(arms))[::-1]
    ax.errorbar(t.set_index("arm").loc[arms, col] * scale, y, xerr=t.set_index("arm").loc[arms, col + "_ci"] * scale,
                fmt="o", color=color, ms=6, mec=ps.SURFACE, mew=1.2, capsize=0, elinewidth=1.2)
    ax.set_yticks(y, labels)
    ax.grid(axis="y", visible=False)


def main():
    ps.apply()
    import matplotlib.pyplot as plt
    out = {}

    # E1 ----------------------------------------------------------------------------------------------
    e1 = load("e1_speed_liquidity")
    arms = ["thin", "no_our_mm", "baseline", "deep", "slow_informed", "fast_informed"]
    labels = ["Thin book (no MM, no AMM)", "No automated MM", "Baseline", "Deep book", "Slow informed traders",
              "Fast informed traders"]
    cols = ["mae", "rmse", "spread_c", "depth_top", "t50_med", "t90_med", "t50_frac_reached", "pnl_informed",
            "pnl_noise", "pnl_mm", "pnl_naive_mm", "pnl_amm", "pnl_arb"]
    t1 = table(e1, arms, cols, "sim_e1_speed_liquidity.csv")
    out["e1"] = t1
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharey=True)
    dotplot(axes[0], t1, "mae", arms, labels, ps.BLUE, 100)
    axes[0].set_title("Pricing error |mid - truth| (pp)")
    dotplot(axes[1], t1, "spread_c", arms, labels, ps.BLUE)
    axes[1].set_title("Quoted spread (cents)")
    dotplot(axes[2], t1, "t50_med", arms, labels, ps.BLUE)
    axes[2].set_title("Half-life after news (s, median)")
    ps.save(fig, "fig_sim_e1_liquidity_speed.png")

    # E2 ----------------------------------------------------------------------------------------------
    e2r = pd.read_csv(os.path.join(RUNS, "e2_calibration_main", "calib_rows.csv"))
    rows = []
    for a, g in e2r.groupby("arm"):
        for frac, gg in g.groupby("frac"):
            gg = gg[(gg.mid > 0) & (gg.mid < 1) & (gg.truth > 0.005) & (gg.truth < 0.995)]
            cl = gg.seed.astype(str) + ":" + gg.market
            reg_out = calibration_regression(gg.mid.values, gg.outcome.values, clusters=cl.values)
            # price vs truth: slope of logit(mid) on logit(truth); < 1 = compressed toward 50%
            x, y = logit(gg.truth.values), logit(gg.mid.values)
            A = np.column_stack([np.ones(len(x)), x])
            b, *_ = np.linalg.lstsq(A, y, rcond=None)
            resid = y - A @ b
            se = np.sqrt(np.sum(resid ** 2) / (len(x) - 2) / np.sum((x - x.mean()) ** 2))
            ls = gg[gg.truth < 0.15]
            fv = gg[gg.truth > 0.85]
            rows.append({"arm": a, "frac": frac, "n": len(gg), "slope_vs_outcome": reg_out["b"],
                         "se_vs_outcome": reg_out["se_b"], "slope_price_on_truth": float(b[1]), "se_price_on_truth": float(se),
                         "longshot_overpricing_pp": float(100 * (ls.mid - ls.truth).mean()) if len(ls) else np.nan,
                         "favourite_underpricing_pp": float(100 * (fv.truth - fv.mid).mean()) if len(fv) else np.nan,
                         "mae_pp": float(100 * (gg.mid - gg.truth).abs().mean())})
    t2 = pd.DataFrame(rows)
    t2.to_csv(os.path.join(ps.TAB, "sim_e2_calibration.csv"), index=False)
    out["e2"] = t2
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    bins = np.linspace(0, 1, 11)
    for i, (a, lab) in enumerate([("unbiased_mm", "Unbiased noise flow"), ("biased_mm", "Longshot-biased flow, MM on"),
                                  ("biased_nomm", "Longshot-biased flow, no MM")]):
        g = e2r[(e2r.arm == a) & (e2r.frac == 0.5)]
        g = g.assign(b=pd.cut(g.truth, bins, include_lowest=True))
        s = g.groupby("b", observed=True).apply(lambda x: pd.Series({"x": x.truth.mean(), "y": 100 * (x.mid - x.truth).mean(),
                                                                       "se": 100 * (x.mid - x.truth).std() / np.sqrt(len(x))}))
        ax.errorbar(s.x, s.y, yerr=1.96 * s.se, fmt="o-", color=ps.SERIES[i], ms=5, mec=ps.SURFACE, mew=1.0,
                    capsize=0, label=lab)
    ax.axhline(0, color=ps.MUTED, lw=1.0)
    ax.set_xlabel("True probability (simulation ground truth)")
    ax.set_ylabel("Mid minus truth (pp, 95% CI)")
    ax.set_title("Simulated mispricing by true probability (mid-session)")
    ax.legend(loc="upper right")
    ps.save(fig, "fig_sim_e2_calibration.png")

    # E3 ----------------------------------------------------------------------------------------------
    e3 = load("e3_microstructure")
    t3 = table(e3, ["baseline", "fast_informed"], ["r2_oos", "hit", "b_imb", "b_ofi", "frac_pred_gt_half",
                                                   "taker_edge_c", "taker_n", "corr_truthgap", "mae"],
               "sim_e3_microstructure.csv")
    out["e3"] = t3

    # E4 ----------------------------------------------------------------------------------------------
    e4 = load("e4_arbitrage")
    arms4 = ["no_fees_no_arbs", "no_arbitrageurs", "simple_arbitrageur", "lp_arbitrageur", "both"]
    lab4 = ["No arbitrageurs, no fees", "No arbitrageurs", "Rule-based arbitrageur", "LP arbitrageur", "Both"]
    t4 = table(e4, arms4, ["opp_share", "opp_profit_mean_c", "persist_scans_mean", "opp_cross-venue", "opp_bucket",
                           "opp_ladder", "opp_combo", "pnl_arb", "pnl_arbitrageur", "pnl_mm", "mae", "spread_c"],
               "sim_e4_arbitrage.csv")
    out["e4"] = t4
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), sharey=True)
    dotplot(axes[0], t4, "opp_share", arms4, lab4, ps.BLUE, 100)
    axes[0].set_title("Scans with executable arbitrage (%)")
    dotplot(axes[1], t4, "opp_profit_mean_c", arms4, lab4, ps.BLUE, 0.01)
    axes[1].set_title("Mean locked-in profit ($, after fees)")
    dotplot(axes[2], t4, "persist_scans_mean", arms4, lab4, ps.BLUE, 5.0)
    axes[2].set_title("Mean persistence (s)")
    ps.save(fig, "fig_sim_e4_arbitrage.png")

    # E5 ----------------------------------------------------------------------------------------------
    if os.path.exists(os.path.join(RUNS, "e5_mm_design_seeds32", "results.csv")):
        e5 = load("e5_mm_design", "seeds32")          # 32-seed rerun, arms explicit vs the shipped default
        arms5 = ["mm_tight", "mm_no_inventory_skew", "mm_no_event_logic", "mm_portfolio", "mm_mid_fair", "mm_default",
                 "mm_oracle_fair"]
        lab5 = ["Tight spread (kappa 25)", "No inventory skew", "No scheduled-news logic", "With portfolio skew",
                "Fair value = external mid", "Default automated MM", "Oracle fair value (bound)"]
    else:
        e5 = load("e5_mm_design")
        arms5 = ["mm_tight", "mm_no_inventory_skew", "mm_no_event_logic", "mm_no_portfolio", "mm_mid_fair",
                 "mm_default", "mm_oracle_fair"]
        lab5 = ["Tight spread (kappa 25)", "No inventory skew", "No scheduled-news logic", "No portfolio skew",
                "Fair value = external mid", "Default automated MM", "Oracle fair value (bound)"]
    t5 = table(e5, arms5, ["pnl_mm", "mo5_mm", "sc_mm", "dd_mm", "vol_mm", "pnl_naive_mm", "pnl_amm", "mae"],
               "sim_e5_mm_design.csv")
    out["e5"] = t5
    # Figure: paired (same-seed) differences vs the default MM, i.e. what the text reports. Every arm faces the
    # same information path, so these intervals are much tighter than the level intervals in the table.
    alt5 = [a for a in arms5 if a != "mm_default"]
    alt_lab = [lab for a, lab in zip(arms5, lab5) if a != "mm_default"]
    base5 = e5[e5.arm == "mm_default"].set_index("seed")
    prow = []
    for a in alt5:
        g = e5[e5.arm == a].set_index("seed")
        common = base5.index.intersection(g.index)
        r = {"arm": a, "pairs": len(common)}
        for c, scale in (("pnl_mm", 1.0), ("dd_mm", 0.01), ("mo5_mm", 1.0)):      # dd_mm is in cents
            r[c], r[c + "_ci"] = mean_ci((g.loc[common, c] - base5.loc[common, c]) * scale)
        prow.append(r)
    p5 = pd.DataFrame(prow)
    n5 = int(p5.pairs.min())
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), sharey=True)
    for ax, (c, title) in zip(axes, [("pnl_mm", "Change in P&L per 1 h session ($)"),
                                     ("dd_mm", "Change in max drawdown ($)"),
                                     ("mo5_mm", "Change in 5 s markout (cents)")]):
        dotplot(ax, p5, c, alt5, alt_lab, ps.BLUE)
        ax.axvline(0, color=ps.MUTED, lw=1.0)
        ax.set_title(title)
    axes[0].set_xlabel(f"vs default MM, paired by seed, 95% CI, n = {n5}")
    ps.save(fig, "fig_sim_e5_mm_design.png")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    for k, v in out.items():
        print(f"===== {k}")
        keep = [c for c in v.columns if not c.endswith("_ci")]
        print(v[keep].round(3).to_string())


if __name__ == "__main__":
    main()
