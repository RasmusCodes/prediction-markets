"""Q2 (calibration) and Q5 (liquidity vs efficiency) on resolved Kalshi and Polymarket markets.

Panel: for every resolved binary market and horizon h before its close, the market probability at close - h.
  Kalshi      mid of the hourly candle's closing bid/ask (requires a two-sided book with spread <= 25c);
              the spread is kept as a liquidity measure. Close = market close_time.
  Polymarket  hourly price-history point (CLOB 'p'). Close = min(endDate, closedTime). Outcome = token 0 won.
Inference is clustered by event (strike ladders / multi-outcome events are strongly dependent).

Outputs (paper/tables, paper/figures):
  calibration_summary.csv        venue x horizon: n, events, Brier + Murphy parts, ECE, calibration a/b (cluster SE)
  calibration_eventweighted.csv  same with each event weighted equally
  flb_returns.csv                mean return of buying YES / NO by price bucket (Kalshi net of taker fee, at the ask)
  liquidity_terciles.csv         calibration and accuracy by volume tercile (and Kalshi spread tercile)
  category_calibration.csv       Kalshi by category
  recalibration_oos.csv          train <= 2025-12-31, test 2026: Platt / isotonic Brier change and a fee-aware trade rule
  fig_calibration_reliability.png, fig_flb_returns.png, fig_calibration_horizons.png, fig_liquidity_efficiency.png
"""
from __future__ import annotations

import os
import sqlite3
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pmkt.pricing import calibration as cal  # noqa: E402
from research import plotstyle as ps  # noqa: E402

HIST = os.path.join(ROOT, "data", "hist")
HORIZONS = [1, 6, 24, 72, 168]           # hours before close
SPLIT_TS = pd.Timestamp("2026-01-01", tz="UTC").timestamp()
MAX_SPREAD = 0.25


def ro(path):
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found: raw data are not in the repository; build it with collect/ (see data/DATA.md)")
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


# ------------------------------------------------------------------------------------------------ panels
def kalshi_panel() -> pd.DataFrame:
    k = ro(os.path.join(HIST, "kalshi_hist.sqlite"))
    mk = pd.read_sql("""SELECT m.ticker, m.event_ticker, m.series_ticker, m.open_ts, m.close_ts, m.result, m.volume,
                               e.category, s.fee_type, s.fee_multiplier
                        FROM markets m LEFT JOIN events e USING(event_ticker)
                        LEFT JOIN series s ON s.ticker = m.series_ticker
                        WHERE m.result IN ('yes','no')""", k)
    cd = pd.read_sql("SELECT ticker, end_ts, bid_close, ask_close FROM candles WHERE period=60", k)
    cd = cd.sort_values("end_ts")
    cd["end_ts"] = cd.end_ts.astype("int64")
    mk["y"] = (mk.result == "yes").astype(int)
    mk["category"] = mk.category.fillna("Other")
    rows = []
    for h in HORIZONS + ["mid"]:
        t = mk.copy()
        if h == "mid":
            t["target"] = (t.open_ts + 0.5 * (t.close_ts - t.open_ts))
            tol = 24 * 3600
        else:
            t["target"] = t.close_ts - h * 3600.0
            tol = 6 * 3600 if h <= 24 else 24 * 3600
        t = t[t.target >= t.open_ts].copy()
        t["target"] = t.target.astype("int64")
        t = t.sort_values("target")
        m = pd.merge_asof(t, cd, left_on="target", right_on="end_ts", by="ticker", direction="backward", tolerance=tol)
        ok = (m.bid_close > 0) & (m.ask_close < 1) & (m.ask_close > m.bid_close) & ((m.ask_close - m.bid_close) <= MAX_SPREAD)
        m = m[ok].copy()
        m["p"] = 0.5 * (m.bid_close + m.ask_close)
        m["spread"] = m.ask_close - m.bid_close
        m["horizon"] = str(h)
        rows.append(m[["ticker", "event_ticker", "series_ticker", "category", "close_ts", "volume", "y", "p", "spread",
                       "bid_close", "ask_close", "fee_multiplier", "horizon"]])
    out = pd.concat(rows, ignore_index=True)
    out["venue"] = "Kalshi"
    out = out.rename(columns={"event_ticker": "cluster"})
    return out


SPORT_WORDS = (" vs ", " vs. ", "nba", "nfl", "nhl", "mlb", "ufc", "fc ", "premier league", "champions league",
               "match", "game ", "grand prix", "open:", "tennis", "spread", "o/u")
CRYPTO_WORDS = ("bitcoin", "btc", "ethereum", "eth ", "solana", "xrp", "crypto", "up or down", "doge")
POL_WORDS = ("election", "president", "senate", "governor", "nominee", "primary", "parliament", "prime minister",
             "trump", "biden", "harris", "fed ", "fomc", "rate cut", "tariff")


def pm_category(q: str) -> str:
    s = (q or "").lower()
    if any(w in s for w in CRYPTO_WORDS):
        return "Crypto"
    if any(w in s for w in POL_WORDS):
        return "Politics & macro"
    if any(w in s for w in SPORT_WORDS):
        return "Sports"
    return "Other"


def polymarket_panel() -> pd.DataFrame:
    pm = ro(os.path.join(HIST, "polymarket_hist.sqlite"))
    mk = pd.read_sql("""SELECT condition_id, question, event_slug, winner_index, token0, start_ts, end_ts, closed_ts,
                               volume, sample_group, fees_enabled FROM markets WHERE winner_index IS NOT NULL""", pm)
    px = pd.read_sql("SELECT token_id, t, p FROM prices", pm).sort_values("t")
    px["t"] = px.t.astype("int64")
    mk["ref"] = np.where(mk.closed_ts.notna(), np.minimum(mk.end_ts, mk.closed_ts.fillna(np.inf)), mk.end_ts)
    mk["y"] = (mk.winner_index == 0).astype(int)
    mk["category"] = mk.question.map(pm_category)
    rows = []
    for h in HORIZONS + ["mid"]:
        t = mk.copy()
        if h == "mid":
            t["target"] = t.start_ts + 0.5 * (t.ref - t.start_ts)
            tol = 24 * 3600
        else:
            t["target"] = t.ref - h * 3600.0
            tol = 3 * 3600 if h <= 24 else 24 * 3600
        t = t[(t.target >= t.start_ts) & t.target.notna()].copy()
        t["target"] = t.target.astype("int64")
        t = t.sort_values("target").rename(columns={"token0": "token_id"})
        m = pd.merge_asof(t, px, left_on="target", right_on="t", by="token_id", direction="backward", tolerance=tol)
        m = m[m.p.notna() & (m.p > 0) & (m.p < 1)].copy()
        m["horizon"] = str(h)
        rows.append(m[["condition_id", "event_slug", "category", "ref", "volume", "y", "p", "sample_group",
                       "fees_enabled", "horizon"]])
    out = pd.concat(rows, ignore_index=True).rename(columns={"event_slug": "cluster", "ref": "close_ts",
                                                             "condition_id": "ticker"})
    out["cluster"] = out.cluster.fillna(out.ticker)
    out["venue"] = "Polymarket"
    out["spread"] = np.nan
    return out


# ------------------------------------------------------------------------------------------------ stats
def summarize(d: pd.DataFrame, weights: str | None = None) -> dict:
    w = None
    if weights == "event":
        w = 1.0 / d.groupby("cluster").cluster.transform("size").values
    dec = cal.murphy_decomposition(d.p.values, d.y.values, n_bins=10, w=w)
    reg = cal.calibration_regression(d.p.values, d.y.values, clusters=d.cluster.values, w=w)
    return {"n": len(d), "events": d.cluster.nunique(), "mean_p": float(np.average(d.p, weights=w)),
            "mean_y": float(np.average(d.y, weights=w)), "brier": dec["brier"], "reliability": dec["reliability"],
            "resolution": dec["resolution"], "uncertainty": dec["uncertainty"],
            "ece": cal.ece(d.p.values, d.y.values, 10, "uniform", w), "logloss": cal.log_loss(d.p.values, d.y.values, w),
            "a": reg["a"], "se_a": reg["se_a"], "b": reg["b"], "se_b": reg["se_b"], "z_b1": reg["z_b1"]}


def kalshi_fee(p, mult):
    return 0.07 * np.nan_to_num(mult, nan=1.0) * p * (1 - p)


BUCKETS = [0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90, 0.95, 1.0]


def flb_returns(d: pd.DataFrame, venue: str, n_boot: int = 300) -> pd.DataFrame:
    """Return per $1 staked buying YES (and NO) by entry-price bucket. Kalshi: buy YES at the ask / NO at 1 - bid,
    net of the taker fee. Polymarket: at the recorded price with an assumed 1c half-spread, no fee."""
    d = d.copy()
    if venue == "Kalshi":
        cy = d.ask_close + kalshi_fee(d.ask_close, d.fee_multiplier)
        cn = (1 - d.bid_close) + kalshi_fee(1 - d.bid_close, d.fee_multiplier)
    else:
        cy = np.minimum(d.p + 0.01, 0.999)
        cn = np.minimum(1 - d.p + 0.01, 0.999)
    d["r_yes"] = (d.y - cy) / cy
    d["r_no"] = ((1 - d.y) - cn) / cn
    d["bucket"] = pd.cut(d.p, BUCKETS, include_lowest=True)
    rng = np.random.default_rng(0)
    out = []
    for b, g in d.groupby("bucket", observed=True):
        cl = g.cluster.values
        uc, inv = np.unique(cl, return_inverse=True)
        idx_by = [np.where(inv == i)[0] for i in range(len(uc))]
        def boot(col):
            vals = g[col].values
            bs = []
            for _ in range(n_boot):
                pick = rng.integers(0, len(uc), len(uc))
                ii = np.concatenate([idx_by[j] for j in pick])
                bs.append(vals[ii].mean())
            return np.quantile(bs, [0.025, 0.975])
        lo_y, hi_y = boot("r_yes")
        lo_n, hi_n = boot("r_no")
        out.append({"venue": venue, "bucket": str(b), "p_mid": float(g.p.mean()), "n": len(g), "events": len(uc),
                    "win_rate": float(g.y.mean()), "ret_yes": float(g.r_yes.mean()), "ret_yes_lo": lo_y,
                    "ret_yes_hi": hi_y, "ret_no": float(g.r_no.mean()), "ret_no_lo": lo_n, "ret_no_hi": hi_n})
    return pd.DataFrame(out)


def recalibration_oos(d: pd.DataFrame, venue: str, theta: float = 0.02) -> list[dict]:
    """Time-ordered split at the venue's median close date: fit on the earlier half, evaluate on the later half."""
    split = float(d.close_ts.median())
    tr, te = d[d.close_ts < split], d[d.close_ts >= split]
    if len(tr) < 500 or len(te) < 500:
        return []
    rows = []
    for method in ("platt", "isotonic"):
        g = cal.recalibration_gain(tr.p.values, tr.y.values, te.p.values, te.y.values, method)
        m = (cal.PlattScaler() if method == "platt" else cal.IsotonicCalibrator()).fit(tr.p.values, tr.y.values)
        q = np.clip(m.predict(te.p.values), 1e-4, 1 - 1e-4)
        if venue == "Kalshi":
            cy = te.ask_close.values + kalshi_fee(te.ask_close.values, te.fee_multiplier.values)
            cn = (1 - te.bid_close.values) + kalshi_fee(1 - te.bid_close.values, te.fee_multiplier.values)
        else:
            cy, cn = np.minimum(te.p.values + 0.01, 0.999), np.minimum(1 - te.p.values + 0.01, 0.999)
        buy_y = (q - cy) > theta
        buy_n = ((1 - q) - cn) > theta
        pnl = np.where(buy_y, te.y.values - cy, 0.0) + np.where(buy_n, (1 - te.y.values) - cn, 0.0)
        cost = np.where(buy_y, cy, 0.0) + np.where(buy_n, cn, 0.0)
        n_tr = int(buy_y.sum() + buy_n.sum())
        # cluster bootstrap of the per-trade edge
        traded = (buy_y | buy_n)
        sub = te[traded].assign(pnl=pnl[traded])
        rng = np.random.default_rng(1)
        uc = sub.cluster.unique()
        grp = {c: sub.pnl.values[sub.cluster.values == c] for c in uc}
        bs = []
        for _ in range(400):
            pick = rng.choice(uc, len(uc))
            v = np.concatenate([grp[c] for c in pick]) if len(uc) else np.array([0.0])
            bs.append(v.mean())
        rows.append({"venue": venue, "method": method, "split_utc": pd.Timestamp(split, unit="s").strftime("%Y-%m-%d"),
                     "n_train": len(tr), "n_test": len(te),
                     "brier_raw": g["brier_raw"], "brier_cal": g["brier_cal"], "logloss_raw": g["logloss_raw"],
                     "logloss_cal": g["logloss_cal"], "trades": n_tr, "trade_events": len(uc),
                     "edge_per_contract": float(pnl[traded].mean()) if n_tr else np.nan,
                     "edge_lo": float(np.quantile(bs, 0.025)) if n_tr else np.nan,
                     "edge_hi": float(np.quantile(bs, 0.975)) if n_tr else np.nan,
                     "return_on_cost": float(pnl.sum() / cost.sum()) if cost.sum() > 0 else np.nan})
    return rows


def terciles(d: pd.DataFrame, col: str, label: str) -> list[dict]:
    rows = []
    q = d[col].rank(pct=True)
    d = d.assign(t=np.select([q <= 1 / 3, q <= 2 / 3], ["low", "mid"], "high"))
    for t, g in d.groupby("t"):
        s = summarize(g)
        rows.append({"venue": d.venue.iloc[0], "split": label, "tercile": t, "median": float(g[col].median()), **s})
    return rows


# ------------------------------------------------------------------------------------------------ main
def main():
    ps.apply()
    import matplotlib.pyplot as plt
    print("building panels ...", flush=True)
    K = kalshi_panel()
    P = polymarket_panel()
    D = pd.concat([K, P], ignore_index=True)
    D.to_pickle(os.path.join(ROOT, "data", "processed_calibration_panel.pkl"))
    summ, summ_ew = [], []
    for (v, h), g in D.groupby(["venue", "horizon"]):
        summ.append({"venue": v, "horizon": h, **summarize(g)})
        summ_ew.append({"venue": v, "horizon": h, **summarize(g, "event")})
    S = pd.DataFrame(summ)
    S.to_csv(os.path.join(ps.TAB, "calibration_summary.csv"), index=False)
    pd.DataFrame(summ_ew).to_csv(os.path.join(ps.TAB, "calibration_eventweighted.csv"), index=False)
    print(S.round(4).to_string(), flush=True)

    # reliability diagrams (24 h)
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.0), sharey=True)
    for ax, v in zip(axes, ["Kalshi", "Polymarket"]):
        g = D[(D.venue == v) & (D.horizon == "24")]
        rt = pd.DataFrame(cal.reliability_table(g.p.values, g.y.values, 10, "uniform"))
        rt.to_csv(os.path.join(ps.TAB, f"reliability_{v.lower()}_24h.csv"), index=False)
        ax.plot([0, 1], [0, 1], color=ps.AXIS, lw=1.0, zorder=1)
        ax.errorbar(rt.p_mean, rt.freq, yerr=[rt.freq - rt.ci_lo, rt.ci_hi - rt.freq], fmt="o", color=ps.VENUE_COLOR[v],
                    ecolor=ps.VENUE_COLOR[v], elinewidth=1.0, capsize=0, ms=6, mec=ps.SURFACE, mew=1.2, zorder=3)
        s = S[(S.venue == v) & (S.horizon == "24")].iloc[0]
        ax.set_title(f"{v}: 24 h before close")
        ax.text(0.03, 0.95, f"n = {int(s.n):,} markets, {int(s.events):,} events\nslope b = {s.b:.2f} (SE {s.se_b:.2f})\n"
                f"Brier {s.brier:.3f}, ECE {s.ece:.3f}", transform=ax.transAxes, va="top", fontsize=8.5, color=ps.INK2)
        ax.set_xlabel("Market probability (bin mean)")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    axes[0].set_ylabel("Realised frequency (95% Wilson CI)")
    ps.save(fig, "fig_calibration_reliability.png")

    # slope by horizon
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    order = ["168", "72", "24", "6", "1"]
    for v in ["Kalshi", "Polymarket"]:
        g = S[(S.venue == v) & (S.horizon.isin(order))].set_index("horizon").loc[order]
        x = np.arange(len(order)) + (-0.06 if v == "Kalshi" else 0.06)
        ax.errorbar(x, g.b, yerr=1.96 * g.se_b, fmt="o-", color=ps.VENUE_COLOR[v], capsize=0, ms=6, mec=ps.SURFACE,
                    mew=1.2, label=v)
        ax.text(x[-1] + 0.12, g.b.iloc[-1], v, color=ps.INK2, fontsize=8.5, va="center")
    ax.axhline(1.0, color=ps.MUTED, lw=1.0, ls="--")
    ax.set_xticks(np.arange(len(order)), ["7 d", "3 d", "24 h", "6 h", "1 h"])
    ax.set_xlabel("Horizon before close")
    ax.set_ylabel("Calibration slope b (95% CI, event-clustered)")
    ax.set_title("Calibration slope by horizon (b > 1: prices too compressed toward 50%)")
    ax.legend(loc="upper left")
    ax.set_xlim(-0.4, len(order) - 0.2)
    ps.save(fig, "fig_calibration_horizons.png")

    # favourite-longshot returns (24 h)
    F = pd.concat([flb_returns(D[(D.venue == v) & (D.horizon == "24")], v) for v in ["Kalshi", "Polymarket"]])
    F.to_csv(os.path.join(ps.TAB, "flb_returns.csv"), index=False)
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    for i, v in enumerate(["Kalshi", "Polymarket"]):
        g = F[F.venue == v]
        x = np.arange(len(g)) + (i - 0.5) * 0.18
        ax.errorbar(x, g.ret_yes * 100, yerr=[(g.ret_yes - g.ret_yes_lo) * 100, (g.ret_yes_hi - g.ret_yes) * 100],
                    fmt="o", color=ps.VENUE_COLOR[v], capsize=0, ms=6, mec=ps.SURFACE, mew=1.2,
                    label=v + (" (at ask, net of taker fee)" if v == "Kalshi" else " (price + 1c, no fee)"))
    ax.axhline(0, color=ps.MUTED, lw=1.0)
    labels = ["<5", "5-10", "10-20", "20-35", "35-50", "50-65", "65-80", "80-90", "90-95", ">95"]
    ax.set_xticks(np.arange(len(labels)), labels)
    ax.set_xlabel("Price of YES 24 h before close (cents)")
    ax.set_ylabel("Mean return of buying YES (%, 95% CI)")
    ax.set_title("Return to buying YES by price: the favourite-longshot pattern")
    ax.legend(loc="lower right")
    ps.save(fig, "fig_flb_returns.png")

    # liquidity (Q5)
    L = []
    for v in ["Kalshi", "Polymarket"]:
        g = D[(D.venue == v) & (D.horizon == "24")]
        L += terciles(g, "volume", "volume")
        if v == "Kalshi":
            L += terciles(g, "spread", "spread")
    L = pd.DataFrame(L)
    L.to_csv(os.path.join(ps.TAB, "liquidity_terciles.csv"), index=False)
    print(L[["venue", "split", "tercile", "median", "n", "events", "brier", "reliability", "ece", "b", "se_b"]].round(4).to_string())
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6))
    tord = ["low", "mid", "high"]
    for v, off in (("Kalshi", -0.08), ("Polymarket", 0.08)):
        g = L[(L.venue == v) & (L.split == "volume")].set_index("tercile").loc[tord]
        x = np.arange(3) + off
        axes[0].errorbar(x, g.b, yerr=1.96 * g.se_b, fmt="o-", color=ps.VENUE_COLOR[v], ms=6, mec=ps.SURFACE, mew=1.2, label=v)
        axes[1].plot(x, g.ece * 100, "o-", color=ps.VENUE_COLOR[v], ms=6, mec=ps.SURFACE, mew=1.2, label=v)
    axes[0].axhline(1.0, color=ps.MUTED, lw=1.0, ls="--")
    for ax, t in zip(axes, ["Calibration slope b (95% CI)", "Expected calibration error (pp)"]):
        ax.set_xticks(range(3), ["Low volume", "Mid", "High volume"])
        ax.set_title(t)
    axes[0].legend(loc="upper right")
    ps.save(fig, "fig_liquidity_efficiency.png")

    # Kalshi categories
    C = []
    g24 = D[(D.venue == "Kalshi") & (D.horizon == "24")]
    for c, g in g24.groupby("category"):
        if len(g) >= 300 and g.cluster.nunique() >= 20:
            C.append({"category": c, **summarize(g)})
    for c, g in D[(D.venue == "Polymarket") & (D.horizon == "24")].groupby("category"):
        if len(g) >= 300:
            C.append({"category": "PM: " + c, **summarize(g)})
    pd.DataFrame(C).to_csv(os.path.join(ps.TAB, "category_calibration.csv"), index=False)

    # out-of-sample recalibration + fee-aware trading rule (24 h)
    R = []
    for v in ["Kalshi", "Polymarket"]:
        R += recalibration_oos(D[(D.venue == v) & (D.horizon == "24")], v)
    R = pd.DataFrame(R)
    R.to_csv(os.path.join(ps.TAB, "recalibration_oos.csv"), index=False)
    print(R.round(4).to_string())
    print("done", flush=True)


if __name__ == "__main__":
    main()
