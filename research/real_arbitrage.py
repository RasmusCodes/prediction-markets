"""Q4: do correlated (logically related) markets create executable arbitrage on real venues?

Uses the 13.9 h live capture (top of book every ~4 s for the tracked universe).
Tests at every snapshot, after fees and limited by displayed top-of-book size:
  sell-all   mutually exclusive outcomes: selling YES on every listed outcome (= buying every NO) pays at least
             (m-1) dollars, so sum(bid - fee) > 1 is an arbitrage. Valid for ANY subset of a mutually exclusive set.
  buy-all    exhaustive set: sum(ask + fee) < 1 is an arbitrage. Only evaluated where the tracked set looks complete
             (Kalshi bucket events with fewer than 40 tracked outcomes, i.e. not truncated by the collector cap).
  ladder     Kalshi strike ladders 'above K' in one event: P(>K_lo) >= P(>K_hi), so bid(K_hi) - ask(K_lo) - fees > 0.
Kalshi taker fee 0.07 p(1-p) per contract (exact rate, no per-order rounding: a lower bound on costs); Polymarket
fee-free (most markets). A quote is used only while its market was in the collector's polled set (membership
from the universe table, re-selected every 20 min).
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from research import plotstyle as ps  # noqa: E402

LIVE = os.path.join(ROOT, "data", "live", "live.sqlite")


def kfee(p):
    return 0.07 * p * (1 - p)


def load():
    if not os.path.exists(LIVE):
        raise SystemExit(f"{LIVE} not found: raw data are not in the repository; build it with collect/ (see data/DATA.md)")
    db = sqlite3.connect(f"file:{LIVE}?mode=ro", uri=True)
    u = pd.read_sql("SELECT ts, venue, market, event, group_key, mutually_exclusive, extra FROM universe", db)
    cyc = pd.read_sql("SELECT ts, venue FROM cycles", db)
    return db, u, cyc


def membership(u: pd.DataFrame, venue: str):
    """market -> list of (start, end) intervals during which it was in the polled set."""
    uu = u[u.venue == venue]
    stamps = np.sort(uu.ts.unique())
    nxt = {s: (stamps[i + 1] if i + 1 < len(stamps) else np.inf) for i, s in enumerate(stamps)}
    iv = {}
    for m, g in uu.groupby("market"):
        iv[m] = [(s, nxt[s]) for s in sorted(g.ts.unique())]
    return iv


def event_frame(db, venue, markets, cyc_t, iv, close_ts: dict | None = None):
    """Wide forward-filled bid/ask/size frames on the venue's cycle times, masked outside polled membership and
    from 60 s before each market's close (frozen settling quotes). Empty sides (Kalshi bid 0 / ask 1) are NaN."""
    q = ",".join("?" * len(markets))
    d = pd.read_sql(f"SELECT ts, market, bid, ask, bid_sz, ask_sz FROM quotes WHERE venue=? AND market IN ({q})", db,
                    params=[venue] + list(markets)).sort_values("ts")
    d.loc[(d.bid <= 0) | (d.bid_sz <= 0), ["bid", "bid_sz"]] = np.nan
    d.loc[(d.ask >= 1) | (d.ask_sz <= 0), ["ask", "ask_sz"]] = np.nan
    d[["bid", "ask", "bid_sz", "ask_sz"]] = d[["bid", "ask", "bid_sz", "ask_sz"]].fillna(-1.0)   # keep "empty" in ffill
    out = {}
    for col in ("bid", "ask", "bid_sz", "ask_sz"):
        w = d.pivot_table(index="ts", columns="market", values=col, aggfunc="last")
        w = w.reindex(w.index.union(cyc_t)).ffill().reindex(cyc_t)
        w = w.mask(w < 0)
        for m in w.columns:
            mask = np.zeros(len(cyc_t), bool)
            for a, b in iv.get(m, []):
                mask |= (cyc_t >= a) & (cyc_t < b)
            if close_ts and close_ts.get(m):
                mask &= cyc_t < close_ts[m] - 60
            w.loc[~mask, m] = np.nan
        out[col] = w.reindex(columns=markets)
    return out


def iso_ts(s):
    try:
        return pd.Timestamp(s).timestamp()
    except Exception:
        return None


def runs_of(flag: np.ndarray, t: np.ndarray):
    """Durations (s) of consecutive True runs."""
    durs, start = [], None
    for i, f in enumerate(flag):
        if f and start is None:
            start = i
        if (not f) and start is not None:
            durs.append(t[i - 1] - t[start] + 4.0)
            start = None
    if start is not None:
        durs.append(t[len(flag) - 1] - t[start] + 4.0)
    return durs


def main():
    ps.apply()
    import matplotlib.pyplot as plt
    db, u, cyc = load()
    counts = pd.read_csv(os.path.join(ROOT, "data", "live", "kalshi_event_counts.csv")).set_index("event")
    rows, ladder_rows = [], []
    for venue in ("kalshi", "poly"):
        ct = np.sort(cyc[cyc.venue == venue].ts.unique())
        iv = membership(u, venue)
        uu = u[u.venue == venue].drop_duplicates("market", keep="last")
        close = {m: iso_ts(c) for m, c in zip(uu.market, uu.close_time)} if "close_time" in uu else {}
        fee = kfee if venue == "kalshi" else (lambda p: 0.0 * p)
        me = uu[uu.mutually_exclusive == 1]
        for ev, g in me.groupby("event"):
            mk = sorted(g.market.unique())
            if len(mk) < 2:
                continue
            F = event_frame(db, venue, mk, ct, iv, close)
            bid, ask, bsz, asz = F["bid"].values, F["ask"].values, F["bid_sz"].values, F["ask_sz"].values
            # every leg two-sided and none near-settled (frozen / resolving books are where stale quotes live)
            live = (bid >= 0.03) & (bid <= 0.97) & (ask >= 0.03) & (ask <= 0.97)
            full = np.all(np.isfinite(bid) & np.isfinite(ask), axis=1)
            full_live = full & np.all(live | ~np.isfinite(bid), axis=1)
            if full.sum() < 20:
                continue
            sell = np.nansum(bid - fee(bid), axis=1) - 1.0
            sell_sets = np.nanmin(bsz, axis=1)
            buy = 1.0 - np.nansum(ask + fee(ask), axis=1)
            buy_sets = np.nanmin(asz, axis=1)
            midsum = np.nansum(0.5 * (bid + ask), axis=1)
            n_true = counts.n_markets.get(ev) if venue == "kalshi" and ev in counts.index else None
            complete = venue == "kalshi" and n_true is not None and int(n_true) == len(mk)
            for kind, edge, sets, valid in (("sell-all", sell, sell_sets, True), ("buy-all", buy, buy_sets, complete)):
                if not valid:
                    continue
                f = full & (edge > 0.0)
                f_live = full_live & (edge > 0.0)
                durs = runs_of(f, ct)
                rows.append({"venue": venue, "event": ev, "outcomes": len(mk), "true_outcomes": n_true, "test": kind,
                             "snapshots": int(full.sum()), "arb_snapshots": int(f.sum()),
                             "arb_snapshots_live_legs": int(f_live.sum()), "arb_share": float(f.sum() / full.sum()),
                             "mean_edge_c": float(100 * edge[f].mean()) if f.any() else 0.0,
                             "max_edge_c": float(100 * edge[f].max()) if f.any() else 0.0,
                             "mean_sets": float(np.nanmean(sets[f])) if f.any() else 0.0,
                             "mean_profit_usd": float(np.nanmean(edge[f] * sets[f])) if f.any() else 0.0,
                             "episodes": len(durs), "median_duration_s": float(np.median(durs)) if durs else 0.0,
                             "mean_midsum": float(np.nanmean(midsum[full])), "complete_set": complete})
        if venue == "kalshi":   # strike ladders
            ex = uu.assign(x=uu.extra.map(lambda s: json.loads(s) if s else {}))
            ex["st"] = ex.x.map(lambda d: d.get("strike_type"))
            ex["K"] = ex.x.map(lambda d: d.get("floor_strike"))
            # genuine single-underlying ladders only: tickers ending in -T<strike> (e.g. KXBTCD-...-T68099.99); events
            # that mix "above K" markets on different teams/players/cities are not ladders
            lad = ex[ex.st.isin(["greater", "greater_or_equal"]) & ex.K.notna() &
                     ex.market.str.contains(r"-T\d+(\.\d+)?$", regex=True)]
            for ev, g in lad.groupby("event"):
                g = g.sort_values("K")
                mk = list(g.market)
                if len(mk) < 3:
                    continue
                F = event_frame(db, venue, mk, ct, iv, close)
                bid, ask, bsz, asz = F["bid"].values, F["ask"].values, F["bid_sz"].values, F["ask_sz"].values
                n_pairs = 0
                hits = np.zeros(len(ct), bool)
                edges, profits = [], []
                for i in range(len(mk) - 1):            # adjacent strikes (the binding constraints)
                    e = bid[:, i + 1] - kfee(bid[:, i + 1]) - (ask[:, i] + kfee(ask[:, i]))
                    ok = np.isfinite(e)
                    n_pairs += 1
                    h = ok & (e > 0)
                    hits |= h
                    edges += list(e[h])
                    profits += list(e[h] * np.minimum(bsz[h, i + 1], asz[h, i]))
                both = np.isfinite(bid).sum(axis=1) > 1
                durs = runs_of(hits, ct)
                ladder_rows.append({"venue": venue, "event": ev, "strikes": len(mk), "snapshots": int(both.sum()),
                                    "arb_snapshots": int(hits.sum()), "arb_share": float(hits.sum() / max(both.sum(), 1)),
                                    "mean_edge_c": float(100 * np.mean(edges)) if edges else 0.0,
                                    "mean_profit_usd": float(np.mean(profits)) if profits else 0.0,
                                    "episodes": len(durs), "median_duration_s": float(np.median(durs)) if durs else 0.0})
    A = pd.DataFrame(rows)
    L = pd.DataFrame(ladder_rows)
    A.to_csv(os.path.join(ps.TAB, "real_arb_multioutcome.csv"), index=False)
    L.to_csv(os.path.join(ps.TAB, "real_arb_ladders.csv"), index=False)
    pd.set_option("display.width", 250)
    summ = A.groupby(["venue", "test"]).agg(events=("event", "nunique"), snapshots=("snapshots", "sum"),
                                            arb_snapshots=("arb_snapshots", "sum"),
                                            arb_live_legs=("arb_snapshots_live_legs", "sum"),
                                            events_with_arb=("arb_snapshots", lambda x: int((x > 0).sum())),
                                            mean_edge_c=("mean_edge_c", lambda x: float(x[x > 0].mean()) if (x > 0).any() else 0),
                                            median_dur=("median_duration_s", lambda x: float(x[x > 0].median()) if (x > 0).any() else 0))
    summ["arb_share"] = summ.arb_snapshots / summ.snapshots
    print(summ.round(4).to_string())
    print("midsum (sum of mids) by venue/test:", A.groupby(["venue", "complete_set"]).mean_midsum.describe().round(3).to_string())
    if len(L):
        print("ladders:", len(L), "events; snapshots", int(L.snapshots.sum()), "arb snapshots", int(L.arb_snapshots.sum()),
              "share", round(L.arb_snapshots.sum() / max(L.snapshots.sum(), 1), 5), "events with arb", int((L.arb_snapshots > 0).sum()),
              "mean edge c", round(L[L.arb_snapshots > 0].mean_edge_c.mean(), 2) if (L.arb_snapshots > 0).any() else 0)
    summ.to_csv(os.path.join(ps.TAB, "real_arb_summary.csv"))

    # overround distribution figure (complete Kalshi bucket sets vs Polymarket subsets)
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    k = A[(A.venue == "kalshi") & (A.test == "buy-all")]
    if len(k):
        ax.hist((k.mean_midsum - 1) * 100, bins=25, color=ps.BLUE, alpha=0.9, label="Kalshi complete bucket sets: sum of mids - 100c")
    ax.axvline(0, color=ps.MUTED, lw=1.0)
    ax.set_xlabel("Average overround per event (cents)")
    ax.set_ylabel("Events")
    ax.set_title("Coherence of mutually exclusive sets (live capture)")
    ax.legend(loc="upper right")
    ps.save(fig, "fig_real_overround.png")


if __name__ == "__main__":
    main()
