"""Performance and market-quality metrics computed from a finished simulation (or a replay backtest).

Agent level: settled P&L, P&L path marked at truth, Sharpe of the marked path, max drawdown, volume, fees,
fill counts, maker share, spread capture (fill vs mid at fill), markouts vs truth at several horizons (the
adverse-selection cost of being filled), inventory statistics, and capital carry cost.
Market level: pricing error |mid - truth| (mean, RMSE), quoted spread, top-of-book depth, share of time with
a two-sided book, calibration of mids against realised outcomes.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

TRADE_COLS = ["t", "venue", "market", "price", "qty", "taker_side", "taker", "maker", "taker_kind", "maker_kind",
              "truth", "taker_fee", "maker_fee"]
BOOK_COLS = ["t", "venue", "market", "bid", "ask", "bid_sz", "ask_sz", "last", "volume", "truth"]
AGENT_COLS = ["t", "agent", "venue", "eq_mid", "eq_truth", "cash", "reserved", "gross"]


def frames(sim) -> dict[str, pd.DataFrame]:
    r = sim.recorder
    trades = pd.DataFrame(r.trades, columns=TRADE_COLS)
    books = pd.DataFrame(r.book_rows, columns=BOOK_COLS)
    agents = pd.DataFrame(r.agent_rows, columns=AGENT_COLS)
    pos = pd.DataFrame(r.pos_rows, columns=["t", "agent", "venue", "market", "pos"])
    return {"trades": trades, "books": books, "agents": agents, "pos": pos}


def _truth_lookup(sim):
    w = sim.world

    def f(market, t):
        return w.paths[market][np.clip((np.asarray(t) / w.dt).astype(int), 0, len(w.grid) - 1)]
    return f


def agent_fills(sim, trades: pd.DataFrame, agent: str) -> pd.DataFrame:
    """One row per fill from the agent's point of view: side (+1 bought YES), price, qty, fee, maker flag."""
    tk = trades[trades.taker == agent].assign(side=lambda d: d.taker_side, fee=lambda d: d.taker_fee, maker_flag=0)
    mk = trades[trades.maker == agent].assign(side=lambda d: -d.taker_side, fee=lambda d: d.maker_fee, maker_flag=1)
    return pd.concat([tk, mk]).sort_values("t").reset_index(drop=True)


def markouts(sim, fills: pd.DataFrame, horizons=(5, 30, 120, 600)) -> dict:
    """Average signed (truth_{t+h} - price) * side per contract, in cents; negative = adverse selection."""
    if fills.empty:
        return {f"mo_{h}s": float("nan") for h in horizons}
    look = _truth_lookup(sim)
    out = {}
    for h in horizons:
        vals = []
        for m, g in fills.groupby("market"):
            tr = look(m, np.minimum(g.t.values + h, sim.close_t)) * 100.0
            vals.append(((tr - g.price.values) * g.side.values * g.qty.values))
        v = np.concatenate(vals)
        out[f"mo_{h}s"] = float(v.sum() / fills.qty.sum())
    return out


def spread_capture(books: pd.DataFrame, fills: pd.DataFrame) -> float:
    """Average (mid - price) * side in cents at the last book sample before each fill (positive = earned)."""
    if fills.empty or books.empty:
        return float("nan")
    b = books.dropna(subset=["bid", "ask"]).copy()
    b["mid"] = 0.5 * (b.bid + b.ask)
    tot, q = 0.0, 0.0
    for (v, m), g in fills.groupby(["venue", "market"]):
        bb = b[(b.venue == v) & (b.market == m)]
        if bb.empty:
            continue
        idx = np.searchsorted(bb.t.values, g.t.values, side="right") - 1
        ok = idx >= 0
        mids = bb.mid.values[np.clip(idx, 0, None)]
        tot += float(np.sum(((mids - g.price.values) * g.side.values * g.qty.values)[ok]))
        q += float(g.qty.values[ok].sum())
    return tot / q if q else float("nan")


def path_stats(eq: pd.Series, t: pd.Series, bucket_s: float = 300.0) -> dict:
    """Sharpe of bucketed P&L increments (per bucket, not annualised) and max drawdown of a marked equity path."""
    if len(eq) < 3:
        return {"sharpe_bucket": float("nan"), "max_dd": float("nan")}
    s = pd.Series(eq.values, index=t.values)
    grid = np.arange(t.min(), t.max() + 1e-9, bucket_s)
    sampled = s.reindex(s.index.union(grid)).ffill().loc[grid]
    inc = np.diff(sampled.values)
    sh = float(inc.mean() / inc.std(ddof=1)) if len(inc) > 2 and inc.std(ddof=1) > 0 else float("nan")
    peak = np.maximum.accumulate(s.values)
    return {"sharpe_bucket": sh, "max_dd": float(np.max(peak - s.values))}


def agent_report(sim, fr: dict | None = None, carry_rate: float = 0.04) -> pd.DataFrame:
    fr = fr or frames(sim)
    trades, books, agents = fr["trades"], fr["books"], fr["agents"]
    rows = []
    for aid, ag in sim.agents.items():
        accs = [ex.accounts[aid] for ex in sim.venues.values() if aid in ex.accounts]
        pnl = sum(a.cash + a.reserved - a.cash0 for a in accs)
        fees = sum(a.fees_paid for a in accs)
        vol = sum(a.volume for a in accs)
        f = agent_fills(sim, trades, aid)
        mo = markouts(sim, f)
        a_rows = agents[agents.agent == aid].groupby("t").agg(equity=("eq_truth", "sum"), res=("reserved", "sum"),
                                                              gross=("gross", "sum")).reset_index()
        ps = (path_stats(a_rows["equity"], a_rows["t"]) if len(a_rows)
              else {"sharpe_bucket": float("nan"), "max_dd": float("nan")})
        # capital carry: reserved collateral plus capital tied in open positions, charged at carry_rate p.a.
        if len(a_rows) > 1:
            dt = np.diff(a_rows.t.values)
            locked = a_rows.res.values[:-1]
            carry = float(np.sum(locked * dt) * carry_rate / (365 * 86400))
        else:
            carry = 0.0
        rows.append({"agent": aid, "kind": ag.kind, "pnl": pnl / 100.0, "pnl_after_carry": (pnl - carry) / 100.0,
                     "fees": fees / 100.0, "volume": vol, "fills": len(f),
                     "maker_share": float(f.maker_flag.mean()) if len(f) else float("nan"),
                     "spread_capture_c": spread_capture(books, f) if len(f) else float("nan"),
                     **{k: v for k, v in mo.items()}, **ps,
                     "max_gross": float(a_rows.gross.max()) if len(a_rows) else 0.0})
    return pd.DataFrame(rows)


def market_quality(sim, fr: dict | None = None) -> pd.DataFrame:
    fr = fr or frames(sim)
    b = fr["books"]
    b = b[b.t <= sim.close_t]
    out = []
    for (v, m), g in b.groupby(["venue", "market"]):
        two = g.dropna(subset=["bid", "ask"])
        mid = 0.5 * (two.bid + two.ask) / 100.0
        err = (mid - two.truth).abs()
        out.append({"venue": v, "market": m, "two_sided": len(two) / max(len(g), 1),
                    "spread_c": float((two.ask - two.bid).mean()) if len(two) else float("nan"),
                    "depth_top": float((two.bid_sz + two.ask_sz).mean()) if len(two) else float("nan"),
                    "mae": float(err.mean()) if len(two) else float("nan"),
                    "rmse": float(np.sqrt((err ** 2).mean())) if len(two) else float("nan"),
                    "volume": float(g.volume.max()), "outcome": int(sim.world.outcome(m))})
    return pd.DataFrame(out)


def info_halflife(sim, fr: dict | None = None, min_jump: float = 0.08, horizon: float = 300.0) -> pd.DataFrame:
    """After each truth jump (|dp| >= min_jump within one grid step), time until the mid closes half (and 90%) of
    the gap to the new truth. NaN if never within the horizon."""
    fr = fr or frames(sim)
    b = fr["books"].dropna(subset=["bid", "ask"]).copy()
    b["mid"] = 0.5 * (b.bid + b.ask) / 100.0
    w = sim.world
    rows = []
    for n in w.news:
        i = w.idx(n.t)
        for m, d in w.defs.items():
            if w.fam_of[m] != n.family:
                continue
            p_before, p_after = w.paths[m][max(i - 1, 0)], w.paths[m][min(i + 5, len(w.grid) - 1)]
            if abs(p_after - p_before) < min_jump or n.t > sim.close_t - horizon:
                continue
            for v in sim.venues:
                g = b[(b.venue == v) & (b.market == m) & (b.t >= n.t - 30) & (b.t <= n.t + horizon)]
                pre = g[g.t < n.t]
                post = g[g.t >= n.t]
                if pre.empty or post.empty:
                    continue
                m0 = pre.mid.values[-1]
                gap0 = p_after - m0
                if abs(gap0) < min_jump / 2:
                    continue
                prog = (post.mid.values - m0) / gap0
                def first(th):
                    k = np.argmax(prog >= th)
                    return float(post.t.values[k] - n.t) if np.any(prog >= th) else float("nan")
                rows.append({"t": n.t, "kind": n.kind, "venue": v, "market": m, "jump": float(p_after - p_before),
                             "gap0": float(gap0), "t50": first(0.5), "t90": first(0.9)})
    return pd.DataFrame(rows)
