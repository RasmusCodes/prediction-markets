"""Q3: can market microstructure predict short-term probability changes on a real venue, after costs?

Part A (predictive regressions, Kalshi live capture): for every market with at least 300 recorded quote changes,
a 5 s grid of top-of-book states; features = level-1 imbalance, signed trade-flow imbalance over the past 30 s,
microprice minus mid, and the past 30 s mid change; target = mid change over the next 30 s (cents). Pooled OLS fitted
on the first 60% of the capture (by time) and evaluated on the last 40%, separately for the 15-minute crypto markets
and for everything else. Economic test: a taker who crosses whenever the forecast exceeds half the spread plus the
taker fee, marked to the mid 30 s later.

Part B (replay backtests, ReplayBacktester with queue-position fills, 0.5 s latency, Kalshi fees, settlement at the
official result) on every KXBTC15M / KXETH15M window in the capture:
  naive_mm    joins the best bid and ask (10 contracts) when the spread is at least 2c; flat in the last 90 s
  spot_taker  takes 10 when the spot-implied probability p* beats the ask (bid) by fee + 3c
  spot_mm     quotes p* -/+ 2c (inside or at the touch only), inventory skew 1c per 20 contracts
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
from pmkt.backtest.replay import BUY, SELL, ReplayBacktester  # noqa: E402
from pmkt.core.fees import QuadraticFee  # noqa: E402
from research import plotstyle as ps  # noqa: E402
from research.real_arbitrage import iso_ts, membership  # noqa: E402

LIVE = os.path.join(ROOT, "data", "live", "live.sqlite")
C15 = os.path.join(ROOT, "data", "hist", "crypto15m.sqlite")
H = 30.0


def ro(p):
    if not os.path.exists(p):
        raise SystemExit(f"{p} not found: raw data are not in the repository; build it with collect/ (see data/DATA.md)")
    return sqlite3.connect(f"file:{p}?mode=ro", uri=True)


def kfee_c(p_cents):
    p = np.asarray(p_cents, float) / 100.0
    return 100.0 * 0.07 * p * (1 - p)


# ------------------------------------------------------------------------------------------------ Part A
def build_features(db, u) -> pd.DataFrame:
    cnt = pd.read_sql("SELECT market, COUNT(*) n FROM quotes WHERE venue='kalshi' GROUP BY market", db)
    mk = cnt[cnt.n >= 300].market.tolist()
    iv = membership(u, "kalshi")
    uu = u[u.venue == "kalshi"].drop_duplicates("market", keep="last")
    close = {m: iso_ts(c) for m, c in zip(uu.market, uu.close_time)}
    out = []
    for m in mk:
        q = pd.read_sql("SELECT ts, bid, ask, bid_sz, ask_sz FROM quotes WHERE venue='kalshi' AND market=? ORDER BY ts", db,
                        params=(m,))
        if len(q) < 300:
            continue
        lo, hi = q.ts.min(), q.ts.max()
        if close.get(m):
            hi = min(hi, close[m] - 60)
        grid = np.arange(np.ceil(lo), hi, 5.0)
        if len(grid) < 200:
            continue
        g = q.set_index("ts").reindex(q.ts.tolist() + list(grid)).sort_index()
        g = g[~g.index.duplicated(keep="last")].ffill().reindex(grid)
        ok = (g.bid > 0) & (g.ask < 1) & (g.ask > g.bid) & ((g.ask - g.bid) <= 0.10) & (g.bid_sz > 0) & (g.ask_sz > 0)
        mem = np.zeros(len(grid), bool)
        for a, b in iv.get(m, [(lo, hi + 1)]):
            mem |= (grid >= a) & (grid < b)
        if m.startswith(("KXBTC15M", "KXETH15M")):
            mem[:] = True          # hot-list markets are polled continuously, not via the universe table
        g = g[ok.values & mem]
        if len(g) < 100:
            continue
        t = g.index.values
        mid = 100 * 0.5 * (g.bid.values + g.ask.values)
        tr = pd.read_sql("SELECT ts, size, taker_side FROM trades WHERE venue='kalshi' AND market=? ORDER BY ts", db,
                         params=(m,))
        s = np.where(tr.taker_side.values == "yes", 1.0, -1.0) * tr["size"].values
        ts = tr.ts.values
        cs = np.concatenate([[0.0], np.cumsum(s)])
        ca = np.concatenate([[0.0], np.cumsum(np.abs(s))])
        i1 = np.searchsorted(ts, t, side="right")
        i0 = np.searchsorted(ts, t - 30.0, side="left")
        vol = ca[i1] - ca[i0]
        ofi = np.where(vol > 0, (cs[i1] - cs[i0]) / np.maximum(vol, 1e-9), 0.0)
        imb = ((g.bid_sz - g.ask_sz) / (g.bid_sz + g.ask_sz)).values
        micro = 100 * ((g.bid * g.ask_sz + g.ask * g.bid_sz) / (g.bid_sz + g.ask_sz)).values - mid
        # future / past mid change on the 5 s grid (only where the grid is contiguous)
        tser = pd.Series(mid, index=t)
        fut = tser.reindex(t + H).values - mid
        past = mid - tser.reindex(t - H).values
        spread = 100 * (g.ask - g.bid).values
        out.append(pd.DataFrame({"market": m, "t": t, "mid": mid, "spread": spread, "imb": imb, "ofi": ofi,
                                 "micro": micro, "past": past, "fut": fut, "vol30": vol,
                                 "group": "crypto15m" if m.startswith(("KXBTC15M", "KXETH15M")) else "other"}))
    return pd.concat(out, ignore_index=True).dropna(subset=["fut", "past"])


def predictive(F: pd.DataFrame) -> pd.DataFrame:
    rows = []
    split = F.t.quantile(0.6)
    X = ["imb", "ofi", "micro", "past"]
    for grp, d in F.groupby("group"):
        tr, te = d[d.t < split], d[d.t >= split]
        if len(tr) < 500 or len(te) < 300:
            continue
        A = np.column_stack([np.ones(len(tr)), tr[X].values])
        b, *_ = np.linalg.lstsq(A, tr.fut.values, rcond=None)
        pred = np.column_stack([np.ones(len(te)), te[X].values]) @ b
        y = te.fut.values
        r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
        nz = y != 0
        cost = te.spread.values / 2 + kfee_c(te.mid.values)
        sel = np.abs(pred) > cost
        edge = np.sign(pred[sel]) * y[sel] - cost[sel]
        # cluster bootstrap (by market) of the taker edge
        mk = te.market.values[sel]
        rng = np.random.default_rng(0)
        um = np.unique(mk)
        pos = {x: np.where(mk == x)[0] for x in um}
        bs = [np.mean(np.concatenate([edge[pos[x]] for x in rng.choice(um, len(um))])) for _ in range(300)] if len(um) else [np.nan]
        r = {"group": grp, "markets": d.market.nunique(), "n_train": len(tr), "n_test": len(te), "r2_oos": float(r2),
             "hit_rate": float(np.mean(np.sign(pred[nz]) == np.sign(y[nz]))) if nz.any() else np.nan,
             "share_nonzero_moves": float(nz.mean()), "mean_abs_move_c": float(np.abs(y).mean()),
             "mean_cost_c": float(cost.mean()), "trades_signalled": int(sel.sum()),
             "taker_edge_c": float(edge.mean()) if sel.any() else np.nan,
             "taker_edge_lo": float(np.nanquantile(bs, 0.025)) if sel.any() else np.nan,
             "taker_edge_hi": float(np.nanquantile(bs, 0.975)) if sel.any() else np.nan}
        for name, coef in zip(["const"] + X, b):
            r["b_" + name] = float(coef)
        # univariate correlations on the test set
        for x in X:
            r["corr_" + x] = float(np.corrcoef(te[x].values, y)[0, 1])
        rows.append(r)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------ Part B
class NaiveMM:
    def __init__(self, size=10):
        self.size = size
    def on_quote(self, bt, t, q):
        if bt.close_t - t < 90:
            bt.quote(BUY, None); bt.quote(SELL, None)
            return
        if np.isnan(q.bid) or np.isnan(q.ask) or q.ask - q.bid < 2:
            bt.quote(BUY, None); bt.quote(SELL, None)
            return
        bt.quote(BUY, int(q.bid), self.size)
        bt.quote(SELL, int(q.ask), self.size)
    def on_trade(self, bt, t, tr):
        pass


class SpotTaker:
    def __init__(self, pstar: pd.Series, margin=3.0, size=10):
        self.p, self.margin, self.size = pstar, margin, size
    def _p(self, t):
        i = self.p.index.searchsorted(t, side="right") - 1
        return None if i < 0 else 100 * float(self.p.iloc[i])
    def on_quote(self, bt, t, q):
        if bt.close_t - t < 90:
            return
        p = self._p(t)
        if p is None:
            return
        if not np.isnan(q.ask) and p - q.ask - kfee_c(q.ask) > self.margin:
            bt.take(BUY, int(q.ask), self.size)
        elif not np.isnan(q.bid) and q.bid - p - kfee_c(q.bid) > self.margin:
            bt.take(SELL, int(q.bid), self.size)
    def on_trade(self, bt, t, tr):
        pass


class SpotMM(SpotTaker):
    def __init__(self, pstar, half=2.0, size=10):
        super().__init__(pstar, 0, size)
        self.half = half
    def on_quote(self, bt, t, q):
        p = self._p(t)
        if p is None or bt.close_t - t < 90 or np.isnan(q.bid) or np.isnan(q.ask):
            bt.quote(BUY, None); bt.quote(SELL, None)
            return
        skew = bt.pos / 20.0
        b = int(np.floor(p - self.half - skew))
        a = int(np.ceil(p + self.half - skew))
        b = min(b, int(q.ask) - 1)                 # never cross: at most join/improve
        a = max(a, int(q.bid) + 1)
        bt.quote(BUY, b if 1 <= b <= 98 and b >= q.bid - 5 else None, self.size)
        bt.quote(SELL, a if 2 <= a <= 99 and a <= q.ask + 5 else None, self.size)


def replay_crypto(db) -> pd.DataFrame:
    km = pd.read_sql("SELECT ticker, open_ts, close_ts, result FROM k_markets WHERE result IN ('yes','no')", ro(C15))
    lo, hi = db.execute("SELECT MIN(ts), MAX(ts) FROM spot").fetchone()
    km = km[(km.open_ts >= lo) & (km.close_ts <= hi)]
    sp = pd.read_pickle(os.path.join(ROOT, "data", "processed_speed_live.pkl"))
    rows = []
    fee = QuadraticFee(0.07, 0.0)
    for m in km.itertuples():
        q = pd.read_sql("SELECT ts AS t, bid, ask, bid_sz, ask_sz FROM quotes WHERE venue='kalshi' AND market=? ORDER BY ts",
                        db, params=(m.ticker,))
        tr = pd.read_sql("SELECT ts AS t, price, size AS qty, taker_side FROM trades WHERE venue='kalshi' AND market=? ORDER BY ts",
                         db, params=(m.ticker,))
        if len(q) < 30 or len(tr) < 100:
            continue
        q.loc[(q.bid <= 0) | (q.bid_sz <= 0), "bid"] = np.nan
        q.loc[(q.ask >= 1) | (q.ask_sz <= 0), "ask"] = np.nan
        q["bid"] = (q.bid * 100).round()
        q["ask"] = (q.ask * 100).round()
        tr["price"] = (tr.price * 100).round().astype(int)
        tr["taker_side"] = np.where(tr.taker_side == "yes", BUY, SELL)
        tr["qty"] = tr.qty.astype(float)
        ps_ = sp[sp.ticker == m.ticker].set_index("t").pstar
        res = 1 if m.result == "yes" else 0
        for name, strat in (("naive_mm", NaiveMM()), ("spot_taker", SpotTaker(ps_)), ("spot_mm", SpotMM(ps_))):
            if name != "naive_mm" and ps_.empty:
                continue
            bt = ReplayBacktester(q, tr, fee=fee, latency=0.5, result=res, max_pos=200)
            bt.close_t = m.close_ts
            r = bt.run(strat)
            rows.append({"ticker": m.ticker, "series": m.ticker.split("-")[0], "strategy": name, "pnl": r.pnl,
                         "fees": r.fees, "fills": len(r.fills), "contracts": float(r.fills.qty.sum()) if len(r.fills) else 0.0,
                         "maker_share": float(r.fills.maker.mean()) if len(r.fills) else np.nan, "final_pos": r.position})
    return pd.DataFrame(rows)


def main():
    ps.apply()
    db = ro(LIVE)
    u = pd.read_sql("SELECT ts, venue, market, event, group_key, mutually_exclusive, close_time, extra FROM universe", db)
    print("features ...", flush=True)
    F = build_features(db, u)
    print(F.groupby("group").agg(markets=("market", "nunique"), n=("t", "size")).to_string(), flush=True)
    P = predictive(F)
    P.to_csv(os.path.join(ps.TAB, "real_microstructure_predictive.csv"), index=False)
    pd.set_option("display.width", 250)
    print(P.round(4).T.to_string(), flush=True)
    print("replay ...", flush=True)
    R = replay_crypto(db)
    R.to_csv(os.path.join(ps.TAB, "real_replay_crypto15m_bymarket.csv"), index=False)
    S = R.groupby("strategy").agg(markets=("ticker", "nunique"), total_pnl=("pnl", "sum"), mean_pnl=("pnl", "mean"),
                                  sd_pnl=("pnl", "std"), win_share=("pnl", lambda x: float((x > 0).mean())),
                                  fees=("fees", "sum"), contracts=("contracts", "sum"), maker_share=("maker_share", "mean"))
    S["t_stat"] = S.mean_pnl / (S.sd_pnl / np.sqrt(S.markets))
    S["pnl_per_contract_c"] = 100 * S.total_pnl / S.contracts.replace(0, np.nan)
    S.to_csv(os.path.join(ps.TAB, "real_replay_crypto15m_summary.csv"))
    print(S.round(3).to_string())


if __name__ == "__main__":
    main()
