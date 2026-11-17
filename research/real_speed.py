"""Q1: how fast does a prediction market incorporate public information?

Natural experiment: Kalshi's 15-minute crypto markets (KXBTC15M / KXETH15M) pay $1 if the 60-second average of the
CF Benchmarks index before the window's end is at least the 60-second average before its start (the strike).
The information driving them, the spot price, is public and observable every second. With spot S (Coinbase,
a constituent and close proxy of the index), strike K, per-second log volatility sigma and time left tau, the
spot-implied probability is

    p*(t) = Phi( (ln S_t - ln K + m_t) / (sigma * sqrt(V_t)) )
    V_t = (tau - 60) + 20                 if tau > 60   (remaining path + variance of the final 60 s average)
        = tau^3 / (3 * 60^2)              if tau <= 60  (only the unaveraged remainder is random)
    m_t = the already-fixed part of the average inside the final minute (0 before it)

Measurements
  live (13.9 h, 1 s grid): Kalshi price = 1-second VWAP of trade prints (~100 prints/s), forward-filled; regress
       the cumulative Kalshi change over [t, t+l] on the spot-implied change dp*_t; beta(l)/beta(120 s) is the share
       of a spot-implied move reflected after l seconds. Also the lead-lag cross-correlation.
  history (30 days, 1-minute candles): regress the Kalshi mid change on contemporaneous and lagged dp*.
Volatility: EWMA of squared 1 s (live) or 1 min (history) spot log returns, half-life 10 min, pre-window only.
"""
from __future__ import annotations

import os
import sqlite3
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from research import plotstyle as ps  # noqa: E402

LIVE = os.path.join(ROOT, "data", "live", "live.sqlite")
C15 = os.path.join(ROOT, "data", "hist", "crypto15m.sqlite")
SYM = {"KXBTC15M": "BTC-USD", "KXETH15M": "ETH-USD"}
LAGS = [0, 1, 2, 3, 5, 8, 10, 15, 20, 30, 45, 60, 90, 120]


def ro(p):
    if not os.path.exists(p):
        raise SystemExit(f"{p} not found: raw data are not in the repository; build it with collect/ (see data/DATA.md)")
    return sqlite3.connect(f"file:{p}?mode=ro", uri=True)


def implied_prob(lnS, lnK, tau, sigma, avg_known=None):
    """Vectorised p* given log spot, log strike, seconds left, per-second sigma; avg_known = running mean of ln S
    over the elapsed part of the final minute (None/NaN before it)."""
    tau = np.asarray(tau, float)
    lnS = np.asarray(lnS, float)
    avg = lnS if avg_known is None else np.where(np.isfinite(avg_known), avg_known, lnS)
    V = np.where(tau > 60, (tau - 60) + 20.0, tau ** 3 / (3 * 3600.0))
    mean = np.where(tau > 60, lnS, ((60 - tau) * avg + tau * lnS) / 60.0)
    sd = sigma * np.sqrt(np.maximum(V, 1e-9))
    return norm.cdf((mean - lnK) / np.maximum(sd, 1e-12))


def ewma_sigma(r: np.ndarray, dt: float, halflife_s: float = 600.0) -> np.ndarray:
    """Per-second sigma from returns sampled every dt seconds (EWMA of squares, uses data up to t-1)."""
    lam = 0.5 ** (dt / halflife_s)
    v = np.empty(len(r))
    acc = np.nanmean(r[:50] ** 2) if len(r) > 50 else np.nanmean(r ** 2)
    for i, x in enumerate(r):
        v[i] = acc
        if np.isfinite(x):
            acc = lam * acc + (1 - lam) * x * x
    return np.sqrt(v / dt)


# ------------------------------------------------------------------------------------------------ live (1 s)
def live_panel() -> pd.DataFrame:
    km = pd.read_sql("SELECT ticker, series_ticker, open_ts, close_ts, floor_strike, result FROM k_markets", ro(C15))
    lv = ro(LIVE)
    lo, hi = lv.execute("SELECT MIN(ts), MAX(ts) FROM spot").fetchone()
    km = km[(km.open_ts >= lo) & (km.close_ts <= hi)]
    spot = pd.read_sql("SELECT ts, symbol, price, bid, ask FROM spot", lv)
    out = []
    for sym_series, sym in SYM.items():
        sp = spot[spot.symbol == sym].sort_values("ts")
        if sp.empty:
            continue
        grid = np.arange(np.floor(sp.ts.min()), np.ceil(sp.ts.max()))
        mid = 0.5 * (sp.bid + sp.ask)
        lnS = np.log(pd.Series(mid.values, index=sp.ts.values).groupby(np.floor(sp.ts.values)).last()
                     .reindex(grid).ffill().values)
        r = np.diff(lnS, prepend=lnS[0])
        gap = pd.Series(np.floor(sp.ts.values)).diff().max()
        sig = ewma_sigma(r, 1.0)
        sgrid = pd.DataFrame({"t": grid, "lnS": lnS, "sigma": sig})
        mk = km[km.series_ticker == sym_series]
        tickers = tuple(mk.ticker)
        if not tickers:
            continue
        q = ",".join("?" * len(tickers))
        tr = pd.read_sql(f"SELECT market, ts, price, size FROM trades WHERE venue='kalshi' AND market IN ({q})", lv,
                         params=tickers)
        for m in mk.itertuples():
            tt = tr[tr.market == m.ticker]
            if len(tt) < 200:
                continue
            g = sgrid[(sgrid.t >= m.open_ts) & (sgrid.t < m.close_ts)].copy()
            if len(g) < 600:
                continue
            sec = np.floor(tt.ts.values)
            vw = (pd.DataFrame({"s": sec, "pv": tt.price.values * tt["size"].values, "v": tt["size"].values})
                  .groupby("s").sum())
            kp = (vw.pv / vw.v).reindex(g.t.values).ffill()
            g["k"] = kp.values
            g["ntr"] = pd.Series(tt.groupby(sec).size()).reindex(g.t.values).fillna(0).values
            tau = m.close_ts - g.t.values
            # running average of ln S within the final minute
            in_last = tau <= 60
            run = np.full(len(g), np.nan)
            if in_last.any():
                idx = np.where(in_last)[0]
                run[idx] = np.cumsum(g.lnS.values[idx]) / np.arange(1, len(idx) + 1)
            sigma0 = float(g.sigma.values[0])           # pre-window volatility, held fixed inside the window
            g["pstar"] = implied_prob(g.lnS.values, np.log(m.floor_strike), tau, sigma0, run)
            g["tau"] = tau
            g["ticker"] = m.ticker
            g["series"] = sym_series
            g["result"] = 1 if m.result == "yes" else 0
            out.append(g)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def impulse_response(d: pd.DataFrame, price_col: str = "k", n_boot: int = 200) -> pd.DataFrame:
    """beta(l): regress k_{t+l} - k_{t-1} on dp*_t (all in probability points), pooled, market-clustered bootstrap."""
    rows = []
    rng = np.random.default_rng(0)
    pieces = []
    for tk, g in d.groupby("ticker"):
        g = g.sort_values("t")
        k = g[price_col].values
        ps_ = g.pstar.values
        dp = np.diff(ps_, prepend=np.nan)
        base = np.roll(k, 1)
        base[0] = np.nan
        cols = {"ticker": tk, "dp": dp, "tau": g.tau.values}
        for l in LAGS:
            fut = np.full(len(k), np.nan)
            if l < len(k):
                fut[:len(k) - l] = k[l:]
            cols[f"r{l}"] = fut - base
        pieces.append(pd.DataFrame(cols))
    P = pd.concat(pieces).dropna()
    P = P[(P.tau > 150) & (P.dp.abs() > 1e-4)]       # skip the last 2.5 min (lags would cross the close)
    tk = P.ticker.values
    uniq = np.unique(tk)
    idx = {u: np.where(tk == u)[0] for u in uniq}
    for l in LAGS:
        y, x = P[f"r{l}"].values, P.dp.values
        b = float(np.sum(x * y) / np.sum(x * x))
        bs = []
        for _ in range(n_boot):
            ii = np.concatenate([idx[u] for u in rng.choice(uniq, len(uniq))])
            bs.append(np.sum(x[ii] * y[ii]) / np.sum(x[ii] ** 2))
        rows.append({"lag_s": l, "beta": b, "lo": float(np.quantile(bs, 0.025)), "hi": float(np.quantile(bs, 0.975)),
                     "n": len(P), "markets": len(uniq)})
    R = pd.DataFrame(rows)
    R["share"] = R.beta / R.beta.iloc[-1]
    return R


EV_GRID = np.arange(-10, 91)


def event_study(d: pd.DataFrame, q: float = 0.98, settle: int = 10, n_boot: int = 300) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Large 5-second spot moves (top 1-q of |dlnS_5s| within each series), at least 60 s apart and 180+ s before
    the close. D = p*(t+settle) - p*(t-5) is the information content; Kalshi's catch-up curve is
    (k(t+s) - k(t-5)) / D, sign-aligned, averaged across events. Returns (curve, per-event table)."""
    ev_rows, curves = [], []
    for ser, gs in d.groupby("series"):
        dl = []
        for _, g in gs.groupby("ticker"):
            dl.append(np.abs(g.lnS.values[5:] - g.lnS.values[:-5]))
        thr = np.quantile(np.concatenate(dl), q)
        for tk, g in gs.groupby("ticker"):
            g = g.sort_values("t").reset_index(drop=True)
            lnS, ps_, k, tau = g.lnS.values, g.pstar.values, g.k.values, g.tau.values
            last = -1e9
            for i in range(10, len(g) - 95):
                if tau[i] < 180 or i - last < 60:
                    continue
                if abs(lnS[i] - lnS[i - 5]) < thr:
                    continue
                D = ps_[i + settle] - ps_[i - 5]
                if abs(D) < 0.02 or not np.isfinite(k[i - 5]):
                    continue
                last = i
                sgn = np.sign(D)
                kc = (k[i + EV_GRID] - k[i - 5]) / D
                pc = (ps_[i + EV_GRID] - ps_[i - 5]) / D
                curves.append((ser, kc, pc))
                ev_rows.append({"series": ser, "ticker": tk, "t": g.t.values[i], "D": D, "tau": tau[i],
                                "k_share_30": kc[EV_GRID == 30][0], "k_share_60": kc[EV_GRID == 60][0]})
    E = pd.DataFrame(ev_rows)
    rows = []
    rng = np.random.default_rng(2)
    for ser in sorted(set(c[0] for c in curves)):
        K = np.array([c[1] for c in curves if c[0] == ser])
        Pp = np.array([c[2] for c in curves if c[0] == ser])
        med_k = np.nanmedian(K, 0)
        mean_k = np.nanmean(np.clip(K, -2, 3), 0)
        bs = np.array([np.nanmedian(K[rng.integers(0, len(K), len(K))], 0) for _ in range(n_boot)])
        for j, s in enumerate(EV_GRID):
            rows.append({"series": ser, "s": int(s), "kalshi_median": med_k[j], "kalshi_mean": mean_k[j],
                         "lo": np.quantile(bs[:, j], 0.025), "hi": np.quantile(bs[:, j], 0.975),
                         "pstar_median": float(np.nanmedian(Pp[:, j])), "events": len(K)})
    return pd.DataFrame(rows), E


def first_cross(curve: pd.DataFrame, level: float) -> float:
    c = curve[curve.s >= 0]
    hit = c[c.kalshi_median >= level]
    return float(hit.s.iloc[0]) if len(hit) else float("nan")


def lead_lag(d: pd.DataFrame, max_lag: int = 20) -> pd.DataFrame:
    rows = []
    for l in range(-max_lag, max_lag + 1):
        xs, ys = [], []
        for _, g in d.groupby("ticker"):
            g = g[g.tau > 90]
            dp, dk = np.diff(g.pstar.values), np.diff(g.k.values)
            if l >= 0:
                xs.append(dp[:len(dp) - l]); ys.append(dk[l:])
            else:
                xs.append(dp[-l:]); ys.append(dk[:len(dk) + l])
        x, y = np.concatenate(xs), np.concatenate(ys)
        ok = np.isfinite(x) & np.isfinite(y)
        rows.append({"lag_s": l, "corr": float(np.corrcoef(x[ok], y[ok])[0, 1])})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------ history (1 min)
def hist_minute_regression() -> tuple[pd.DataFrame, dict]:
    c = ro(C15)
    km = pd.read_sql("SELECT ticker, series_ticker, open_ts, close_ts, floor_strike, result FROM k_markets WHERE result IN ('yes','no')", c)
    kc = pd.read_sql("SELECT ticker, end_ts, bid_close, ask_close FROM k_candles", c)
    cb = pd.read_sql("SELECT product, ts, close FROM cb_candles", c)
    rows = []
    for ser, sym in SYM.items():
        s = cb[cb["product"] == sym].sort_values("ts").drop_duplicates("ts")
        s = s.set_index("ts").close
        full = s.reindex(np.arange(s.index.min(), s.index.max() + 60, 60)).ffill()
        lnS = np.log(full.values)
        r = np.diff(lnS, prepend=lnS[0])
        sig = pd.Series(ewma_sigma(r, 60.0), index=full.index)
        lns = pd.Series(lnS, index=full.index)
        mk = km[km.series_ticker == ser].set_index("ticker")
        g = kc[kc.ticker.isin(mk.index)].copy()
        g = g[(g.bid_close > 0) & (g.ask_close < 1) & (g.ask_close > g.bid_close)]
        g["mid"] = 0.5 * (g.bid_close + g.ask_close)
        g["spread"] = g.ask_close - g.bid_close
        g = g.join(mk[["open_ts", "close_ts", "floor_strike", "result"]], on="ticker")
        # Coinbase candle 'ts' is the minute START; the Kalshi candle end_ts is the minute END -> spot at end_ts - 60
        g["lnS"] = lns.reindex(g.end_ts.values - 60).values
        g["sig"] = sig.reindex(g.open_ts.values.astype(int) // 60 * 60).values
        g["tau"] = g.close_ts - g.end_ts
        g = g[(g.tau >= 60) & g.lnS.notna() & g.sig.notna()]
        g["pstar"] = implied_prob(g.lnS.values, np.log(g.floor_strike.values), g.tau.values, g.sig.values)
        g["series"] = ser
        rows.append(g)
    H = pd.concat(rows).sort_values(["ticker", "end_ts"])
    H["dmid"] = H.groupby("ticker").mid.diff()
    H["dp0"] = H.groupby("ticker").pstar.diff()
    H["dp1"] = H.groupby("ticker").dp0.shift(1)
    H["dp2"] = H.groupby("ticker").dp0.shift(2)
    H["gap"] = H.groupby("ticker").pstar.shift(1) - H.groupby("ticker").mid.shift(1)   # last minute's mispricing
    X = H.dropna(subset=["dmid", "dp0", "dp1", "dp2", "gap"])
    X = X[X.spread <= 0.05]
    A = np.column_stack([np.ones(len(X)), X.dp0, X.dp1, X.dp2, X.gap])
    beta, *_ = np.linalg.lstsq(A, X.dmid.values, rcond=None)
    # market-clustered bootstrap
    rng = np.random.default_rng(1)
    tk = X.ticker.values
    uniq = np.unique(tk)
    pos = {u: np.where(tk == u)[0] for u in uniq}
    bs = []
    for _ in range(200):
        ii = np.concatenate([pos[u] for u in rng.choice(uniq, len(uniq))])
        bb, *_ = np.linalg.lstsq(A[ii], X.dmid.values[ii], rcond=None)
        bs.append(bb)
    bs = np.array(bs)
    names = ["const", "dp*_t", "dp*_t-1", "dp*_t-2", "gap_t-1"]
    T = pd.DataFrame({"term": names, "coef": beta, "lo": np.quantile(bs, 0.025, 0), "hi": np.quantile(bs, 0.975, 0)})
    resid = X.dmid.values - A @ beta
    info = {"n": len(X), "markets": len(uniq), "r2": float(1 - resid.var() / X.dmid.var()),
            "days": float((H.end_ts.max() - H.end_ts.min()) / 86400),
            "mean_abs_gap_pp": float(100 * X.gap.abs().mean()),
            "brier_mid_at_5min": None}
    # accuracy check of the spot model vs Kalshi 5 minutes before close
    last5 = H[(H.tau >= 290) & (H.tau <= 310)].dropna(subset=["mid", "pstar"])
    yy = (last5.result == "yes").astype(int)
    info["brier_mid_5min"] = float(np.mean((last5.mid - yy) ** 2))
    info["brier_pstar_5min"] = float(np.mean((last5.pstar - yy) ** 2))
    info["n_5min"] = len(last5)
    return T, info


def main():
    ps.apply()
    import matplotlib.pyplot as plt
    print("live panel ...", flush=True)
    d = live_panel()
    print(d.groupby("series").ticker.nunique().to_dict(), len(d), flush=True)
    d.to_pickle(os.path.join(ROOT, "data", "processed_speed_live.pkl"))
    out = {}
    for ser, g in d.groupby("series"):
        R = impulse_response(g)
        R["series"] = ser
        out[ser] = R
        print(ser, R[["lag_s", "beta", "lo", "hi", "share"]].round(3).to_string(), flush=True)
    IR = pd.concat(out.values())
    IR.to_csv(os.path.join(ps.TAB, "speed_impulse_response_live.csv"), index=False)
    LL = lead_lag(d[d.series == "KXBTC15M"])
    LL.to_csv(os.path.join(ps.TAB, "speed_leadlag_live_btc.csv"), index=False)
    # mispricing vs spot model inside windows (live)
    mis = d[(d.tau > 120)].assign(gap=lambda x: (x.pstar - x.k).abs())
    print("live mean |p* - kalshi| (pp):", round(100 * mis.gap.mean(), 2), "median", round(100 * mis.gap.median(), 2))
    EVc, EVt = event_study(d)
    EVc.to_csv(os.path.join(ps.TAB, "speed_event_curve_live.csv"), index=False)
    EVt.to_csv(os.path.join(ps.TAB, "speed_events_live.csv"), index=False)
    halfl = {}
    for ser, c in EVc.groupby("series"):
        halfl[ser] = {"events": int(c.events.iloc[0]), "t50_s": first_cross(c, 0.5), "t90_s": first_cross(c, 0.9),
                      "share_at_0s": float(c[c.s == 0].kalshi_median.iloc[0]),
                      "share_at_10s": float(c[c.s == 10].kalshi_median.iloc[0]),
                      "share_at_30s": float(c[c.s == 30].kalshi_median.iloc[0]),
                      "share_at_60s": float(c[c.s == 60].kalshi_median.iloc[0])}
    print("event study:", halfl, flush=True)
    pd.DataFrame(halfl).T.to_csv(os.path.join(ps.TAB, "speed_event_summary.csv"))
    T, info = hist_minute_regression()
    T.to_csv(os.path.join(ps.TAB, "speed_minute_regression_hist.csv"), index=False)
    print(T.round(4).to_string(), info, flush=True)
    pd.Series(info).to_csv(os.path.join(ps.TAB, "speed_minute_regression_info.csv"))

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))
    ax = axes[0]
    for i, (ser, c) in enumerate(EVc.groupby("series")):
        col = ps.SERIES[i]
        lab = "BTC" if "BTC" in ser else "ETH"
        ax.fill_between(c.s, c.lo, c.hi, color=col, alpha=0.15, lw=0)
        ax.plot(c.s, c.kalshi_median, "-", color=col, lw=1.8, label=f"Kalshi {lab} 15-min (n={int(c.events.iloc[0])})")
        ax.text(c.s.iloc[-1] + 1.5, c.kalshi_median.iloc[-1], lab, color=ps.INK2, fontsize=8.5, va="center")
    c0 = EVc[EVc.series == EVc.series.iloc[0]]
    ax.plot(c0.s, c0.pstar_median, "--", color=ps.MUTED, lw=1.2, label="Spot-implied p* (BTC)")
    ax.axhline(1.0, color=ps.MUTED, lw=0.8, ls=":")
    ax.axvline(0, color=ps.AXIS, lw=0.8)
    ax.set_xlabel("Seconds after a large 5 s spot move")
    ax.set_ylabel("Kalshi move / spot-implied move (median)")
    ax.set_title("Catch-up after spot moves")
    ax.legend(loc="lower right")
    ax = axes[1]
    ax.bar(LL.lag_s, LL["corr"], width=0.8, color=ps.BLUE)
    ax.axvline(0, color=ps.MUTED, lw=0.8)
    ax.set_xlabel("Lag l (s): corr(dp*_t, dKalshi_t+l)")
    ax.set_title("Lead-lag, BTC (1 s changes)")
    ps.save(fig, "fig_speed_impulse.png")
    print("done", flush=True)


if __name__ == "__main__":
    main()
