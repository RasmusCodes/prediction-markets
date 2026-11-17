"""Metric functions for the simulation experiments (run inside worker processes by pmkt.backtest.runner)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from pmkt.backtest.metrics import agent_report, frames, info_halflife, market_quality
from pmkt.backtest.runner import summarize_sim
def _with_monitor(sim, ctx):
    return next((a for a in sim.agents.values() if a.kind == "monitor"), None)


def e1_metrics(sim, ctx) -> dict:
    """Information speed, pricing efficiency, liquidity, P&L by trader type (default bundle + extras)."""
    return summarize_sim(sim, ctx)


def e2_metrics(sim, ctx) -> dict:
    """Calibration sample: one row per market with mid / truth at fixed fractions of the session, plus outcome.
    Returned as a JSON-able dict of lists so the parent can pool across seeds."""
    fr = frames(sim)
    b = fr["books"].dropna(subset=["bid", "ask"])
    rows = []
    for (v, m), g in b.groupby(["venue", "market"]):
        g = g.sort_values("t")
        for frac in (0.25, 0.5, 0.9):
            tt = frac * sim.close_t
            k = np.searchsorted(g.t.values, tt, side="right") - 1
            if k < 0:
                continue
            r = g.iloc[k]
            rows.append({"market": m, "frac": frac, "mid": 0.5 * (r.bid + r.ask) / 100.0, "spread": (r.ask - r.bid) / 100.0,
                         "truth": float(r.truth), "outcome": int(sim.world.outcome(m))})
    out = summarize_sim(sim, ctx)
    out["calib_rows"] = rows
    return out


def e3_metrics(sim, ctx, horizon: float = 30.0) -> dict:
    """Microstructure predictability: features at each 5 s sample -> next-`horizon` mid change; train on the first
    half of the session, test on the second half (per run)."""
    fr = frames(sim)
    b = fr["books"].dropna(subset=["bid", "ask"]).copy()
    tr = fr["trades"]
    b["mid"] = 0.5 * (b.bid + b.ask)
    feats = []
    for (v, m), g in b.groupby(["venue", "market"]):
        g = g.sort_values("t").reset_index(drop=True)
        if len(g) < 50:
            continue
        t = g.t.values
        mid = g.mid.values
        step = int(round(horizon / sim.recorder.dt))
        fut = np.full(len(g), np.nan)
        fut[:-step] = mid[step:] - mid[:-step]
        imb = (g.bid_sz - g.ask_sz) / (g.bid_sz + g.ask_sz).replace(0, np.nan)
        tt = tr[(tr.venue == v) & (tr.market == m)]
        signed = (tt.taker_side * tt.qty).values
        ts = tt.t.values
        cs = np.concatenate([[0.0], np.cumsum(signed)])
        lo = np.searchsorted(ts, t - 30.0, side="left")
        hi = np.searchsorted(ts, t, side="right")
        flow = cs[hi] - cs[lo]
        vol = np.concatenate([[0.0], np.cumsum(np.abs(signed))])
        tot = vol[hi] - vol[lo]
        ofi = np.where(tot > 0, flow / np.maximum(tot, 1e-9), 0.0)
        truth_gap = g.truth.values * 100.0 - mid
        feats.append(pd.DataFrame({"t": t, "fut": fut, "imb": imb.values, "ofi": ofi, "spread": (g.ask - g.bid).values,
                                   "half": 0.5 * (g.ask - g.bid).values, "truth_gap": truth_gap}))
    if not feats:
        return {"n": 0}
    d = pd.concat(feats).dropna()
    d = d[d.t < sim.close_t - horizon]
    split = 0.5 * sim.close_t
    trn, tst = d[d.t < split], d[d.t >= split]
    X = ["imb", "ofi"]
    out = {"n_train": len(trn), "n_test": len(tst)}
    if len(trn) > 50 and len(tst) > 50:
        A = np.column_stack([np.ones(len(trn)), trn[X].values])
        beta, *_ = np.linalg.lstsq(A, trn.fut.values, rcond=None)
        pred = np.column_stack([np.ones(len(tst)), tst[X].values]) @ beta
        ss_res = np.sum((tst.fut.values - pred) ** 2)
        ss_tot = np.sum((tst.fut.values - tst.fut.values.mean()) ** 2)
        nz = tst.fut.values != 0
        out.update({"r2_oos": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
                    "hit": float(np.mean(np.sign(pred[nz]) == np.sign(tst.fut.values[nz]))) if nz.any() else float("nan"),
                    "b_imb": float(beta[1]), "b_ofi": float(beta[2]),
                    "frac_pred_gt_half": float(np.mean(np.abs(pred) > tst.half.values)),
                    "corr_truthgap": float(np.corrcoef(tst.truth_gap, tst.fut)[0, 1])})
        # naive cost-aware taker rule on the test half: cross when |pred| > half-spread, earn realised move - half
        sel = np.abs(pred) > tst.half.values
        if sel.any():
            out["taker_edge_c"] = float(np.mean(np.sign(pred[sel]) * tst.fut.values[sel] - tst.half.values[sel]))
            out["taker_n"] = int(sel.sum())
    out.update({k: v for k, v in summarize_sim(sim, ctx).items() if k in ("mae", "spread_c")})
    return out


def e4_metrics(sim, ctx) -> dict:
    """Arbitrage incidence from the read-only monitor, plus P&L of arbitrage-taking agents."""
    mon = _with_monitor(sim, ctx)
    out = summarize_sim(sim, ctx)
    if mon is None:
        return out
    log = pd.DataFrame(mon.log, columns=["t", "family", "kind", "profit", "capital"])
    out["scans"] = mon.n_scans
    out["opp_scans"] = int(log.t.nunique()) if len(log) else 0
    out["opp_share"] = out["opp_scans"] / max(mon.n_scans, 1)
    out["opp_profit_mean_c"] = float(log.profit.mean()) if len(log) else 0.0
    out["opp_profit_sum_c"] = float(log.profit.sum()) if len(log) else 0.0
    for k in ("cross-venue", "bucket", "ladder", "combo", "binary"):
        sub = log[log.kind.str.contains(k)] if len(log) else log
        out[f"opp_{k}"] = int(len(sub))
    # persistence: run lengths of consecutive scans with an opportunity, per family
    runs = []
    if len(log):
        times = sorted(log.t.unique())
        for f, g in log.groupby("family"):
            ts = sorted(g.t.unique())
            r = 1
            for a, b2 in zip(ts[:-1], ts[1:]):
                if abs(b2 - a - mon.period) < 1e-6:
                    r += 1
                else:
                    runs.append(r); r = 1
            runs.append(r)
    out["persist_scans_mean"] = float(np.mean(runs)) if runs else 0.0
    return out
