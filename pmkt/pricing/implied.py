"""Market-implied probabilities from order books, and de-vigging of multi-outcome price vectors."""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq


def mid(bid: float | None, ask: float | None) -> float | None:
    if bid is None or ask is None:
        return None
    return 0.5 * (bid + ask)


def microprice(bid, bid_sz, ask, ask_sz):
    """Size-weighted mid: leans toward the side with less resting size (where the next trade is likelier to move)."""
    if bid is None or ask is None:
        return None
    if not bid_sz and not ask_sz:
        return 0.5 * (bid + ask)
    return (bid * ask_sz + ask * bid_sz) / (bid_sz + ask_sz)


def depth_weighted_mid(bids, asks, levels: int = 3):
    """Mid of the size-weighted average prices of the top `levels` on each side."""
    if not bids or not asks:
        return None
    b = bids[:levels]
    a = asks[:levels]
    pb = sum(p * q for p, q in b) / max(sum(q for _, q in b), 1e-12)
    pa = sum(p * q for p, q in a) / max(sum(q for _, q in a), 1e-12)
    return 0.5 * (pb + pa)


def imbalance(bid_sz, ask_sz) -> float:
    """Level-1 order-book imbalance in [-1, 1] (positive = more resting bids)."""
    tot = (bid_sz or 0) + (ask_sz or 0)
    return 0.0 if tot <= 0 else ((bid_sz or 0) - (ask_sz or 0)) / tot


def implied_probability(snap: dict, method: str = "micro", fallback: float | None = None) -> float | None:
    """Probability in [0,1] from an exchange snapshot dict (prices in cents)."""
    bid, ask = snap.get("bid"), snap.get("ask")
    if method == "micro":
        v = microprice(bid, snap.get("bid_sz"), ask, snap.get("ask_sz"))
    elif method == "depth":
        v = depth_weighted_mid(snap.get("bids"), snap.get("asks"))
    elif method == "last":
        v = snap.get("last")
    else:
        v = mid(bid, ask)
    if v is None:
        v = snap.get("last") if snap.get("last") is not None else fallback
        if v is None:
            return None
    return float(v) / 100.0


# ------------------------------------------------------------------------------------------- multi-outcome
def overround(prices) -> float:
    """sum(p) - 1 for a mutually exclusive, exhaustive set (asks > 0 means no buy-all arbitrage)."""
    return float(np.sum(prices) - 1.0)


def normalize(prices, method: str = "proportional"):
    """Map a vector of (possibly non-coherent) prices to probabilities summing to one.

    proportional: p_i / sum p
    power:        p_i^k with k solving sum p_i^k = 1 (shrinks longshots more when sum > 1)
    shin:         Shin (1993) model with a fraction z of insiders; z solved so the implied probabilities sum to 1
    """
    p = np.clip(np.asarray(prices, float), 1e-9, None)
    s = p.sum()
    if method == "proportional" or len(p) < 2:
        return p / s
    if method == "power":
        if abs(s - 1) < 1e-12:
            return p
        f = lambda k: np.sum(p ** k) - 1.0
        lo, hi = (1.0, 50.0) if s > 1 else (0.02, 1.0)
        if f(lo) * f(hi) > 0:
            return p / s
        return p ** brentq(f, lo, hi)
    if method == "shin":
        pi = p / s if s < 1 else p  # Shin applies to an overround book; for underround fall back to proportional
        if s <= 1:
            return p / s
        def implied(z):
            return (np.sqrt(z ** 2 + 4 * (1 - z) * pi ** 2 / s) - z) / (2 * (1 - z))
        g = lambda z: implied(z).sum() - 1.0
        if g(0.0) * g(0.99) > 0:
            return p / s
        z = brentq(g, 0.0, 0.99)
        q = implied(z)
        return q / q.sum()
    raise ValueError(method)


def coherent_band(bids, asks):
    """For a mutually exclusive, exhaustive set: executable buy-all cost and sell-all proceeds (cents)."""
    buy_all = float(np.sum(asks)) if all(a is not None for a in asks) else None
    sell_all = float(np.sum(bids)) if all(b is not None for b in bids) else None
    return buy_all, sell_all
