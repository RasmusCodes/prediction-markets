"""Cost-function automated market makers for prediction markets.

LMSR (Hanson 2003/2007):      C(q) = b ln sum_i exp(q_i / b),  p_i = softmax(q / b),  worst-case loss b ln n.
LS-LMSR (Othman, Pennock, Reeves & Sandholm 2013): b(q) = alpha * sum_i q_i, so liquidity deepens as
                              volume accrues; prices sum to more than one (a built-in spread / vig).

`ladder()` turns an AMM state into limit orders on a 1-cent grid: the quantity it would sell between
consecutive ticks is posted at the upper tick (and symmetrically for bids), so an AMM can provide liquidity
inside a central limit order book alongside other market makers.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.special import logsumexp


class LMSR:
    def __init__(self, b: float, n: int = 2, q0=None):
        self.b = float(b)
        self.q = np.zeros(n) if q0 is None else np.asarray(q0, float).copy()

    def cost(self, q=None) -> float:
        q = self.q if q is None else q
        return float(self.b * logsumexp(np.asarray(q) / self.b))

    def prices(self, q=None) -> np.ndarray:
        q = self.q if q is None else np.asarray(q)
        z = q / self.b
        e = np.exp(z - z.max())
        return e / e.sum()

    def cost_to_trade(self, i: int, shares: float) -> float:
        q2 = self.q.copy()
        q2[i] += shares
        return self.cost(q2) - self.cost()

    def trade(self, i: int, shares: float) -> float:
        c = self.cost_to_trade(i, shares)
        self.q[i] += shares
        return c

    def shares_to_price(self, i: int, target: float) -> float:
        """Shares of outcome i to buy (negative = sell) to move p_i to target (binary closed form)."""
        if len(self.q) != 2:
            raise NotImplementedError
        j = 1 - i
        cur = self.q[i] - self.q[j]
        want = self.b * math.log(target / (1 - target))
        return want - cur

    @property
    def max_loss(self) -> float:
        return self.b * math.log(len(self.q))


class LSLMSR:
    def __init__(self, alpha: float = 0.05, n: int = 2, q0=None):
        self.alpha = float(alpha)
        self.q = np.full(n, 10.0) if q0 is None else np.asarray(q0, float).copy()

    def b(self, q=None) -> float:
        q = self.q if q is None else q
        return self.alpha * float(np.sum(q))

    def cost(self, q=None) -> float:
        q = self.q if q is None else np.asarray(q)
        b = self.b(q)
        return float(b * logsumexp(q / b))

    def prices(self, q=None) -> np.ndarray:
        """Marginal prices dC/dq_i (they sum to slightly more than one)."""
        q = self.q if q is None else np.asarray(q, float)
        b = self.b(q)
        z = q / b
        e = np.exp(z - z.max())
        s = e / e.sum()
        lse = logsumexp(z)
        return self.alpha * lse + (s * np.sum(q) - np.sum(q * s)) / np.sum(q)

    def cost_to_trade(self, i: int, shares: float) -> float:
        q2 = self.q.copy()
        q2[i] += shares
        return self.cost(q2) - self.cost()

    def trade(self, i: int, shares: float) -> float:
        c = self.cost_to_trade(i, shares)
        self.q[i] += shares
        return c


def ladder(amm, levels: int = 5, min_size: int = 1) -> tuple[list, list]:
    """Binary AMM -> (bids, asks) as [(price_cents, contracts)] on the 1c grid around its current price."""
    p_yes = float(amm.prices()[0]) if not isinstance(amm, LSLMSR) else float(amm.prices()[0] / amm.prices().sum())
    cur = p_yes * 100.0
    asks, bids = [], []
    if isinstance(amm, LMSR):
        b = amm.b
        lo_tick = int(math.ceil(cur))
        for k in range(lo_tick, min(lo_tick + levels, 99)):
            p1, p2 = max(cur, k) / 100.0, (k + 1) / 100.0
            if p2 >= 0.995:
                break
            size = b * (math.log(p2 / (1 - p2)) - math.log(p1 / (1 - p1)))
            if size >= min_size:
                asks.append((k + 1, int(size)))
        hi_tick = int(math.floor(cur))
        for k in range(hi_tick, max(hi_tick - levels, 1), -1):
            p1, p2 = min(cur, k) / 100.0, (k - 1) / 100.0
            if p2 <= 0.005:
                break
            size = b * (math.log(p1 / (1 - p1)) - math.log(p2 / (1 - p2)))
            if size >= min_size:
                bids.append((k - 1, int(size)))
        return bids, asks
    # LS-LMSR: numerical: shares bought until the marginal YES price crosses each tick
    q = amm.q.copy()
    for k in range(int(math.ceil(cur)) + 1, min(int(math.ceil(cur)) + 1 + levels, 99)):
        lo, hi = 0.0, 1e6
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            q2 = q.copy(); q2[0] += mid
            pp = amm.prices(q2)
            if pp[0] / pp.sum() * 100 < k:
                lo = mid
            else:
                hi = mid
        if lo >= min_size:
            asks.append((k, int(lo)))
        q[0] += lo
    q = amm.q.copy()
    for k in range(int(math.floor(cur)) - 1, max(int(math.floor(cur)) - 1 - levels, 1), -1):
        lo, hi = 0.0, 1e6
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            q2 = q.copy(); q2[1] += mid
            pp = amm.prices(q2)
            if pp[0] / pp.sum() * 100 > k:
                lo = mid
            else:
                hi = mid
        if lo >= min_size:
            bids.append((k, int(lo)))
        q[1] += lo
    return bids, asks
