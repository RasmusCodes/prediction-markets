"""Dynamic, liquidity-sensitive quoting.

Quotes are built in log-odds space, which keeps them inside (0, 1) and scales the price-space spread with
p(1-p) the way the information content of a binary contract does.

  reservation  r = x_hat - gamma * sigma_x^2 * tau_eff * (q / q_unit)                (Avellaneda-Stoikov 2008)
  half-spread  d = 0.5 * gamma * sigma_x^2 * tau_eff + (1/gamma) ln(1 + gamma/kappa)  (AS optimal spread)
                 + c_vol * sigma_x * sqrt(h) + c_tox * toxicity + c_evt * event_sd + c_depth * thinness
  quotes       bid = sigmoid(r - d), ask = sigmoid(r + d) -> rounded outward to the 1c grid, at least
               `min_ticks` wide and wide enough to cover maker fees.

Also: online log-odds volatility, Kyle's lambda by OLS of mid changes on signed volume, and a VPIN-style
order-flow toxicity measure (volume-bucketed absolute imbalance).
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from .normal import logit, sigmoid, slogit, ssigmoid


class LogOddsVol:
    """EWMA of squared log-odds changes per unit time (sigma_x^2 per second)."""

    def __init__(self, halflife_s: float = 300.0, prior: float = 1e-4):
        self.hl, self.var = halflife_s, prior
        self.last_x: float | None = None
        self.last_t: float | None = None

    def update(self, p: float, t: float) -> float:
        x = float(logit(p))
        if self.last_x is not None and t > self.last_t:
            dt = t - self.last_t
            inst = (x - self.last_x) ** 2 / dt
            w = 1 - 0.5 ** (dt / self.hl)
            self.var = (1 - w) * self.var + w * min(inst, 50 * self.var + 1e-3)
        self.last_x, self.last_t = x, t
        return self.var

    @property
    def sigma(self) -> float:
        return math.sqrt(self.var)


class Toxicity:
    """VPIN-style: mean |buy - sell| / bucket volume over the last n volume buckets."""

    def __init__(self, bucket: float = 200.0, n: int = 20):
        self.bucket, self.buf = bucket, deque(maxlen=n)
        self.b = self.s = 0.0
        self._sum = 0.0

    def update(self, qty: float, taker_buy: bool) -> float:
        rem = float(qty)
        while rem > 0:
            room = self.bucket - (self.b + self.s)
            x = min(room, rem)
            if taker_buy:
                self.b += x
            else:
                self.s += x
            rem -= x
            if self.b + self.s >= self.bucket - 1e-9:
                if len(self.buf) == self.buf.maxlen:
                    self._sum -= self.buf[0]
                v = abs(self.b - self.s) / self.bucket
                self.buf.append(v)
                self._sum += v
                self.b = self.s = 0.0
        return self.value

    @property
    def value(self) -> float:
        return self._sum / len(self.buf) if self.buf else 0.0


def kyle_lambda(dmid: np.ndarray, signed_vol: np.ndarray) -> tuple[float, float]:
    """OLS slope of mid changes on signed volume, with its standard error."""
    x, y = np.asarray(signed_vol, float), np.asarray(dmid, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 3 or np.var(x) == 0:
        return float("nan"), float("nan")
    xc = x - x.mean()
    lam = float(np.sum(xc * (y - y.mean())) / np.sum(xc * xc))
    resid = y - y.mean() - lam * xc
    se = float(np.sqrt(np.sum(resid ** 2) / (len(x) - 2) / np.sum(xc * xc)))
    return lam, se


@dataclass
class QuoteParams:
    gamma: float = 1.0            # risk aversion (per inventory unit, log-odds units)
    kappa: float = 8.0            # arrival-intensity decay per unit of log-odds distance
    q_unit: float = 50.0          # contracts per inventory unit
    unwind_s: float = 300.0       # horizon over which inventory risk is charged (caps T - t)
    c_vol: float = 1.0
    vol_h: float = 30.0           # seconds of volatility charged into the spread
    c_tox: float = 0.6
    c_evt: float = 0.5
    c_depth: float = 0.0
    min_ticks: int = 2
    max_half_logit: float = 0.8
    size: int = 20
    levels: int = 1
    level_step_ticks: int = 2
    max_inventory: float = 300


class LiquiditySensitiveQuoter:
    def __init__(self, params: QuoteParams | None = None, maker_fee_cents: float = 0.0):
        self.p = params or QuoteParams()
        self.maker_fee = maker_fee_cents

    def logit_quotes(self, p_hat: float, q: float, sigma2: float, t_left: float, toxicity: float = 0.0,
                     event_sd_logit: float = 0.0, thinness: float = 0.0, extra_skew: float = 0.0):
        P = self.p
        x = float(logit(p_hat))
        tau = max(min(t_left, P.unwind_s), 1.0)
        r = x - P.gamma * sigma2 * tau * (q / P.q_unit) + extra_skew
        d = 0.5 * P.gamma * sigma2 * tau + (1.0 / P.gamma) * math.log(1.0 + P.gamma / P.kappa)
        d += P.c_vol * math.sqrt(max(sigma2, 0.0) * P.vol_h) + P.c_tox * toxicity + P.c_evt * event_sd_logit
        d += P.c_depth * thinness
        d = min(d, P.max_half_logit)
        return r, d

    def quotes(self, p_hat: float, q: float, sigma2: float, t_left: float, **kw) -> tuple[int | None, int | None]:
        """Integer-cent (bid, ask); None on a side when inventory is at its limit."""
        r, d = self.logit_quotes(p_hat, q, sigma2, t_left, **kw)
        pb, pa = ssigmoid(r - d) * 100.0, ssigmoid(r + d) * 100.0
        fee = self.maker_fee
        bid = int(math.floor(pb - fee + 1e-9))
        ask = int(math.ceil(pa + fee - 1e-9))
        center = 0.5 * (pb + pa)
        while ask - bid < self.p.min_ticks:
            if center - bid <= ask - center:
                bid -= 1
            else:
                ask += 1
        bid = max(1, min(98, bid))
        ask = max(bid + 1, min(99, ask))
        if q >= self.p.max_inventory:
            bid = None
        if q <= -self.p.max_inventory:
            ask = None
        return bid, ask

    def ladder(self, p_hat: float, q: float, sigma2: float, t_left: float, **kw):
        """Multi-level quotes [(side, price, size)] around the top-of-book quotes."""
        bid, ask = self.quotes(p_hat, q, sigma2, t_left, **kw)
        out = []
        P = self.p
        for lvl in range(P.levels):
            sz = int(round(P.size * (1.0 + 0.5 * lvl)))
            if bid is not None and bid - lvl * P.level_step_ticks >= 1:
                out.append((1, bid - lvl * P.level_step_ticks, sz))
            if ask is not None and ask + lvl * P.level_step_ticks <= 99:
                out.append((-1, ask + lvl * P.level_step_ticks, sz))
        return out
