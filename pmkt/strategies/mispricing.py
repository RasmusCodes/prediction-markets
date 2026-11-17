"""Statistical mispricing strategies (not riskless): trade when a model's fair value differs from the
executable price by more than fees plus a threshold, sized by fractional Kelly and capped by risk limits.

Models
- "calibration": fair = g(market mid) where g is a recalibration map fitted on held-out history (corrects
  systematic biases such as the favourite-longshot pattern).
- "copula": for a conjunction market A AND B, fair = Gaussian-copula joint of the A and B fair values with a
  latent correlation estimated online from co-movement of their log-odds (or a supplied prior).
- "consistency": buckets priced off the strike ladder, P(K_j < X <= K_j+1) = P(>K_j) - P(>K_j+1).
- "leadlag": the same event on two venues; trade the lagging venue toward the leading venue's mid.
"""
from __future__ import annotations

import math

import numpy as np

from ..core.orderbook import BUY, MAX_TICK, SELL
from ..pricing.correlation import EWMACov, copula_joint
from ..risk.risk import RiskManager
from ..sim.agents import Agent


class MispricingStrategy(Agent):
    kind = "mispricing"

    def __init__(self, aid, capital, listings, model: str = "copula", threshold: float = 0.03, kelly: float = 0.25,
                 max_pos: int = 300, max_order: int = 60, calib_map=None, rho_prior: float | None = None,
                 risk: RiskManager | None = None, passive: bool = False, **kw):
        kw.setdefault("rate", 0.5)
        super().__init__(aid, capital, listings, **kw)
        self.model, self.threshold, self.kelly = model, threshold, kelly
        self.max_pos, self.max_order = max_pos, max_order
        self.calib_map, self.rho_prior = calib_map, rho_prior
        self.risk = risk or RiskManager()
        self.passive = passive
        self.cov = {}
        self.signals = []        # (t, venue, market, fair, bid, ask)

    def on_start(self):
        w = self.sim.world
        self.by_fam = {}
        for v, m in self.listings:
            self.by_fam.setdefault(w.fam_of[m], set()).add(m)
        if self.model == "copula":
            for f, ms in self.by_fam.items():
                base = sorted(x for x in ms if not x.endswith(".AB"))
                if len(base) == 2:
                    self.cov[f] = EWMACov(base, halflife=400.0, prior_var=2e-5)

    def _mid(self, v, m):
        b = self.book(v, m)
        mid = b.mid()
        return None if mid is None else mid / 100.0

    def fair(self, v, m) -> float | None:
        w = self.sim.world
        if self.model == "calibration":
            mid = self._mid(v, m)
            return None if mid is None or self.calib_map is None else float(self.calib_map(np.array([mid]))[0])
        if self.model == "copula":
            if not m.endswith(".AB"):
                return None
            f = w.fam_of[m]
            a, b = f + ".A", f + ".B"
            pa, pb = self._mid(v, a), self._mid(v, b)
            if pa is None or pb is None:
                return None
            if self.rho_prior is not None:
                rho = self.rho_prior
            else:
                cov = self.cov.get(f)
                if cov is None:
                    return None
                rho = float(np.clip(cov.corr()[0, 1], -0.95, 0.95))
            return float(copula_joint(pa, pb, rho))
        if self.model == "consistency":
            d = w.defs[m]
            if d.kind != "bucket":
                return None
            fam = w.families[d.family]
            lo, hi = d.meta["lo"], d.meta["hi"]
            above = lambda K: 1.0 if K == -math.inf else (0.0 if K == math.inf else self._mid(v, f"{fam.id}>{K:g}"))
            a, b = above(lo), above(hi)
            if a is None or b is None:
                return None
            return max(0.0, a - b)
        if self.model == "leadlag":
            others = [(vv, mm) for vv, mm in self.listings if mm == m and vv != v]
            mids = [self._mid(vv, mm) for vv, mm in others]
            mids = [x for x in mids if x is not None]
            return float(np.mean(mids)) if mids else None
        return None

    def on_wake(self):
        if self.model == "copula":
            for f, cov in self.cov.items():
                v0 = next(v for v, m in self.listings if self.sim.world.fam_of[m] == f)
                cov.update({m: self._mid(v0, m) for m in cov.names})
        for v, m in self.listings:
            if not self.is_open(v, m):
                continue
            fair = self.fair(v, m)
            if fair is None:
                continue
            bb, bq, ba, aq = self.book(v, m).bbo()
            pos = self.pos(v, m)
            self.signals.append((self.sim.now, v, m, fair, bb or None, ba if ba <= MAX_TICK else None))
            fc = fair * 100.0
            thr = self.threshold * 100.0
            if ba <= MAX_TICK and fc - ba - self.fee_per_contract(v, ba) > thr and pos < self.max_pos:
                c = ba / 100.0
                f_star = (fair - c) / (1 - c)
                bankroll = self.sim.venues[v].accounts[self.id].cash
                target = int(self.kelly * f_star * bankroll / max(ba, 1))
                qty = min(self.max_order, max(0, target - max(pos, 0)), aq, self.max_pos - pos)
                if qty > 0:
                    self.submit(v, m, BUY, ba, qty, "IOC", tag=self.model)
            elif bb and bb - fc - self.fee_per_contract(v, bb) > thr and pos > -self.max_pos:
                c = 1 - bb / 100.0                      # buying NO at 100 - bid
                f_star = ((1 - fair) - c) / (1 - c)
                bankroll = self.sim.venues[v].accounts[self.id].cash
                target = int(self.kelly * f_star * bankroll / max(100 - bb, 1))
                qty = min(self.max_order, max(0, target - max(-pos, 0)), bq, self.max_pos + pos)
                if qty > 0:
                    self.submit(v, m, SELL, bb, qty, "IOC", tag=self.model)
