"""Automated market maker with inventory-aware, liquidity-sensitive quoting.

Per market: fair value from public information (FairValue), quotes from LiquiditySensitiveQuoter
(Avellaneda-Stoikov reservation price and spread in log-odds, plus volatility, toxicity and event terms).

Features
- inventory skew on the position netted across venues (the same event listed twice is one risk);
- optional portfolio skew: multi-asset reservation price r_i = x_i - gamma*tau*sum_j Cov_ij q_j using an EWMA
  covariance of log-odds changes inside each family of related markets;
- scheduled-news handling: quotes widen as a known announcement approaches and are pulled `pull_s` seconds
  before it (adverse-selection protection), then restored after `resume_s`;
- jump guard: after a detected discontinuity quotes are widened for `jump_cool_s` seconds;
- every quote passes the RiskManager (position, family worst-case loss, price collar, kill switch);
- `fair_mode`: "model" (default, public info only), "mid" (external microprice only), or "oracle"
  (true probability, an upper bound used only in research).
"""
from __future__ import annotations

import math

import numpy as np

from ..core.orderbook import BUY, SELL
from ..execution.oms import QuoteManager
from ..pricing.correlation import EWMACov
from ..pricing.events import event_move_distribution
from ..pricing.liquidity import LiquiditySensitiveQuoter, QuoteParams
from ..pricing.normal import Phi_inv
from ..risk.risk import RiskManager
from ..sim.agents import Agent
from .fair_value import FairValue


class MarketMaker(Agent):
    kind = "mm"
    wants_tape = True

    def __init__(self, aid, capital, listings, params: QuoteParams | None = None, fair_mode: str = "model",
                 risk: RiskManager | None = None, pull_s: float = 20.0, resume_s: float = 10.0,
                 widen_window_s: float = 300.0, jump_cool_s: float = 20.0, portfolio: bool = False,
                 requote_on_trade: bool = True, requote_min_s: float = 0.3, **kw):
        kw.setdefault("rate", 1.0)
        kw.setdefault("latency", 0.03)
        super().__init__(aid, capital, listings, **kw)
        self.params = params or QuoteParams()
        self.quoter = LiquiditySensitiveQuoter(self.params)
        self.fair_mode = fair_mode
        self.risk = risk or RiskManager()
        self.pull_s, self.resume_s, self.widen_window_s = pull_s, resume_s, widen_window_s
        self.jump_cool_s = jump_cool_s
        self.portfolio = portfolio
        self.qm = QuoteManager(self)
        self.markets = sorted({m for _, m in self.listings})
        self.fv = {m: FairValue() for m in self.markets}
        self.cov = {}
        self.fam_markets = {}
        self.n_pulls = 0
        self.requote_on_trade, self.requote_min_s = requote_on_trade, requote_min_s
        self.last_requote: dict[str, float] = {}
        self.n_event_requotes = 0
        self.log_quotes = False
        self.quote_log = []          # (t, venue, market, fair, bid, ask, inventory) when log_quotes is on

    def on_start(self):
        w = self.sim.world
        for m in self.markets:
            self.fam_markets.setdefault(w.fam_of[m], []).append(m)
        if self.portfolio:
            for f, ms in self.fam_markets.items():
                if len(ms) > 1:
                    self.cov[f] = EWMACov(ms, halflife=300.0, prior_var=2e-5)
        self.sched = {}
        for n in w.scheduled_news():
            self.sched.setdefault(n.family, []).append((n.t, n.var))

    # ------------------------------------------------------------------ information
    def on_tape(self, venue, f):
        fv = self.fv.get(f.market)
        if fv is None:
            return
        before = fv.p
        fv.observe_trade(self.sim.now, f.price, f.qty, f.taker_side)
        # event-driven requote: a print that moves fair value by >= half a tick refreshes that market at once
        t = self.sim.now
        if (self.requote_on_trade and not self.sim.closed and abs(fv.p - before) * 100.0 >= 0.5
                and t - self.last_requote.get(f.market, -1e9) >= self.requote_min_s):
            self.last_requote[f.market] = t
            self.n_event_requotes += 1
            self._quote_markets({f.market})

    def fair(self, m: str) -> float:
        if self.fair_mode == "oracle":
            return self.sim.truth(m)
        return self.fv[m].p

    def net_positions(self) -> dict:
        pos = {}
        for v, ex in self.sim.venues.items():
            a = ex.accounts.get(self.id)
            if a:
                for m, q in a.pos.items():
                    if q:
                        pos[m] = pos.get(m, 0) + q
        return pos

    def equity_at_fair(self) -> float:
        marks = {m: self.fair(m) * 100 for m in self.markets}
        return sum(ex.accounts[self.id].equity(marks) for ex in self.sim.venues.values() if self.id in ex.accounts)

    def _event_state(self, m: str, t: float):
        """('pull' | 'widen' | None, event_sd_logit) for the family's next scheduled announcement."""
        fam = self.sim.world.fam_of[m]
        evs = self.sched.get(fam, [])
        for tau, v in evs:
            if tau - self.pull_s <= t <= tau + self.resume_s:
                return "pull", 0.0
        nxt = [(tau, v) for tau, v in evs if tau > t]
        if nxt:
            tau, v = min(nxt)
            if tau - t <= self.widen_window_s:
                p = min(max(self.fair(m), 0.02), 0.98)
                remaining = v + max(self.fv[m].sigma2, 1e-6) * (self.sim.T - t)
                _, _, emove = event_move_distribution(p, v, max(remaining, v * 1.01))
                return "widen", emove / (p * (1 - p)) * (1 - (tau - t) / self.widen_window_s)
        return None, 0.0

    # ------------------------------------------------------------------ quoting loop
    def on_wake(self):
        sim, t = self.sim, self.sim.now
        self.qm.new_wake()
        if self.risk.update_equity(self.equity_at_fair()):
            for v, m in self.listings:
                self.qm.sync(v, m, [])
            return
        for v, m in self.listings:  # refresh fair values from every venue's external book first
            if self.is_open(v, m):
                bid, bq, ask, aq = self.external_bbo(v, m)
                self.fv[m].observe_book(t, bid, bq, ask, aq)
        self._quote_markets(None)

    def _quote_markets(self, only: set | None):
        sim, t = self.sim, self.sim.now
        if self.risk.killed:
            return
        if only is not None:
            self.qm.new_wake()
        pos = self.net_positions()
        marks = {m: self.fair(m) * 100.0 for m in self.markets}
        gross = float(sum(abs(q) for q in pos.values()))
        if self.portfolio and only is None:   # covariance sampled on the regular clock only
            for f, cov in self.cov.items():
                cov.update({m: self.fair(m) for m in cov.names})
        t_left = max(sim.close_t - t, 1.0)
        evt_cache = {}
        for v, m in self.listings:
            if (only is not None and m not in only) or not self.is_open(v, m):
                continue
            if m not in evt_cache:
                evt_cache[m] = self._event_state(m, t)
            state, evt_sd = evt_cache[m]
            if state == "pull":
                self.n_pulls += 1
                self.qm.sync(v, m, [])
                continue
            p_hat = self.fair(m) if self.fair_mode != "mid" else self._mid_fair(v, m)
            fv = self.fv[m]
            q = pos.get(m, 0)
            extra = 0.0
            fam = sim.world.fam_of[m]
            if self.portfolio and fam in self.cov:
                cov = self.cov[fam]
                i = cov.ix[m]
                C = cov.cov()
                tau = min(t_left, self.params.unwind_s)
                cross = sum(C[i, cov.ix[n]] * pos.get(n, 0) for n in cov.names if n != m)
                extra = -self.params.gamma * tau * cross / self.params.q_unit
            jump_recent = (t - fv.last_jump_t) < self.jump_cool_s
            ladder = self.quoter.ladder(p_hat, q, fv.sigma2, t_left, toxicity=fv.tox.value,
                                        event_sd_logit=evt_sd + (0.3 if jump_recent else 0.0), extra_skew=extra)
            ok_quotes = []
            for side, price, size in ladder:
                ok, allowed, _ = self.risk.check(m, side, price, size, p_hat * 100.0, pos, marks, gross)
                if ok and allowed > 0:
                    ok_quotes.append((side, price, allowed))
            self.qm.sync(v, m, ok_quotes)
            if ok_quotes and self.log_quotes:
                bids = [p for s, p, _ in ok_quotes if s == BUY]
                asks = [p for s, p, _ in ok_quotes if s == SELL]
                self.quote_log.append((t, v, m, p_hat, max(bids) if bids else None, min(asks) if asks else None, q))

    def _mid_fair(self, v, m) -> float:
        bid, bq, ask, aq = self.external_bbo(v, m)
        if bid is None or ask is None:
            return self.fv[m].p
        return ((bid * aq + ask * bq) / (bq + aq) if bq + aq > 0 else 0.5 * (bid + ask)) / 100.0
