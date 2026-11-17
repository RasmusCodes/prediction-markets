"""Fair value for one market, learned only from public information (no access to the simulator's truth).

Inputs: other participants' quotes (microprice of the external book, observation noise growing with the
spread) and every public trade print. A trade is treated as a noisy observation of value located a little
beyond the trade price in the aggressor's direction (Glosten-Milgrom intuition: a buy is evidence of higher
value), with the push growing in sqrt(size). A jump-robust log-odds Kalman filter fuses the two; its process
noise tracks realised log-odds volatility. Toxicity (VPIN-style) and a jump flag are maintained alongside.
"""
from __future__ import annotations

import math

from ..core.orderbook import BUY
from ..pricing.bayes import LogitKalman
from ..pricing.events import JumpDetector
from ..pricing.liquidity import LogOddsVol, Toxicity
from ..pricing.normal import logit


class FairValue:
    def __init__(self, p0: float = 0.5, trade_push: float = 0.08, trade_R: float = 0.02, book_R: float = 0.05,
                 tox_bucket: float = 150.0, size_ref: float = 15.0):
        self.size_ref = size_ref
        self.kf = LogitKalman(p0, var0=0.6, q=2e-5, jump_k=4.0, jump_inflate=20.0)
        self.vol = LogOddsVol(halflife_s=300.0, prior=2e-5)
        self.tox = Toxicity(bucket=tox_bucket, n=15)
        self.jd = JumpDetector(window=150, k=6.0, min_abs=0.25)
        self.trade_push, self.trade_R, self.book_R = trade_push, trade_R, book_R
        self.n_obs = 0
        self.last_jump_t = -1e9

    def _predict(self, t: float) -> None:
        self.kf.q = max(self.vol.var, 2e-6)
        self.kf.predict(t)

    def observe_book(self, t: float, bid, bq, ask, aq) -> None:
        if bid is None or ask is None:
            return
        micro = (bid * aq + ask * bq) / (bq + aq) if (bq + aq) > 0 else 0.5 * (bid + ask)
        p = min(max(micro / 100.0, 0.01), 0.99)
        self._predict(t)
        self.kf.update_prob(p, spread=(ask - bid) / 100.0, base_R=self.book_R)
        self.vol.update(self.kf.p, t)
        if self.jd.update(self.kf.p):
            self.last_jump_t = t
        self.n_obs += 1

    def observe_trade(self, t: float, price: int, qty: int, taker_side: int) -> None:
        p = min(max(price / 100.0, 0.01), 0.99)
        s = 1.0 if taker_side == BUY else -1.0
        push = self.trade_push * math.sqrt(min(qty, 200) / 10.0)
        self._predict(t)
        # bigger prints carry more information: observation noise shrinks with size
        self.kf.update(float(logit(p)) + s * push, self.trade_R / (1.0 + qty / self.size_ref))
        self.vol.update(self.kf.p, t)
        self.tox.update(qty, taker_side == BUY)
        if self.jd.update(self.kf.p):
            self.last_jump_t = t
        self.n_obs += 1

    @property
    def p(self) -> float:
        return self.kf.p

    @property
    def sigma2(self) -> float:
        return self.vol.var
