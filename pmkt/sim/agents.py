"""Trader population for the agent-based simulator (the 'competing traders').

Agent           base class: latency, wake process, live-order tracking that survives latency (pending/live/
                cancelling states), helpers for books and positions.
NoiseTrader     uninformed flow; optional favourite-longshot preference (buys the cheap side more often) and a
                share of passive limit orders that expire.
InformedTrader  sees the true probability with private noise and a persistent bias; trades when the edge
                beats its threshold plus the venue's fee; reacts to unscheduled news after a delay.
MomentumTrader  chases recent mid-price moves.
NaiveMarketMaker fixed-spread quotes around the last trade with a linear inventory skew.
AMMAgent        LMSR / LS-LMSR cost-function market maker that posts its curve into the order book.
SimpleArbitrageur competing arbitrageur: bucket sums, ladder monotonicity and cross-venue duplicates.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from ..core.orderbook import BUY, MAX_TICK, SELL
from ..pricing import amm as amm_mod
from ..pricing.normal import logit, sigmoid, slogit, ssigmoid


@dataclass
class OrderState:
    venue: str
    market: str
    side: int
    price: int
    qty: int
    rem: int
    status: str = "pending"   # pending | live | cancelling
    tif: str = "GTC"
    tag: str = ""


class Agent:
    kind = "agent"
    wants_news = False

    def __init__(self, aid: str, capital: dict, listings=None, latency: float = 0.05, jitter: float = 0.02,
                 rate: float = 0.1, seed: int = 0):
        self.id = aid
        self.capital = dict(capital)
        self.listings = list(listings or [])
        self.lat, self.jit, self.rate = latency, jitter, rate
        self.rng = np.random.default_rng(seed)
        self.sim = None
        self.orders: dict[int, OrderState] = {}
        self.n_fills = 0

    # ------------------------------------------------------------------ plumbing
    def attach(self, sim) -> None:
        self.sim = sim

    def latency(self) -> float:
        return self.lat + (float(self.rng.exponential(self.jit)) if self.jit > 0 else 0.0)

    def next_wake(self, now: float):
        return now + float(self.rng.exponential(1.0 / self.rate)) if self.rate > 0 else None

    def news_delay(self):
        return None

    # hooks
    def on_start(self): pass
    def on_wake(self): pass
    def on_news(self, news): pass
    def on_close(self): self.orders.clear()

    def on_order_result(self, venue, market, res) -> None:
        st = self.orders.get(res.oid)
        if st is None:
            return
        if res.status == "rejected" or not res.resting or st.rem <= 0:
            self.orders.pop(res.oid, None)
        else:
            st.status = "live" if st.status == "pending" else st.status

    def on_cancel_result(self, venue, oid, ok) -> None:
        st = self.orders.get(oid)
        if st is None:
            return
        if ok or st.rem <= 0:
            self.orders.pop(oid, None)
        else:
            st.status = "live"   # cancel bounced (e.g. rate limit); order still working

    def on_fill(self, venue, f, is_taker) -> None:
        self.n_fills += 1
        oid = f.taker_oid if is_taker else f.maker_oid
        st = self.orders.get(oid)
        if st is not None:
            st.rem -= f.qty
            if st.rem <= 0 and st.status != "pending":
                self.orders.pop(oid, None)

    # ------------------------------------------------------------------ helpers
    def submit(self, venue, market, side, price, qty, tif="GTC", post_only=False, tag="") -> int | None:
        price, qty = int(price), int(qty)
        if qty <= 0 or not (1 <= price <= MAX_TICK):
            return None
        oid = self.sim.submit(self, venue, market, side, price, qty, tif, post_only, tag)
        self.orders[oid] = OrderState(venue, market, side, price, qty, qty, "pending", tif, tag)
        return oid

    def cancel(self, oid: int) -> None:
        st = self.orders.get(oid)
        if st is None or st.status == "cancelling":
            return
        st.status = "cancelling"
        self.sim.cancel(self, st.venue, oid)

    def cancel_market(self, venue, market) -> None:
        for oid, st in list(self.orders.items()):
            if st.venue == venue and st.market == market and st.tif == "GTC":
                self.cancel(oid)

    def pos(self, venue, market) -> int:
        a = self.sim.venues[venue].accounts.get(self.id)
        return a.pos.get(market, 0) if a else 0

    def book(self, venue, market):
        return self.sim.venues[venue].markets[market].book

    def is_open(self, venue, market) -> bool:
        return self.sim.venues[venue].markets[market].status == "open"

    def ref_price(self, venue, market, default: float = 50.0) -> float:
        b = self.book(venue, market)
        m = b.mid()
        if m is not None:
            return m
        return float(b.last_price) if b.last_price is not None else default

    def fee_per_contract(self, venue, price, taker=True) -> float:
        fee = self.sim.venues[venue].fee
        return (fee.taker(price, 100) if taker else fee.maker(price, 100)) / 100.0

    def external_bbo(self, venue, market):
        """Best bid/ask with this agent's own resting quantity removed (what everyone else is showing)."""
        b = self.book(venue, market)
        own_b, own_a = {}, {}
        for st in self.orders.values():
            if st.venue == venue and st.market == market and st.status != "pending":
                d = own_b if st.side == BUY else own_a
                d[st.price] = d.get(st.price, 0) + st.rem
        bid = ask = None
        bq = aq = 0
        p = b.best_bid
        while p >= 1:
            q = b.bid_qty[p] - own_b.get(p, 0)
            if q > 0:
                bid, bq = p, q
                break
            p -= 1
        p = b.best_ask
        while p <= MAX_TICK:
            q = b.ask_qty[p] - own_a.get(p, 0)
            if q > 0:
                ask, aq = p, q
                break
            p += 1
        return bid, bq, ask, aq


def _geom(rng, mean: float) -> int:
    return int(rng.geometric(1.0 / max(mean, 1.0)))


class NoiseTrader(Agent):
    kind = "noise"

    def __init__(self, aid, capital, listings, mean_size=8.0, longshot_bias=0.0, limit_prob=0.3, limit_life=90.0,
                 aggression=2, **kw):
        super().__init__(aid, capital, listings, **kw)
        self.mean_size, self.beta = mean_size, longshot_bias
        self.limit_prob, self.limit_life, self.aggr = limit_prob, limit_life, aggression

    def on_wake(self):
        if not self.listings:
            return
        v, m = self.listings[self.rng.integers(len(self.listings))]
        if not self.is_open(v, m):
            return
        b = self.book(v, m)
        p = self.ref_price(v, m)
        pb = 0.5 + self.beta * (50.0 - p) / 50.0          # longshot preference: buy the cheap side
        side = BUY if self.rng.random() < pb else SELL
        qty = _geom(self.rng, self.mean_size)
        bb, _, ba, _ = b.bbo()
        if self.rng.random() < self.limit_prob:
            off = int(self.rng.integers(0, 4))
            if side == BUY:
                px = (bb if bb else int(p) - 1) - off
                px = min(px, (ba - 1) if ba <= MAX_TICK else 98)
            else:
                px = (ba if ba <= MAX_TICK else int(math.ceil(p)) + 1) + off
                px = max(px, (bb + 1) if bb else 2)
            px = max(1, min(99, px))
            oid = self.submit(v, m, side, px, qty, "GTC")
            if oid is not None:
                self.sim.schedule(self.sim.now + self.limit_life * float(self.rng.uniform(0.5, 1.5)), self._expire, oid)
        else:
            if side == BUY:
                px = min(99, (ba if ba <= MAX_TICK else int(p) + 1) + self.aggr)
            else:
                px = max(1, (bb if bb else int(p) - 1) - self.aggr)
            self.submit(v, m, side, px, qty, "IOC")

    def _expire(self, oid):
        if oid in self.orders and not self.sim.closed:
            self.cancel(oid)


class InformedTrader(Agent):
    kind = "informed"
    wants_news = True

    def __init__(self, aid, capital, listings, noise_sd=0.25, bias_sd=0.05, threshold=0.03, max_pos=300,
                 max_order=60, size_per_edge=800.0, react_prob=1.0, react_delay=(0.5, 3.0), **kw):
        super().__init__(aid, capital, listings, **kw)
        self.noise_sd, self.threshold = noise_sd, threshold
        self.max_pos, self.max_order, self.size_per_edge = max_pos, max_order, size_per_edge
        self.react_prob, self.react_delay = react_prob, react_delay
        self.bias = {m: float(self.rng.normal(0, bias_sd)) for _, m in self.listings}

    def news_delay(self):
        if self.rng.random() < self.react_prob:
            return float(self.rng.uniform(*self.react_delay))
        return None

    def belief(self, market) -> float:
        x = slogit(self.sim.truth(market)) + self.bias.get(market, 0.0) + float(self.rng.normal(0, self.noise_sd))
        return ssigmoid(x)

    def _trade(self, v, m):
        if not self.is_open(v, m):
            return
        b = self.book(v, m)
        bb, bq, ba, aq = b.bbo()
        bel = self.belief(m) * 100.0
        pos = self.pos(v, m)
        thr = self.threshold * 100.0
        if ba <= MAX_TICK:
            edge = bel - ba - self.fee_per_contract(v, ba)
            if edge > thr and pos < self.max_pos:
                limit = int(min(99, math.floor(bel - thr)))
                qty = int(min(self.max_order, max(1, self.size_per_edge * edge / 100.0), self.max_pos - pos))
                if limit >= ba and qty > 0:
                    self.submit(v, m, BUY, limit, qty, "IOC")
                    return
        if bb:
            edge = bb - bel - self.fee_per_contract(v, bb)
            if edge > thr and pos > -self.max_pos:
                limit = int(max(1, math.ceil(bel + thr)))
                qty = int(min(self.max_order, max(1, self.size_per_edge * edge / 100.0), self.max_pos + pos))
                if limit <= bb and qty > 0:
                    self.submit(v, m, SELL, limit, qty, "IOC")

    def on_wake(self):
        if self.listings:
            v, m = self.listings[self.rng.integers(len(self.listings))]
            self._trade(v, m)

    def on_news(self, news):
        fam = news.family
        for v, m in self.listings:
            if self.sim.world.fam_of.get(m) == fam:
                self._trade(v, m)


class MomentumTrader(Agent):
    kind = "momentum"

    def __init__(self, aid, capital, listings, lookback=60.0, threshold=3.0, size=10, max_pos=200, **kw):
        super().__init__(aid, capital, listings, **kw)
        self.lookback, self.threshold, self.size, self.max_pos = lookback, threshold, size, max_pos
        self.hist = {lm: deque() for lm in self.listings}

    def on_wake(self):
        now = self.sim.now
        for lm in self.listings:
            v, m = lm
            if not self.is_open(v, m):
                continue
            mid = self.book(v, m).mid()
            if mid is None:
                continue
            h = self.hist[lm]
            h.append((now, mid))
            while h and h[0][0] < now - self.lookback:
                h.popleft()
        v, m = self.listings[self.rng.integers(len(self.listings))]
        h = self.hist[(v, m)]
        if len(h) < 2 or not self.is_open(v, m):
            return
        chg = h[-1][1] - h[0][1]
        pos = self.pos(v, m)
        bb, _, ba, _ = self.book(v, m).bbo()
        if chg > self.threshold and pos < self.max_pos and ba <= MAX_TICK:
            self.submit(v, m, BUY, min(99, ba + 1), self.size, "IOC")
        elif chg < -self.threshold and pos > -self.max_pos and bb:
            self.submit(v, m, SELL, max(1, bb - 1), self.size, "IOC")


class NaiveMarketMaker(Agent):
    kind = "naive_mm"

    def __init__(self, aid, capital, listings, half_spread=3, size=20, max_inv=300, skew_ticks_per_100=2.0,
                 p0=None, **kw):
        super().__init__(aid, capital, listings, **kw)
        self.hs, self.size, self.max_inv, self.skew = half_spread, size, max_inv, skew_ticks_per_100
        # opens at the public prior (the listing's opening line), then anchors on the last trade
        self.anchor = {(v, m): 100.0 * (p0 or {}).get(m, 0.5) for v, m in self.listings}

    def on_wake(self):
        for v, m in self.listings:
            if not self.is_open(v, m):
                continue
            b = self.book(v, m)
            anchor = b.last_price if b.last_price is not None else self.anchor.get((v, m), 50.0)
            self.anchor[(v, m)] = anchor
            q = self.pos(v, m)
            c = anchor - self.skew * q / 100.0
            bid, ask = int(round(c - self.hs)), int(round(c + self.hs))
            bid, ask = max(1, min(98, bid)), max(2, min(99, ask))
            if ask <= bid:
                ask = bid + 1
            want = []
            if q < self.max_inv:
                want.append((BUY, bid))
            if q > -self.max_inv:
                want.append((SELL, ask))
            live = [(oid, st) for oid, st in self.orders.items() if st.venue == v and st.market == m]
            have = {(st.side, st.price) for _, st in live if st.status != "cancelling"}
            if have == set(want):
                continue
            for oid, st in live:
                if (st.side, st.price) not in want:
                    self.cancel(oid)
            for side, px in want:
                if (side, px) not in have:
                    self.submit(v, m, side, px, self.size, "GTC", post_only=True)


class AMMAgent(Agent):
    """Cost-function market maker (LMSR, or LS-LMSR when alpha is given) quoting its curve as limit orders."""
    kind = "amm"

    def __init__(self, aid, capital, listings, b=150.0, alpha=None, levels=4, p0=None, **kw):
        super().__init__(aid, capital, listings, **kw)
        self.levels = levels
        self.makers = {}
        for v, m in self.listings:
            start = (p0 or {}).get(m, 0.5)
            if alpha:
                mk = amm_mod.LSLMSR(alpha=alpha, q0=[100.0, 100.0])
                # tilt the seed inventory toward the starting price
                tilt = mk.b() * math.log(start / (1 - start))
                mk.q = np.array([100.0 + max(tilt, 0), 100.0 + max(-tilt, 0)])
            else:
                mk = amm_mod.LMSR(b=b)
                mk.q[0] += mk.shares_to_price(0, start)
            self.makers[(v, m)] = mk

    def on_fill(self, venue, f, is_taker):
        super().on_fill(venue, f, is_taker)
        mk = self.makers.get((venue, f.market))
        if mk is None:
            return
        # cost-function bookkeeping: q counts shares the AMM has sold. Selling YES -> q_yes += qty,
        # buying YES (= selling NO) -> q_no += qty.
        our_side = f.taker_side if is_taker else -f.taker_side
        if our_side == SELL:
            mk.q[0] += f.qty
        else:
            mk.q[1] += f.qty

    def on_wake(self):
        for (v, m), mk in self.makers.items():
            if not self.is_open(v, m):
                continue
            bids, asks = amm_mod.ladder(mk, levels=self.levels)
            want = {(BUY, p): q for p, q in bids} | {(SELL, p): q for p, q in asks}
            live = [(oid, st) for oid, st in self.orders.items() if st.venue == v and st.market == m]
            have = {}
            for oid, st in live:
                if st.status == "cancelling":
                    continue
                key = (st.side, st.price)
                if key in want and abs(st.rem - want[key]) <= max(2, 0.3 * want[key]) and key not in have:
                    have[key] = oid
                else:
                    self.cancel(oid)
            for key, q in want.items():
                if key not in have and q > 0:
                    self.submit(v, m, key[0], key[1], q, "GTC", post_only=True)


class SimpleArbitrageur(Agent):
    """Competitor that takes obvious, rule-based arbitrage (no optimisation, no leg management)."""
    kind = "arbitrageur"

    def __init__(self, aid, capital, groups=(), ladders=(), duplicates=(), min_edge=1.0, max_qty=50, **kw):
        super().__init__(aid, capital, [], **kw)
        self.groups, self.ladders, self.duplicates = list(groups), list(ladders), list(duplicates)
        self.min_edge, self.max_qty = min_edge, max_qty

    def _fee(self, v, p):
        return self.fee_per_contract(v, p)

    def on_wake(self):
        for v, mids in self.groups:          # mutually exclusive, exhaustive set on one venue
            if not all(self.is_open(v, m) for m in mids):
                continue
            bbos = [self.book(v, m).bbo() for m in mids]
            if all(x[2] <= MAX_TICK for x in bbos):
                cost = sum(x[2] + self._fee(v, x[2]) for x in bbos)
                if cost < 100 - self.min_edge:
                    q = min(self.max_qty, *(x[3] for x in bbos))
                    for m, x in zip(mids, bbos):
                        self.submit(v, m, BUY, x[2], q, "IOC")
            if all(x[0] for x in bbos):
                proceeds = sum(x[0] - self._fee(v, x[0]) for x in bbos)
                if proceeds > 100 + self.min_edge:
                    q = min(self.max_qty, *(x[1] for x in bbos))
                    for m, x in zip(mids, bbos):
                        self.submit(v, m, SELL, x[0], q, "IOC")
        for v, mids in self.ladders:         # mids ordered by increasing strike: P(>K) must fall
            for lo, hi in zip(mids[:-1], mids[1:]):
                if not (self.is_open(v, lo) and self.is_open(v, hi)):
                    continue
                _, _, a_lo, aq = self.book(v, lo).bbo()
                b_hi, bq, _, _ = self.book(v, hi).bbo()
                if a_lo <= MAX_TICK and b_hi and b_hi - a_lo - self._fee(v, a_lo) - self._fee(v, b_hi) > self.min_edge:
                    q = min(self.max_qty, aq, bq)
                    self.submit(v, lo, BUY, a_lo, q, "IOC")
                    self.submit(v, hi, SELL, b_hi, q, "IOC")
        for (v1, m1), (v2, m2) in self.duplicates:
            for (va, ma), (vb, mb) in (((v1, m1), (v2, m2)), ((v2, m2), (v1, m1))):
                if not (self.is_open(va, ma) and self.is_open(vb, mb)):
                    continue
                bb, bq, _, _ = self.book(va, ma).bbo()
                _, _, ba, aq = self.book(vb, mb).bbo()
                if bb and ba <= MAX_TICK and bb - ba - self._fee(va, bb) - self._fee(vb, ba) > self.min_edge:
                    q = min(self.max_qty, bq, aq)
                    self.submit(vb, mb, BUY, ba, q, "IOC")
                    self.submit(va, ma, SELL, bb, q, "IOC")
