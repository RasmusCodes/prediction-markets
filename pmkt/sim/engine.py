"""Discrete-event simulation of one or more venues trading the markets of a World.

Time is in seconds. Every agent action travels with that agent's latency: an order decided at t (looking at
the book as it is at t) reaches the exchange at t + latency, so slow quotes can be picked off by fast
traders after news. Fills are pushed to both counterparties when they happen. Unscheduled news is pushed to
subscribed agents after their own reaction delay; scheduled news times are public in advance.

The world path is fixed by the scenario seed, and every agent draws from its own RNG stream, so two runs
that differ only in one strategy's parameters face the same information path.
"""
from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, field

import numpy as np

from ..core.exchange import Exchange
from ..core.orderbook import BUY, SELL, MAX_TICK


@dataclass
class Listing:
    venue: str
    market: str


class Recorder:
    """Samples books, truth and agent state on a fixed clock; keeps the full trade tape."""

    def __init__(self, sim: "Simulator", dt: float = 5.0, track_agents: list[str] | None = None):
        self.sim, self.dt = sim, dt
        self.track = track_agents
        self.t: list[float] = []
        self.book_rows: list[tuple] = []     # (t, venue, market, bid, ask, bid_sz, ask_sz, last, volume, truth)
        self.agent_rows: list[tuple] = []    # (t, agent, venue, equity_mid, equity_truth, cash, reserved, gross_pos)
        self.pos_rows: list[tuple] = []      # (t, agent, venue, market, pos)
        self.trades: list[tuple] = []        # (t, venue, market, price, qty, taker_side, taker, maker, taker_kind, maker_kind, truth)

    def sample(self) -> None:
        sim = self.sim
        t = sim.now
        self.t.append(t)
        marks_mid: dict[tuple, float] = {}
        marks_truth: dict[str, float] = {}
        for v, ex in sim.venues.items():
            for m, mk in ex.markets.items():
                b = mk.book
                bb, bq, ba, aq = b.bbo()
                truth = sim.world.prob(m, t) if m in sim.world.paths else None
                self.book_rows.append((t, v, m, bb or None, ba if ba <= MAX_TICK else None, bq, aq, b.last_price,
                                       b.volume, truth))
                mid = b.mid()
                marks_mid[(v, m)] = mid if mid is not None else (b.last_price if b.last_price is not None else
                                                                 (truth * 100 if truth is not None else 50.0))
                if truth is not None:
                    marks_truth[m] = truth * 100
        ids = self.track if self.track is not None else list(sim.agents)
        for aid in ids:
            for v, ex in sim.venues.items():
                a = ex.accounts.get(aid)
                if a is None:
                    continue
                mm = {m: marks_mid[(v, m)] for m in ex.markets}
                eq_mid = a.equity(mm)
                eq_truth = a.equity({m: marks_truth.get(m, mm[m]) for m in ex.markets})
                gross = sum(abs(q) for q in a.pos.values())
                self.agent_rows.append((t, aid, v, eq_mid, eq_truth, a.cash, a.reserved, gross))
                for m, q in a.pos.items():
                    if q:
                        self.pos_rows.append((t, aid, v, m, q))

    def on_fill(self, venue: str, f) -> None:
        sim = self.sim
        truth = sim.world.prob(f.market, sim.now) if f.market in sim.world.paths else None
        tk = sim.agents[f.taker].kind if f.taker in sim.agents else "?"
        mk = sim.agents[f.maker].kind if f.maker in sim.agents else "?"
        self.trades.append((sim.now, venue, f.market, f.price, f.qty, f.taker_side, f.taker, f.maker, tk, mk, truth,
                            f.taker_fee, f.maker_fee))


class Simulator:
    def __init__(self, world, venues: dict[str, Exchange], seed: int = 0, record_dt: float = 5.0,
                 close_t: float | None = None, track_agents: list[str] | None = None):
        self.world = world
        self.venues = venues
        self.now = 0.0
        self.T = world.T
        self.close_t = close_t if close_t is not None else world.T - 1.0
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self._q: list = []
        self._seq = itertools.count()
        self.agents: dict = {}
        self.news_subs: list = []
        self.tape_subs: list = []
        self.n_events = 0
        self.closed = False
        self.settled = False
        self.recorder = Recorder(self, record_dt, track_agents)
        for ex in venues.values():
            ex.listeners.append(self._on_fill)
        for n in world.news:
            if n.kind == "jump":
                self.schedule(n.t, self._news, n)
        self.schedule(self.close_t, self._close_all)
        self.schedule(0.0, self._sample)

    # ------------------------------------------------------------------ scheduling
    def schedule(self, t: float, fn, *args) -> None:
        heapq.heappush(self._q, (t, next(self._seq), fn, args))

    def add_agent(self, agent) -> None:
        self.agents[agent.id] = agent
        agent.attach(self)
        for v, cash in agent.capital.items():
            self.venues[v].open_account(agent.id, int(cash))
        agent.on_start()
        nxt = agent.next_wake(self.now)
        if nxt is not None:
            self.schedule(nxt, self._wake, agent)
        if getattr(agent, "wants_news", False):
            self.news_subs.append(agent)
        if getattr(agent, "wants_tape", False):
            self.tape_subs.append(agent)

    def run(self, until: float | None = None) -> None:
        until = self.T if until is None else until
        q = self._q
        while q and q[0][0] <= until:
            t, _, fn, args = heapq.heappop(q)
            self.now = t
            fn(*args)
            self.n_events += 1
        self.now = max(self.now, until)
        if until >= self.T and not self.settled:
            self.settle_all()

    # ------------------------------------------------------------------ handlers
    def _sample(self) -> None:
        self.recorder.sample()
        if self.now + self.recorder.dt <= self.close_t:
            self.schedule(self.now + self.recorder.dt, self._sample)

    def _wake(self, agent) -> None:
        if self.closed:
            return
        agent.on_wake()
        nxt = agent.next_wake(self.now)
        if nxt is not None and nxt < self.close_t:
            self.schedule(nxt, self._wake, agent)

    def _news(self, news) -> None:
        for a in self.news_subs:
            d = a.news_delay()
            if d is not None:
                self.schedule(self.now + d, self._deliver_news, a, news)

    def _deliver_news(self, agent, news) -> None:
        if not self.closed:
            agent.on_news(news)

    def _on_fill(self, venue: str, f) -> None:
        self.recorder.on_fill(venue, f)
        ta = self.agents.get(f.taker)
        if ta is not None:
            ta.on_fill(venue, f, True)
        ma = self.agents.get(f.maker)
        if ma is not None:
            ma.on_fill(venue, f, False)
        for a in self.tape_subs:  # public trade prints
            a.on_tape(venue, f)

    def _arrive_order(self, agent, venue, market, side, price, qty, tif, post_only, oid, tag) -> None:
        if self.closed:
            return
        res = self.venues[venue].submit(agent.id, market, side, price, qty, tif=tif, post_only=post_only,
                                        t=self.now, oid=oid, tag=tag)
        agent.on_order_result(venue, market, res)

    def _arrive_cancel(self, agent, venue, oid) -> None:
        if self.closed:
            return
        ok = self.venues[venue].cancel(agent.id, oid, t=self.now)
        agent.on_cancel_result(venue, oid, ok)

    def _close_all(self) -> None:
        self.recorder.sample()
        for ex in self.venues.values():
            for m in ex.markets:
                ex.close_market(m)
        self.closed = True
        for a in self.agents.values():
            a.on_close()

    def settle_all(self) -> None:
        for ex in self.venues.values():
            for m in ex.markets:
                ex.settle(m, self.world.outcome(m))
        self.settled = True
        self.now = self.T
        self.recorder.sample()

    # ------------------------------------------------------------------ agent-facing API
    def submit(self, agent, venue: str, market: str, side: int, price: int, qty: int, tif: str = "GTC",
               post_only: bool = False, tag: str = "") -> int:
        ex = self.venues[venue]
        oid = ex.next_oid()
        self.schedule(self.now + agent.latency(), self._arrive_order, agent, venue, market, side, int(price), int(qty),
                      tif, post_only, oid, tag)
        return oid

    def cancel(self, agent, venue: str, oid: int) -> None:
        self.schedule(self.now + agent.latency(), self._arrive_cancel, agent, venue, oid)

    def book(self, venue: str, market: str):
        return self.venues[venue].markets[market].book

    def truth(self, market: str) -> float:
        return self.world.prob(market, self.now)

    def account(self, agent_id: str, venue: str):
        return self.venues[venue].accounts.get(agent_id)
