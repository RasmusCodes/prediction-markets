"""Historical replay backtester for recorded real-market data (top-of-book quotes + trade prints).

The recorded market is treated as exogenous (our orders do not move it), which is the standard assumption for
small size; fills are modelled conservatively:

- Taker orders execute against the first recorded quote at or after decision time + latency, at most the
  displayed size, at the displayed price, plus the venue fee.
- Passive orders join the back of the queue when they arrive (decision time + latency). A resting bid at p
  fills only from later trade prints whose aggressor sold: a print below p fills us outright (the market
  traded through our level); a print at p first consumes the quantity that was displayed ahead of us. If the
  displayed size at our level later drops below our queue position without trades (cancels ahead), the queue
  position shrinks to the displayed size. Cancels take effect after the latency.
- Settlement at the market's result when known, else marked at the last mid.

Strategy interface: an object with on_quote(bt, t, q) / on_trade(bt, t, tr) callbacks that call
bt.quote(side, price, qty) (replace our resting order on that side; price None cancels) or
bt.take(side, limit, qty). Prices in cents, sizes in contracts.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..core.fees import FeeSchedule, ZeroFee

BUY, SELL = 1, -1


@dataclass
class Resting:
    side: int
    price: int
    qty: int
    active_t: float           # when it reaches the venue
    queue: float = -1.0       # quantity ahead; set on activation
    cancel_t: float | None = None


@dataclass
class ReplayResult:
    fills: pd.DataFrame
    pnl: float
    fees: float
    position: int
    settle_price: float
    n_orders: int
    path: pd.DataFrame = field(default_factory=pd.DataFrame)


class ReplayBacktester:
    def __init__(self, quotes: pd.DataFrame, trades: pd.DataFrame, fee: FeeSchedule | None = None,
                 latency: float = 0.25, result: int | None = None, max_pos: int = 500, maker_fee: bool = True):
        """quotes: columns t, bid, ask, bid_sz, ask_sz (cents / contracts), sorted by t.
        trades: columns t, price, qty, taker_side (+1 buyer-initiated, -1 seller-initiated), sorted by t."""
        self.q = quotes.reset_index(drop=True)
        self.tr = trades.reset_index(drop=True)
        self.fee = fee or ZeroFee()
        self.lat = latency
        self.result = result
        self.max_pos = max_pos
        self.maker_fee = maker_fee
        self.cash = 0.0
        self.pos = 0
        self.fees = 0.0
        self.fills = []
        self.resting: dict[int, Resting] = {}
        self.pending_takes = []
        self.n_orders = 0
        self.now = 0.0
        self.cur = None

    # -------------------------------------------------------------------------------- strategy API
    def quote(self, side: int, price: int | None, qty: int = 0) -> None:
        old = self.resting.get(side)
        if old is not None and old.cancel_t is None:
            if price is not None and old.price == price and old.qty == qty:
                return
            old.cancel_t = self.now + self.lat
        if price is not None and qty > 0:
            room = self.max_pos - self.pos if side == BUY else self.max_pos + self.pos
            qty = min(qty, max(room, 0))
            if qty > 0:
                self.resting[side * 1000 + self.n_orders] = Resting(side, int(price), int(qty), self.now + self.lat)
                self.n_orders += 1

    def take(self, side: int, limit: int, qty: int) -> None:
        room = self.max_pos - self.pos if side == BUY else self.max_pos + self.pos
        qty = min(qty, max(room, 0))
        if qty > 0:
            self.pending_takes.append((self.now + self.lat, side, int(limit), int(qty)))
            self.n_orders += 1

    # -------------------------------------------------------------------------------- internals
    def _fill(self, t, side, price, qty, maker):
        fee = (self.fee.maker(price, qty) if (maker and self.maker_fee) else (0 if maker else self.fee.taker(price, qty)))
        self.cash -= side * price * qty + fee
        self.pos += side * qty
        self.fees += fee
        self.fills.append((t, side, price, qty, int(maker), fee))

    def _activate_and_cancel(self, t, q):
        for k, r in list(self.resting.items()):
            if r.cancel_t is not None and r.cancel_t <= t:
                del self.resting[k]
                continue
            if r.active_t <= t and r.queue < 0:
                best = q.bid if r.side == BUY else q.ask
                shown = q.bid_sz if r.side == BUY else q.ask_sz
                if best is None or (np.isnan(best) if isinstance(best, float) else False):
                    r.queue = 0.0
                elif (r.side == BUY and r.price > best) or (r.side == SELL and r.price < best):
                    r.queue = 0.0                     # improving the touch: first in line
                elif r.price == best:
                    r.queue = float(shown or 0)
                else:
                    r.queue = float("inf")            # behind better levels; fills only on trade-through
            elif r.queue > 0 and r.queue != float("inf"):
                best = q.bid if r.side == BUY else q.ask
                shown = q.bid_sz if r.side == BUY else q.ask_sz
                if best == r.price and shown is not None and shown < r.queue:
                    r.queue = float(shown)            # cancels ahead of us
                elif (r.side == BUY and best is not None and best < r.price) or \
                        (r.side == SELL and best is not None and best > r.price):
                    r.queue = 0.0                     # level ahead vanished: we are now at the front

    def _on_trade(self, t, tr):
        for k, r in list(self.resting.items()):
            if r.active_t > t or r.queue < 0 or (r.cancel_t is not None and r.cancel_t <= t):
                continue
            hit = (r.side == BUY and tr.taker_side == SELL and tr.price <= r.price) or \
                  (r.side == SELL and tr.taker_side == BUY and tr.price >= r.price)
            if not hit:
                continue
            avail = tr.qty
            through = (tr.price < r.price) if r.side == BUY else (tr.price > r.price)
            if not through:
                if r.queue == float("inf"):
                    continue
                eat = min(r.queue, avail)
                r.queue -= eat
                avail -= eat
            if avail <= 0:
                continue
            x = int(min(r.qty, avail))
            if x > 0:
                self._fill(t, r.side, r.price, x, True)
                r.qty -= x
                if r.qty <= 0:
                    del self.resting[k]

    def _process_takes(self, t, q):
        keep = []
        for (ta, side, limit, qty) in self.pending_takes:
            if ta > t:
                keep.append((ta, side, limit, qty))
                continue
            px, sz = (q.ask, q.ask_sz) if side == BUY else (q.bid, q.bid_sz)
            if px is None or (isinstance(px, float) and np.isnan(px)):
                continue
            if (side == BUY and px <= limit) or (side == SELL and px >= limit):
                x = int(min(qty, sz or 0))
                if x > 0:
                    self._fill(t, side, int(px), x, False)
        self.pending_takes = keep

    def run(self, strategy, sample_every: float = 30.0) -> ReplayResult:
        qi, ti = 0, 0
        nq, nt = len(self.q), len(self.tr)
        path, last_sample = [], -1e18
        qt = self.q.t.values
        tt = self.tr.t.values if nt else np.array([])
        while qi < nq or ti < nt:
            if ti < nt and (qi >= nq or tt[ti] < qt[qi]):
                tr = self.tr.iloc[ti]
                self.now = float(tr.t)
                self._on_trade(self.now, tr)
                strategy.on_trade(self, self.now, tr)
                ti += 1
            else:
                q = self.q.iloc[qi]
                self.now = float(q.t)
                self.cur = q
                self._activate_and_cancel(self.now, q)
                self._process_takes(self.now, q)
                strategy.on_quote(self, self.now, q)
                qi += 1
                if self.now - last_sample >= sample_every:
                    mid = 0.5 * (q.bid + q.ask) if q.bid == q.bid and q.ask == q.ask else np.nan
                    path.append((self.now, self.pos, self.cash, self.cash + self.pos * (mid if mid == mid else 0)))
                    last_sample = self.now
        if self.result is not None:
            settle = 100.0 * self.result
        else:
            last = self.q.iloc[-1]
            settle = 0.5 * (last.bid + last.ask)
        pnl = self.cash + self.pos * settle
        fills = pd.DataFrame(self.fills, columns=["t", "side", "price", "qty", "maker", "fee"])
        return ReplayResult(fills, pnl / 100.0, self.fees / 100.0, self.pos, settle, self.n_orders,
                            pd.DataFrame(path, columns=["t", "pos", "cash", "equity_mid"]))
