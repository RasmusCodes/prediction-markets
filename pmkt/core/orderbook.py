"""Limit order book with price-time priority for one binary contract.

The book is kept in YES-price terms: a bid is "buy YES" (equivalently sell NO) and an ask is "sell YES"
(equivalently buy NO at 100 - price). Prices are integer cents 1..99, quantities integer contracts.

Implementation notes
- One FIFO deque per price level plus an aggregate-quantity array; best bid/ask are cached and
  rescanned only when a level empties (99 levels, so the scan is cheap).
- Cancels are lazy: the order is marked dead and its quantity removed from the aggregate at once; the
  dead entry is popped when matching reaches it. Aggregates are therefore always exact.
- Self-trade prevention: an incoming order never trades with the same trader's resting order; the
  resting order is cancelled instead (cancel-resting policy) and reported in `stp_cancelled`.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

BUY, SELL = 1, -1
MIN_TICK, MAX_TICK = 1, 99


class Order:
    __slots__ = ("oid", "trader", "market", "side", "price", "qty", "rem", "tif", "post_only", "ts",
                 "dead", "fee_res", "res_unit", "tag")

    def __init__(self, oid: int, trader: str, market: str, side: int, price: int, qty: int,
                 tif: str = "GTC", post_only: bool = False, ts: float = 0.0, tag: str = ""):
        self.oid, self.trader, self.market, self.side = oid, trader, market, side
        self.price, self.qty, self.rem = price, qty, qty
        self.tif, self.post_only, self.ts, self.tag = tif, post_only, ts, tag
        self.dead = False
        self.fee_res = 0      # fee reservation still held for this order (cents)
        self.res_unit = 0     # collateral reserved per unfilled contract (cents)

    def __repr__(self) -> str:
        s = "B" if self.side == BUY else "S"
        return f"Order({self.oid} {self.trader} {s}{self.rem}/{self.qty}@{self.price} {self.tif})"


@dataclass(slots=True)
class Fill:
    market: str
    price: int
    qty: int
    taker_side: int
    taker_oid: int
    maker_oid: int
    taker: str
    maker: str
    ts: float
    taker_fee: int = 0
    maker_fee: int = 0


@dataclass
class MatchResult:
    fills: list = field(default_factory=list)
    resting: bool = False
    rejected: str | None = None
    stp_cancelled: list = field(default_factory=list)


class OrderBook:
    def __init__(self, market: str):
        self.market = market
        self.bids: list = [None] * (MAX_TICK + 2)
        self.asks: list = [None] * (MAX_TICK + 2)
        self.bid_qty = [0] * (MAX_TICK + 2)
        self.ask_qty = [0] * (MAX_TICK + 2)
        self.best_bid = 0            # 0 = no bid
        self.best_ask = MAX_TICK + 1  # 100 = no ask
        self.orders: dict[int, Order] = {}
        self.last_price: int | None = None
        self.volume = 0
        self.n_trades = 0
        self.version = 0              # bumps on every change (lets scanners skip unchanged books)

    # ------------------------------------------------------------------ queries
    def bbo(self) -> tuple[int, int, int, int]:
        """(best_bid, bid_size, best_ask, ask_size); bid 0 / ask 100 when that side is empty."""
        bb, ba = self.best_bid, self.best_ask
        return bb, (self.bid_qty[bb] if bb else 0), ba, (self.ask_qty[ba] if ba <= MAX_TICK else 0)

    def mid(self) -> float | None:
        if self.best_bid and self.best_ask <= MAX_TICK:
            return 0.5 * (self.best_bid + self.best_ask)
        return None

    def microprice(self) -> float | None:
        bb, bq, ba, aq = self.bbo()
        if bb and ba <= MAX_TICK and bq + aq > 0:
            return (bb * aq + ba * bq) / (bq + aq)
        return self.mid()

    def depth(self, n: int = 10) -> tuple[list, list]:
        """Aggregated levels: bids best-first [(price, qty)], asks best-first."""
        bids, asks = [], []
        p = self.best_bid
        while p >= MIN_TICK and len(bids) < n:
            if self.bid_qty[p] > 0:
                bids.append((p, self.bid_qty[p]))
            p -= 1
        p = self.best_ask
        while p <= MAX_TICK and len(asks) < n:
            if self.ask_qty[p] > 0:
                asks.append((p, self.ask_qty[p]))
            p += 1
        return bids, asks

    def queue_ahead(self, oid: int) -> int:
        """Live quantity ahead of this order at its price level (0 if it is first or not found)."""
        o = self.orders.get(oid)
        if o is None:
            return 0
        lvl = (self.bids if o.side == BUY else self.asks)[o.price]
        ahead = 0
        for x in lvl:
            if x is o:
                return ahead
            if not x.dead:
                ahead += x.rem
        return ahead

    def available(self, side: int, limit: int, exclude_trader: str | None = None) -> int:
        """Quantity an incoming order of `side` with `limit` could execute against."""
        tot = 0
        if side == BUY:
            rng, levels = range(self.best_ask, min(limit, MAX_TICK) + 1), self.asks
        else:
            rng, levels = range(self.best_bid, max(limit, MIN_TICK) - 1, -1), self.bids
        for p in rng:
            lvl = levels[p]
            if lvl:
                for x in lvl:
                    if not x.dead and x.trader != exclude_trader:
                        tot += x.rem
        return tot

    # ------------------------------------------------------------------ mutations
    def _rescan_bid(self) -> None:
        p = self.best_bid
        while p >= MIN_TICK and self.bid_qty[p] == 0:
            p -= 1
        self.best_bid = p if p >= MIN_TICK else 0

    def _rescan_ask(self) -> None:
        p = self.best_ask
        while p <= MAX_TICK and self.ask_qty[p] == 0:
            p += 1
        self.best_ask = p

    def _rest(self, o: Order) -> None:
        self.version += 1
        if o.side == BUY:
            if self.bids[o.price] is None:
                self.bids[o.price] = deque()
            self.bids[o.price].append(o)
            self.bid_qty[o.price] += o.rem
            if o.price > self.best_bid:
                self.best_bid = o.price
        else:
            if self.asks[o.price] is None:
                self.asks[o.price] = deque()
            self.asks[o.price].append(o)
            self.ask_qty[o.price] += o.rem
            if o.price < self.best_ask:
                self.best_ask = o.price
        self.orders[o.oid] = o

    def cancel(self, oid: int) -> Order | None:
        o = self.orders.pop(oid, None)
        if o is None or o.dead:
            return None
        o.dead = True
        self.version += 1
        if o.side == BUY:
            self.bid_qty[o.price] -= o.rem
            if o.price == self.best_bid and self.bid_qty[o.price] == 0:
                self._rescan_bid()
        else:
            self.ask_qty[o.price] -= o.rem
            if o.price == self.best_ask and self.ask_qty[o.price] == 0:
                self._rescan_ask()
        return o

    def add(self, o: Order) -> MatchResult:
        res = MatchResult()
        if not (MIN_TICK <= o.price <= MAX_TICK) or o.qty <= 0:
            res.rejected = "bad_price_or_qty"
            return res
        crosses = (o.side == BUY and o.price >= self.best_ask) or (o.side == SELL and o.price <= self.best_bid)
        if o.post_only and crosses:
            res.rejected = "post_only_would_cross"
            return res
        if o.tif == "FOK" and self.available(o.side, o.price, exclude_trader=o.trader) < o.qty:
            res.rejected = "fok_insufficient"
            return res
        if crosses:
            self._match(o, res)
        if o.rem > 0:
            if o.tif == "GTC":
                self._rest(o)
                res.resting = True
            else:
                o.dead = True  # IOC/FOK remainder is cancelled
        return res

    def _match(self, o: Order, res: MatchResult) -> None:
        if o.side == BUY:
            levels, qtys = self.asks, self.ask_qty
            def best(): return self.best_ask
            def ok(p): return p <= o.price and p <= MAX_TICK
            rescan = self._rescan_ask
        else:
            levels, qtys = self.bids, self.bid_qty
            def best(): return self.best_bid
            def ok(p): return p >= o.price and p >= MIN_TICK
            rescan = self._rescan_bid
        while o.rem > 0:
            p = best()
            if not ok(p):
                break
            lvl = levels[p]
            while lvl and o.rem > 0:
                m = lvl[0]
                if m.dead:
                    lvl.popleft()
                    continue
                if m.trader == o.trader:  # self-trade prevention: cancel the resting order
                    lvl.popleft()
                    m.dead = True
                    qtys[p] -= m.rem
                    self.orders.pop(m.oid, None)
                    res.stp_cancelled.append(m)
                    continue
                x = o.rem if o.rem < m.rem else m.rem
                m.rem -= x
                o.rem -= x
                qtys[p] -= x
                res.fills.append(Fill(self.market, p, x, o.side, o.oid, m.oid, o.trader, m.trader, o.ts))
                self.last_price = p
                self.volume += x
                self.n_trades += 1
                self.version += 1
                if m.rem == 0:
                    lvl.popleft()
                    m.dead = True
                    self.orders.pop(m.oid, None)
            if qtys[p] == 0:
                rescan()

    def check(self) -> None:
        """Internal consistency assertions (used by tests)."""
        for side_levels, qtys in ((self.bids, self.bid_qty), (self.asks, self.ask_qty)):
            for p in range(MIN_TICK, MAX_TICK + 1):
                lvl = side_levels[p]
                live = sum(x.rem for x in lvl if not x.dead) if lvl else 0
                assert live == qtys[p], (self.market, p, live, qtys[p])
        bb = max([p for p in range(MIN_TICK, MAX_TICK + 1) if self.bid_qty[p] > 0], default=0)
        ba = min([p for p in range(MIN_TICK, MAX_TICK + 1) if self.ask_qty[p] > 0], default=MAX_TICK + 1)
        assert bb == self.best_bid and ba == self.best_ask, (bb, self.best_bid, ba, self.best_ask)
        assert self.best_bid < self.best_ask, "crossed book"
