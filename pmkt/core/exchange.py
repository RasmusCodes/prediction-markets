"""Exchange (venue) simulator: markets, fully collateralised accounts, fees, settlement, set conversion.

Accounting model (integer cents, provably solvent)
- Every order is fully collateralised when it is accepted: a buy of q YES at limit L reserves q*L, a sell
  (= buy of NO at 100-L) reserves q*(100-L), plus a per-contract fee bound. This is gross margining of
  resting orders, which is how a fully-funded venue such as Kalshi treats open orders; it is
  conservative for sell-to-close orders.
- A fill releases the reservation, pays the actual price (takers keep any price improvement) and nets
  positions: a YES and a NO held in the same market merge into 100c of cash immediately. Positions are
  stored as net YES contracts (negative = long NO).
- Settlement pays 100c per winning contract. Minting a full set of a convertible mutually-exclusive event
  costs 100c per set; redeeming one pays 100c. Total cash + fees is conserved through settlement
  (tests/test_exchange.py checks this invariant under random order flow).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .fees import FeeSchedule, ZeroFee
from .orderbook import BUY, SELL, MAX_TICK, MIN_TICK, Fill, Order, OrderBook


@dataclass
class MarketSpec:
    id: str
    title: str = ""
    kind: str = "binary"          # binary | ladder | bucket | combo
    family: str = ""
    event: str | None = None      # mutually-exclusive group id (multi-outcome)
    close_t: float = float("inf")
    meta: dict = field(default_factory=dict)


class Market:
    __slots__ = ("spec", "book", "status", "outcome")

    def __init__(self, spec: MarketSpec):
        self.spec = spec
        self.book = OrderBook(spec.id)
        self.status = "open"
        self.outcome: bool | None = None


class Account:
    __slots__ = ("trader", "cash", "reserved", "pos", "fees_paid", "volume", "n_orders", "n_cancels",
                 "n_fills", "n_rejects", "rl_tokens", "rl_t", "cash0", "settled_pnl")

    def __init__(self, trader: str, cash: int):
        self.trader = trader
        self.cash = int(cash)
        self.cash0 = int(cash)
        self.reserved = 0
        self.pos: dict[str, int] = {}
        self.fees_paid = 0
        self.volume = 0
        self.n_orders = self.n_cancels = self.n_fills = self.n_rejects = 0
        self.rl_tokens = None
        self.rl_t = 0.0
        self.settled_pnl = 0

    def equity(self, marks: dict[str, float]) -> float:
        """Cash + reservations + positions marked at `marks` (YES price in cents)."""
        v = float(self.cash + self.reserved)
        for m, q in self.pos.items():
            if q:
                px = marks.get(m, 50.0)
                v += q * px if q > 0 else (-q) * (100.0 - px)
        return v


@dataclass
class SubmitResult:
    oid: int
    status: str              # filled | partial | resting | done | rejected
    fills: list
    reason: str | None = None
    resting: bool = False


class Exchange:
    def __init__(self, venue: str, fee: FeeSchedule | None = None, msg_rate: float | None = None,
                 msg_burst: int = 40, allow_conversion: bool = False):
        self.venue = venue
        self.fee = fee or ZeroFee()
        self.msg_rate, self.msg_burst = msg_rate, msg_burst
        self.allow_conversion = allow_conversion
        self.markets: dict[str, Market] = {}
        self.events: dict[str, list[str]] = {}
        self.convertible: dict[str, bool] = {}
        self.accounts: dict[str, Account] = {}
        self.fees_collected = 0
        self.tape: list[Fill] = []
        self.listeners: list = []          # callables(venue, fill)
        self.order_listeners: list = []    # callables(venue, event, order) for book-change notifications
        self._resting: dict[int, Order] = {}
        self._oid = 0

    # ------------------------------------------------------------------ setup
    def add_market(self, spec: MarketSpec) -> Market:
        m = Market(spec)
        self.markets[spec.id] = m
        if spec.event:
            self.events.setdefault(spec.event, []).append(spec.id)
        return m

    def set_event_convertible(self, event: str, convertible: bool = True) -> None:
        self.convertible[event] = convertible and self.allow_conversion

    def open_account(self, trader: str, cash: int) -> Account:
        a = self.accounts.get(trader)
        if a is None:
            a = self.accounts[trader] = Account(trader, cash)
        return a

    def book(self, market: str) -> OrderBook:
        return self.markets[market].book

    # ------------------------------------------------------------------ helpers
    def _rate_ok(self, a: Account, t: float) -> bool:
        if self.msg_rate is None:
            return True
        if a.rl_tokens is None:
            a.rl_tokens, a.rl_t = float(self.msg_burst), t
        a.rl_tokens = min(float(self.msg_burst), a.rl_tokens + (t - a.rl_t) * self.msg_rate)
        a.rl_t = t
        if a.rl_tokens >= 1.0:
            a.rl_tokens -= 1.0
            return True
        return False

    def next_oid(self) -> int:
        self._oid += 1
        return self._oid

    def _release(self, a: Account, o: Order) -> None:
        amt = o.rem * o.res_unit + o.fee_res
        a.reserved -= amt
        a.cash += amt
        o.fee_res = 0

    def _apply_fill_side(self, a: Account, o: Order, f: int, p: int, fee: int) -> None:
        m = o.market
        rel = f * o.res_unit
        a.reserved -= rel
        a.cash += rel
        x = a.pos.get(m, 0)
        if o.side == BUY:
            a.cash -= f * p
            merge = min(f, -x) if x < 0 else 0
            a.pos[m] = x + f
        else:
            a.cash -= f * (100 - p)
            merge = min(f, x) if x > 0 else 0
            a.pos[m] = x - f
        a.cash += 100 * merge
        take = fee if fee <= o.fee_res else o.fee_res
        o.fee_res -= take
        a.reserved -= take
        a.cash -= fee - take
        a.fees_paid += fee
        a.volume += f
        a.n_fills += 1

    # ------------------------------------------------------------------ trading API
    def submit(self, trader: str, market: str, side: int, price: int, qty: int, tif: str = "GTC",
               post_only: bool = False, t: float = 0.0, oid: int | None = None, tag: str = "") -> SubmitResult:
        oid = oid if oid is not None else self.next_oid()
        a = self.accounts[trader]
        mk = self.markets.get(market)
        if mk is None or mk.status != "open":
            a.n_rejects += 1
            return SubmitResult(oid, "rejected", [], "market_closed")
        if not self._rate_ok(a, t):
            a.n_rejects += 1
            return SubmitResult(oid, "rejected", [], "rate_limited")
        price, qty = int(price), int(qty)
        if not (MIN_TICK <= price <= MAX_TICK) or qty <= 0 or side not in (BUY, SELL):
            a.n_rejects += 1
            return SubmitResult(oid, "rejected", [], "bad_order")
        o = Order(oid, trader, market, side, price, qty, tif, post_only, t, tag)
        o.res_unit = price if side == BUY else 100 - price
        lo, hi = (MIN_TICK, price) if side == BUY else (price, MAX_TICK)
        o.fee_res = qty * self.fee.per_contract_bound(lo, hi)
        need = qty * o.res_unit + o.fee_res
        if need > a.cash:
            a.n_rejects += 1
            return SubmitResult(oid, "rejected", [], "insufficient_funds")
        a.cash -= need
        a.reserved += need
        a.n_orders += 1
        res = mk.book.add(o)
        if res.rejected:
            self._release(a, o)
            a.n_orders -= 1
            a.n_rejects += 1
            return SubmitResult(oid, "rejected", [], res.rejected)
        for c in res.stp_cancelled:  # resting orders of the same trader removed by self-trade prevention
            self._resting.pop(c.oid, None)
            self._release(self.accounts[c.trader], c)
            self._notify_order("cancel", c)
        for fl in res.fills:
            fl.taker_fee = self.fee.taker(fl.price, fl.qty)
            fl.maker_fee = self.fee.maker(fl.price, fl.qty)
            fl.ts = t
            self._apply_fill_side(a, o, fl.qty, fl.price, fl.taker_fee)
            mo = self._resting.get(fl.maker_oid)
            ma = self.accounts[fl.maker]
            self._apply_fill_side(ma, mo, fl.qty, fl.price, fl.maker_fee)
            if mo.rem == 0:
                self._release(ma, mo)
                self._resting.pop(fl.maker_oid, None)
            self.fees_collected += fl.taker_fee + fl.maker_fee
            self.tape.append(fl)
            for cb in self.listeners:
                cb(self.venue, fl)
        if res.resting:
            self._resting[oid] = o
            self._notify_order("rest", o)
        else:
            self._release(a, o)  # IOC/FOK remainder or fully filled: free what is left
        if not res.fills:
            status = "resting" if res.resting else "done"
        elif o.rem == 0:
            status = "filled"
        else:
            status = "partial"
        return SubmitResult(oid, status, res.fills, None, res.resting)

    def _notify_order(self, ev: str, o: Order) -> None:
        for cb in self.order_listeners:
            cb(self.venue, ev, o)

    def cancel(self, trader: str, oid: int, t: float = 0.0) -> bool:
        o = self._resting.get(oid)
        if o is None or o.trader != trader:
            return False
        a = self.accounts[trader]
        if not self._rate_ok(a, t):
            a.n_rejects += 1
            return False
        if self.markets[o.market].book.cancel(oid) is None:
            return False
        self._resting.pop(oid, None)
        self._release(a, o)
        a.n_cancels += 1
        self._notify_order("cancel", o)
        return True

    def cancel_all(self, trader: str, market: str | None = None, t: float = 0.0) -> int:
        ids = [oid for oid, o in self._resting.items() if o.trader == trader and (market is None or o.market == market)]
        n = 0
        for oid in ids:
            o = self._resting.pop(oid)
            if self.markets[o.market].book.cancel(oid) is not None:
                self._release(self.accounts[trader], o)
                self.accounts[trader].n_cancels += 1
                self._notify_order("cancel", o)
                n += 1
        return n

    def open_orders(self, trader: str, market: str | None = None) -> list[Order]:
        return [o for o in self._resting.values() if o.trader == trader and (market is None or o.market == market)]

    # ------------------------------------------------------------------ conversion (multi-outcome)
    def mint_set(self, trader: str, event: str, n: int) -> bool:
        if not self.convertible.get(event) or n <= 0:
            return False
        a = self.accounts[trader]
        if a.cash < 100 * n:
            return False
        a.cash -= 100 * n
        for m in self.events[event]:
            x = a.pos.get(m, 0)
            merge = min(n, -x) if x < 0 else 0
            a.pos[m] = x + n
            a.cash += 100 * merge
        return True

    def redeem_set(self, trader: str, event: str, n: int) -> bool:
        if not self.convertible.get(event) or n <= 0:
            return False
        a = self.accounts[trader]
        mids = self.events[event]
        if any(a.pos.get(m, 0) < n for m in mids):
            return False
        for m in mids:
            a.pos[m] -= n
        a.cash += 100 * n
        return True

    # ------------------------------------------------------------------ lifecycle
    def close_market(self, market: str) -> None:
        mk = self.markets[market]
        if mk.status != "open":
            return
        for oid in [oid for oid, o in self._resting.items() if o.market == market]:
            o = self._resting.pop(oid)
            mk.book.cancel(oid)
            self._release(self.accounts[o.trader], o)
        mk.status = "closed"

    def settle(self, market: str, yes: bool) -> None:
        self.close_market(market)
        mk = self.markets[market]
        if mk.status == "settled":
            return
        mk.outcome = bool(yes)
        for a in self.accounts.values():
            q = a.pos.get(market, 0)
            if q > 0 and yes:
                a.cash += 100 * q
            elif q < 0 and not yes:
                a.cash += 100 * (-q)
            if q:
                a.pos[market] = 0
        mk.status = "settled"

    # ------------------------------------------------------------------ reporting
    def total_cash(self) -> int:
        return sum(a.cash + a.reserved for a in self.accounts.values())

    def open_interest(self, market: str) -> int:
        return sum(q for a in self.accounts.values() if (q := a.pos.get(market, 0)) > 0)

    def snapshot(self, market: str, depth: int = 10) -> dict:
        mk = self.markets[market]
        b = mk.book
        bids, asks = b.depth(depth)
        bb, bq, ba, aq = b.bbo()
        return {"venue": self.venue, "market": market, "status": mk.status, "bid": bb or None,
                "ask": ba if ba <= MAX_TICK else None, "bid_sz": bq, "ask_sz": aq, "mid": b.mid(),
                "last": b.last_price, "volume": b.volume, "bids": bids, "asks": asks,
                "outcome": mk.outcome}
