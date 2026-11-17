"""Order management: quote reconciliation and smart order routing.

QuoteManager keeps a desired quote set in sync with live orders while sending as few messages as possible:
an existing order at the right side and price is kept (it keeps its queue position) unless partial fills have
shrunk it below `min_rem_frac` of the desired size; everything else is cancelled and the missing quotes are
posted post-only. A per-wake message budget mirrors exchange rate limits.

route_order walks every venue's book for the same market, ranks price levels by fee-adjusted cost, and splits
a parent order into IOC child orders that respect a limit price on the all-in (price + fee) basis.
"""
from __future__ import annotations

from ..core.orderbook import BUY, MAX_TICK, SELL


class QuoteManager:
    def __init__(self, agent, min_rem_frac: float = 0.35, msg_budget: int = 60):
        self.agent = agent
        self.min_rem_frac = min_rem_frac
        self.msg_budget = msg_budget
        self.msgs_this_wake = 0
        self.n_msgs = 0

    def new_wake(self) -> None:
        self.msgs_this_wake = 0

    def _spend(self) -> bool:
        if self.msgs_this_wake >= self.msg_budget:
            return False
        self.msgs_this_wake += 1
        self.n_msgs += 1
        return True

    def sync(self, venue: str, market: str, desired, post_only: bool = True) -> int:
        a = self.agent
        want = {}
        for side, price, size in desired:
            if 1 <= price <= MAX_TICK and size > 0:
                want[(side, int(price))] = want.get((side, int(price)), 0) + int(size)
        kept = set()
        sent = 0
        for oid, st in list(a.orders.items()):
            if st.venue != venue or st.market != market or st.tif != "GTC" or st.status == "cancelling":
                continue
            key = (st.side, st.price)
            if key in want and key not in kept and st.rem >= self.min_rem_frac * want[key]:
                kept.add(key)
            elif self._spend():
                a.cancel(oid)
                sent += 1
        for key, size in want.items():
            if key in kept:
                continue
            if not self._spend():
                break
            a.submit(venue, market, key[0], key[1], size, "GTC", post_only=post_only)
            sent += 1
        return sent


def route_order(sim, listings, side: int, qty: int, limit_all_in: float, taker: bool = True):
    """Split a parent order across venues by fee-adjusted price. Returns [(venue, market, price, qty)]."""
    levels = []
    for venue, market in listings:
        ex = sim.venues[venue]
        if ex.markets[market].status != "open":
            continue
        bids, asks = ex.markets[market].book.depth(10)
        book_side = asks if side == BUY else bids
        for price, size in book_side:
            fee = (ex.fee.taker(price, 100) if taker else ex.fee.maker(price, 100)) / 100.0
            all_in = price + fee if side == BUY else price - fee
            levels.append((all_in, venue, market, price, size))
    levels.sort(key=lambda x: x[0], reverse=(side == SELL))
    out, left = [], qty
    agg: dict[tuple, list] = {}
    for all_in, venue, market, price, size in levels:
        if left <= 0:
            break
        if (side == BUY and all_in > limit_all_in) or (side == SELL and all_in < limit_all_in):
            break
        take = min(left, size)
        key = (venue, market)
        if key in agg:
            agg[key][0] = price if side == BUY else min(agg[key][0], price)
            agg[key][1] += take
        else:
            agg[key] = [price, take]
        left -= take
    for (venue, market), (price, q) in agg.items():
        out.append((venue, market, price, q))
    return out
