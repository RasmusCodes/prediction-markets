"""Arbitrage detection by linear programming over joint outcome states (a Dutch-book search).

For a family of related contracts with states s and payoff matrix M (M[s, m] = 1 if market m's YES pays in
state s), and executable book levels on every venue that lists them:

    variables  x_lk >= 0  buy at ask level k of listing l   (<= displayed size)
               y_lk >= 0  sell at bid level k of listing l  (<= displayed size)
    profit_s = sum x (100 M[s,m_l] - ask - fee) + sum y (bid - fee - 100 M[s,m_l])
    maximise   t   subject to   P_s + profit_s >= t  for every state s,   capital <= K

P_s is the settlement value of positions already held in the family, so the same program finds pure
arbitrage from a flat book (P = 0) and repairs a legged-out position (raises the worst case). Selling YES
is buying NO, so its capital use is (100 - bid). One program covers multi-outcome sums, strike-ladder
monotonicity, bucket-vs-ladder consistency, conjunction (Frechet) bounds and cross-venue duplicates.
Fees enter at the exact per-contract rate; integer rounding and per-order fee rounding are re-checked
before any order is sent.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linprog

from ..core.orderbook import BUY, MAX_TICK, SELL
from ..sim.agents import Agent


@dataclass
class Opportunity:
    family: str
    profit: float                 # guaranteed improvement in worst-case settlement value, cents (after fees)
    capital: float
    orders: list                  # [(venue, market, side, limit_price, qty)]
    worst_before: float = 0.0
    worst_after: float = 0.0
    kind: str = ""
    t: float = 0.0
    legs: list = field(default_factory=list)


def _fee_rate(ex, price: int) -> float:
    return ex.fee.taker(price, 10_000) / 10_000.0


def scan_family(sim, family: str, struct, listings, positions: dict | None = None, levels: int = 3,
                capital: float = 5_000_00, min_profit: float = 5.0) -> Opportunity | None:
    """struct: FamilyStructure; listings: [(venue, market)] for markets in the family; positions netted by market."""
    M = struct.M
    col = {m: j for j, m in enumerate(struct.markets)}
    nS = M.shape[0]
    P = np.zeros(nS)
    if positions:
        q = np.array([positions.get(m, 0.0) for m in struct.markets])
        P = 100.0 * (M @ q + np.sum(np.maximum(-q, 0.0)))
    legs = []   # (venue, market, side, price, size, unit_payoff_vec, cash_per_contract, capital_per_contract)
    for v, m in listings:
        ex = sim.venues[v]
        mk = ex.markets.get(m)
        if mk is None or mk.status != "open" or m not in col:
            continue
        bids, asks = mk.book.depth(levels)
        pay = 100.0 * M[:, col[m]]
        for price, size in asks:
            f = _fee_rate(ex, price)
            legs.append((v, m, BUY, price, size, pay, -(price + f), price + f))
        for price, size in bids:
            f = _fee_rate(ex, price)
            legs.append((v, m, SELL, price, size, -pay, price - f, 100 - price + f))
    if not legs:
        return None
    n = len(legs)
    # variables: z_1..z_n (leg quantities), t
    c = np.zeros(n + 1)
    c[-1] = -1.0
    c[:n] = 1e-6                                     # tie-break toward fewer contracts
    A, b = [], []
    for s in range(nS):
        row = np.zeros(n + 1)
        for i, leg in enumerate(legs):
            row[i] = -(leg[5][s] + leg[6])           # -(payoff_s + cash) per contract
        row[-1] = 1.0
        A.append(row)
        b.append(P[s])
    cap_row = np.zeros(n + 1)
    cap_row[:n] = [leg[7] for leg in legs]
    A.append(cap_row)
    b.append(capital)
    bounds = [(0, leg[4]) for leg in legs] + [(None, None)]
    res = linprog(c, A_ub=np.array(A), b_ub=np.array(b), bounds=bounds, method="highs")
    if res.status != 0:
        return None
    z = np.floor(res.x[:n] + 1e-9)
    if z.sum() <= 0:
        return None
    # exact re-check with integer sizes and per-order fee rounding
    orders = {}
    for i, leg in enumerate(legs):
        if z[i] <= 0:
            continue
        v, m, side, price = leg[:4]
        key = (v, m, side)
        o = orders.setdefault(key, {"qty": 0, "limit": price, "cash": 0.0, "legs": []})
        o["qty"] += int(z[i])
        o["limit"] = max(o["limit"], price) if side == BUY else min(o["limit"], price)
        o["cash"] += (-price if side == BUY else price) * z[i]
        o["legs"].append((price, int(z[i])))
    vals = P.copy()
    cap_used = 0.0
    for (v, m, side), o in orders.items():
        ex = sim.venues[v]
        fee = sum(ex.fee.taker(px, q) for px, q in o["legs"])
        pay = 100.0 * M[:, col[m]] * o["qty"]
        vals += (pay if side == BUY else -pay) + o["cash"] - fee
        cap_used += sum((px if side == BUY else 100 - px) * q for px, q in o["legs"]) + fee
    worst_before, worst_after = float(P.min()), float(vals.min())
    gain = worst_after - worst_before
    if gain < min_profit:
        return None
    kinds = sorted({sim.world.defs[m].kind for (_, m, _) in orders})
    venues = sorted({v for (v, _, _) in orders})
    kind = ("cross-venue " if len(venues) > 1 else "") + "/".join(kinds)
    return Opportunity(family, gain, cap_used, [(v, m, s, o["limit"], o["qty"]) for (v, m, s), o in orders.items()],
                       worst_before, worst_after, kind, sim.now)


class ArbScanner:
    """Read-only scanner used by the dashboard and the research code."""

    def __init__(self, sim, structures: dict, levels: int = 3, capital: float = 5_000_00, min_profit: float = 1.0):
        self.sim, self.struct, self.levels = sim, structures, levels
        self.capital, self.min_profit = capital, min_profit
        self.listings = {}
        for f, s in structures.items():
            ls = [(v, m) for v, ex in sim.venues.items() for m in s.markets if m in ex.markets]
            self.listings[f] = ls

    def scan(self) -> list[Opportunity]:
        out = []
        for f, s in self.struct.items():
            o = scan_family(self.sim, f, s, self.listings[f], None, self.levels, self.capital, self.min_profit)
            if o is not None:
                out.append(o)
        return sorted(out, key=lambda o: -o.profit)


class ArbMonitor(Agent):
    """Read-only: scans every `period` seconds and logs executable arbitrage (after fees) at displayed depth."""
    kind = "monitor"

    def __init__(self, structures, period: float = 5.0, min_profit: float = 1.0):
        super().__init__("monitor", {}, [], rate=0.0)
        self.structures, self.period, self.min_profit = structures, period, min_profit
        self.log = []          # (t, family, kind, profit_cents, capital_cents)
        self.latest: list[Opportunity] = []
        self.n_scans = 0

    def next_wake(self, now):
        return now + self.period

    def on_start(self):
        self.scanner = ArbScanner(self.sim, self.structures, levels=3, capital=10_000_00, min_profit=self.min_profit)

    def on_wake(self):
        self.n_scans += 1
        self.latest = self.scanner.scan()
        for o in self.latest:
            self.log.append((self.sim.now, o.family, o.kind, o.profit, o.capital))


class ArbitrageStrategy(Agent):
    kind = "arb"

    def __init__(self, aid, capital, structures: dict, levels: int = 3, max_capital: float = 3_000_00,
                 min_profit: float = 10.0, repair_min: float = 50.0, **kw):
        kw.setdefault("rate", 2.0)
        kw.setdefault("latency", 0.02)
        super().__init__(aid, capital, [], **kw)
        self.struct = structures
        self.levels, self.max_capital = levels, max_capital
        self.min_profit, self.repair_min = min_profit, repair_min
        self.found: list[Opportunity] = []
        self.n_sent = 0

    def on_start(self):
        self.fam_listings = {f: [(v, m) for v, ex in self.sim.venues.items() for m in s.markets if m in ex.markets]
                             for f, s in self.struct.items()}
        self.seen_version = {}

    def _version(self, f) -> tuple:
        return tuple(self.sim.venues[v].markets[m].book.version for v, m in self.fam_listings[f])

    def net_positions(self) -> dict:
        pos = {}
        for ex in self.sim.venues.values():
            a = ex.accounts.get(self.id)
            if a:
                for m, q in a.pos.items():
                    if q:
                        pos[m] = pos.get(m, 0) + q
        return pos

    def on_wake(self):
        if any(st.status == "pending" for st in self.orders.values()):
            return  # previous legs still in flight
        pos = self.net_positions()
        for f, s in self.struct.items():
            if not self.fam_listings[f]:
                continue
            ver = self._version(f)
            if self.seen_version.get(f) == ver:
                continue  # nothing changed in this family's books since the last solve
            self.seen_version[f] = ver
            fam_pos = {m: pos.get(m, 0) for m in s.markets}
            flat = not any(fam_pos.values())
            opp = scan_family(self.sim, f, s, self.fam_listings[f], None if flat else fam_pos, self.levels,
                              self.max_capital, self.min_profit if flat else self.repair_min)
            if opp is None:
                continue
            opp.kind = opp.kind if flat else "repair " + opp.kind
            self.found.append(opp)
            for v, m, side, limit, qty in opp.orders:
                self.submit(v, m, side, limit, qty, "IOC", tag="arb")
                self.n_sent += 1
