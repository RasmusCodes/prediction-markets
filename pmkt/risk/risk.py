"""Pre-trade and portfolio risk for correlated binary markets.

Scenario risk is exact: every family of related markets has a finite set of joint outcome states (e.g. which
bucket an index lands in, or the four A/B combinations of a pair), so the settlement P&L of the whole book
in each state is a matrix product. Worst-case loss, expected P&L under any probability vector and CVaR over
families follow directly; families are treated as independent for the portfolio distribution (Monte Carlo).

Pre-trade checks: max order size, price collar around fair value (fat-finger guard), per-market position
limit, per-family worst-case-loss limit evaluated *with* the new order, and a drawdown kill switch.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.orderbook import BUY


@dataclass
class RiskLimits:
    max_order: int = 300
    max_pos: int = 800                 # contracts per (market), netted across venues
    max_family_loss: float = 2_500_00  # cents of worst-case settlement loss per family
    collar_ticks: int = 35             # reject orders priced further than this from fair value
    max_drawdown: float = 5_000_00     # cents from peak equity before the kill switch trips
    max_gross: int = 20_000            # total contracts across markets


@dataclass
class FamilyStructure:
    family: str
    markets: list           # market ids (columns of M)
    M: np.ndarray           # states x markets, 1 if the market's YES pays in that state


class RiskManager:
    def __init__(self, limits: RiskLimits | None = None, structures: dict | None = None):
        self.lim = limits or RiskLimits()
        self.struct: dict[str, FamilyStructure] = structures or {}
        self.fam_of = {m: f for f, s in self.struct.items() for m in s.markets}
        self.peak = None
        self.killed = False
        self.kill_reason = ""
        self.n_rejects: dict[str, int] = {}

    # --------------------------------------------------------------------------------------- scenarios
    def family_payoff(self, family: str, positions: dict[str, float], cash_flows: float = 0.0) -> np.ndarray:
        """Settlement value (cents) of the positions in each joint state of the family."""
        s = self.struct[family]
        q = np.array([positions.get(m, 0.0) for m in s.markets])
        # long YES pays 100 if YES; long NO (q<0) pays 100 if NO -> value = 100*(q*Y + max(-q,0))
        yes = s.M @ q
        no_leg = np.sum(np.maximum(-q, 0.0))
        return 100.0 * (yes + no_leg) + cash_flows

    def family_pnl_states(self, family: str, positions: dict, marks: dict) -> np.ndarray:
        """P&L per state relative to marking the positions at `marks` (cents per contract YES price)."""
        s = self.struct[family]
        val_now = 0.0
        for m in s.markets:
            q = positions.get(m, 0.0)
            px = marks.get(m, 50.0)
            val_now += q * px if q > 0 else -q * (100.0 - px)
        return self.family_payoff(family, positions) - val_now

    def worst_case_loss(self, family: str, positions: dict, marks: dict) -> float:
        return float(max(0.0, -np.min(self.family_pnl_states(family, positions, marks))))

    def expected_pnl(self, family: str, positions: dict, marks: dict, state_probs: np.ndarray) -> float:
        return float(state_probs @ self.family_pnl_states(family, positions, marks))

    def portfolio_cvar(self, positions: dict, marks: dict, state_probs: dict, alpha: float = 0.05,
                       n: int = 20000, seed: int = 0) -> dict:
        """Monte Carlo over independent families; returns mean, VaR and CVaR of settlement P&L (cents)."""
        rng = np.random.default_rng(seed)
        tot = np.zeros(n)
        for f in self.struct:
            pnl = self.family_pnl_states(f, positions, marks)
            if not np.any(pnl):
                continue
            pr = np.clip(state_probs[f], 0, None)
            pr = pr / pr.sum()
            tot += pnl[rng.choice(len(pnl), size=n, p=pr)]
        q = np.quantile(tot, alpha)
        return {"mean": float(tot.mean()), "var": float(-q), "cvar": float(-tot[tot <= q].mean()) if np.any(tot <= q) else 0.0}

    # --------------------------------------------------------------------------------------- pre-trade
    def _reject(self, why: str):
        self.n_rejects[why] = self.n_rejects.get(why, 0) + 1
        return False, 0, why

    def check(self, market: str, side: int, price: int, qty: int, fair_cents: float | None,
              positions: dict, marks: dict, gross: float = 0.0):
        """-> (ok, allowed_qty, reason). Positions are netted across venues by market id."""
        L = self.lim
        if self.killed:
            return self._reject("killed")
        qty = min(qty, L.max_order)
        if fair_cents is not None and abs(price - fair_cents) > L.collar_ticks:
            return self._reject("collar")
        cur = positions.get(market, 0)
        room = L.max_pos - cur if side == BUY else L.max_pos + cur
        qty = min(qty, max(0, room))
        if qty <= 0:
            return self._reject("position_limit")
        if gross + qty > L.max_gross and ((side == BUY and cur >= 0) or (side != BUY and cur <= 0)):
            return self._reject("gross_limit")
        fam = self.fam_of.get(market)
        if fam is not None:
            # cheap bound first: a family can never lose more than 100c per contract held in it
            bound = 100.0 * (sum(abs(positions.get(m, 0)) for m in self.struct[fam].markets) + qty)
            if bound <= L.max_family_loss:
                return True, qty, "ok"
            newpos = dict(positions)
            newpos[market] = cur + (qty if side == BUY else -qty)
            mk = dict(marks)
            mk.setdefault(market, price)
            wcl = self.worst_case_loss(fam, newpos, mk)
            if wcl > L.max_family_loss and wcl > self.worst_case_loss(fam, positions, mk):
                return self._reject("family_loss")
        return True, qty, "ok"

    def update_equity(self, equity: float) -> bool:
        """Track drawdown; returns True if the kill switch is (now) engaged."""
        if self.peak is None or equity > self.peak:
            self.peak = equity
        if not self.killed and self.peak - equity > self.lim.max_drawdown:
            self.killed = True
            self.kill_reason = f"drawdown {self.peak - equity:.0f}c > {self.lim.max_drawdown:.0f}c"
        return self.killed


def structures_from_world(world) -> dict[str, FamilyStructure]:
    out = {}
    for fid, fam in world.families.items():
        ids, M, _ = fam.states()
        out[fid] = FamilyStructure(fid, ids, M)
    return out
