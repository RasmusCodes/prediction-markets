"""Fee schedules. All amounts are integer cents; prices are integer cents in [1, 99].

QuadraticFee reproduces the published Kalshi formula  fee = roundup(rate * C * P * (1 - P))  dollars, i.e.
ceil(rate * C * P(1-P) * 100) cents with P in dollars (general rate 0.07; some series also charge makers,
e.g. 0.0175). Rounding is applied per fill, which is what makes small orders comparatively expensive
(one contract at 50c pays 2c, not 1.75c). Verify the current schedule before using these numbers live.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


class FeeSchedule:
    name = "zero"

    def taker(self, price: int, qty: int) -> int:
        return 0

    def maker(self, price: int, qty: int) -> int:
        return 0

    def per_contract_bound(self, lo: int, hi: int) -> int:
        """Upper bound on the fee per contract for any fill price in [lo, hi] (used for reservations)."""
        p = min(max(50, lo), hi)  # P(1-P) peaks at 50c
        return max(self.taker(p, 1), self.maker(p, 1), 0)

    def describe(self) -> str:
        return self.name


@dataclass
class ZeroFee(FeeSchedule):
    name: str = "zero"


@dataclass
class QuadraticFee(FeeSchedule):
    taker_rate: float = 0.07
    maker_rate: float = 0.0
    name: str = "quadratic"

    @staticmethod
    def _fee(rate: float, price: int, qty: int) -> int:
        if rate == 0.0 or qty <= 0:
            return 0
        P = price / 100.0
        return int(math.ceil(rate * qty * P * (1.0 - P) * 100.0 - 1e-9))

    def taker(self, price: int, qty: int) -> int:
        return self._fee(self.taker_rate, price, qty)

    def maker(self, price: int, qty: int) -> int:
        return self._fee(self.maker_rate, price, qty)

    def describe(self) -> str:
        return f"quadratic(taker={self.taker_rate}, maker={self.maker_rate})"


@dataclass
class LinearFee(FeeSchedule):
    """Proportional to notional: bps of price paid, rounded up per fill. Negative maker_bps = rebate."""
    taker_bps: float = 0.0
    maker_bps: float = 0.0
    name: str = "linear"

    @staticmethod
    def _fee(bps: float, price: int, qty: int) -> int:
        if bps == 0.0 or qty <= 0:
            return 0
        raw = bps * 1e-4 * price * qty
        return int(math.ceil(raw - 1e-9)) if raw > 0 else -int(math.floor(-raw + 1e-9))

    def taker(self, price: int, qty: int) -> int:
        return self._fee(self.taker_bps, price, qty)

    def maker(self, price: int, qty: int) -> int:
        return self._fee(self.maker_bps, price, qty)

    def per_contract_bound(self, lo: int, hi: int) -> int:
        return max(self.taker(hi, 1), self.maker(hi, 1), 0)

    def describe(self) -> str:
        return f"linear(taker={self.taker_bps}bp, maker={self.maker_bps}bp)"


def fee_from_spec(spec: dict | str | None) -> FeeSchedule:
    if spec is None or spec == "zero":
        return ZeroFee()
    if isinstance(spec, str):
        if spec == "kalshi":
            return QuadraticFee(0.07, 0.0)
        if spec == "kalshi_maker":
            return QuadraticFee(0.07, 0.0175)
        raise ValueError(f"unknown fee spec {spec!r}")
    kind = spec.get("kind", "quadratic")
    if kind == "quadratic":
        return QuadraticFee(spec.get("taker_rate", 0.07), spec.get("maker_rate", 0.0))
    if kind == "linear":
        return LinearFee(spec.get("taker_bps", 0.0), spec.get("maker_bps", 0.0))
    if kind == "zero":
        return ZeroFee()
    raise ValueError(f"unknown fee kind {kind!r}")
