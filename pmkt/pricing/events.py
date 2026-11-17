"""Event-driven probability updates.

- ScheduledRisk: known future information events (data releases, debates, verdicts) with an expected
  amount of revealed latent variance; the market maker widens or pulls quotes ahead of them.
- event_move_distribution: distribution of the post-event probability in the latent-Gaussian model, which
  gives the expected absolute price move of an announcement before it happens.
- JumpDetector: robust (MAD-scaled) detector of discontinuous moves in log-odds, used to flag news and
  re-anchor fair value.
- news_update: Bayes in log-odds for an observed report with a likelihood ratio.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .normal import Phi, Phi_inv, logit, sigmoid, slogit


@dataclass
class ScheduledRisk:
    events: list = field(default_factory=list)   # [(time, revealed_variance)]

    def pending_within(self, t: float, horizon: float) -> float:
        return float(sum(v for (tau, v) in self.events if t < tau <= t + horizon))

    def next_event(self, t: float):
        fut = [(tau, v) for (tau, v) in self.events if tau > t]
        return min(fut) if fut else None


_GH: dict[int, tuple] = {}


def _gauss_hermite(n: int):
    if n not in _GH:
        x, w = np.polynomial.hermite_e.hermegauss(n)
        _GH[n] = (x, w / w.sum())
    return _GH[n]


def event_move_distribution(p: float, v_event: float, v_total: float, n: int = 41):
    """Post-event probabilities and weights (Gauss-Hermite) when an event reveals v_event of the v_total
    remaining latent variance. Returns (p_after, weights, expected |dp|)."""
    v_event = min(max(v_event, 0.0), v_total * 0.999999)
    z0 = Phi_inv(p)
    x, w = _gauss_hermite(n)
    # latent distance-to-threshold in sd units of total remaining: z0 * sqrt(v_total)
    after = Phi((z0 * math.sqrt(v_total) + x * math.sqrt(v_event)) / math.sqrt(v_total - v_event))
    return after, w, float(np.sum(w * np.abs(after - p)))


def news_update(p: float, likelihood_ratio: float) -> float:
    return float(sigmoid(logit(p) + math.log(max(likelihood_ratio, 1e-300))))


class JumpDetector:
    """Flags log-odds moves larger than k robust standard deviations (and at least min_abs).

    The scale is an EWMA of |change| (x 1.2533 to match a normal sd), updated only with non-jump moves so that
    one jump does not mask the next; O(1) per update."""

    def __init__(self, window: int = 200, k: float = 6.0, min_abs: float = 0.15):
        self.k, self.min_abs = k, min_abs
        self.alpha = 2.0 / (window + 1.0)
        self.scale: float | None = None
        self.n = 0
        self.last: float | None = None

    def update(self, p: float) -> bool:
        x = slogit(float(p))
        jump = False
        if self.last is not None:
            d = abs(x - self.last)
            if self.n >= 20 and self.scale is not None:
                jump = d > max(self.k * 1.2533 * self.scale, self.min_abs)
            if not jump:
                self.scale = d if self.scale is None else (1 - self.alpha) * self.scale + self.alpha * d
                self.n += 1
        self.last = x
        return jump
