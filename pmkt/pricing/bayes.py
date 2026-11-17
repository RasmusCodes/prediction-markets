"""Bayesian probability models: base rates, signal fusion in log-odds, Glosten-Milgrom learning from order
flow, and a Kalman filter on the log-odds of the event (the fair-value engine used by the market maker)."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats

from .normal import logit, sigmoid, ssigmoid


@dataclass
class BetaBinomial:
    """Conjugate base-rate model: P(event) ~ Beta(a, b), updated with observed successes/failures."""
    a: float = 1.0
    b: float = 1.0

    def update(self, successes: float, failures: float) -> "BetaBinomial":
        return BetaBinomial(self.a + successes, self.b + failures)

    @property
    def mean(self) -> float:
        return self.a / (self.a + self.b)

    @property
    def var(self) -> float:
        s = self.a + self.b
        return self.a * self.b / (s * s * (s + 1))

    def interval(self, level: float = 0.9) -> tuple[float, float]:
        lo = (1 - level) / 2
        return float(stats.beta.ppf(lo, self.a, self.b)), float(stats.beta.ppf(1 - lo, self.a, self.b))

    @staticmethod
    def from_mean_strength(mean: float, strength: float) -> "BetaBinomial":
        return BetaBinomial(mean * strength, (1 - mean) * strength)


class LogOddsBelief:
    """Bayes' rule in log-odds: posterior logit = prior logit + sum of log-likelihood ratios."""

    def __init__(self, p0: float = 0.5):
        self.x = float(logit(p0))

    @property
    def p(self) -> float:
        return float(sigmoid(self.x))

    def update_lr(self, likelihood_ratio: float) -> float:
        self.x += math.log(max(likelihood_ratio, 1e-300))
        return self.p

    def update_gaussian_signal(self, s: float, mu_yes: float, mu_no: float, sd: float) -> float:
        """Signal s ~ N(mu_yes, sd) if the event happens, N(mu_no, sd) otherwise."""
        llr = ((s - mu_no) ** 2 - (s - mu_yes) ** 2) / (2 * sd * sd)
        self.x += llr
        return self.p

    def update_binary_signal(self, says_yes: bool, accuracy: float) -> float:
        """A source that is right with probability `accuracy` reports yes/no."""
        lr = accuracy / (1 - accuracy)
        self.x += math.log(lr) if says_yes else -math.log(lr)
        return self.p


class GlostenMilgrom:
    """Bayesian dealer (Glosten & Milgrom 1985) for a binary asset worth 1 or 0.

    A fraction mu of arriving traders are informed and act on a private signal that is correct with
    probability q; the rest buy or sell with probability 1/2. Quotes are the dealer's posterior value
    conditional on the next trade being a buy (ask) or a sell (bid); the belief updates after each trade.
    """

    def __init__(self, p0: float = 0.5, mu: float = 0.3, q: float = 0.8):
        self.p, self.mu, self.q = float(p0), float(mu), float(q)

    def _lik(self, buy: bool) -> tuple[float, float]:
        noise = (1 - self.mu) * 0.5
        if buy:
            return self.mu * self.q + noise, self.mu * (1 - self.q) + noise   # P(buy|V=1), P(buy|V=0)
        return self.mu * (1 - self.q) + noise, self.mu * self.q + noise

    def quote(self) -> tuple[float, float]:
        l1, l0 = self._lik(True)
        ask = self.p * l1 / (self.p * l1 + (1 - self.p) * l0)
        l1, l0 = self._lik(False)
        bid = self.p * l1 / (self.p * l1 + (1 - self.p) * l0)
        return bid, ask

    def update(self, buy: bool, n: int = 1) -> float:
        l1, l0 = self._lik(buy)
        x = logit(self.p) + n * math.log(l1 / l0)
        self.p = float(sigmoid(x))
        return self.p


class LogitKalman:
    """Local-level Kalman filter on x = logit(p) with a jump-robust update.

    State: x_t = x_{t-1} + w,  w ~ N(0, q * dt).  Observation: z = x + v, v ~ N(0, R).
    R is supplied per observation (e.g. larger when the quoted spread is wide or the quote is stale).
    Innovations beyond `jump_k` standard deviations inflate the state variance (the information arrived
    as a jump), which lets the filter re-anchor quickly after news instead of averaging over it.
    """

    def __init__(self, p0: float = 0.5, var0: float = 1.0, q: float = 1e-4, jump_k: float = 4.0,
                 jump_inflate: float = 25.0):
        self.x, self.P, self.q = float(logit(p0)), float(var0), float(q)
        self.jump_k, self.jump_inflate = jump_k, jump_inflate
        self.t: float | None = None
        self.n_jumps = 0

    def predict(self, t: float) -> None:
        if self.t is not None and t > self.t:
            self.P += self.q * (t - self.t)
        self.t = t

    def update(self, z_logit: float, R: float, t: float | None = None) -> float:
        if t is not None:
            self.predict(t)
        S = self.P + R
        innov = z_logit - self.x
        if innov * innov > (self.jump_k ** 2) * S:
            self.P *= self.jump_inflate
            self.n_jumps += 1
            S = self.P + R
        K = self.P / S
        self.x += K * innov
        self.P *= (1 - K)
        return self.p

    def update_prob(self, p_obs: float, spread: float | None = None, t: float | None = None,
                    base_R: float = 0.02) -> float:
        """Observe a market probability; the observation noise grows with the quoted spread (in prob units)."""
        p_obs = min(max(p_obs, 0.005), 0.995)
        dlogit_dp = 1.0 / (p_obs * (1 - p_obs))
        sd = base_R ** 0.5 + (0.5 * spread * dlogit_dp if spread else 0.0)
        return self.update(float(logit(p_obs)), sd * sd, t)

    @property
    def p(self) -> float:
        return ssigmoid(self.x)

    def interval(self, z: float = 1.645) -> tuple[float, float]:
        s = math.sqrt(max(self.P, 0.0))
        return ssigmoid(self.x - z * s), ssigmoid(self.x + z * s)


def posterior_from_trades(p0: float, signed_flow: np.ndarray, mu: float = 0.3, q: float = 0.8) -> np.ndarray:
    """Path of the Glosten-Milgrom belief over a sequence of trade signs (+1 buy / -1 sell)."""
    gm = GlostenMilgrom(p0, mu, q)
    out = np.empty(len(signed_flow))
    for i, s in enumerate(signed_flow):
        out[i] = gm.update(s > 0)
    return out
