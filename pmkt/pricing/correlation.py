"""Correlation models for related binary events.

- Gaussian copula: P(A and B) = Phi2(Phi^-1(pA), Phi^-1(pB); rho); implied rho from a combo price.
- Frechet-Hoeffding bounds: the no-arbitrage band for a conjunction given its marginals.
- Tetrachoric correlation from paired binary outcomes.
- Online EWMA covariance of log-odds changes (used for portfolio-level inventory risk).
- Lead-lag cross-correlation of log-odds changes (information transmission between related markets).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from .normal import Phi_inv, bvn_cdf, logit


def frechet_bounds(pA: float, pB: float) -> tuple[float, float]:
    return max(0.0, pA + pB - 1.0), min(pA, pB)


def copula_joint(pA, pB, rho):
    """P(A and B) under a Gaussian copula with latent correlation rho."""
    return bvn_cdf(Phi_inv(pA), Phi_inv(pB), rho)


def copula_conditional(pA: float, pB: float, rho: float, a_true: bool = True) -> float:
    """P(B | A) (or P(B | not A)) under the copula."""
    j = float(copula_joint(pA, pB, rho))
    return j / pA if a_true else (pB - j) / (1 - pA)


def implied_rho(pA: float, pB: float, pAB: float) -> float | None:
    """Latent correlation that reproduces an observed conjunction price; None if outside the Frechet band."""
    lo, hi = frechet_bounds(pA, pB)
    if not (lo < pAB < hi):
        return None
    f = lambda r: float(copula_joint(pA, pB, r)) - pAB
    return float(brentq(f, -0.9999, 0.9999))


def tetrachoric(y1, y2) -> float:
    """MLE of the latent correlation of two binary variables (thresholds fixed at the marginal rates)."""
    y1, y2 = np.asarray(y1, int), np.asarray(y2, int)
    n11 = np.sum((y1 == 1) & (y2 == 1)); n10 = np.sum((y1 == 1) & (y2 == 0))
    n01 = np.sum((y1 == 0) & (y2 == 1)); n00 = np.sum((y1 == 0) & (y2 == 0))
    n = n11 + n10 + n01 + n00
    p1, p2 = (n11 + n10) / n, (n11 + n01) / n
    p1, p2 = np.clip(p1, 1e-4, 1 - 1e-4), np.clip(p2, 1e-4, 1 - 1e-4)

    def nll(r):
        j = float(np.clip(copula_joint(p1, p2, r), 1e-12, None))
        cells = np.clip([j, p1 - j, p2 - j, 1 - p1 - p2 + j], 1e-12, None)
        return -(n11 * np.log(cells[0]) + n10 * np.log(cells[1]) + n01 * np.log(cells[2]) + n00 * np.log(cells[3]))
    return float(minimize_scalar(nll, bounds=(-0.999, 0.999), method="bounded").x)


class EWMACov:
    """Exponentially weighted covariance of log-odds changes across markets (half-life in observations)."""

    def __init__(self, names: list[str], halflife: float = 200.0, prior_var: float = 1e-3):
        self.names = list(names)
        self.ix = {n: i for i, n in enumerate(self.names)}
        k = len(self.names)
        self.lam = 0.5 ** (1.0 / halflife)
        self.S = np.eye(k) * prior_var
        self.last: dict[str, float] = {}

    def update(self, probs: dict[str, float]) -> None:
        d = np.zeros(len(self.names))
        seen = False
        for n, p in probs.items():
            if n not in self.ix or p is None:
                continue
            x = float(logit(p))
            if n in self.last:
                d[self.ix[n]] = x - self.last[n]
                seen = True
            self.last[n] = x
        if seen:
            self.S = self.lam * self.S + (1 - self.lam) * np.outer(d, d)

    def cov(self) -> np.ndarray:
        return self.S

    def corr(self) -> np.ndarray:
        s = np.sqrt(np.clip(np.diag(self.S), 1e-18, None))
        return self.S / np.outer(s, s)


def logodds_changes(p: np.ndarray) -> np.ndarray:
    return np.diff(logit(np.asarray(p, float)))


def lead_lag(x: np.ndarray, y: np.ndarray, max_lag: int = 10) -> dict[int, float]:
    """corr(dx_t, dy_{t+k}) for k in [-max_lag, max_lag] on aligned series; k > 0 means x leads y."""
    dx, dy = logodds_changes(x), logodds_changes(y)
    out = {}
    for k in range(-max_lag, max_lag + 1):
        if k >= 0:
            a, b = dx[:len(dx) - k], dy[k:]
        else:
            a, b = dx[-k:], dy[:len(dy) + k]
        if len(a) > 3 and np.std(a) > 0 and np.std(b) > 0:
            out[k] = float(np.corrcoef(a, b)[0, 1])
    return out


def pca_factors(changes: np.ndarray, k: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Leading principal components of a T x N matrix of log-odds changes: (explained share, loadings N x k)."""
    X = changes - changes.mean(0)
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    share = s ** 2 / np.sum(s ** 2)
    return share[:k], Vt[:k].T
