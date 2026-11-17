"""Small numerical toolkit shared by the world model and the pricing engine."""
from __future__ import annotations

import math

import numpy as np
from scipy import special

EPS = 1e-9


def logit(p):
    if isinstance(p, float):  # scalar fast path (hot loop in the simulator)
        p = 1e-6 if p < 1e-6 else (1 - 1e-6 if p > 1 - 1e-6 else p)
        return math.log(p / (1.0 - p))
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p) - np.log1p(-p)


def slogit(p: float) -> float:
    p = 1e-6 if p < 1e-6 else (1 - 1e-6 if p > 1 - 1e-6 else p)
    return math.log(p / (1.0 - p))


def ssigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def sigmoid(x):
    return special.expit(x)


def Phi(x):
    return special.ndtr(x)


def Phi_inv(p):
    return special.ndtri(np.clip(p, 1e-12, 1 - 1e-12))


def bvn_cdf(h, k, rho):
    """P(X <= h, Y <= k) for a standard bivariate normal with correlation rho (vectorised).

    Uses Owen's T representation (Owen 1956): exact up to scipy's owens_t accuracy, much faster and less noisy
    than the quasi-Monte-Carlo in scipy.stats.multivariate_normal for the 2-D case.
    """
    h, k, rho = np.broadcast_arrays(np.asarray(h, float), np.asarray(k, float), np.asarray(rho, float))
    out = np.empty(h.shape, float)
    rho_c = np.clip(rho, -0.999999, 0.999999)
    hh = np.where(np.abs(h) < 1e-12, 1e-12, h)
    kk = np.where(np.abs(k) < 1e-12, 1e-12, k)
    s = np.sqrt(1.0 - rho_c ** 2)
    a_h = (kk - rho_c * hh) / (hh * s)
    a_k = (hh - rho_c * kk) / (kk * s)
    beta = np.where((hh * kk < 0) | ((hh * kk == 0) & (hh + kk < 0)), 0.5, 0.0)
    out[...] = 0.5 * Phi(hh) + 0.5 * Phi(kk) - special.owens_t(hh, a_h) - special.owens_t(kk, a_k) - beta
    # degenerate correlations
    hi = rho >= 0.999999
    lo = rho <= -0.999999
    if np.any(hi):
        out[hi] = Phi(np.minimum(h, k))[hi]
    if np.any(lo):
        out[lo] = np.maximum(0.0, Phi(h) + Phi(k) - 1.0)[lo]
    return np.clip(out, 0.0, 1.0)


def bvn_upper(h, k, rho):
    """P(X > h, Y > k) for a standard bivariate normal."""
    return bvn_cdf(-np.asarray(h, float), -np.asarray(k, float), rho)


def poisson_weights(lam_tau: float, nmax: int = 60, tol: float = 1e-12) -> np.ndarray:
    """Poisson(lam_tau) pmf truncated where the tail mass is below tol (renormalised)."""
    if lam_tau <= 0:
        return np.array([1.0])
    n = np.arange(nmax + 1)
    logw = n * np.log(lam_tau) - lam_tau - special.gammaln(n + 1)
    w = np.exp(logw)
    c = np.cumsum(w)
    cut = int(np.searchsorted(c, 1 - tol)) + 1
    w = w[:max(cut, 1)]
    return w / w.sum()
