"""Ground-truth world: latent information processes whose conditional probabilities are exact martingales.

Each event family is driven by latent Gaussian state variables observed on a fixed time grid. An event's
outcome is a threshold condition on the terminal latent value, so the "true" probability at time t is the
conditional probability given the information revealed so far:

  X(t) = X(0) + sigma W(t) + sum of unscheduled jumps (Poisson rate lam, N(0, s^2)) + scheduled news jumps
         (known time tau_k, N(0, v_k))
  p(t) = P(X(T) > c | X(t)) = sum_n Pois(n; lam (T-t)) * Phi((X(t) - c) / sqrt(sigma^2 (T-t) + V_sched(t) + n s^2))

Families
  BinaryFamily  one market  1{X_T > 0}
  LadderFamily  strike ladder 1{X_T > K_j}  plus mutually-exclusive buckets 1{K_j <= X_T < K_j+1}
  PairFamily    two correlated events A, B (common factor, loadings a_A, a_B) plus the combo "A and B";
                diffusion + scheduled common news only, so the joint law is exactly bivariate normal.

Paths are pre-simulated on the grid from the family's own RNG stream, so a given seed produces the same
world regardless of what the traders do (common random numbers for strategy comparisons).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..pricing.normal import Phi, Phi_inv, bvn_upper, poisson_weights


@dataclass
class NewsEvent:
    t: float
    family: str
    kind: str          # "scheduled" | "jump"
    size: float        # latent jump that happened (agents only learn its effect through prices / truth)
    var: float         # variance revealed (scheduled) or jump variance (unscheduled)


@dataclass
class MarketDef:
    id: str
    family: str
    kind: str                 # binary | ladder | bucket | combo
    title: str
    event: str | None = None  # mutually-exclusive group
    meta: dict = field(default_factory=dict)


class Family:
    id: str
    T: float

    def markets(self) -> list[MarketDef]:
        raise NotImplementedError

    def prob_path(self) -> dict[str, np.ndarray]:
        raise NotImplementedError

    def outcomes(self) -> dict[str, bool]:
        raise NotImplementedError

    def states(self) -> tuple[list[str], np.ndarray, np.ndarray]:
        """(market ids, payoff matrix [n_states x n_markets] of 0/1, true terminal state probabilities at t=0)."""
        raise NotImplementedError


def _simulate_univariate(rng, grid, sigma, lam, jump_sd, sched):
    """Increments of a latent process on the grid; returns (path increments cumulative, news list)."""
    dt = grid[1] - grid[0]
    n = len(grid)
    inc = rng.normal(0.0, sigma * np.sqrt(dt), n)
    inc[0] = 0.0
    news = []
    T = grid[-1]
    if lam > 0:
        k = rng.poisson(lam * T)
        for tj in np.sort(rng.uniform(0, T, k)):
            i = min(int(np.ceil(tj / dt)), n - 1)
            j = rng.normal(0.0, jump_sd)
            inc[i] += j
            news.append(("jump", grid[i], j, jump_sd ** 2))
    for tau, v in sched:
        i = min(int(np.ceil(tau / dt)), n - 1)
        j = rng.normal(0.0, np.sqrt(v))
        inc[i] += j
        news.append(("scheduled", grid[i], j, v))
    return np.cumsum(inc), news


def _remaining_sched_var(grid, sched):
    """V_sched(t_i): variance of scheduled news strictly after grid time t_i."""
    out = np.zeros(len(grid))
    dt = grid[1] - grid[0]
    for tau, v in sched:
        i = min(int(np.ceil(tau / dt)), len(grid) - 1)
        out[:i] += v  # news landing at grid index i is revealed at t_i, so it is still pending before i
    return out


def _prob_above(x, c, tau, sigma, lam, jump_sd, vs):
    """Vectorised P(X_T > c | X_t = x) with Poisson-mixture jumps; tau, vs arrays aligned with x."""
    x, tau, vs = np.asarray(x, float), np.asarray(tau, float), np.asarray(vs, float)
    out = np.zeros_like(x)
    lam_max = lam * float(np.max(tau)) if lam > 0 else 0.0
    nmax = int(lam_max + 10 * np.sqrt(lam_max + 1) + 10)
    if lam > 0:
        ns = np.arange(nmax + 1)
        from scipy import special
        logw = ns[None, :] * np.log(np.maximum(lam * tau[:, None], 1e-300)) - lam * tau[:, None] - special.gammaln(ns + 1)[None, :]
        w = np.exp(logw)
        w[tau <= 0, :] = 0.0
        w[tau <= 0, 0] = 1.0
        var = sigma ** 2 * tau[:, None] + vs[:, None] + ns[None, :] * jump_sd ** 2
    else:
        w = np.ones((len(x), 1))
        var = (sigma ** 2 * tau + vs)[:, None]
    z = (x[:, None] - c) / np.sqrt(np.maximum(var, 1e-300))
    pz = np.where(var > 1e-14, Phi(z), (x[:, None] > c).astype(float))
    out = np.sum(w * pz, axis=1) / np.sum(w, axis=1)
    return np.clip(out, 0.0, 1.0)


class BinaryFamily(Family):
    def __init__(self, fid: str, grid: np.ndarray, p0: float, sigma: float = 0.6, lam: float = 0.0,
                 jump_sd: float = 0.0, sched=(), rng=None, title: str = ""):
        self.id, self.grid, self.T = fid, grid, grid[-1]
        self.sigma, self.lam, self.jump_sd, self.sched = sigma, lam, jump_sd, list(sched)
        self.title = title or f"Will event {fid} happen?"
        rng = rng or np.random.default_rng()
        self.vs = _remaining_sched_var(grid, self.sched)
        x0 = self._solve_x0(p0)
        path, news = _simulate_univariate(rng, grid, sigma, lam, jump_sd, self.sched)
        self.x = x0 + path
        self.news = [NewsEvent(t, fid, k, j, v) for k, t, j, v in news]
        self._p = _prob_above(self.x, 0.0, self.T - grid, sigma, lam, jump_sd, self.vs)

    def _solve_x0(self, p0: float) -> float:
        lo, hi = -10.0, 10.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            p = _prob_above(np.array([mid]), 0.0, np.array([self.T]), self.sigma, self.lam, self.jump_sd,
                            np.array([self.vs[0]]))[0]
            lo, hi = (mid, hi) if p < p0 else (lo, mid)
        return 0.5 * (lo + hi)

    def markets(self):
        return [MarketDef(self.id, self.id, "binary", self.title)]

    def prob_path(self):
        return {self.id: self._p}

    def outcomes(self):
        return {self.id: bool(self.x[-1] > 0.0)}

    def states(self):
        p = float(self._p[0])
        return [self.id], np.array([[0.0], [1.0]]), np.array([1 - p, p])


class LadderFamily(Family):
    """Strike ladder on one latent variable (e.g. 'index above K at expiry') plus the bucket partition."""

    def __init__(self, fid: str, grid: np.ndarray, strikes, x0: float = 0.0, sigma: float = 0.6,
                 lam: float = 0.0, jump_sd: float = 0.0, sched=(), rng=None, title: str = "",
                 buckets: bool = True, ladder: bool = True, unit: str = ""):
        self.id, self.grid, self.T = fid, grid, grid[-1]
        self.strikes = sorted(float(k) for k in strikes)
        self.sigma, self.lam, self.jump_sd, self.sched = sigma, lam, jump_sd, list(sched)
        self.title, self.unit = title or f"Level of {fid} at expiry", unit
        self.with_buckets, self.with_ladder = buckets, ladder
        rng = rng or np.random.default_rng()
        self.vs = _remaining_sched_var(grid, self.sched)
        path, news = _simulate_univariate(rng, grid, sigma, lam, jump_sd, self.sched)
        self.x = x0 + path
        self.news = [NewsEvent(t, fid, k, j, v) for k, t, j, v in news]
        tau = self.T - grid
        self._above = {K: _prob_above(self.x, K, tau, sigma, lam, jump_sd, self.vs) for K in self.strikes}

    def _ladder_id(self, K):
        return f"{self.id}>{K:g}"

    def _bucket_ids(self):
        """Buckets partition the line as (-inf,K1], (K1,K2], ..., (Kk, inf); ladder '>K' = union of buckets above K."""
        ks = self.strikes
        ids = [f"{self.id}(<={ks[0]:g})"]
        ids += [f"{self.id}({a:g},{b:g}]" for a, b in zip(ks[:-1], ks[1:])]
        ids += [f"{self.id}(>{ks[-1]:g})"]
        return ids

    def markets(self):
        out = []
        if self.with_ladder:
            for K in self.strikes:
                out.append(MarketDef(self._ladder_id(K), self.id, "ladder", f"{self.title} above {K:g}{self.unit}?",
                                     meta={"strike": K}))
        if self.with_buckets:
            ev = f"{self.id}:buckets"
            ks = [-np.inf] + self.strikes + [np.inf]
            for mid, lo, hi in zip(self._bucket_ids(), ks[:-1], ks[1:]):
                out.append(MarketDef(mid, self.id, "bucket", f"{self.title} in ({lo:g}, {hi:g}]{self.unit}?",
                                     event=ev, meta={"lo": lo, "hi": hi}))
        return out

    def prob_path(self):
        out = {}
        if self.with_ladder:
            for K in self.strikes:
                out[self._ladder_id(K)] = self._above[K]
        if self.with_buckets:
            ab = [np.ones_like(self.x)] + [self._above[K] for K in self.strikes] + [np.zeros_like(self.x)]
            for mid, a, b in zip(self._bucket_ids(), ab[:-1], ab[1:]):
                out[mid] = np.clip(a - b, 0.0, 1.0)
        return out

    def outcomes(self):
        xT = self.x[-1]
        out = {}
        if self.with_ladder:
            for K in self.strikes:
                out[self._ladder_id(K)] = bool(xT > K)
        if self.with_buckets:
            ks = [-np.inf] + self.strikes + [np.inf]
            for mid, lo, hi in zip(self._bucket_ids(), ks[:-1], ks[1:]):
                out[mid] = bool(lo < xT <= hi) if np.isfinite(hi) else bool(xT > lo)
        return out

    def states(self):
        """States = which bucket X_T lands in (k+1 states)."""
        ids = [m.id for m in self.markets()]
        ks = [-np.inf] + self.strikes + [np.inf]
        nS = len(ks) - 1
        M = np.zeros((nS, len(ids)))
        for s in range(nS):
            lo, hi = ks[s], ks[s + 1]
            for j, m in enumerate(self.markets()):
                if m.kind == "ladder":
                    M[s, j] = 1.0 if lo >= m.meta["strike"] else 0.0
                else:
                    M[s, j] = 1.0 if (m.meta["lo"] == lo) else 0.0
        a = [1.0] + [float(self._above[K][0]) for K in self.strikes] + [0.0]
        probs = np.clip(np.array(a[:-1]) - np.array(a[1:]), 0, 1)
        return ids, M, probs / probs.sum()


class PairFamily(Family):
    """Two correlated binary events and their conjunction (a 'combo' / parlay market)."""

    def __init__(self, fid: str, grid: np.ndarray, pA: float, pB: float, rho: float = 0.6,
                 sigma: float = 0.6, sched=(), rng=None, titles=("A", "B")):
        self.id, self.grid, self.T = fid, grid, grid[-1]
        self.rho, self.sigma, self.sched = rho, sigma, list(sched)
        self.titles = titles
        a = np.sqrt(max(rho, 0.0))           # equal loadings on the common factor: corr = a^2 for diffusion
        self.aA = self.aB = a
        rng = rng or np.random.default_rng()
        dt = grid[1] - grid[0]
        n = len(grid)
        fF = rng.normal(0, sigma * np.sqrt(dt), n); fF[0] = 0
        eA = rng.normal(0, sigma * np.sqrt(dt), n); eA[0] = 0
        eB = rng.normal(0, sigma * np.sqrt(dt), n); eB[0] = 0
        self.news = []
        for tau, v in self.sched:  # scheduled common news hits the factor
            i = min(int(np.ceil(tau / dt)), n - 1)
            j = rng.normal(0, np.sqrt(v))
            fF[i] += j
            self.news.append(NewsEvent(grid[i], fid, "scheduled", j, v))
        F, EA, EB = np.cumsum(fF), np.cumsum(eA), np.cumsum(eB)
        tau = self.T - grid
        vF = sigma ** 2 * tau + _remaining_sched_var(grid, self.sched)
        vE = sigma ** 2 * tau
        self.VA = self.aA ** 2 * vF + (1 - self.aA ** 2) * vE
        self.VB = self.aB ** 2 * vF + (1 - self.aB ** 2) * vE
        self.C = self.aA * self.aB * vF
        # thresholds so that P(A), P(B) at t=0 equal pA, pB: X(0)=0, outcome 1{X_T > c}
        self.cA = -Phi_inv(pA) * np.sqrt(self.VA[0])
        self.cB = -Phi_inv(pB) * np.sqrt(self.VB[0])
        self.xA = self.aA * F + np.sqrt(1 - self.aA ** 2) * EA
        self.xB = self.aB * F + np.sqrt(1 - self.aB ** 2) * EB
        sA, sB = np.sqrt(np.maximum(self.VA, 1e-300)), np.sqrt(np.maximum(self.VB, 1e-300))
        hA, hB = (self.cA - self.xA) / sA, (self.cB - self.xB) / sB
        live = self.VA > 1e-14
        self._pA = np.where(live, Phi(-hA), (self.xA > self.cA).astype(float))
        self._pB = np.where(live, Phi(-hB), (self.xB > self.cB).astype(float))
        r = np.where(live, self.C / np.maximum(sA * sB, 1e-300), 0.0)
        self._pAB = np.where(live, bvn_upper(hA, hB, r), ((self.xA > self.cA) & (self.xB > self.cB)).astype(float))

    def markets(self):
        a, b = self.titles
        return [MarketDef(f"{self.id}.A", self.id, "binary", f"{a}?"),
                MarketDef(f"{self.id}.B", self.id, "binary", f"{b}?"),
                MarketDef(f"{self.id}.AB", self.id, "combo", f"{a} AND {b}?")]

    def prob_path(self):
        return {f"{self.id}.A": self._pA, f"{self.id}.B": self._pB, f"{self.id}.AB": self._pAB}

    def outcomes(self):
        A, B = bool(self.xA[-1] > self.cA), bool(self.xB[-1] > self.cB)
        return {f"{self.id}.A": A, f"{self.id}.B": B, f"{self.id}.AB": A and B}

    def states(self):
        ids = [m.id for m in self.markets()]
        M = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 1]], float)
        pA, pB, pAB = self._pA[0], self._pB[0], self._pAB[0]
        probs = np.array([1 - pA - pB + pAB, pA - pAB, pB - pAB, pAB])
        return ids, M, np.clip(probs, 0, 1) / np.clip(probs, 0, 1).sum()

    def implied_rho(self) -> float:
        return float(self.C[0] / np.sqrt(self.VA[0] * self.VB[0]))


class World:
    """A set of families on a common grid. Provides O(1) truth lookups by market id and time."""

    def __init__(self, families: list[Family], grid: np.ndarray):
        self.families = {f.id: f for f in families}
        self.grid = grid
        self.dt = float(grid[1] - grid[0])
        self.T = float(grid[-1])
        self.defs: dict[str, MarketDef] = {}
        self.paths: dict[str, np.ndarray] = {}
        self.fam_of: dict[str, str] = {}
        for f in families:
            for m in f.markets():
                self.defs[m.id] = m
                self.fam_of[m.id] = f.id
            self.paths.update(f.prob_path())
        self.news = sorted([n for f in families for n in getattr(f, "news", [])], key=lambda n: n.t)
        self._outcomes = {}
        for f in families:
            self._outcomes.update(f.outcomes())

    def idx(self, t: float) -> int:
        return min(max(int(t / self.dt), 0), len(self.grid) - 1)

    def prob(self, market: str, t: float) -> float:
        return float(self.paths[market][self.idx(t)])

    def outcome(self, market: str) -> bool:
        return self._outcomes[market]

    def scheduled_news(self) -> list[NewsEvent]:
        return [n for n in self.news if n.kind == "scheduled"]


def make_grid(T: float, dt: float = 1.0) -> np.ndarray:
    n = int(round(T / dt)) + 1
    return np.linspace(0.0, T, n)
