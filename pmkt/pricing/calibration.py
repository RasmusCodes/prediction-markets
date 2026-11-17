"""Calibration diagnostics and recalibration maps for probability forecasts.

Scores: Brier, log loss, Murphy (1973) decomposition, expected calibration error.
Tests: calibration regression  y ~ Bernoulli(sigmoid(a + b * logit(p)))  — b > 1 means prices are too
compressed toward 0.5 (longshots overpriced / favourites underpriced, the favourite-longshot pattern);
b < 1 means prices are too extreme. Cluster-robust SEs because markets inside one event are dependent.
Recalibration: Platt scaling, isotonic regression (pool-adjacent-violators), beta calibration (Kull et al. 2017).
"""
from __future__ import annotations

import numpy as np

from .logistic import LogisticRegression
from .normal import logit, sigmoid


def brier(p, y, w=None) -> float:
    p, y = np.asarray(p, float), np.asarray(y, float)
    return float(np.average((p - y) ** 2, weights=w))


def log_loss(p, y, w=None) -> float:
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, float)
    return float(-np.average(y * np.log(p) + (1 - y) * np.log(1 - p), weights=w))


def _bin_edges(p, n_bins, strategy):
    if strategy == "quantile":
        e = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
        e[0], e[-1] = 0.0, 1.0
        return e
    return np.linspace(0, 1, n_bins + 1)


def reliability_table(p, y, n_bins: int = 10, strategy: str = "uniform", w=None) -> list[dict]:
    p, y = np.asarray(p, float), np.asarray(y, float)
    w = np.ones(len(p)) if w is None else np.asarray(w, float)
    edges = _bin_edges(p, n_bins, strategy)
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        m = idx == b
        if not m.any():
            continue
        ww = w[m]
        n_eff = ww.sum() ** 2 / np.sum(ww ** 2)
        f = float(np.average(y[m], weights=ww))
        z = 1.96
        den = 1 + z * z / n_eff
        c = (f + z * z / (2 * n_eff)) / den
        h = z * np.sqrt(f * (1 - f) / n_eff + z * z / (4 * n_eff * n_eff)) / den
        rows.append({"lo": float(edges[b]), "hi": float(edges[b + 1]), "n": int(m.sum()), "weight": float(ww.sum()),
                     "p_mean": float(np.average(p[m], weights=ww)), "freq": f, "ci_lo": float(c - h),
                     "ci_hi": float(c + h)})
    return rows


def ece(p, y, n_bins: int = 10, strategy: str = "uniform", w=None) -> float:
    rows = reliability_table(p, y, n_bins, strategy, w)
    tot = sum(r["weight"] for r in rows)
    return float(sum(r["weight"] / tot * abs(r["p_mean"] - r["freq"]) for r in rows))


def murphy_decomposition(p, y, n_bins: int = 10, w=None) -> dict:
    """Brier ~= reliability - resolution + uncertainty (exact when forecasts take only the bin values)."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    w = np.ones(len(p)) if w is None else np.asarray(w, float)
    ybar = np.average(y, weights=w)
    rows = reliability_table(p, y, n_bins, "uniform", w)
    tot = sum(r["weight"] for r in rows)
    rel = sum(r["weight"] * (r["p_mean"] - r["freq"]) ** 2 for r in rows) / tot
    res = sum(r["weight"] * (r["freq"] - ybar) ** 2 for r in rows) / tot
    unc = ybar * (1 - ybar)
    return {"brier": brier(p, y, w), "reliability": float(rel), "resolution": float(res), "uncertainty": float(unc)}


def calibration_regression(p, y, clusters=None, w=None) -> dict:
    """Fit y ~ sigmoid(a + b logit p); returns a, b with (cluster-robust) SEs and tests of a=0, b=1."""
    x = logit(np.clip(np.asarray(p, float), 0.005, 0.995))
    m = LogisticRegression().fit(x, y, w=w, names=["logit_p"], clusters=clusters)
    a, b = m.coef_
    sa, sb = m.se_
    return {"a": float(a), "b": float(b), "se_a": float(sa), "se_b": float(sb),
            "z_a0": float(a / sa), "z_b1": float((b - 1) / sb), "n": int(len(x)),
            "n_clusters": int(len(np.unique(clusters))) if clusters is not None else None}


class PlattScaler:
    def fit(self, p, y, w=None):
        self.m = LogisticRegression().fit(logit(np.clip(p, 0.005, 0.995)), y, w=w)
        return self

    def predict(self, p):
        return self.m.predict_proba(logit(np.clip(p, 0.005, 0.995)))


class BetaCalibrator:
    """Kull, Silva Filho & Flach (2017): logistic regression on [ln p, -ln(1-p)]."""

    def fit(self, p, y, w=None):
        p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
        self.m = LogisticRegression(l2=1e-6).fit(np.column_stack([np.log(p), -np.log1p(-p)]), y, w=w)
        return self

    def predict(self, p):
        p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
        return self.m.predict_proba(np.column_stack([np.log(p), -np.log1p(-p)]))


class IsotonicCalibrator:
    """Monotone non-decreasing map fitted by pool-adjacent-violators; linear interpolation between knots."""

    def fit(self, p, y, w=None):
        p, y = np.asarray(p, float), np.asarray(y, float)
        w = np.ones(len(p)) if w is None else np.asarray(w, float)
        o = np.argsort(p, kind="mergesort")
        xs, ys, ws = p[o], y[o], w[o]
        ux, inv = np.unique(xs, return_inverse=True)       # pool tied forecasts first
        sw = np.bincount(inv, ws)
        sy = np.bincount(inv, ws * ys) / sw
        vals, wts, cnt = [], [], []
        for i in range(len(ux)):                            # pool adjacent violators
            vals.append(sy[i]); wts.append(sw[i]); cnt.append(1)
            while len(vals) > 1 and vals[-2] > vals[-1]:
                v2, w2, c2 = vals.pop(), wts.pop(), cnt.pop()
                v1, w1, c1 = vals.pop(), wts.pop(), cnt.pop()
                vals.append((v1 * w1 + v2 * w2) / (w1 + w2)); wts.append(w1 + w2); cnt.append(c1 + c2)
        fitted = np.repeat(np.asarray(vals), cnt)
        self.x_, self.y_ = ux, fitted
        return self

    def predict(self, p):
        return np.interp(np.asarray(p, float), self.x_, self.y_)


def cluster_bootstrap(stat_fn, data_idx_by_cluster: list[np.ndarray], n_boot: int = 500, seed: int = 0,
                      level: float = 0.95) -> tuple[float, float, np.ndarray]:
    """Percentile CI of stat_fn(indices) resampling whole clusters with replacement."""
    rng = np.random.default_rng(seed)
    G = len(data_idx_by_cluster)
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, G, G)
        idx = np.concatenate([data_idx_by_cluster[g] for g in pick])
        vals.append(stat_fn(idx))
    vals = np.asarray(vals)
    a = (1 - level) / 2
    return float(np.nanquantile(vals, a)), float(np.nanquantile(vals, 1 - a)), vals


def recalibration_gain(p_train, y_train, p_test, y_test, method: str = "platt") -> dict:
    """Out-of-sample Brier / log loss before and after fitting a recalibration map on the training split."""
    cal = {"platt": PlattScaler, "isotonic": IsotonicCalibrator, "beta": BetaCalibrator}[method]().fit(p_train, y_train)
    q = np.clip(cal.predict(p_test), 1e-4, 1 - 1e-4)
    return {"method": method, "brier_raw": brier(p_test, y_test), "brier_cal": brier(q, y_test),
            "logloss_raw": log_loss(p_test, y_test), "logloss_cal": log_loss(q, y_test)}
