"""Logistic regression by Newton-Raphson (IRLS) with optional ridge penalty, observation weights and
cluster-robust (sandwich) standard errors. No scikit-learn dependency."""
from __future__ import annotations

import numpy as np

from .normal import sigmoid


class LogisticRegression:
    def __init__(self, l2: float = 0.0, fit_intercept: bool = True, max_iter: int = 100, tol: float = 1e-10):
        self.l2, self.fit_intercept, self.max_iter, self.tol = l2, fit_intercept, max_iter, tol
        self.coef_: np.ndarray | None = None
        self.cov_: np.ndarray | None = None
        self.names: list[str] | None = None

    def _design(self, X):
        X = np.asarray(X, float)
        if X.ndim == 1:
            X = X[:, None]
        if self.fit_intercept:
            X = np.column_stack([np.ones(len(X)), X])
        return X

    def fit(self, X, y, w=None, names=None, clusters=None):
        A = self._design(X)
        y = np.asarray(y, float)
        w = np.ones(len(y)) if w is None else np.asarray(w, float)
        n, k = A.shape
        pen = np.full(k, self.l2)
        if self.fit_intercept:
            pen[0] = 0.0
        beta = np.zeros(k)
        for _ in range(self.max_iter):
            eta = A @ beta
            p = sigmoid(eta)
            W = w * p * (1 - p)
            g = A.T @ (w * (y - p)) - pen * beta
            H = (A * W[:, None]).T @ A + np.diag(pen)
            step = np.linalg.solve(H + 1e-12 * np.eye(k), g)
            beta += step
            if np.max(np.abs(step)) < self.tol:
                break
        self.coef_ = beta
        p = sigmoid(A @ beta)
        W = w * p * (1 - p)
        H = (A * W[:, None]).T @ A + np.diag(pen)
        Hinv = np.linalg.pinv(H)
        if clusters is not None:
            # cluster-robust sandwich: H^-1 (sum_g s_g s_g') H^-1 with small-sample factor G/(G-1)
            s = A * (w * (y - p))[:, None]
            cl = np.asarray(clusters)
            _, inv = np.unique(cl, return_inverse=True)
            G = inv.max() + 1
            Sg = np.zeros((G, k))
            np.add.at(Sg, inv, s)
            meat = Sg.T @ Sg * (G / max(G - 1, 1))
            self.cov_ = Hinv @ meat @ Hinv
        else:
            self.cov_ = Hinv
        self.names = (["const"] if self.fit_intercept else []) + (list(names) if names else
                                                                    [f"x{i}" for i in range(k - self.fit_intercept)])
        self.loglik_ = float(np.sum(w * (y * np.log(np.clip(p, 1e-15, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-15, 1)))))
        self.n_ = n
        return self

    @property
    def se_(self) -> np.ndarray:
        return np.sqrt(np.clip(np.diag(self.cov_), 0, None))

    def predict_proba(self, X) -> np.ndarray:
        return sigmoid(self._design(X) @ self.coef_)

    def summary(self) -> list[dict]:
        z = self.coef_ / np.where(self.se_ > 0, self.se_, np.nan)
        from scipy.stats import norm
        pv = 2 * norm.sf(np.abs(z))
        return [{"name": n, "coef": float(c), "se": float(s), "z": float(zz), "p": float(pp)}
                for n, c, s, zz, pp in zip(self.names, self.coef_, self.se_, z, pv)]

    def aic(self) -> float:
        return 2 * len(self.coef_) - 2 * self.loglik_
