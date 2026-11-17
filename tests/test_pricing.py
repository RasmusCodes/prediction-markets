import numpy as np

from pmkt.pricing import amm, bayes, calibration, correlation, events, implied, liquidity
from pmkt.pricing.logistic import LogisticRegression
from pmkt.pricing.normal import logit, sigmoid


def test_logistic_recovers_coefficients_and_cluster_se_is_wider():
    rng = np.random.default_rng(0)
    G, m = 400, 10
    g = np.repeat(np.arange(G), m)
    u = np.repeat(rng.normal(0, 1.0, G), m)           # shared shock inside each cluster
    x = rng.normal(size=G * m)
    y = rng.random(G * m) < sigmoid(0.5 + 1.5 * x + u)
    naive = LogisticRegression().fit(x, y)
    cl = LogisticRegression().fit(x, y, clusters=g)
    assert abs(naive.coef_[1] - 1.5 / np.sqrt(1 + 0.35 * 1.0)) < 0.3   # attenuated by the random effect
    assert cl.se_[0] > naive.se_[0] * 1.3              # intra-cluster correlation inflates the intercept SE


def test_calibration_regression_detects_compression():
    rng = np.random.default_rng(1)
    true = rng.uniform(0.02, 0.98, 20000)
    y = rng.random(20000) < true
    compressed = sigmoid(0.6 * logit(true))          # prices too close to 0.5 -> slope b ~ 1/0.6
    r = calibration.calibration_regression(compressed, y)
    assert 1.5 < r["b"] < 1.85 and r["z_b1"] > 5
    iso = calibration.IsotonicCalibrator().fit(compressed, y)
    q = iso.predict(np.linspace(0.05, 0.95, 50))
    assert np.all(np.diff(q) >= -1e-12)
    g = calibration.recalibration_gain(compressed[:10000], y[:10000], compressed[10000:], y[10000:], "platt")
    assert g["brier_cal"] < g["brier_raw"]
    d = calibration.murphy_decomposition(true, y, n_bins=20)
    assert abs(d["brier"] - (d["reliability"] - d["resolution"] + d["uncertainty"])) < 0.01


def test_bayesian_models():
    bb = bayes.BetaBinomial(1, 1).update(30, 70)
    assert abs(bb.mean - 31 / 102) < 1e-12
    gm = bayes.GlostenMilgrom(0.5, mu=0.4, q=0.9)
    bid, ask = gm.quote()
    assert bid < 0.5 < ask
    for _ in range(40):
        gm.update(True)
    assert gm.p > 0.99
    kf = bayes.LogitKalman(0.5, var0=1.0, q=1e-4)
    for t in range(200):
        kf.update_prob(0.8, spread=0.02, t=float(t))
    assert abs(kf.p - 0.8) < 0.01
    lo, hi = kf.interval()
    assert lo < kf.p < hi


def test_correlation_and_implied():
    pA, pB, rho = 0.6, 0.4, 0.5
    j = float(correlation.copula_joint(pA, pB, rho))
    lo, hi = correlation.frechet_bounds(pA, pB)
    assert lo < j < hi
    assert abs(correlation.implied_rho(pA, pB, j) - rho) < 1e-6
    assert abs(float(correlation.copula_joint(pA, pB, 0.0)) - pA * pB) < 1e-9
    rng = np.random.default_rng(3)
    z = rng.multivariate_normal([0, 0], [[1, 0.6], [0.6, 1]], 20000)
    est = correlation.tetrachoric(z[:, 0] > 0.3, z[:, 1] > -0.2)
    assert abs(est - 0.6) < 0.05
    for m in ("proportional", "power", "shin"):
        q = implied.normalize([0.55, 0.30, 0.25], m)
        assert abs(q.sum() - 1) < 1e-9
    assert implied.microprice(40, 100, 42, 10) > 41


def test_events_and_liquidity():
    after, w, emove = events.event_move_distribution(0.5, 0.5, 1.0)
    assert abs(np.sum(w * after) - 0.5) < 1e-6        # martingale: expected post-event price = current price
    assert 0.1 < emove < 0.4
    q = liquidity.LiquiditySensitiveQuoter(liquidity.QuoteParams(min_ticks=2))
    b0, a0 = q.quotes(0.5, 0, 1e-4, 600)
    b1, a1 = q.quotes(0.5, 200, 1e-4, 600)
    assert b0 < 50 < a0 and a0 - b0 >= 2
    assert b1 <= b0 and a1 <= a0                       # long inventory skews both quotes down
    tox = liquidity.Toxicity(bucket=10, n=5)
    for _ in range(10):
        tox.update(5, True)
    assert tox.value == 1.0


def test_amm_lmsr_and_lslmsr():
    m = amm.LMSR(b=100)
    assert np.allclose(m.prices(), [0.5, 0.5])
    c = m.trade(0, 50)
    assert 25 < c < 50 and m.prices()[0] > 0.5
    bids, asks = amm.ladder(m, levels=3)
    assert asks[0][0] > m.prices()[0] * 100 > bids[0][0]
    ls = amm.LSLMSR(alpha=0.05, q0=[100, 100])
    assert ls.prices().sum() > 1.0                    # LS-LMSR charges a spread
    ls.trade(0, 50)
    assert ls.prices()[0] > ls.prices()[1]
