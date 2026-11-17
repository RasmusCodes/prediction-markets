import numpy as np
from scipy.stats import multivariate_normal

from pmkt.pricing.normal import bvn_cdf
from pmkt.world import BinaryFamily, LadderFamily, PairFamily, World, make_grid


def test_bvn_cdf_matches_scipy():
    rng = np.random.default_rng(0)
    for _ in range(40):
        h, k = rng.normal(0, 1.5, 2)
        r = rng.uniform(-0.95, 0.95)
        ref = multivariate_normal(mean=[0, 0], cov=[[1, r], [r, 1]]).cdf([h, k])
        assert abs(bvn_cdf(h, k, r) - ref) < 2e-4
    assert abs(bvn_cdf(0.0, 0.0, 0.5) - (0.25 + np.arcsin(0.5) / (2 * np.pi))) < 1e-9


def test_truth_is_a_calibrated_martingale():
    grid = make_grid(600, 2.0)
    p0, outs, mids = 0.3, [], []
    for s in range(1200):
        f = BinaryFamily("X", grid, p0, sigma=0.05, lam=0.004, jump_sd=0.6, sched=[(300, 0.3)],
                         rng=np.random.default_rng(s))
        mids.append(f.prob_path()["X"][150])
        outs.append(f.outcomes()["X"])
    assert abs(np.mean(mids) - p0) < 0.03          # E[p_t] = p_0
    assert abs(np.mean(outs) - p0) < 0.035         # P(outcome) = p_0


def test_ladder_buckets_coherent_and_pair_respects_frechet():
    grid = make_grid(900, 1.0)
    lad = LadderFamily("L", grid, strikes=[-0.5, 0.0, 0.5], sigma=0.03, lam=0.002, jump_sd=0.4,
                       rng=np.random.default_rng(1))
    w = World([lad], grid)
    P = w.paths
    buckets = [m for m in w.defs if w.defs[m].kind == "bucket"]
    s = sum(P[m] for m in buckets)
    assert np.allclose(s, 1.0, atol=1e-9)
    assert np.all(P["L>-0.5"] >= P["L>0"] - 1e-12) and np.all(P["L>0"] >= P["L>0.5"] - 1e-12)
    outs = lad.outcomes()
    assert sum(outs[m] for m in buckets) == 1
    ids, M, probs = lad.states()
    assert np.isclose(probs.sum(), 1.0) and M.shape == (4, len(ids))

    pair = PairFamily("P", grid, 0.6, 0.5, rho=0.5, sigma=0.03, sched=[(450, 0.4)], rng=np.random.default_rng(2))
    pp = pair.prob_path()
    a, b, ab = pp["P.A"], pp["P.B"], pp["P.AB"]
    assert np.all(ab <= np.minimum(a, b) + 1e-9)
    assert np.all(ab >= np.maximum(0, a + b - 1) - 1e-9)
    assert abs(a[0] - 0.6) < 1e-6 and abs(b[0] - 0.5) < 1e-6
    o = pair.outcomes()
    assert o["P.AB"] == (o["P.A"] and o["P.B"])
