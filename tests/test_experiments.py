"""Sanity checks. Run with:  pytest -q"""

import numpy as np

from marketsim.experiments import (
    cuped_adjust, ground_truth, naive_ab, pre_period_covariates, switchback, wls_cluster,
)
from marketsim.market import MarketConfig, make_regions, make_world, simulate


def small_market(seed=0, **kw):
    cfg = MarketConfig(n_regions=8, days=7, **kw)
    rng = np.random.default_rng(seed)
    regions = make_regions(cfg, rng)
    return cfg, regions, make_world(cfg, regions, rng), rng


def test_no_interference_means_naive_ab_is_unbiased():
    """With effectively unlimited Dashers there is no shared resource to fight over,
    so the naive A/B test should match the truth. This shows the bias comes from
    interference, not from a bug in the estimator."""
    cfg, regions, world, rng = small_market(seed=1, congestion_minutes=0.0)
    truth = ground_truth(world)["conversion"]["effect"]
    est = naive_ab(world, rng)["conversion"]
    assert abs(est["effect"] - truth) < 3 * est["se"]


def test_with_interference_naive_ab_overstates_fee_cut():
    cfg, regions, world, rng = small_market(seed=2)
    truth = ground_truth(world)["conversion"]["effect"]
    est = naive_ab(world, rng)["conversion"]["effect"]
    assert est > 2 * truth


def test_fee_cut_raises_eta_in_full_rollout():
    cfg, regions, world, rng = small_market(seed=3)
    t = ground_truth(world)
    assert t["eta_treatment"] > t["eta_control"]


def test_switchback_close_to_truth():
    cfg, regions, world, rng = small_market(seed=4)
    pre = make_world(cfg, regions, rng)
    T, R = world.sessions.shape
    cov = pre_period_covariates(simulate(pre, np.zeros((T, R)), rng), T, 4)
    truth = ground_truth(world)["conversion"]["effect"]
    sb = switchback(world, rng, 4, 0, cov)["conversion"]
    assert abs(sb["effect_cuped"] - truth) < 4 * sb["se_cuped"]


def test_cuped_keeps_mean_and_reduces_variance():
    rng = np.random.default_rng(0)
    x = rng.normal(size=5000)
    y = 2 * x + rng.normal(size=5000)
    w = np.ones_like(x)
    y_adj, theta = cuped_adjust(y, x, w)
    assert abs(theta - 2) < 0.1
    assert abs(y_adj.mean() - y.mean()) < 1e-9
    assert y_adj.var() < 0.3 * y.var()


def test_cluster_robust_se_larger_when_clustered():
    rng = np.random.default_rng(0)
    G, m = 200, 20
    cl = np.repeat(np.arange(G), m)
    d = np.repeat(rng.integers(0, 2, G), m).astype(float)
    y = np.repeat(rng.normal(size=G), m) + rng.normal(size=G * m) * 0.3
    _, se_iid, se_cr, _ = wls_cluster(y, d, np.ones_like(y), cl)
    assert se_cr > 2 * se_iid
