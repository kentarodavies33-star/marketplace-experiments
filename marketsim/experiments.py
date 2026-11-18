"""Experiment designs and estimators.

* ground_truth   - what really happens if the fee cut ships to everyone (vs. nobody)
* naive_ab       - classic user-level 50/50 split inside every region and period
* switchback     - randomise whole (region, time-block) units to treatment or control
* CUPED          - variance reduction using each unit's pre-experiment outcome
* cluster-robust standard errors - correct inference when rows inside a block are correlated
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from .market import PERIODS_PER_DAY, MarketConfig, World, simulate

METRICS = ("conversion", "revenue")


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------

def _pooled(out: dict, arm: str, metric: str) -> float:
    sessions = out[f"sessions_{arm}"].sum()
    num = out[f"orders_{arm}"].sum() if metric == "conversion" else out[f"revenue_{arm}"].sum()
    return num / sessions


def ground_truth(world: World) -> dict:
    """Global treatment effect: everyone treated vs. everyone in control, same world,
    no sampling noise. This is the number a perfect experiment would recover."""
    T, R = world.sessions.shape
    all_c = simulate(world, np.zeros((T, R)), stochastic=False)
    all_t = simulate(world, np.ones((T, R)), stochastic=False)
    res = {}
    for m in METRICS:
        c, t = _pooled(all_c, "c", m), _pooled(all_t, "t", m)
        res[m] = dict(control=c, treatment=t, effect=t - c, lift=t / c - 1)
    res["eta_control"] = float(np.average(all_c["eta"], weights=world.sessions + 1e-9))
    res["eta_treatment"] = float(np.average(all_t["eta"], weights=world.sessions + 1e-9))
    return res


# ---------------------------------------------------------------------------
# Naive user-level A/B test
# ---------------------------------------------------------------------------

def naive_ab(world: World, rng: np.random.Generator) -> dict:
    T, R = world.sessions.shape
    out = simulate(world, np.full((T, R), 0.5), rng)
    res = {}
    n_t, n_c = out["sessions_t"].sum(), out["sessions_c"].sum()
    for m in METRICS:
        c, t = _pooled(out, "c", m), _pooled(out, "t", m)
        if m == "conversion":
            se = np.sqrt(t * (1 - t) / n_t + c * (1 - c) / n_c)
        else:  # revenue per session: variance from per-session revenue values
            rev_t = out["revenue_t"].sum() / max(out["orders_t"].sum(), 1)
            rev_c = out["revenue_c"].sum() / max(out["orders_c"].sum(), 1)
            p_t, p_c = t / rev_t, c / rev_c
            se = np.sqrt(rev_t**2 * p_t * (1 - p_t) / n_t + rev_c**2 * p_c * (1 - p_c) / n_c)
        res[m] = dict(control=c, treatment=t, effect=t - c, lift=t / c - 1, se=se)
    res["raw"] = out
    return res


# ---------------------------------------------------------------------------
# Switchback
# ---------------------------------------------------------------------------

def switchback_assignment(n_periods: int, n_regions: int, block_periods: int,
                          rng: np.random.Generator):
    """Each region's timeline is cut into blocks of `block_periods`; each block is a
    coin flip. Returns (treat 0/1 array, block id array) both shaped (periods, regions)."""
    block_of_period = np.arange(n_periods) // block_periods
    n_blocks = block_of_period.max() + 1
    coin = rng.integers(0, 2, (n_blocks, n_regions))
    treat = coin[block_of_period].astype(float)
    cluster = block_of_period[:, None] * n_regions + np.arange(n_regions)[None, :]
    position = (np.arange(n_periods) % block_periods)[:, None] * np.ones((1, n_regions), int)
    return treat, cluster, position


def wls_cluster(y, d, w, clusters):
    """Weighted least squares of y on [1, d]. Returns the treatment coefficient, its
    plain (iid) standard error and its cluster-robust standard error."""
    X = np.column_stack([np.ones_like(d), d])
    W = w / w.mean()
    XtWX = X.T @ (X * W[:, None])
    bread = np.linalg.inv(XtWX)
    beta = bread @ (X.T @ (W * y))
    resid = y - X @ beta
    n, k = X.shape

    # iid SE: treats every period-region row as independent
    sigma2 = np.sum(W * resid**2) / (n - k)
    se_iid = np.sqrt(sigma2 * bread[1, 1])

    # Cluster-robust (sandwich) SE, clustered on switchback units
    scores = X * (W * resid)[:, None]
    _, inv = np.unique(clusters, return_inverse=True)
    G = inv.max() + 1
    cluster_scores = np.zeros((G, k))
    np.add.at(cluster_scores, inv, scores)
    meat = cluster_scores.T @ cluster_scores
    V = bread @ meat @ bread * (G / (G - 1)) * ((n - 1) / (n - k))
    return beta[1], se_iid, np.sqrt(V[1, 1]), beta[0]


def cuped_adjust(y, x, w):
    """CUPED: y_adj = y - theta * (x - mean(x)), theta = cov(y, x) / var(x).
    theta is estimated on both arms pooled, so the treatment effect stays unbiased."""
    xm = np.average(x, weights=w)
    ym = np.average(y, weights=w)
    theta = np.average((x - xm) * (y - ym), weights=w) / np.average((x - xm) ** 2, weights=w)
    return y - theta * (x - xm), theta


def pre_period_covariates(pre_out: dict, n_periods: int, block_periods: int) -> dict:
    """CUPED covariate for every experiment period: the same region's outcome during the
    same switchback block-slot of the week (e.g. Tuesday 6-8pm), pooled over all
    pre-period weeks, when everyone was in control. Pooling over the whole block keeps the
    covariate from being swamped by 30-minute noise."""
    week = 7 * PERIODS_PER_DAY
    if week % block_periods:
        raise ValueError("block_periods must divide the number of periods in a week (336)")
    sessions = pre_out["sessions_c"] + pre_out["sessions_t"]
    totals = {"conversion": pre_out["orders_c"] + pre_out["orders_t"],
              "revenue": pre_out["revenue_c"] + pre_out["revenue_t"]}
    n_weeks = sessions.shape[0] // week
    R = sessions.shape[1]
    n_slots = week // block_periods

    def by_slot(a):
        a = a[: n_weeks * week].reshape(n_weeks, n_slots, block_periods, R)
        return a.sum(axis=(0, 2))  # (slots, regions)

    s = by_slot(sessions)
    slot = (np.arange(n_periods) % week) // block_periods
    cov = {}
    for m, num in totals.items():
        with np.errstate(invalid="ignore", divide="ignore"):
            cov[m] = np.nan_to_num(by_slot(num) / s)[slot]
    return cov


def switchback(world: World, rng: np.random.Generator, block_periods: int = 4,
               burn_in: int = 0, pre_covariates: dict | None = None) -> dict:
    """Run a switchback and analyse it with and without CUPED.

    block_periods: block length in 30-minute periods (4 = 2 hours).
    burn_in: periods at the start of each block to drop, so backlog left over from the
        previous block (carryover) doesn't contaminate the comparison.
    """
    T, R = world.sessions.shape
    treat, cluster, position = switchback_assignment(T, R, block_periods, rng)
    out = simulate(world, treat, rng)

    sessions = out["sessions_t"] + out["sessions_c"]
    keep = (sessions > 0) & (position >= burn_in)
    d = treat[keep]
    w = sessions[keep]
    cl = cluster[keep]
    res = {"treat": treat, "raw": out}
    for m in METRICS:
        num = (out["orders_t"] + out["orders_c"]) if m == "conversion" else (out["revenue_t"] + out["revenue_c"])
        y = (num / np.where(sessions > 0, sessions, 1))[keep]
        est, se_iid, se_cr, ctrl = wls_cluster(y, d, w, cl)
        r = dict(effect=est, se_iid=se_iid, se=se_cr, control=ctrl, lift=est / ctrl)
        if pre_covariates is not None:
            x = pre_covariates[m][keep]
            y_adj, theta = cuped_adjust(y, x, w)
            est_c, _, se_c, _ = wls_cluster(y_adj, d, w, cl)
            r.update(effect_cuped=est_c, se_cuped=se_c, lift_cuped=est_c / ctrl, theta=theta)
        res[m] = r
    return res


def ci(est, se, alpha=0.05):
    z = stats.norm.ppf(1 - alpha / 2)
    return est - z * se, est + z * se
