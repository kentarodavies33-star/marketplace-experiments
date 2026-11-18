"""A small three-sided delivery marketplace: customers, Dashers and restaurants.

How the market works, one region and one 30-minute period at a time:

* Customers open the app (sessions). Each session converts to an order with a
  probability that falls with the delivery fee and with the quoted ETA, and rises
  with restaurant selection in the region.
* Dashers supply delivery capacity. When orders get close to capacity, ETAs rise
  sharply, and orders that can't be served carry over to the next period as backlog.
* Restaurants set baseline selection (more restaurants, more conversion) and prep
  time (part of the ETA).

The important part is that conversion depends on ETA, ETA depends on total orders,
and total orders depend on conversion. So every period is solved as an equilibrium.
Because treatment and control customers in the same region share one pool of
Dashers, a treatment that changes demand also changes ETAs for the control group.
That is the interference that breaks a naive A/B test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PERIODS_PER_DAY = 48  # 30-minute periods


@dataclass
class MarketConfig:
    n_regions: int = 20
    days: int = 14
    base_sessions: float = 70.0      # mean app sessions per region per period at profile = 1
    # Pricing
    control_fee: float = 3.99
    treatment_fee: float = 2.99      # the change being tested: a $1 delivery-fee cut
    basket_size: float = 30.0
    take_rate: float = 0.15          # commission DoorDash keeps from the restaurant
    # Customer behaviour (logit conversion model)
    base_conversion: float = 0.22    # conversion at control fee and a 35-minute ETA
    fee_sensitivity: float = 0.30    # logit drop per $1 of fee
    eta_sensitivity: float = 0.45    # logit drop per 10 minutes of ETA
    selection_effect: float = 0.25   # logit gain per log-unit of restaurant count
    region_alpha_sd: float = 0.30    # persistent region differences in conversion
    # Supply side
    deliveries_per_dasher: float = 1.0   # deliveries one Dasher completes in a period
    supply_tightness: tuple = (0.80, 1.05)  # Dasher supply relative to expected demand
    peak_flatness: float = 0.45      # 0 = supply tracks demand peaks exactly, 1 = flat
    travel_minutes: float = 14.0
    congestion_minutes: float = 32.0  # extra ETA when utilisation hits 100%
    congestion_power: float = 4.0
    # Noise
    day_noise_sd: float = 0.10       # region-day demand shocks
    dasher_response: float = 0.0     # carryover: how strongly Dashers react to last period's busyness
    seed: int | None = None

    @property
    def n_periods(self) -> int:
        return self.days * PERIODS_PER_DAY


def demand_profile(n_periods: int, start_day: int = 0) -> np.ndarray:
    """Relative demand for each period: lunch and dinner peaks, busier weekends."""
    t = np.arange(n_periods)
    hour = (t % PERIODS_PER_DAY) / 2.0
    day = start_day + t // PERIODS_PER_DAY
    daily = (0.12
             + 0.95 * np.exp(-(((hour - 12.0) / 1.3) ** 2))
             + 1.50 * np.exp(-(((hour - 18.5) / 1.7) ** 2)))
    weekend = np.where(day % 7 >= 4, 1.15, 1.0)  # Fri, Sat, Sun
    return daily * weekend


@dataclass
class World:
    """Everything that happens regardless of treatment: who opens the app, which
    Dashers log on, and the fixed traits of each region."""
    config: MarketConfig
    sessions: np.ndarray          # (periods, regions) app sessions
    dashers: np.ndarray           # (periods, regions) active Dashers
    region_alpha: np.ndarray      # (regions,) logit intercepts
    prep_minutes: np.ndarray      # (regions,)
    restaurants: np.ndarray       # (regions,)


def make_regions(cfg: MarketConfig, rng: np.random.Generator) -> dict:
    size = rng.lognormal(0.0, 0.35, cfg.n_regions)
    restaurants = np.maximum(15, np.round(90 * size * rng.lognormal(0, 0.3, cfg.n_regions)))
    logit0 = np.log(cfg.base_conversion / (1 - cfg.base_conversion))
    alpha = (logit0 + cfg.fee_sensitivity * cfg.control_fee
             + rng.normal(0, cfg.region_alpha_sd, cfg.n_regions))
    return dict(
        size=size,
        restaurants=restaurants,
        region_alpha=alpha,
        prep_minutes=rng.uniform(9, 15, cfg.n_regions),
        tightness=rng.uniform(*cfg.supply_tightness, cfg.n_regions),
    )


def make_world(cfg: MarketConfig, regions: dict, rng: np.random.Generator,
               start_day: int = 0) -> World:
    """Draw sessions and Dasher supply for one stretch of days."""
    T, R = cfg.n_periods, cfg.n_regions
    prof = demand_profile(T, start_day)[:, None]
    day_idx = np.arange(T) // PERIODS_PER_DAY
    day_shock = rng.lognormal(0, cfg.day_noise_sd, (cfg.days, R))[day_idx]
    mean_sessions = cfg.base_sessions * regions["size"][None, :] * prof * day_shock
    sessions = rng.poisson(mean_sessions)

    # Dashers are scheduled against a forecast of control-level orders, but supply
    # is flatter than demand (people don't all log on at 6:30pm), so peaks are tight.
    flat_prof = (1 - cfg.peak_flatness) * prof + cfg.peak_flatness * prof.mean()
    expected_orders = cfg.base_sessions * regions["size"][None, :] * flat_prof * cfg.base_conversion
    dasher_mean = regions["tightness"][None, :] * expected_orders / cfg.deliveries_per_dasher
    dashers = rng.poisson(np.maximum(dasher_mean, 0.5))

    return World(
        config=cfg,
        sessions=sessions,
        dashers=dashers,
        region_alpha=regions["region_alpha"],
        prep_minutes=regions["prep_minutes"],
        restaurants=regions["restaurants"],
    )


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def simulate(world: World, treat_share: np.ndarray, rng: np.random.Generator | None = None,
             stochastic: bool = True) -> dict:
    """Run the market for every period under a treatment assignment.

    treat_share: (periods, regions) array in [0, 1], the share of sessions that see
        the treatment fee. 0.5 everywhere = a user-level A/B test, 0/1 blocks = a
        switchback, all 1 or all 0 = a full rollout or no change.
    rng: random generator for splitting sessions and drawing orders.
    stochastic: if False, use expected values instead of random draws (used for the
        ground-truth counterfactuals so that the truth has no sampling noise).
    """
    cfg = world.config
    T, R = world.sessions.shape
    rng = rng if rng is not None else np.random.default_rng()
    sel = cfg.selection_effect * np.log(world.restaurants / 90.0)
    base_eta = world.prep_minutes + cfg.travel_minutes

    out = {k: np.zeros((T, R)) for k in
           ["sessions_t", "sessions_c", "orders_t", "orders_c", "eta", "util", "backlog"]}
    backlog = np.zeros(R)
    prev_util = np.full(R, 0.8)

    for t in range(T):
        n = world.sessions[t].astype(float)
        share = treat_share[t]
        if stochastic:
            n_t = rng.binomial(world.sessions[t], share).astype(float)
        else:
            n_t = n * share
        n_c = n - n_t
        # Busy periods attract extra Dashers in the next period (and quiet ones drive them off).
        supply_boost = np.clip(1 + cfg.dasher_response * (prev_util - 0.8), 0.5, 2.0)
        capacity = np.maximum(world.dashers[t] * supply_boost * cfg.deliveries_per_dasher, 0.5)

        def conv(fee, eta):
            return _sigmoid(world.region_alpha + sel - cfg.fee_sensitivity * fee
                            - cfg.eta_sensitivity * (eta - 35.0) / 10.0)

        def eta_of(orders):
            util = (orders + backlog) / capacity
            return base_eta + cfg.congestion_minutes * np.minimum(util, 2.5) ** cfg.congestion_power, util

        # Solve orders = n_t * p_t(eta(orders)) + n_c * p_c(eta(orders)) by bisection.
        # The right-hand side falls as orders rise, so there is exactly one solution.
        lo, hi = np.zeros(R), n.copy() + 1e-9
        for _ in range(32):
            mid = 0.5 * (lo + hi)
            eta, _ = eta_of(mid)
            excess = n_t * conv(cfg.treatment_fee, eta) + n_c * conv(cfg.control_fee, eta) - mid
            lo = np.where(excess > 0, mid, lo)
            hi = np.where(excess > 0, hi, mid)
        orders_eq = 0.5 * (lo + hi)
        eta, util = eta_of(orders_eq)
        p_t, p_c = conv(cfg.treatment_fee, eta), conv(cfg.control_fee, eta)

        if stochastic:
            o_t = rng.binomial(n_t.astype(int), p_t).astype(float)
            o_c = rng.binomial(n_c.astype(int), p_c).astype(float)
        else:
            o_t, o_c = n_t * p_t, n_c * p_c

        # Congestion and backlog follow the expected load (the equilibrium), so the
        # ground truth and the noisy experiments live in exactly the same market.
        backlog = np.maximum(0.0, orders_eq + backlog - capacity)

        out["sessions_t"][t], out["sessions_c"][t] = n_t, n_c
        out["orders_t"][t], out["orders_c"][t] = o_t, o_c
        out["eta"][t], out["util"][t], out["backlog"][t] = eta, util, backlog
        prev_util = util

    out["revenue_t"] = out["orders_t"] * (cfg.treatment_fee + cfg.take_rate * cfg.basket_size)
    out["revenue_c"] = out["orders_c"] * (cfg.control_fee + cfg.take_rate * cfg.basket_size)
    out["treat_share"] = treat_share
    return out
