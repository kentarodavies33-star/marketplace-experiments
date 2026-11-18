"""Run the whole study and save figures and a results summary.

    python run_experiments.py            # full run (about 3-4 minutes)
    python run_experiments.py --quick    # fewer simulations, for a fast check

Outputs go to results/: figures/*.png, summary.md and summary.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from marketsim.experiments import (
    ci, ground_truth, naive_ab, pre_period_covariates, switchback,
)
from marketsim.market import PERIODS_PER_DAY, MarketConfig, make_regions, make_world, simulate

OUT = Path("results")
FIG = OUT / "figures"

# Chart styling: validated categorical palette, recessive axes and grid
INK, INK_2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
C_NAIVE, C_SWITCH, C_CUPED = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False,
})

BLOCK = 4     # switchback block = 4 x 30 min = 2 hours
BURN_IN = 0


def one_market(cfg: MarketConfig, seed: int, block=BLOCK, burn_in=BURN_IN, run_naive=True):
    """Build a market, run every design on it, return estimates and the truth."""
    rng = np.random.default_rng(seed)
    regions = make_regions(cfg, rng)
    pre_world = make_world(cfg, regions, rng, start_day=0)
    world = make_world(cfg, regions, rng, start_day=cfg.days)
    T, R = world.sessions.shape
    pre_out = simulate(pre_world, np.zeros((T, R)), rng)
    cov = pre_period_covariates(pre_out, T, block)

    truth = ground_truth(world)
    res = {"truth": truth}
    if run_naive:
        res["naive"] = naive_ab(world, rng)
    res["switch"] = switchback(world, rng, block, burn_in, cov)
    res["world"] = world
    return res


def lift_and_ci(est, se, base):
    lo, hi = ci(est, se)
    return est / base, lo / base, hi / base


# ---------------------------------------------------------------------------
# 1. One experiment, start to finish
# ---------------------------------------------------------------------------

def single_experiment(cfg, seed=7):
    r = one_market(cfg, seed)
    rows = {}
    for m in ("conversion", "revenue"):
        t = r["truth"][m]
        n = r["naive"][m]
        s = r["switch"][m]
        rows[m] = {
            "truth": t["lift"],
            "naive": lift_and_ci(n["effect"], n["se"], n["control"]),
            "switchback": lift_and_ci(s["effect"], s["se"], s["control"]),
            "switchback_cuped": lift_and_ci(s["effect_cuped"], s["se_cuped"], s["control"]),
        }

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    labels = ["Naive A/B test", "Switchback", "Switchback + CUPED"]
    keys = ["naive", "switchback", "switchback_cuped"]
    colors = [C_NAIVE, C_SWITCH, C_CUPED]
    titles = {"conversion": "Orders per session: lift from a $1 fee cut",
              "revenue": "Revenue per session: lift from a $1 fee cut"}
    for ax, m in zip(axes, ("conversion", "revenue")):
        for i, (k, c) in enumerate(zip(keys, colors)):
            est, lo, hi = rows[m][k]
            y = 2 - i
            ax.plot([lo * 100, hi * 100], [y, y], color=c, lw=2, solid_capstyle="round")
            ax.plot(est * 100, y, "o", color=c, ms=9, mec=SURFACE, mew=2)
            ax.annotate(f"{est:+.1%}", (est * 100, y), xytext=(0, 9), textcoords="offset points",
                        ha="center", va="bottom", color=INK_2, fontsize=9,
                        bbox=dict(fc=SURFACE, ec="none", pad=1.5), zorder=5)
        truth = rows[m]["truth"] * 100
        ax.axvline(truth, color=INK, lw=1.5, ls="--")
        ax.annotate(f"True effect of full rollout: {rows[m]['truth']:+.1%}", (truth, 2.55),
                    xytext=(4, 0), textcoords="offset points", fontsize=9, color=INK)
        ax.axvline(0, color=AXIS, lw=1)
        ax.set_title(titles[m])
        ax.set_xlabel("Estimated lift (%), with 95% confidence interval")
        ax.set_ylim(-0.6, 2.9)
        ax.grid(axis="y", visible=False)
        lo_x = min(rows[m][k][1] for k in keys) * 100
        hi_x = max(rows[m][k][2] for k in keys) * 100
        ax.set_xlim(min(lo_x, truth, 0) - 3, max(hi_x, truth) + 8)
    axes[0].set_yticks([2, 1, 0], labels)
    fig.tight_layout()
    fig.savefig(FIG / "1_headline_one_experiment.png", dpi=160)
    plt.close(fig)

    # Mechanism: ETAs by time of day under each world
    world = r["world"]
    T, R = world.sessions.shape
    all_c = simulate(world, np.zeros((T, R)), stochastic=False)
    all_t = simulate(world, np.ones((T, R)), stochastic=False)
    ab = r["naive"]["raw"]
    hour = (np.arange(T) % PERIODS_PER_DAY) / 2

    def by_hour(eta):
        w = world.sessions
        return np.array([np.average(eta[hour == h], weights=w[hour == h] + 1e-9)
                         for h in np.unique(hour)])

    hours = np.unique(hour)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    series = [("Nobody gets the fee cut", by_hour(all_c["eta"]), MUTED, "-"),
              ("Everybody gets the fee cut", by_hour(all_t["eta"]), INK, "-"),
              ("A/B test: both groups share the same Dashers", by_hour(ab["eta"]), C_NAIVE, "-")]
    for name, y, c, ls in series:
        ax.plot(hours, y, color=c, lw=2, ls=ls, label=name)
    ax.set_xticks(range(0, 25, 3), [f"{h}:00" for h in range(0, 25, 3)])
    ax.set_xlim(0, 24)
    ax.set_ylabel("Average delivery ETA (minutes)")
    ax.set_title("Why the A/B test is biased: the control group's ETA gets worse too")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "2_mechanism_eta_by_hour.png", dpi=160)
    plt.close(fig)

    ab_control_eta = float(np.average(ab["eta"], weights=world.sessions + 1e-9))
    return rows, dict(eta_control_world=r["truth"]["eta_control"],
                      eta_treatment_world=r["truth"]["eta_treatment"],
                      eta_during_ab=ab_control_eta)


# ---------------------------------------------------------------------------
# 2. Repeat many times to measure bias, precision and coverage
# ---------------------------------------------------------------------------

def monte_carlo(cfg, n_reps):
    recs = []
    for i in range(n_reps):
        r = one_market(cfg, 1000 + i)
        for m in ("conversion", "revenue"):
            t = r["truth"][m]
            n, s = r["naive"][m], r["switch"][m]
            base = t["control"]
            for method, est, se in [("naive", n["effect"], n["se"]),
                                    ("switchback", s["effect"], s["se"]),
                                    ("switchback_iid_se", s["effect"], s["se_iid"]),
                                    ("switchback_cuped", s["effect_cuped"], s["se_cuped"])]:
                lo, hi = ci(est, se)
                recs.append(dict(rep=i, metric=m, method=method, truth=t["effect"],
                                 est=est, se=se, err_lift=(est - t["effect"]) / base,
                                 covered=lo <= t["effect"] <= hi,
                                 truth_lift=t["lift"], est_lift=est / base,
                                 wrong_sign=np.sign(est) != np.sign(t["effect"]) and (lo > 0 or hi < 0)))
        print(f"  simulation {i + 1}/{n_reps}", end="\r")
    print()
    return recs


def summarise(recs):
    out = {}
    for m in ("conversion", "revenue"):
        out[m] = {}
        for method in ("naive", "switchback", "switchback_iid_se", "switchback_cuped"):
            rs = [r for r in recs if r["metric"] == m and r["method"] == method]
            err = np.array([r["err_lift"] for r in rs])
            se_lift = np.array([r["se"] / (r["truth"] / r["truth_lift"]) for r in rs])
            out[m][method] = dict(
                mean_truth_lift=float(np.mean([r["truth_lift"] for r in rs])),
                mean_est_lift=float(np.mean([r["est_lift"] for r in rs])),
                bias_pp=float(err.mean() * 100),
                sd_pp=float(err.std(ddof=1) * 100),
                rmse_pp=float(np.sqrt(np.mean(err**2)) * 100),
                avg_se_pp=float(se_lift.mean() * 100),
                coverage=float(np.mean([r["covered"] for r in rs])),
                confidently_wrong_sign=float(np.mean([r["wrong_sign"] for r in rs])),
            )
    return out


def plot_monte_carlo(recs):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    methods = [("naive", "Naive A/B test", C_NAIVE),
               ("switchback", "Switchback", C_SWITCH),
               ("switchback_cuped", "Switchback + CUPED", C_CUPED)]
    titles = {"conversion": "Orders per session", "revenue": "Revenue per session"}
    rng = np.random.default_rng(0)
    for ax, m in zip(axes, ("conversion", "revenue")):
        for i, (key, label, c) in enumerate(methods):
            errs = np.array([r["err_lift"] for r in recs if r["metric"] == m and r["method"] == key]) * 100
            y = 2 - i + rng.uniform(-0.18, 0.18, len(errs))
            ax.scatter(errs, y, s=26, color=c, alpha=0.75, edgecolor=SURFACE, linewidth=0.8, label=label)
            ax.plot([errs.mean()] * 2, [2 - i - 0.3, 2 - i + 0.3], color=INK, lw=2)
        ax.axvline(0, color=INK, lw=1.5, ls="--")
        ax.set_yticks([2, 1, 0], [l for _, l, _ in methods])
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("Estimated lift minus true lift (percentage points)")
        ax.set_title(f"{titles[m]}: error across {len(errs)} simulated experiments")
    axes[1].set_yticklabels([])
    axes[0].annotate("dashed line = no error\nblack tick = average error", (0.99, 0.02),
                     xycoords="axes fraction", ha="right", va="bottom", fontsize=8.5, color=INK_2)
    fig.tight_layout()
    fig.savefig(FIG / "3_monte_carlo_errors.png", dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. How long should switchback blocks be?
# ---------------------------------------------------------------------------

def block_length_sweep(cfg, n_reps, blocks=(1, 2, 4, 8, 16)):
    rows = []
    for b in blocks:
        errs, ses = [], []
        for i in range(n_reps):
            r = one_market(cfg, 5000 + i, block=b, run_naive=False)
            t = r["truth"]["conversion"]
            s = r["switch"]["conversion"]
            errs.append((s["effect_cuped"] - t["effect"]) / t["control"])
            ses.append(s["se_cuped"] / t["control"])
        errs = np.array(errs) * 100
        rows.append(dict(block_periods=b, block_hours=b / 2, bias_pp=float(errs.mean()),
                         bias_se_pp=float(errs.std(ddof=1) / np.sqrt(n_reps)),
                         avg_se_pp=float(np.mean(ses) * 100)))
        print(f"  block length {b / 2:g}h done")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    x = [r["block_hours"] for r in rows]
    axes[0].errorbar(x, [r["bias_pp"] for r in rows], yerr=[1.96 * r["bias_se_pp"] for r in rows],
                     color=C_SWITCH, lw=2, marker="o", ms=8, mec=SURFACE, mew=2, capsize=0)
    axes[0].axhline(0, color=INK, lw=1.2, ls="--")
    axes[0].set_title("Bias: carryover between blocks is small here")
    axes[0].set_ylabel("Average error (pp of lift)")
    axes[1].plot(x, [r["avg_se_pp"] for r in rows], color=C_SWITCH, lw=2, marker="o", ms=8,
                 mec=SURFACE, mew=2)
    axes[1].set_title("Noise: longer blocks mean fewer units")
    axes[1].set_ylabel("Standard error (pp of lift)")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(x, [f"{v:g}h" for v in x])
        ax.set_xlabel("Switchback block length")
    fig.tight_layout()
    fig.savefig(FIG / "4_block_length_tradeoff.png", dpi=160)
    plt.close(fig)
    return rows


def write_summary(single, mech, mc, sweep, n_reps):
    names = {"naive": "Naive A/B test", "switchback": "Switchback (cluster-robust SE)",
             "switchback_iid_se": "Switchback, but with plain (iid) SE",
             "switchback_cuped": "Switchback + CUPED"}
    lines = ["# Results", "",
             f"_Generated by `run_experiments.py`. Monte Carlo uses {n_reps} independent simulated markets._", "",
             "## One experiment (seed 7)", "",
             "| Metric | True lift | Naive A/B | Switchback | Switchback + CUPED |",
             "|---|---|---|---|---|"]
    for m, label in (("conversion", "Orders per session"), ("revenue", "Revenue per session")):
        r = single[m]
        f = lambda k: f"{r[k][0]:+.1%} [{r[k][1]:+.1%}, {r[k][2]:+.1%}]"
        lines.append(f"| {label} | {r['truth']:+.1%} | {f('naive')} | {f('switchback')} | {f('switchback_cuped')} |")
    lines += ["", "Average delivery ETA (session-weighted):", "",
              f"- Nobody gets the fee cut: {mech['eta_control_world']:.1f} min",
              f"- Everybody gets the fee cut: {mech['eta_treatment_world']:.1f} min",
              f"- During the A/B test (both groups): {mech['eta_during_ab']:.1f} min", "",
              "## Across many simulated experiments", ""]
    for m, label in (("conversion", "Orders per session"), ("revenue", "Revenue per session")):
        lines += [f"### {label}", "",
                  "| Method | Avg estimated lift | Avg true lift | Bias (pp) | RMSE (pp) | Avg SE (pp) | 95% CI coverage |",
                  "|---|---|---|---|---|---|---|"]
        for k, v in mc[m].items():
            lines.append(f"| {names[k]} | {v['mean_est_lift']:+.1%} | {v['mean_truth_lift']:+.1%} | "
                         f"{v['bias_pp']:+.2f} | {v['rmse_pp']:.2f} | {v['avg_se_pp']:.2f} | {v['coverage']:.0%} |")
        lines.append("")
    cuped_gain = 1 - (mc["conversion"]["switchback_cuped"]["avg_se_pp"] / mc["conversion"]["switchback"]["avg_se_pp"]) ** 2
    lines += [f"CUPED cuts the variance of the switchback estimate by about {cuped_gain:.0%} "
              f"(orders metric), which is like running the experiment {1 / (1 - cuped_gain):.1f}x as long.", "",
              "## Switchback block length (orders per session, with CUPED)", "",
              "| Block length | Bias (pp) | Avg SE (pp) |", "|---|---|---|"]
    for r in sweep:
        lines.append(f"| {r['block_hours']:g}h | {r['bias_pp']:+.2f} ± {1.96 * r['bias_se_pp']:.2f} | {r['avg_se_pp']:.2f} |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    (OUT / "summary.json").write_text(json.dumps(
        dict(single=single, mechanism=mech, monte_carlo=mc, block_sweep=sweep), indent=2, default=float))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="fewer simulations")
    p.add_argument("--reps", type=int, default=None)
    args = p.parse_args()
    n_reps = args.reps or (8 if args.quick else 40)
    sweep_reps = 4 if args.quick else 15

    FIG.mkdir(parents=True, exist_ok=True)
    cfg = MarketConfig()
    t0 = time.time()
    print("1/3 One experiment, start to finish")
    single, mech = single_experiment(cfg)
    print(f"2/3 Monte Carlo: {n_reps} simulated experiments")
    recs = monte_carlo(cfg, n_reps)
    mc = summarise(recs)
    plot_monte_carlo(recs)
    print("3/3 Switchback block-length sweep")
    sweep = block_length_sweep(cfg, sweep_reps)
    write_summary(single, mech, mc, sweep, n_reps)
    print(f"Done in {time.time() - t0:.0f}s. See results/summary.md and results/figures/.")


if __name__ == "__main__":
    main()
