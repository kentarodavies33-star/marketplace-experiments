"""Burn-in study: does dropping the first period(s) of each switchback block
remove carryover bias, and how much precision does it cost?

    python burn_in_study.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from marketsim.market import MarketConfig
from run_experiments import one_market

BLOCK = 4                  # 2-hour switchback blocks
BURN_INS = [0, 1, 2]       # periods dropped at the start of each block (0, 30 min, 1 hour)
REPS = 20                 # simulated markets per setting
MODELS = {
    "No carryover (original model)": MarketConfig(dasher_response=0.0),
    "Dashers react to last period": MarketConfig(dasher_response=1.5),
}

results = {}
for name, cfg in MODELS.items():
    results[name] = []
    for burn_in in BURN_INS:
        errors, ses = [], []
        for i in range(REPS):
            r = one_market(cfg, seed=9000 + i, block=BLOCK, burn_in=burn_in, run_naive=False)
            truth = r["truth"]["conversion"]
            est = r["switch"]["conversion"]
            errors.append((est["effect_cuped"] - truth["effect"]) / truth["control"] * 100)
            ses.append(est["se_cuped"] / truth["control"] * 100)
        errors = np.array(errors)
        row = dict(burn_in=burn_in, bias=errors.mean(),
                   bias_ci=1.96 * errors.std(ddof=1) / np.sqrt(REPS), se=np.mean(ses))
        results[name].append(row)
        print(f"{name:32s} burn-in {burn_in}: bias {row['bias']:+.2f} ± {row['bias_ci']:.2f} pp, "
              f"SE {row['se']:.2f} pp")

# Chart: bias and standard error, side by side
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
for (name, rows), color in zip(results.items(), ["#2a78d6", "#eb6834"]):
    x = [r["burn_in"] * 30 for r in rows]
    axes[0].errorbar(x, [r["bias"] for r in rows], yerr=[r["bias_ci"] for r in rows],
                     marker="o", color=color, label=name, capsize=0, lw=2)
    axes[1].plot(x, [r["se"] for r in rows], marker="o", color=color, label=name, lw=2)
axes[0].axhline(0, color="black", ls="--", lw=1)
axes[0].set_title("Bias (closer to 0 is better)")
axes[0].set_ylabel("Avg error in lift (pp)")
axes[1].set_title("Standard error (lower is better)")
axes[1].set_ylabel("SE of lift (pp)")
for ax in axes:
    ax.set_xticks([b * 30 for b in BURN_INS])
    ax.set_xlabel("Minutes dropped at the start of each block")
axes[1].legend(frameon=False)
fig.tight_layout()
fig.savefig("results/figures/5_burn_in_tradeoff.png", dpi=160)
print("Saved results/figures/5_burn_in_tradeoff.png")
