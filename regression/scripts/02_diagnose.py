from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import REPORTS

d = np.load(REPORTS / "dataset.npz", allow_pickle=True)
X, y, trial = d["X"], d["y"], d["trial"]
names = list(dict.fromkeys(trial.tolist()))
total = X.sum(axis=1)

# --- Plot 1: does e-skin track force at all?
fig, axes = plt.subplots(len(names), 1, figsize=(11, 2.0 * len(names)), sharex=False)
for ax, name in zip(np.atleast_1d(axes), names):
    m = trial == name
    ax.plot(y[m], lw=1, label="F_combined (N)")
    ax2 = ax.twinx()
    ax2.plot(total[m] / total[m].max(), lw=1, color="tab:orange", alpha=.7,
             label="e-skin total (norm)")
    ax.set_title(name, fontsize=8); ax.tick_params(labelsize=7)
fig.tight_layout(); fig.savefig(REPORTS / "fig_timeseries.png", dpi=110)
plt.close(fig)

# --- Plot 2: hysteresis + cross-trial consistency
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
dF = np.gradient(y)
for name in names:
    m = trial == name
    a2.scatter(total[m], y[m], s=2, alpha=.25, label=name)
up, dn = dF > 0.5, dF < -0.5
a1.scatter(total[up], y[up], s=2, alpha=.2, label="loading")
a1.scatter(total[dn], y[dn], s=2, alpha=.2, label="unloading")
a1.set(xlabel="e-skin total", ylabel="F_combined (N)", title="Hysteresis")
a1.legend(markerscale=6)
a2.set(xlabel="e-skin total", ylabel="F_combined (N)", title="Per-trial consistency")
a2.legend(fontsize=6, markerscale=6, ncol=2)
fig.tight_layout(); fig.savefig(REPORTS / "fig_hysteresis.png", dpi=110)
plt.close(fig)

print("saturation fraction (taxels at >=98% of observed max):",
      round(float((X >= 0.98 * X.max()).mean()), 4))
for name in names:
    m = trial == name
    r = np.corrcoef(total[m], y[m])[0, 1]
    print(f"  {name}: n={m.sum():5d}  F_range=[{y[m].min():7.2f}, {y[m].max():7.2f}]  r = {r:.3f}")
print(f"POOLED r = {np.corrcoef(total, y)[0,1]:.3f}")
