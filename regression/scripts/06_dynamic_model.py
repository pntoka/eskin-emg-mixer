"""Ridge regression on the 8 split YL_grasp_dynamic trials: leave-one-trial-out
cross-validation across e-skin feature representations (raw256, sqrt256,
total, and the 5-scalar summary features), plus detailed predicted-vs-actual
plots and correlations for the best/worst held-out trials -- same approach
as 04_archive_model.py, applied to this richer single-session dynamic set.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import joblib
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import REPORTS, TAXELS, FloorScaler, scalar_features

ALPHAS = np.logspace(-2, 5, 20)

d = np.load(REPORTS / "dataset.npz", allow_pickle=True)
X_all, y_all, trial_all = d["X"], d["y"], d["trial"]
m = np.array([str(t).startswith("YL_dynamic_") for t in trial_all])
X, y, trial = X_all[m], y_all[m], trial_all[m]
names = sorted(set(trial.tolist()))
print(f"YL_dynamic samples: {len(y):,} across {len(names)} trials")

sat_value = float(X.max())
print(f"saturation-ceiling proxy (max observed taxel value): {sat_value:.1f}")


def model():
    return make_pipeline(FloorScaler(), RidgeCV(alphas=ALPHAS))


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def score(yt, yp):
    r, _ = pearsonr(yt, yp)
    return dict(r2=round(r2_score(yt, yp), 3), pearson_r=round(float(r), 3),
                rmse=round(rmse(yt, yp), 2), mae=round(mean_absolute_error(yt, yp), 2))


FEATS = {
    "raw256": lambda Z: Z,
    "sqrt256": lambda Z: np.sqrt(Z),
    "total": lambda Z: Z.sum(axis=1, keepdims=True),
    "scalars5": lambda Z: scalar_features(Z, sat_value).to_numpy(),
}

# ---------- leave-one-trial-out cross-validation (the real test) ----------
gkf = GroupKFold(n_splits=len(names))
rows = []
loto_pred = {}
for fname, fx in FEATS.items():
    pred = np.zeros_like(y)
    for tr, te in gkf.split(X, y, groups=trial):
        pred[te] = model().fit(fx(X[tr]), y[tr]).predict(fx(X[te]))
    rows.append(dict(features=fname, split="leave-one-trial-out", **score(y, pred)))
    loto_pred[fname] = pred
    for name in names:
        te = np.where(trial == name)[0]
        print(f"  LOTO held-out={name} features={fname}: {score(y[te], pred[te])}")

res = pd.DataFrame(rows)
print()
print(res.to_string(index=False))

with open(REPORTS / "results_dynamic.md", "w") as fh:
    fh.write("# YL_grasp_dynamic split-trial model results\n\n")
    fh.write(f"Samples: {len(y):,} across {len(names)} trials "
             f"({', '.join(names)})\n\n")
    fh.write(res.to_markdown(index=False))
    fh.write("\n")

# ---------- per-trial breakdown for the best feature set ----------
best = "raw256"
per_trial_r2 = {name: r2_score(y[trial == name], loto_pred[best][trial == name])
                 for name in names}
per_trial_r = {name: float(pearsonr(y[trial == name], loto_pred[best][trial == name])[0])
               for name in names}
ranked = sorted(names, key=lambda n: per_trial_r2[n])
worst, best_t = ranked[0], ranked[-1]
print(f"\n[{best}] worst held-out: {worst} (R2={per_trial_r2[worst]:.3f}, r={per_trial_r[worst]:.3f})")
print(f"[{best}] best  held-out: {best_t} (R2={per_trial_r2[best_t]:.3f}, r={per_trial_r[best_t]:.3f})")

# ---------- predicted vs actual timeseries: best + worst held-out trials ----------
fig, axes = plt.subplots(2, 1, figsize=(11, 7))
for ax, name in zip(axes, [best_t, worst]):
    te = np.where(trial == name)[0]
    ax.plot(y[te], label="actual F_combined (N)", lw=1)
    ax.plot(loto_pred[best][te], label=f"predicted ({best}, leave-one-trial-out)", lw=1, alpha=.8)
    ax.set_title(f"held out: {name}  (R2={per_trial_r2[name]:.3f}, "
                 f"pearson r={per_trial_r[name]:.3f})", fontsize=9)
    ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(REPORTS / "fig_dynamic_pred_vs_actual.png", dpi=110)
plt.close(fig)

# ---------- scatter: predicted vs actual, all trials ----------
fig, ax = plt.subplots(figsize=(6.5, 6.5))
cmap = plt.get_cmap("tab10")
for i, name in enumerate(names):
    te = np.where(trial == name)[0]
    ax.scatter(y[te], loto_pred[best][te], s=3, alpha=.3, color=cmap(i % 10), label=name)
lims = [min(y.min(), loto_pred[best].min()), max(y.max(), loto_pred[best].max())]
ax.plot(lims, lims, "k--", lw=1, label="perfect")
ax.set(xlabel="actual F_combined (N)", ylabel="predicted F_combined (N)",
       title=f"Leave-one-trial-out predictions ({best})")
ax.legend(fontsize=6, markerscale=4, ncol=2)
fig.tight_layout()
fig.savefig(REPORTS / "fig_dynamic_scatter.png", dpi=110)
plt.close(fig)

# ---------- final model: fit on all 8 trials pooled ----------
final_raw256 = model().fit(FEATS["raw256"](X), y)
joblib.dump(final_raw256, REPORTS / "dynamic_model_raw256.joblib")
with open(REPORTS / "dynamic_model_metadata.json", "w") as fh:
    json.dump({
        "features": "256 taxels, per-trial baseline-subtracted "
                     "(common.subtract_baseline) before the model",
        "taxel_order": TAXELS,
        "trained_on": names,
        "n_samples": int(len(y)),
        "target": "F_combined = F1_N + F2_N (Newtons)",
        "leave_one_trial_out_r2": rows[0]["r2"],
        "leave_one_trial_out_pearson_r": rows[0]["pearson_r"],
    }, fh, indent=2)
print(f"\nSaved final pooled dynamic model -> {REPORTS/'dynamic_model_raw256.joblib'}")
