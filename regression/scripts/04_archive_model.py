"""Archive-only regression model: the 10 target_force "hold" trials showed
almost no within-trial force variation (narrow bands by design), so e-skin
has nothing to correlate against there. The 2 archive/ sessions are the
only data with real dynamic range (unstructured squeeze/release), so this
script builds and evaluates a model on archive data alone, as a baseline to
extend once more dynamic trials are collected.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import REPORTS, TAXELS, FloorScaler

ARCHIVE = ["archive_165516", "archive_173729"]
ALPHAS = np.logspace(-2, 5, 20)

d = np.load(REPORTS / "dataset.npz", allow_pickle=True)
X_all, y_all, trial_all = d["X"], d["y"], d["trial"]
m = np.isin(trial_all, ARCHIVE)
X, y, trial = X_all[m], y_all[m], trial_all[m]
print(f"archive-only samples: {len(y):,} ({(trial==ARCHIVE[0]).sum():,} / "
      f"{(trial==ARCHIVE[1]).sum():,})")


def model():
    return make_pipeline(FloorScaler(), RidgeCV(alphas=ALPHAS))


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def score(yt, yp):
    return dict(r2=round(r2_score(yt, yp), 3), rmse=round(rmse(yt, yp), 2),
                mae=round(mean_absolute_error(yt, yp), 2))


FEATS = {
    "raw256": lambda Z: Z,
    "sqrt256": lambda Z: np.sqrt(Z),
    "total": lambda Z: Z.sum(axis=1, keepdims=True),
}

# ---------- within-session ceiling (blocked 70/30 + 1s gap) ----------
GAP = 100
rows = []
for fname, fx in FEATS.items():
    for name in ARCHIVE:
        idx = np.where(trial == name)[0]
        n = len(idx); cut = int(0.7 * n)
        tr, te = idx[:cut], idx[cut + GAP:]
        f = model().fit(fx(X[tr]), y[tr])
        s = score(y[te], f.predict(fx(X[te])))
        rows.append(dict(features=fname, split=f"within:{name}", **s))

# ---------- leave-one-session-out (the real cross-trial test here) ----------
loso_pred = {}
for fname, fx in FEATS.items():
    pred = np.full_like(y, np.nan)
    for held_out in ARCHIVE:
        te = np.where(trial == held_out)[0]
        tr = np.where(trial != held_out)[0]
        f = model().fit(fx(X[tr]), y[tr])
        pred[te] = f.predict(fx(X[te]))
    s = score(y, pred)
    rows.append(dict(features=fname, split="leave-one-session-out", **s))
    loso_pred[fname] = pred
    for held_out in ARCHIVE:
        te = np.where(trial == held_out)[0]
        s_i = score(y[te], pred[te])
        print(f"  LOSO held-out={held_out} features={fname}: {s_i}")

res = pd.DataFrame(rows)
print()
print(res.to_string(index=False))

with open(REPORTS / "results_archive.md", "w") as fh:
    fh.write("# Archive-only model results\n\n")
    fh.write(f"Samples: {len(y):,} across {ARCHIVE}\n\n")
    fh.write(res.to_markdown(index=False))
    fh.write("\n")

# ---------- plots: predicted vs actual timeseries for each held-out session
# "total" (single summed-taxel scalar) is the feature set that actually
# generalizes cross-session (raw256/sqrt256 overfit each session's specific
# contact pattern -- LOSO R2 goes negative for both, see table above) --
# so that's what's shown here as the honest current-best result, with
# raw256 included alongside for comparison since it's the within-session
# ceiling and the one with more headroom once more sessions are added.
PRIMARY = "total"
fig, axes = plt.subplots(len(ARCHIVE), 1, figsize=(11, 8))
for ax, held_out in zip(axes, ARCHIVE):
    te = np.where(trial == held_out)[0]
    ax.plot(y[te], label="actual F_combined (N)", lw=1)
    ax.plot(loso_pred[PRIMARY][te], label="predicted (total feature, trained on other session)",
            lw=1, alpha=.8)
    ax.plot(loso_pred["raw256"][te], label="predicted (raw256, for comparison)",
            lw=1, alpha=.5, color="tab:green")
    r2_p = r2_score(y[te], loso_pred[PRIMARY][te])
    r2_raw = r2_score(y[te], loso_pred["raw256"][te])
    ax.set_title(f"held out: {held_out}  (total R2={r2_p:.3f}, raw256 R2={r2_raw:.3f})", fontsize=9)
    ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(REPORTS / "fig_archive_loso_timeseries.png", dpi=110)
plt.close(fig)

# ---------- scatter: predicted vs actual, both held-out sessions
fig, ax = plt.subplots(figsize=(6, 6))
for held_out, color in zip(ARCHIVE, ["tab:blue", "tab:orange"]):
    te = np.where(trial == held_out)[0]
    ax.scatter(y[te], loso_pred[PRIMARY][te], s=3, alpha=.3, color=color, label=held_out)
lims = [min(y.min(), loso_pred[PRIMARY].min()), max(y.max(), loso_pred[PRIMARY].max())]
ax.plot(lims, lims, "k--", lw=1, label="perfect")
ax.set(xlabel="actual F_combined (N)", ylabel="predicted F_combined (N)",
       title=f"Leave-one-session-out predictions ({PRIMARY} feature)")
ax.legend(fontsize=8, markerscale=4)
fig.tight_layout()
fig.savefig(REPORTS / "fig_archive_scatter.png", dpi=110)
plt.close(fig)

# ---------- final models: fit on ALL archive data pooled, for future extension
# Save both: "total" is the current best cross-session generalizer; raw256
# is kept too since more/varied sessions may let its spatial detail start
# generalizing instead of overfitting one session's contact geometry.
final_total = model().fit(FEATS["total"](X), y)
final_raw256 = model().fit(FEATS["raw256"](X), y)
joblib.dump(final_total, REPORTS / "archive_model_total.joblib")
joblib.dump(final_raw256, REPORTS / "archive_model_raw256.joblib")
with open(REPORTS / "archive_model_metadata.json", "w") as fh:
    json.dump({
        "recommended_current_model": "archive_model_total.joblib",
        "why": ("raw256 fits each session's contact pattern well in-session "
                 "(R2 0.63/0.76) but does not transfer to the other session "
                 "(LOSO R2 -0.22/-2.36); the simple summed-taxel 'total' "
                 "feature generalizes better cross-session (LOSO R2 "
                 "0.40/0.47) despite being much simpler -- likely because "
                 "only 2 sessions/contact geometries exist so far, too few "
                 "for the 256-dim model to learn a session-invariant "
                 "spatial pattern rather than overfitting one session's."),
        "raw256_model": {
            "features": "256 taxels, per-trial baseline-subtracted "
                         "(common.subtract_baseline) before the model",
            "taxel_order": TAXELS,
        },
        "total_model": {
            "features": "sum of all 256 baseline-subtracted taxels (single scalar)",
        },
        "trained_on": ARCHIVE,
        "n_samples": int(len(y)),
        "target": "F_combined = F1_N + F2_N (Newtons)",
        "preprocessing_required_at_inference": (
            "Subtract per-session per-taxel 5th percentile from raw taxel "
            "readings before predicting -- both models were trained on "
            "baseline-subtracted values, not raw ADC counts."
        ),
    }, fh, indent=2)
print(f"\nSaved final pooled-archive models -> "
      f"{REPORTS/'archive_model_total.joblib'}, {REPORTS/'archive_model_raw256.joblib'}")
