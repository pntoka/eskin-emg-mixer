from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_absolute_error

from common import REPORTS, FORCE_HZ, FloorScaler

d = np.load(REPORTS / "dataset.npz", allow_pickle=True)
X, y, trial = d["X"], d["y"], d["trial"]
names = list(dict.fromkeys(trial.tolist()))
ALPHAS = np.logspace(-2, 5, 20)


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

# ---------- Step 3: WITHIN-TRIAL CEILING (blocked 70/30 + gap) ----------
GAP = int(1.0 * FORCE_HZ)
rows = []
for fname, fx in FEATS.items():
    per = []
    for name in names:
        m = np.where(trial == name)[0]
        n = len(m); cut = int(0.7 * n)
        tr, te = m[:cut], m[cut + GAP:]
        if len(te) < 50:
            continue
        f = model().fit(fx(X[tr]), y[tr])
        per.append(score(y[te], f.predict(fx(X[te]))))
    rows.append(dict(features=fname, split="within-trial",
                     **{k: round(float(np.mean([p[k] for p in per])), 3) for k in per[0]}))

# ---------- Step 4: CROSS-TRIAL (GroupKFold on trial_id) ----------
n_splits = min(5, len(names))
gkf = GroupKFold(n_splits=n_splits)
oof_pred = {}
for fname, fx in FEATS.items():
    pred = np.zeros_like(y)
    for tr, te in gkf.split(X, y, groups=trial):
        pred[te] = model().fit(fx(X[tr]), y[tr]).predict(fx(X[te]))
    rows.append(dict(features=fname, split="cross-trial", **score(y, pred)))
    oof_pred[fname] = pred

res = pd.DataFrame(rows)
print(res.to_string(index=False))

# ---------- Step 5: DOES A 5-SECOND CALIBRATION GRASP RESCUE IT? ----------
CAL = int(5.0 * FORCE_HZ)
cal_rows = []
best = "raw256"
for name in names:
    m = np.where(trial == name)[0]
    if len(m) < CAL + 200:
        continue
    p, t = oof_pred[best][m], y[m]
    a, b = np.polyfit(p[:CAL], t[:CAL], 1)
    rest = slice(CAL, None)
    cal_rows.append(dict(trial=name,
                         r2_raw=round(r2_score(t[rest], p[rest]), 3),
                         r2_cal=round(r2_score(t[rest], a * p[rest] + b), 3),
                         rmse_raw=round(rmse(t[rest], p[rest]), 2),
                         rmse_cal=round(rmse(t[rest], a * p[rest] + b), 2),
                         gain=round(float(a), 3), offset=round(float(b), 2)))
cal = pd.DataFrame(cal_rows)
print("\n--- per-trial affine recalibration (5s calibration window) ---")
print(cal.to_string(index=False))
print(f"\nmean R2  raw={cal.r2_raw.mean():.3f}  calibrated={cal.r2_cal.mean():.3f}")

with open(REPORTS / "results.md", "w") as fh:
    fh.write("# Hour-1 results\n\n## Model comparison\n\n")
    fh.write(res.to_markdown(index=False))
    fh.write("\n\n## Per-session affine recalibration\n\n")
    fh.write(cal.to_markdown(index=False))
    fh.write(f"\n\nmean R2  raw={cal.r2_raw.mean():.3f}  calibrated={cal.r2_cal.mean():.3f}\n")

# --- predicted-vs-actual plot for the best and worst held-out trials
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

per_trial_r2 = {name: r2_score(y[trial == name], oof_pred[best][trial == name]) for name in names}
worst = min(per_trial_r2, key=per_trial_r2.get)
best_t = max(per_trial_r2, key=per_trial_r2.get)
fig, axes = plt.subplots(2, 1, figsize=(10, 6))
for ax, name in zip(axes, [best_t, worst]):
    m = trial == name
    ax.plot(y[m], label="actual", lw=1)
    ax.plot(oof_pred[best][m], label="predicted (cross-trial, raw256)", lw=1)
    ax.set_title(f"{name}  (R2={per_trial_r2[name]:.3f})", fontsize=9)
    ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(REPORTS / "fig_pred_vs_actual.png", dpi=110)
plt.close(fig)
print(f"\nbest cross-trial R2: {best_t} ({per_trial_r2[best_t]:.3f})")
print(f"worst cross-trial R2: {worst} ({per_trial_r2[worst]:.3f})")
