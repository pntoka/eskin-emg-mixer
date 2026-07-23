# Hour-1 Feasibility Test: e-skin → grip force regression

**Audience:** a Claude Code instance running inside the project repo (the one containing `data/`, `archive/`, `src/processing/align.py`).

**Goal:** in ~60 minutes of compute + iteration, answer one question with existing recorded data:

> Can we predict `F_combined = F1_N + F2_N` from a single 16×16 e-skin frame well enough to drive a real-time force visualiser — and does that mapping survive being applied to a *trial it was not trained on*?

Everything else (temporal models, CNNs, ONNX export, live streaming) is deliberately out of scope. Do not build it.

---

## 1. The one thing that matters

There are only three possible outcomes, and they lead to completely different next steps. The whole hour is designed to distinguish between them:

| Outcome | Within-trial R² | Cross-trial R² | Diagnosis | Next step |
|---|---|---|---|---|
| **A. Green light** | > 0.9 | > 0.85 | Mapping is stable and transferable | Go straight to live testing |
| **B. Calibration-limited** | > 0.9 | collapses (< 0.6) | Sensor works; per-donning offset/gain drifts | Add a per-session calibration grasp — **this is very likely the outcome, and it is still a success** |
| **C. Physics-limited** | < 0.7 | < 0.7 | Saturation / hysteresis / bad alignment | Stop modelling; fix alignment, check saturation, consider temporal features |

Outcome B is the one people misread as failure. It is not. It means the model is fine and you need a 5-second calibration squeeze at the start of each session. **Step 5 below explicitly tests whether that fix works**, so you leave the hour knowing not just the problem but the remedy.

---

## 2. Scope decisions (already made — do not revisit)

| Decision | Choice | Why |
|---|---|---|
| Which trials | The **14 `target_force` trials in `data/` only** | Force varies continuously — the useful regime. `max_effort` trials are saturation-dominated and would corrupt the fit. |
| `archive/` | **Skip** | No manifest, unresolved force-bias question. Not worth the hour. |
| Alignment | **Bin-average e-skin frames onto the 100 Hz force grid** | Uses all e-skin data, denoises, mirrors the force stream's own 1 kHz→100 Hz block averaging. |
| Target | **`F_combined` only** | F2's calibration is a placeholder; the sum is still physically meaningful. Never model F1/F2 separately. |
| Model | **Ridge regression on 256 taxels** | Sub-microsecond inference (it is a dot product), no hyperparameter rabbit hole, and it is the honest baseline any fancier model must beat. |
| Split | **GroupKFold by `trial_id`** | Random sample splits are invalid here — at 100 Hz consecutive frames are near-duplicates, and a shuffled split can flatter the score by ~10×. |
| E-skin baseline | **Per-trial per-taxel 5th percentile, subtracted** | This is legitimate rather than leaky: at deployment you will always have a no-contact rest period to compute it from. |

Two trials need special handling and the code below does it automatically:
- `PT_target_45_003` has a ~356 s span with a long stall before abort — the stalled segment is dead weight. It is dropped in favour of the redo `PT_target_45_004`.
- `AM_max_squeeze_n_5` / `_old` have suspect subject attribution — irrelevant here since they are `max_effort` and already excluded.

---

## 3. Deliverables

Create these; do not refactor anything existing.

```
scripts/hour1/
  common.py              # loading, alignment, features
  01_build_dataset.py    # → reports/hour1/dataset.npz
  02_diagnose.py         # → reports/hour1/fig_*.png, lag table
  03_ceiling_test.py     # → reports/hour1/results.md
reports/hour1/
  dataset.npz
  fig_timeseries.png
  fig_hysteresis.png
  fig_pred_vs_actual.png
  results.md             # THE deliverable — the table from §1 filled in
```

---

## 4. Step 0 — Recon (5 min)

Before writing anything, verify assumptions. `src/processing/align.py` already has e-skin/force CSV loaders and elapsed-time handling — **read it first and reuse its loaders if their API is clean**, ignoring the EMG-specific parts. The code below falls back to plain `pd.read_csv` if reuse is awkward; do not spend more than 5 minutes deciding.

Check and report:
1. Exact column names in one `eskin.csv` and one `forces.csv` (confirm `R00_C00`…`R15_C15`, `F1_N`, `F2_N`, `elapsed_s`).
2. The **max ADC value** — `eskin[taxels].max().max()` across a `max_effort` trial. This is the saturation ceiling; you need the number.
3. Resting force level: `force['F1_N'].head(50).median()`. If it sits near 0, bias is already applied. If near ~4.0 N, it is not. Report which.
4. Confirm 14 trials have `task_kind == "target_force"` in `manifest.json`.

---

## 5. Step 1 — Build the dataset (15 min)

### `scripts/hour1/common.py`

```python
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

TAXELS = [f"R{r:02d}_C{c:02d}" for r in range(16) for c in range(16)]
FORCE_HZ = 100.0

# Trials to skip: long-stall aborted trial superseded by its redo.
SKIP_TRIALS = {"PT_target_45_003_20260722_153749"}


def iter_target_force_trials(data_dir: Path):
    """Yield (trial_dir, manifest) for every task_kind == 'target_force' trial."""
    for d in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        mpath = d / "manifest.json"
        if not mpath.exists():
            continue
        man = json.loads(mpath.read_text())
        if man.get("task_kind") != "target_force":
            continue
        if d.name in SKIP_TRIALS:
            print(f"  skip (known bad): {d.name}")
            continue
        yield d, man


def align_to_force_grid(eskin: pd.DataFrame, force: pd.DataFrame):
    """Average all e-skin frames falling within each force sample's window.

    Returns (E, t, F, coverage) where E is (n_force, 256) and coverage is the
    fraction of force samples that had at least one real e-skin frame.
    """
    et = eskin["elapsed_s"].to_numpy(float)
    ft = force["elapsed_s"].to_numpy(float)
    E_raw = eskin[TAXELS].to_numpy(float)
    n = len(ft)

    # Bin edges = midpoints between consecutive force timestamps.
    edges = np.empty(n + 1)
    edges[1:-1] = 0.5 * (ft[:-1] + ft[1:])
    edges[0] = ft[0] - 0.5 * (ft[1] - ft[0])
    edges[-1] = ft[-1] + 0.5 * (ft[-1] - ft[-2])

    idx = np.searchsorted(edges, et, side="right") - 1
    ok = (idx >= 0) & (idx < n)
    idx, E_raw = idx[ok], E_raw[ok]

    sums = np.zeros((n, len(TAXELS)))
    counts = np.zeros(n)
    np.add.at(sums, idx, E_raw)
    np.add.at(counts, idx, 1)

    have = counts > 0
    E = np.full((n, len(TAXELS)), np.nan)
    E[have] = sums[have] / counts[have, None]

    # Gaps (e-skin rate drift) -> nearest available frame in time.
    if (~have).any():
        good = np.where(have)[0]
        bad = np.where(~have)[0]
        nearest = good[np.abs(bad[:, None] - good[None, :]).argmin(axis=1)]
        E[bad] = E[nearest]

    F = force["F1_N"].to_numpy(float) + force["F2_N"].to_numpy(float)
    return E, ft, F, float(have.mean())


def estimate_lag_s(eskin_sum: np.ndarray, F: np.ndarray, max_lag_s: float = 1.0):
    """Cross-correlation lag between e-skin total and force, in seconds.

    Positive => e-skin leads force (shift e-skin forward to align).
    An uncorrected lag directly caps achievable R², so always check this.
    """
    a = (eskin_sum - eskin_sum.mean()) / (eskin_sum.std() + 1e-12)
    b = (F - F.mean()) / (F.std() + 1e-12)
    corr = signal.correlate(a, b, mode="full")
    lags = signal.correlation_lags(len(a), len(b), mode="full")
    m = np.abs(lags) <= int(max_lag_s * FORCE_HZ)
    return float(lags[m][np.argmax(corr[m])] / FORCE_HZ)


def subtract_baseline(E: np.ndarray) -> np.ndarray:
    """Per-taxel resting offset removal (5th percentile within the trial)."""
    return np.clip(E - np.percentile(E, 5, axis=0, keepdims=True), 0, None)


def scalar_features(E: np.ndarray, sat_value: float) -> pd.DataFrame:
    """Cheap interpretable summaries — the baseline any model must beat."""
    return pd.DataFrame({
        "total": E.sum(axis=1),
        "top16": np.sort(E, axis=1)[:, -16:].sum(axis=1),
        "area": (E > 0.05 * sat_value).sum(axis=1),
        "peak": E.max(axis=1),
        "sat_frac": (E >= 0.98 * sat_value).mean(axis=1),
    })
```

### `scripts/hour1/01_build_dataset.py`

```python
from pathlib import Path
import numpy as np
import pandas as pd

from common import (TAXELS, iter_target_force_trials, align_to_force_grid,
                    estimate_lag_s, subtract_baseline)

DATA = Path("data")
OUT = Path("reports/hour1"); OUT.mkdir(parents=True, exist_ok=True)

Xs, ys, trials, subjects, targets, times = [], [], [], [], [], []
report = []

for d, man in iter_target_force_trials(DATA):
    eskin = pd.read_csv(d / "eskin.csv")
    force = pd.read_csv(d / "forces.csv")
    E, t, F, cov = align_to_force_grid(eskin, force)
    lag = estimate_lag_s(E.sum(axis=1), F)
    E = subtract_baseline(E)

    Xs.append(E); ys.append(F); times.append(t)
    trials += [d.name] * len(F)
    subjects += [man.get("subject_id", "?")] * len(F)
    targets += [man.get("target_force_n", np.nan)] * len(F)
    report.append(dict(trial=d.name, subject=man.get("subject_id"),
                       target_n=man.get("target_force_n"), n=len(F),
                       coverage=round(cov, 3), lag_s=round(lag, 3),
                       F_min=round(F.min(), 2), F_max=round(F.max(), 2)))
    print(f"  {d.name}: n={len(F)} cov={cov:.2f} lag={lag:+.3f}s")

X = np.vstack(Xs); y = np.concatenate(ys)
np.savez_compressed(OUT / "dataset.npz", X=X, y=y,
                    trial=np.array(trials), subject=np.array(subjects),
                    target=np.array(targets, float), t=np.concatenate(times),
                    taxels=np.array(TAXELS))

df = pd.DataFrame(report)
df.to_csv(OUT / "trial_summary.csv", index=False)
print(df.to_string(index=False))
print(f"\nTotal paired samples: {len(y):,} across {len(df)} trials")
```

**Stop and check before continuing:**
- `coverage` should be ~1.0 for every trial. Below ~0.9 means the e-skin rate drifted badly in that trial — flag it.
- `lag_s` should be small and *consistent* across trials. If the median is meaningfully non-zero (say > 20 ms), re-run with e-skin shifted by that amount before proceeding — otherwise you are measuring alignment error, not model quality.
- Total should land near ~25,000 samples.

---

## 6. Step 2 — Two plots and a saturation number (10 min)

### `scripts/hour1/02_diagnose.py`

```python
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("reports/hour1")
d = np.load(OUT / "dataset.npz", allow_pickle=True)
X, y, trial = d["X"], d["y"], d["trial"]
names = list(dict.fromkeys(trial.tolist()))
total = X.sum(axis=1)

# --- Plot 1: does e-skin track force at all? (the single most useful plot)
fig, axes = plt.subplots(len(names), 1, figsize=(11, 2.0 * len(names)), sharex=False)
for ax, name in zip(np.atleast_1d(axes), names):
    m = trial == name
    ax.plot(y[m], lw=1, label="F_combined (N)")
    ax2 = ax.twinx()
    ax2.plot(total[m] / total[m].max(), lw=1, color="tab:orange", alpha=.7,
             label="e-skin total (norm)")
    ax.set_title(name, fontsize=8); ax.tick_params(labelsize=7)
fig.tight_layout(); fig.savefig(OUT / "fig_timeseries.png", dpi=110)

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
fig.tight_layout(); fig.savefig(OUT / "fig_hysteresis.png", dpi=110)

print("saturation fraction (taxels at >=98% of observed max):",
      round(float((X >= 0.98 * X.max()).mean()), 4))
for name in names:
    m = trial == name
    print(f"  {name}: r = {np.corrcoef(total[m], y[m])[0,1]:.3f}")
print(f"POOLED r = {np.corrcoef(total, y)[0,1]:.3f}")
```

**How to read this in 60 seconds:**
- `fig_timeseries.png` — do the orange and blue traces move together? If yes, feasibility is basically confirmed and the rest is quantification.
- **Per-trial `r` high but POOLED `r` much lower ⇒ Outcome B.** This is the tell, and you will see it here before you fit a single model.
- `fig_hysteresis.png` left panel — a wide loop between loading and unloading bounds how well any memoryless model can ever do.
- Right panel — if each trial forms its own separate line rather than one shared curve, per-trial gain/offset varies and Step 5's calibration fix is the answer.

---

## 7. Step 3–5 — Ceiling test, cross-trial test, and the calibration fix (25 min)

### `scripts/hour1/03_ceiling_test.py`

```python
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_absolute_error

OUT = Path("reports/hour1")
d = np.load(OUT / "dataset.npz", allow_pickle=True)
X, y, trial = d["X"], d["y"], d["trial"]
names = list(dict.fromkeys(trial.tolist()))
ALPHAS = np.logspace(-2, 5, 20)

def model():
    return make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))

def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))

def score(yt, yp):
    return dict(r2=round(r2_score(yt, yp), 3), rmse=round(rmse(yt, yp), 2),
                mae=round(mean_absolute_error(yt, yp), 2))

FEATS = {
    "raw256":  lambda Z: Z,
    "sqrt256": lambda Z: np.sqrt(Z),                 # linearises FSR response
    "total":   lambda Z: Z.sum(axis=1, keepdims=True) # scalar baseline
}

# ---------- Step 3: WITHIN-TRIAL CEILING (blocked 70/30 + gap) ----------
GAP = 100  # 1 s at 100 Hz — breaks autocorrelation across the split boundary
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
gkf = GroupKFold(n_splits=5)
oof_pred, oof_true, oof_trial = {}, None, None
for fname, fx in FEATS.items():
    pred = np.zeros_like(y)
    for tr, te in gkf.split(X, y, groups=trial):
        pred[te] = model().fit(fx(X[tr]), y[tr]).predict(fx(X[te]))
    rows.append(dict(features=fname, split="cross-trial", **score(y, pred)))
    oof_pred[fname] = pred

res = pd.DataFrame(rows)
print(res.to_string(index=False))

# ---------- Step 5: DOES A 5-SECOND CALIBRATION GRASP RESCUE IT? ----------
# For each held-out trial: use its first 5 s to fit y = a*pred + b, then score
# the REMAINDER with and without that correction. This directly simulates the
# per-session calibration you would do live.
CAL = 500  # 5 s at 100 Hz
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
print("\n--- per-trial affine recalibration ---")
print(cal.to_string(index=False))
print(f"\nmean R2  raw={cal.r2_raw.mean():.3f}  calibrated={cal.r2_cal.mean():.3f}")

with open(OUT / "results.md", "w") as fh:
    fh.write("# Hour-1 results\n\n## Model comparison\n\n")
    fh.write(res.to_markdown(index=False))
    fh.write("\n\n## Per-session affine recalibration\n\n")
    fh.write(cal.to_markdown(index=False))
```

Then plot predicted vs actual for the worst and best held-out trials (`fig_pred_vs_actual.png`) — **always trust the time-series plot over the aggregate R²**, since a good R² can hide a systematic lag or a saturating ceiling that is obvious visually.

---

## 8. Reading the results

Fill in the §1 table from `results.md` and report:

1. **Does `raw256` beat `total`?** If ridge on 256 taxels barely improves on the single summed scalar, the spatial pattern is not carrying much information — useful to know, and it means the visualiser can run on a trivially simple model.
2. **Within-trial vs cross-trial R²** — this is the outcome classifier.
3. **`r2_raw` vs `r2_cal`** — if the mean jumps substantially (e.g. 0.5 → 0.9), you have both diagnosed Outcome B *and* proven the fix. Design the live session to start with a short calibration squeeze at a known force, fit gain+offset, and apply it for the rest of the session.
4. **`gain`/`offset` spread across trials** — how much per-session variation you are actually correcting for.

---

## 9. Latency (do not spend time on this)

For the record, so nobody optimises the wrong thing: a ridge model on 256 inputs is a single dot product — well under a microsecond, and even tree ensembles and small nets land in the microsecond-to-low-millisecond range. **The model is never the bottleneck.** Real-time lag will come from serial polling of the 256 taxels, buffering, and any smoothing you apply.

Two rules for later:
- Do the serial polling on its own thread/process with a ring buffer, so acquisition jitter does not stall rendering.
- For smoothing, avoid long moving averages — a 20-sample window at 100 Hz adds ~100 ms of visible lag. Use a **One-Euro filter** (Casiez, Roussel & Vogel, CHI 2012), which smooths hard when force is steady and opens up when it changes fast, giving far less lag for the same jitter reduction.

---

## 10. Explicit non-goals for this hour

Do not build: temporal/lag-window models, CNNs, GRU/LSTM, `archive/` ingestion, `max_effort` handling, leave-one-subject-out (only 4 subjects with non-overlapping force ranges — it would confound subject with force range and mislead you), ONNX export, or any live streaming. If Steps 1–5 finish early, spend the remaining time on **more diagnostic plots**, not more models.

---

## 11. Report back with

- The filled §1 table and which outcome (A/B/C) applies.
- `reports/hour1/results.md` contents.
- Median lag across trials and any trial with coverage < 0.9.
- Saturation fraction in the target-force trials.
- One sentence: is this worth taking to live data, and what must change first?
