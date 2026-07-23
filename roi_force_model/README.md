# e-skin ROI → grip force (F1+F2)

Predict combined grip force `F_combined = F1_N + F2_N` from the e-skin
**contact-ROI reading** at each timestep. Dependency-free — uses only the
repo's `.venv` (numpy / scipy / pandas / matplotlib); ridge is closed-form and
the model is saved as plain JSON. **No scikit-learn / joblib needed.**

Built on the two YL trials:
- `data/YL_grasp_dynamic_002_20260723_155859` — free-form dynamic grasp
  (~5 min, force sweeps 0–80 N). **Training trial** (only one with real range).
- `data/YL_max_squeeze_001_20260723_155707` — 5× max squeeze (mostly saturated).

## The one thing to know: the e-skin saturates ~30 N

Above ~30 N of grip force the ROI reading is **flat** — mean, peak, and even
the single hottest cell stop responding (corr with force ≈ 0). So force is
physically **unresolvable** above that knee; it is a hardware ceiling, not a
model weakness. We therefore:
1. **Train only on below-knee samples** (< 30 N) — saturated samples otherwise
   corrupt the fit; and
2. **Gate at inference on the e-skin reading itself** (`roi_mean ≥ ~203 ⇒
   saturated, unresolved`) — at deployment there's no force label to know what's
   in range, so the gate reads the sensor, not the answer.

Below the knee, on **held-out** samples (unseen in time), the model predicts
force to **RMSE ≈ 5 N** — trustworthy as a soft/medium force estimate.

## What "ROI reading" means

Summing all 256 taxels drowns the contact in baseline noise, so we isolate the
**contact patch (ROI)** — the largest connected cluster of high-peak cells
(reuses `src/processing/eskin.detect_roi`, per PROJECT.md). Per-frame features,
all over the ROI only: `roi_mean`, `roi_area` (cells in contact), `roi_peak`,
`roi_p90` — ROI-size-normalised so they transfer across trials — expanded to a
degree-2 polynomial, then closed-form ridge.

## Run

```bash
# from repo root, in the project .venv
python roi_force_model/train.py       # -> outputs/roi_force_model.json + metrics
python roi_force_model/figures.py     # -> outputs/slide_1|2|3_*.png (for slides)
python roi_force_model/predict.py data/<trial_id>          # inference (honors the gate)
python roi_force_model/predict.py data/<trial_id> --csv out.csv
```

Inference from code:
```python
from roi_force_model.pipeline import build_samples, load_model, predict_force
feats, F, t, aux = build_samples("data/<trial_id>")   # align + baseline + ROI + features
pred_N = predict_force(load_model("roi_force_model/outputs/roi_force_model.json"), feats)
```

**Per-trial preprocessing is required.** The model was trained on
baseline-subtracted ROI features, not raw counts. `build_samples` re-estimates
the per-taxel resting baseline **and** the contact ROI on the new trial (taxel
offsets and grip location drift between recordings), so give it a full trial
folder, not a single frame. `forces.csv` is used only for the time grid +
optional scoring.

## Files

| file | role |
|------|------|
| `pipeline.py` | load / align (200 Hz e-skin → 100 Hz force) / baseline / ROI / features / ridge / save-load |
| `train.py`    | train the below-knee gated model, time-blocked held-out eval (`build_and_eval` is the single source of truth for the numbers) |
| `figures.py`  | 3 slide figures (saturation, accuracy, ROI explainer) |
| `predict.py`  | CLI inference; flags saturated samples as unresolved |
| `outputs/`    | `roi_force_model.json` + slide PNGs |

## Caveats

- **One subject (YL).** Cross-subject generalisation is untested — needs a few
  more dynamic trials across hands + leave-one-subject-out before trusting on a
  new person.
- **Hysteresis.** The ROI→force map differs on loading vs unloading, so the
  gate occasionally lets a genuinely-high-force sample through on fast release.
- The full range (past the knee) needs a **controlled calibration ramp**, not
  more grasp trials — see PROJECT.md.
