"""Shared pipeline for the e-skin ROI -> grip-force (F1+F2) regression.

Dependency-free by design: numpy / scipy / pandas / matplotlib only (no
scikit-learn / joblib), so it runs in the repo's .venv as-is. Ridge
regression is implemented in closed form and models are serialized to plain
JSON (see save_model / load_model), which doubles as human-readable model
documentation.

What lives here:
  * load_trial            -- read a data/<trial_id>/ folder
  * align_to_force_grid   -- pair ~200 Hz e-skin with 100 Hz force (box-average)
  * subtract_baseline     -- per-taxel resting-offset removal (per session)
  * detect ROI            -- reuses src/processing/eskin.detect_roi
  * roi_base_features     -- the ROI "reading": mean / area / peak / p90 / sum
  * poly2 + RidgeNP       -- feature expansion + closed-form ridge
  * metrics / save / load / predict_force

The ROI is the *contact patch* (largest connected cluster of high-peak
cells), not the full 16x16 grid -- see PROJECT.md ("E-skin: ROI, not
full-grid sum"). The model's features are ROI-size-normalised (mean, area,
peak, percentile) so they transfer across trials whose ROI covers a
different number of cells.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# repo root = parent of roi_force_model/ -- put it on sys.path so we can reuse
# the canonical ROI detector rather than re-implementing it.
ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.processing.eskin import detect_roi  # noqa: E402

DATA = ROOT / "data"
TAXELS = [f"R{r:02d}_C{c:02d}" for r in range(16) for c in range(16)]
FORCE_HZ = 100.0
CONTACT_THR = 5.0        # baseline-subtracted taxel counts considered "in contact"
BASELINE_PCT = 5.0       # per-taxel resting offset percentile
BASE_FEATURES = ["roi_mean", "roi_area", "roi_peak", "roi_p90"]


# --------------------------------------------------------------------------- #
# loading + alignment
# --------------------------------------------------------------------------- #
def load_trial(trial_dir: Path):
    """Return (trial_id, eskin_df, force_df, manifest_or_None)."""
    trial_dir = Path(trial_dir)
    man_path = trial_dir / "manifest.json"
    man = json.loads(man_path.read_text()) if man_path.exists() else None
    eskin = pd.read_csv(trial_dir / "eskin.csv")
    force = pd.read_csv(trial_dir / "forces.csv")
    tid = man["trial_id"] if man else trial_dir.name
    return tid, eskin, force, man


def align_to_force_grid(eskin: pd.DataFrame, force: pd.DataFrame):
    """Average all e-skin frames inside each force sample's time window.

    e-skin is ~200 Hz, force is 100 Hz; both share the trial-start elapsed_s
    origin. Returns (E, t, F, coverage): E is (n_force, 256) taxel readings on
    the force time grid, t is force elapsed_s, F = F1_N + F2_N, coverage is the
    fraction of force samples that had >=1 real e-skin frame (rest are
    nearest-filled). Mirrors regression/scripts/common.align_to_force_grid.
    """
    et = eskin["elapsed_s"].to_numpy(float)
    ft = force["elapsed_s"].to_numpy(float)
    E_raw = eskin[TAXELS].to_numpy(float)
    n = len(ft)

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
    if (~have).any():                       # nearest-fill force samples with no frame
        good = np.where(have)[0]
        bad = np.where(~have)[0]
        E[bad] = E[good[np.abs(bad[:, None] - good[None, :]).argmin(axis=1)]]

    F = force["F1_N"].to_numpy(float) + force["F2_N"].to_numpy(float)
    return E, ft, F, float(have.mean())


def subtract_baseline(E: np.ndarray) -> np.ndarray:
    """Per-taxel resting-offset removal (5th-percentile over the trial), clipped
    at 0. Computed per session -- taxel rest offsets drift between recordings,
    so this is a required per-trial step at both train and inference time."""
    return np.clip(E - np.percentile(E, BASELINE_PCT, axis=0, keepdims=True), 0, None)


# --------------------------------------------------------------------------- #
# ROI + features
# --------------------------------------------------------------------------- #
def roi_from_trial(E_baseline: np.ndarray):
    """Detect the whole-trial contact ROI from baseline-subtracted readings.

    Returns (roi_mask (16,16) bool, peak_map (16,16)). For a single continuous
    grasp one trial-wide ROI is correct (the hand grips one patch); for
    multi-rep trials the grip can shift, but a trial-wide ROI is still a fine
    superset for a per-sample force model.
    """
    frames = E_baseline.reshape(-1, 16, 16)
    return detect_roi(frames)


def roi_base_features(E_baseline: np.ndarray, roi_mask: np.ndarray) -> pd.DataFrame:
    """Per-frame ROI 'reading'. All are computed over the ROI cells only.

    roi_mean / roi_peak / roi_p90 are ROI-size-invariant (comparable across
    trials with different ROI sizes); roi_area is the dynamic in-contact cell
    count within the ROI (force = pressure x area, so area carries independent
    information once pressure saturates). roi_sum is included for plotting /
    the naive single-feature baseline, not the primary feature set.
    """
    frames = E_baseline.reshape(-1, 16, 16)
    if not roi_mask.any():                  # degenerate: no contact detected
        roi_mask = np.ones((16, 16), bool)
    roi_vals = frames[:, roi_mask]          # (n, n_roi_cells)
    return pd.DataFrame({
        "roi_mean": roi_vals.mean(axis=1),
        "roi_area": (roi_vals > CONTACT_THR).sum(axis=1).astype(float),
        "roi_peak": roi_vals.max(axis=1),
        "roi_p90": np.percentile(roi_vals, 90, axis=1),
        "roi_sum": roi_vals.sum(axis=1),
    })


def build_samples(trial_dir: Path):
    """End-to-end: folder -> (base_features_df, F, t, info). Runs align ->
    baseline -> ROI -> ROI features. `info` has coverage, roi_cells, span."""
    tid, eskin, force, man = load_trial(trial_dir)
    E, t, F, cov = align_to_force_grid(eskin, force)
    Eb = subtract_baseline(E)
    roi, peak = roi_from_trial(Eb)
    feats = roi_base_features(Eb, roi)
    info = dict(trial_id=tid, coverage=cov, roi_cells=int(roi.sum()),
                span_s=float(t[-1] - t[0]), n=len(F), task_kind=(man or {}).get("task_kind"))
    return feats, F, t, dict(info=info, roi_mask=roi, peak_map=peak,
                             E_baseline=Eb, manifest=man)


# --------------------------------------------------------------------------- #
# feature expansion + closed-form ridge
# --------------------------------------------------------------------------- #
def poly2(X: np.ndarray, names: list[str]):
    """Degree-2 expansion: raw + squares + pairwise interactions (no bias).
    Returns (Xexp, expanded_names)."""
    cols, cnames = [X], list(names)
    for i in range(X.shape[1]):             # squares
        cols.append((X[:, i] ** 2)[:, None]); cnames.append(f"{names[i]}^2")
    for i in range(X.shape[1]):             # pairwise interactions
        for j in range(i + 1, X.shape[1]):
            cols.append((X[:, i] * X[:, j])[:, None]); cnames.append(f"{names[i]}*{names[j]}")
    return np.hstack(cols), cnames


class RidgeNP:
    """Closed-form ridge with a floored standard-scaler front-end.

    The scale floor stops near-constant expanded columns (float noise after
    baseline subtraction) from being blown up to huge magnitudes -- the same
    failure the regression/ pipeline's FloorScaler guards against.
    """
    def __init__(self, alpha: float = 1.0, scale_floor: float = 1e-6):
        self.alpha = float(alpha)
        self.scale_floor = float(scale_floor)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.mean_ = X.mean(axis=0)
        self.scale_ = np.maximum(X.std(axis=0), self.scale_floor)
        Xs = (X - self.mean_) / self.scale_
        self.intercept_ = float(y.mean())
        yc = y - self.intercept_
        n_feat = Xs.shape[1]
        A = Xs.T @ Xs + self.alpha * np.eye(n_feat)
        self.coef_ = np.linalg.solve(A, Xs.T @ yc)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xs = (X - self.mean_) / self.scale_
        return Xs @ self.coef_ + self.intercept_


# --------------------------------------------------------------------------- #
# metrics + serialization
# --------------------------------------------------------------------------- #
def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return dict(
        r2=round(1 - ss_res / ss_tot if ss_tot > 0 else float("nan"), 4),
        rmse=round(float(np.sqrt(np.mean((y_true - y_pred) ** 2))), 3),
        mae=round(float(np.mean(np.abs(y_true - y_pred))), 3),
        n=int(len(y_true)),
    )


def save_model(path: Path, ridge: RidgeNP, base_features: list[str],
               expanded_names: list[str], extra: dict):
    """Serialize a trained model to JSON (weights + full preprocessing spec)."""
    doc = {
        "model": "ridge-poly2 on e-skin ROI features -> F1+F2 (Newtons)",
        "base_features": base_features,
        "expanded_features": expanded_names,
        "alpha": ridge.alpha,
        "scaler_mean": ridge.mean_.tolist(),
        "scaler_scale": ridge.scale_.tolist(),
        "coef": ridge.coef_.tolist(),
        "intercept": ridge.intercept_,
        "preprocessing": {
            "1_align": "average e-skin frames onto the 100 Hz force grid",
            "2_baseline": f"subtract per-taxel {BASELINE_PCT}th-percentile (per trial), clip>=0",
            "3_roi": "detect_roi: largest connected cluster of high-peak cells",
            "4_features": "ROI mean/area/peak/p90, then degree-2 poly + this scaler",
            "note": "baseline + ROI are re-estimated per trial at inference time",
        },
        "contact_thr": CONTACT_THR,
        **extra,
    }
    Path(path).write_text(json.dumps(doc, indent=2))
    return doc


def load_model(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def predict_force(model: dict, base_feats: pd.DataFrame) -> np.ndarray:
    """Apply a loaded JSON model to freshly-extracted base ROI features."""
    X = base_feats[model["base_features"]].to_numpy(float)
    Xexp, _ = poly2(X, model["base_features"])
    Xs = (Xexp - np.array(model["scaler_mean"])) / np.array(model["scaler_scale"])
    return Xs @ np.array(model["coef"]) + model["intercept"]
