"""Loading, alignment, baseline, and idle-trimming helpers for the Hour-1
e-skin -> force feasibility pipeline. Self-contained (plain pd.read_csv)
rather than importing src/processing/align.py, which pulls in unrelated
EMG/ROI machinery this pipeline doesn't need.
"""
from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"

TAXELS = [f"R{r:02d}_C{c:02d}" for r in range(16) for c in range(16)]
FORCE_HZ = 100.0

# task_kind == "target_force" trials, capped at <=45N per user's override
# (60N/75N trials and all max_effort trials excluded); PT_target_45_003 is
# INCLUDED (the source doc skips it for its long stall -- we keep it and
# trim the stall out instead, see trim_idle_mask below).
DATA_TRIALS = [
    "P1_002_20260721_155648",
    "narges_target_10_002_20260722_173151",
    "P1_003_20260721_155726",
    "narges_target_15_003_20260722_173534",
    "PT_target_15_001_20260722_153324",
    "P1_002_20260721_161939",
    "AM_hold_001_20260721_165006",
    "PT_target_30_002_20260722_153507",
    "PT_target_45_003_20260722_153749",
    "PT_target_45_004_20260722_154404",
]
# Legacy pre-manifest sessions from archive/ -- no task/rep metadata.
ARCHIVE_TRIALS = ["archive_165516", "archive_173729"]

# Dynamic-grasping trials split out of the single long
# YL_grasp_dynamic_002_20260723_155859 recording (see
# 05_split_dynamic_session.py) at rest-gap boundaries -- no manifest, same
# shape as the archive/ sessions. Recorded through the current GUI, whose
# ForceReader already applies config/force_bias_calibration.json's BIAS1/
# BIAS2 at acquisition time, so (unlike ARCHIVE_TRIALS) no bias correction
# is applied again here.
DYNAMIC_TRIALS = sorted(p.name for p in DATA.glob("YL_dynamic_*") if p.is_dir())

REP_PAD_S = 2.0          # padding kept around each labelled rep window
ARCHIVE_ACTIVE_MARGIN = 3.0   # N above resting baseline to call "active"
ARCHIVE_ACTIVE_PAD_S = 2.0


def _load_bias():
    with open(DATA / "force_bias_calibration.json") as fh:
        return json.load(fh)


BIAS = _load_bias()


def iter_trials():
    """Yield (trial_id, eskin_df, force_df, manifest_or_None) for every
    trial in the Hour-1 set (10 data/ + 2 archive/), forces bias-corrected
    for archive sessions."""
    for t in DATA_TRIALS:
        d = DATA / t
        man = json.loads((d / "manifest.json").read_text())
        eskin = pd.read_csv(d / "eskin.csv")
        force = pd.read_csv(d / "forces.csv")
        yield t, eskin, force, man

    for t in ARCHIVE_TRIALS:
        d = DATA / t
        eskin = pd.read_csv(d / "eskin.csv")
        force = pd.read_csv(d / "forces.csv").copy()
        # archive/summer_school_project 2.py's raw_to_newton() bakes in
        # whatever BIAS1/BIAS2 was active at recording time (default 0.0
        # before any "Calibrate Zero" click); apply the saved calibration
        # on top per the user's instruction.
        force["F1_N"] = force["F1_N"] - BIAS["BIAS1"]
        force["F2_N"] = force["F2_N"] - BIAS["BIAS2"]
        yield t, eskin, force, None

    for t in DYNAMIC_TRIALS:
        d = DATA / t
        eskin = pd.read_csv(d / "eskin.csv")
        force = pd.read_csv(d / "forces.csv")
        yield t, eskin, force, None


def rep_windows_s(manifest: dict) -> list:
    """Rep windows as (rep_no, start_s, end_s) in seconds from trial start.
    Same logic as src/processing/align.py's _rep_windows_s."""
    origin = datetime.fromisoformat(manifest["start_wall_time"])
    windows = []
    for rep in manifest.get("repetitions", []):
        start = (datetime.fromisoformat(rep["start_wall_time"]) - origin).total_seconds()
        end = (datetime.fromisoformat(rep["end_wall_time"]) - origin).total_seconds()
        windows.append((rep["rep"], start, end))
    return windows


def align_to_force_grid(eskin: pd.DataFrame, force: pd.DataFrame):
    """Average all e-skin frames falling within each force sample's window.

    Returns (E, t, F, coverage) where E is (n_force, 256) and coverage is the
    fraction of force samples that had at least one real e-skin frame.
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

    if (~have).any():
        good = np.where(have)[0]
        bad = np.where(~have)[0]
        nearest = good[np.abs(bad[:, None] - good[None, :]).argmin(axis=1)]
        E[bad] = E[nearest]

    F = force["F1_N"].to_numpy(float) + force["F2_N"].to_numpy(float)
    return E, ft, F, float(have.mean())


def estimate_lag_s(eskin_sum: np.ndarray, F: np.ndarray, max_lag_s: float = 1.0) -> float:
    """Cross-correlation lag between e-skin total and force, in seconds.
    Positive => e-skin leads force (shift e-skin forward to align)."""
    a = (eskin_sum - eskin_sum.mean()) / (eskin_sum.std() + 1e-12)
    b = (F - F.mean()) / (F.std() + 1e-12)
    corr = signal.correlate(a, b, mode="full")
    lags = signal.correlation_lags(len(a), len(b), mode="full")
    m = np.abs(lags) <= int(max_lag_s * FORCE_HZ)
    return float(lags[m][np.argmax(corr[m])] / FORCE_HZ)


def subtract_baseline(E: np.ndarray) -> np.ndarray:
    """Per-taxel resting offset removal (5th percentile over the WHOLE
    trial, i.e. before idle-trimming -- trimming would remove the very
    rest stretches this percentile relies on)."""
    return np.clip(E - np.percentile(E, 5, axis=0, keepdims=True), 0, None)


def trim_idle_mask(t: np.ndarray, F: np.ndarray, manifest: dict | None) -> np.ndarray:
    """Boolean keep-mask dropping long no-grasp/baseline stretches.

    data/ trials (have a manifest): keep each labelled rep window +/- REP_PAD_S.
    archive/ trials (no manifest): no rep boundaries exist, so instead keep
    windows where F_combined is more than ARCHIVE_ACTIVE_MARGIN N above the
    trial's resting baseline (5th percentile of F), +/- ARCHIVE_ACTIVE_PAD_S.
    """
    if manifest is not None:
        reps = rep_windows_s(manifest)
        mask = np.zeros_like(t, dtype=bool)
        for _, start_s, end_s in reps:
            mask |= (t >= start_s - REP_PAD_S) & (t <= end_s + REP_PAD_S)
        return mask

    rest = np.percentile(F, 5)
    active = F > (rest + ARCHIVE_ACTIVE_MARGIN)
    if not active.any():
        return active
    pad_n = int(round(ARCHIVE_ACTIVE_PAD_S * FORCE_HZ))
    idx = np.where(active)[0]
    mask = np.zeros_like(active)
    for i in idx:
        mask[max(0, i - pad_n): min(len(mask), i + pad_n + 1)] = True
    return mask


class FloorScaler:
    """StandardScaler with a variance floor.

    Plain StandardScaler blows up on this data: many taxel columns are
    essentially-but-not-exactly zero within a single trial's training rows
    (float noise ~1e-13 from the baseline subtraction), so the column std
    is near machine epsilon; any real signal appearing in that column at
    test/predict time then gets scaled to ~1e14 and Ridge's (small but
    nonzero) coefficient on it produces predictions in the trillions. A
    floor keeps near-constant columns from being scaled up out of all
    proportion.
    """
    def __init__(self, floor: float = 5.0):
        self.floor = floor

    def fit(self, X, y=None):
        self.mean_ = X.mean(axis=0)
        self.scale_ = np.maximum(X.std(axis=0), self.floor)
        return self

    def transform(self, X):
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


def scalar_features(E: np.ndarray, sat_value: float) -> pd.DataFrame:
    """Cheap interpretable summaries."""
    return pd.DataFrame({
        "total": E.sum(axis=1),
        "top16": np.sort(E, axis=1)[:, -16:].sum(axis=1),
        "area": (E > 0.05 * sat_value).sum(axis=1),
        "peak": E.max(axis=1),
        "sat_frac": (E >= 0.98 * sat_value).mean(axis=1),
    })
