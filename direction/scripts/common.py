"""Shared constants and segmentation/averaging helpers for the direction-clustering
pipeline. Self-contained (plain pd.read_csv), independent of regression/scripts/ and
src/processing/align.py, so the two pipelines stay decoupled.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]          # direction/
DATA = Path(__file__).resolve().parents[2] / "data"  # top-level data/, not direction/data
REPORTS = ROOT / "reports"

TAXELS = [f"R{r:02d}_C{c:02d}" for r in range(16) for c in range(16)]

# trial_id -> (axis, sign)
TRIALS = {
    "YL_positive_x_001_20260723_172701": ("x", 1),
    "YL_negative_x_001_20260723_171051": ("x", -1),
    "YL_positive_y_003_20260723_162436": ("y", 1),
    "YL_negative_y_002_20260723_162231": ("y", -1),
    "YL_positive_z_001_20260723_161834": ("z", 1),
    "YL_negative_z_001_20260723_162033": ("z", -1),
}

SMOOTH_WINDOW_ESKIN = 41       # ~0.2s @ ~200Hz eskin
ESKIN_REST_MARGIN_FRAC = 0.08  # fraction of trial's (p95-p5) eskin-total range that counts as "resting"
ESKIN_ACTIVE_MIN_FRAC = 0.15   # segment must peak above baseline+this*range to count as a real attempt
MIN_GAP_S = 0.5                # min duration of a resting run to count as a real inter-attempt gap

# Explicit per-trial segmentation overrides, keyed by trial_id. Only used when
# 01_segment_attempts.py flags a trial as likely mis-segmented; never applied
# silently. Supported keys: "rest_frac", "active_frac" (override the defaults
# above for that trial_id).
OVERRIDES: dict[str, dict] = {}


def direction_label(axis: str, sign: int) -> str:
    return f"{'positive' if sign > 0 else 'negative'}_{axis}"


def _resting_runs(resting: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous True runs in a boolean array, as (start_idx, end_idx) half-open."""
    n = len(resting)
    runs = []
    i = 0
    while i < n:
        if resting[i]:
            j = i
            while j < n and resting[j]:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


def find_segments_eskin(
    et: np.ndarray,
    Etot: np.ndarray,
    min_gap_s: float = MIN_GAP_S,
    rest_frac: float = ESKIN_REST_MARGIN_FRAC,
    smooth_window: int = SMOOTH_WINDOW_ESKIN,
):
    """Rest-gap segmentation driven only by e-skin-total (force is not used --
    it was found to under-report some attempts whose e-skin signal was clearly
    a normal press/release, see direction/reports/results.md). Mirrors
    regression/scripts/05_split_dynamic_session.py's approach, applied to
    e-skin-total instead of force, with a relative (fraction-of-range) margin
    since e-skin-total's absolute resting level varies hugely trial to trial.

    Returns (segments, baseline, rng) where baseline = 5th percentile and
    rng = (95th - 5th) percentile of the smoothed signal. Each segment dict
    has idx_a, idx_b, t0, t1, duration, Etot_max, Etot_min (indices into the
    eskin grid).
    """
    se = pd.Series(Etot).rolling(smooth_window, center=True, min_periods=1).median().to_numpy()
    p5, p95 = np.percentile(se, 5), np.percentile(se, 95)
    baseline = float(p5)
    rng = float(p95 - p5)
    resting = se <= baseline + rest_frac * rng

    runs = _resting_runs(resting)
    gaps = [r for r in runs if (et[r[1] - 1] - et[r[0]]) >= min_gap_s]

    n = len(et)
    edges = [0] + [(g[0] + g[1]) // 2 for g in gaps] + [n]

    segments = []
    for k in range(len(edges) - 1):
        a, b = edges[k], edges[k + 1]
        if b <= a:
            continue
        seg_E = Etot[a:b]
        segments.append(dict(idx_a=a, idx_b=b, t0=float(et[a]), t1=float(et[b - 1]),
                              duration=float(et[b - 1] - et[a]), n=b - a,
                              Etot_max=float(seg_E.max()), Etot_min=float(seg_E.min())))
    return segments, baseline, rng


def active_window(t: np.ndarray, signal: np.ndarray, idx_a: int, idx_b: int,
                   threshold: float):
    """Within [idx_a, idx_b), find the first/last index where signal clears
    threshold. Returns (t_active_start, t_active_end, n_active) or None if
    nothing in the window clears the threshold."""
    seg = signal[idx_a:idx_b]
    active = seg > threshold
    if not active.any():
        return None
    idx = np.where(active)[0]
    lo, hi = idx[0], idx[-1]
    return float(t[idx_a + lo]), float(t[idx_a + hi]), int(hi - lo + 1)


def attempt_heatmap(eskin_df: pd.DataFrame, taxel_baseline: np.ndarray,
                     t_active_start: float, t_active_end: float) -> tuple[np.ndarray, int]:
    """Mean of taxel columns over rows with elapsed_s in [t_active_start,
    t_active_end], minus the trial's whole-trial per-taxel baseline, clipped
    at 0. Returns (feature_256, n_frames_used)."""
    et = eskin_df["elapsed_s"].to_numpy(float)
    mask = (et >= t_active_start) & (et <= t_active_end)
    n = int(mask.sum())
    mean_frame = eskin_df.loc[mask, TAXELS].to_numpy(float).mean(axis=0)
    feature = np.clip(mean_frame - taxel_baseline, 0, None)
    return feature, n
