"""Post-hoc processing of recorded e-skin CSVs (wall_time, elapsed_s, plus
256 R{row:02d}_C{col:02d} taxel columns -- see SessionRecorder._write_eskin_csv)."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

GRID_N = 16
CONTACT_THRESHOLD = 5.0  # taxel counts (post-tare) considered "in contact"
ROI_ACT_FRAC = 0.25      # a cell joins the ROI if its peak >= this * grid peak

_COLUMNS = [f"R{r:02d}_C{c:02d}" for r in range(GRID_N) for c in range(GRID_N)]


@dataclass
class EskinFeatures:
    peak_total_load: float
    peak_contact_area_cells: int
    centroid_row: float
    centroid_col: float


def load_eskin_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def frame_matrix(row: pd.Series) -> np.ndarray:
    return row[_COLUMNS].to_numpy(dtype=float).reshape(GRID_N, GRID_N)


def frames_array(df: pd.DataFrame) -> np.ndarray:
    """All taxel frames as a ``(n_frames, 16, 16)`` array."""
    return df[_COLUMNS].to_numpy(dtype=float).reshape(-1, GRID_N, GRID_N)


# ---------------------------------------------------------------------------
# Region-of-interest (ROI) detection
#
# Summing all 256 taxels drowns a real contact in baseline noise from cells
# that never touch anything. Instead we isolate the actual contact patch.
#
# Metric = per-cell PEAK pressure (95th percentile over the window's frames):
# untouched cells read ~0, contact cells reach high pressure, so this
# separates contact from no-contact for BOTH transient (squeeze) and
# sustained (hold) grasps. Temporal std is deliberately NOT used -- steady
# contact in a hold barely varies, so std would pick noisy/flaky cells and
# miss the real contact. A largest-connected-component step then drops
# isolated faulty/edge cells, keeping the one contiguous patch.
# ---------------------------------------------------------------------------

def _largest_component(binary: np.ndarray) -> np.ndarray:
    """Keep only the largest 4-connected cluster of True cells."""
    visited = np.zeros_like(binary, dtype=bool)
    best: list = []
    for r in range(binary.shape[0]):
        for c in range(binary.shape[1]):
            if binary[r, c] and not visited[r, c]:
                stack, comp = [(r, c)], []
                visited[r, c] = True
                while stack:
                    y, x = stack.pop()
                    comp.append((y, x))
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if (0 <= ny < binary.shape[0] and 0 <= nx < binary.shape[1]
                                and binary[ny, nx] and not visited[ny, nx]):
                            visited[ny, nx] = True
                            stack.append((ny, nx))
                if len(comp) > len(best):
                    best = comp
    mask = np.zeros_like(binary, dtype=bool)
    for y, x in best:
        mask[y, x] = True
    return mask


def detect_roi(frames: np.ndarray, act_frac: float = ROI_ACT_FRAC):
    """ROI = largest contiguous cluster of high-peak-pressure cells.

    ``frames``: ``(n, 16, 16)``. Returns ``(roi_mask (16,16) bool, peak_map
    (16,16))`` where ``peak_map`` is the per-cell 95th-percentile pressure.
    """
    peak = np.percentile(frames, 95, axis=0)
    if peak.max() <= 0:
        return np.zeros_like(peak, dtype=bool), peak
    candidates = peak >= act_frac * peak.max()
    return _largest_component(candidates), peak


def detect_roi_per_rep(frames: np.ndarray, t: np.ndarray, reps,
                       act_frac: float = ROI_ACT_FRAC):
    """One ROI per rep window -- each rep is a fresh grip that may land on a
    different patch, so a single trial-wide ROI would be wrong.

    ``reps``: iterable of ``(rep_no, start_s, end_s)`` in the same time base as
    ``t``. Returns ``[(rep_no, mask, peak_map), ...]``. Falls back to a single
    whole-trial ROI (rep_no=0) if ``reps`` is empty.
    """
    if not reps:
        mask, peak = detect_roi(frames, act_frac)
        return [(0, mask, peak)]
    out = []
    for rep_no, start_s, end_s in reps:
        sel = (t >= start_s) & (t <= end_s)
        if sel.sum() < 2:                       # no e-skin frames in this window
            zero = np.zeros(frames.shape[1:])
            out.append((rep_no, zero.astype(bool), zero))
        else:
            mask, peak = detect_roi(frames[sel], act_frac)
            out.append((rep_no, mask, peak))
    return out


def roi_signal(frames: np.ndarray, t: np.ndarray, reps, rep_rois):
    """Per-frame e-skin scalar: sum over the ROI of the rep each frame belongs
    to; rest frames use the union of all rep ROIs (they read ~0 anyway).

    Returns ``(signal (n,), roi_union (16,16) bool)``.
    """
    union = np.zeros(frames.shape[1:], dtype=bool)
    for _, mask, _ in rep_rois:
        union |= mask
    mask_by_rep = {rep_no: mask for rep_no, mask, _ in rep_rois}
    fallback = union if union.any() else np.ones(frames.shape[1:], dtype=bool)

    signal = np.empty(frames.shape[0])
    for i in range(frames.shape[0]):
        mask = fallback
        for rep_no, start_s, end_s in reps:
            if start_s <= t[i] <= end_s and mask_by_rep.get(rep_no, np.zeros(1)).any():
                mask = mask_by_rep[rep_no]
                break
        signal[i] = frames[i][mask].sum()
    return signal, union


def extract_features(df: pd.DataFrame, contact_threshold: float = CONTACT_THRESHOLD) -> EskinFeatures:
    values = df[_COLUMNS].to_numpy(dtype=float)  # (n_frames, 256)
    totals = values.sum(axis=1)
    contact_counts = (values > contact_threshold).sum(axis=1)

    peak_idx = int(np.argmax(totals))
    peak_frame = values[peak_idx].reshape(GRID_N, GRID_N)
    rows, cols = np.nonzero(peak_frame > contact_threshold)
    if len(rows):
        weights = peak_frame[rows, cols]
        centroid_row = float(np.average(rows, weights=weights))
        centroid_col = float(np.average(cols, weights=weights))
    else:
        centroid_row = centroid_col = float("nan")

    return EskinFeatures(
        peak_total_load=float(totals.max()),
        peak_contact_area_cells=int(contact_counts.max()),
        centroid_row=centroid_row,
        centroid_col=centroid_col,
    )
