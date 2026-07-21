"""Post-hoc processing of recorded e-skin CSVs (wall_time, elapsed_s, plus
256 R{row:02d}_C{col:02d} taxel columns -- see SessionRecorder._write_eskin_csv)."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

GRID_N = 16
CONTACT_THRESHOLD = 5.0  # taxel counts (post-tare) considered "in contact"

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
