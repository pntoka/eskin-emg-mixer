"""Trial-level temporal alignment of the three recorded streams -- e-skin,
force, and EMG -- onto one common time origin (the trial start), keeping each
stream at its NATIVE sample rate. This is *alignment*, not fusion: no
resampling onto a shared grid and no merged table (see ``correlate.py`` for
that). The goal is an inspectable, per-stream view so the streams can be
compared and validated against the labelled rep windows.

Common time base
----------------
``forces.csv`` and ``eskin.csv`` both carry ``elapsed_s`` measured from the
same trial-start instant (SessionRecorder sets one origin), so we use
``elapsed_s`` directly. The EMG txt has no timestamps, so it is anchored at
elapsed 0 (trial start) and timed off its nominal 2000 Hz rate. The manifest's
rep windows (wall-clock) are converted to elapsed seconds relative to
``start_wall_time`` -- the same origin -- so everything shares one axis.

Sync caveat: EMG and the e-skin/force boards are independently triggered, so
sample-perfect sync is not guaranteed. :func:`anchoring_offset` reports the
residual EMG-vs-rep-window offset as a validation, not a correction.

Force merge: F1 and F2 are the two handle load cells; they are SUMMED into
``F_combined`` (they act on opposite sides of the grip and add to the total
grasp force), following ``forces.load_forces_csv``.
"""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import eskin as eskin_proc
from . import emg_txt
from . import forces as forces_proc
from .emg_c3d import rms_envelope

# EMG channel-quality thresholds (WaveX, uV)
EMG_RAIL = 3290.0        # |value| at/above this == amplifier rail
EMG_CLIP_MAX = 0.15      # reject channels railing more than this fraction
EMG_REP_CORR_MIN = 0.30  # min correlation with the rep on/off pattern to be "real"
ENV_WINDOW_S = 0.10      # EMG envelope smoothing window
GAP_THRESHOLD_S = 1.0    # a stream is "gappy" if consecutive samples exceed this


@dataclass
class AlignedTrial:
    trial_dir: Path
    manifest: dict
    reps: list                        # [(rep_no, start_s, end_s), ...] in elapsed seconds

    # EMG (native rate; may be absent)
    emg_present: bool
    emg_t: np.ndarray                 # (N,) seconds from trial start
    emg: np.ndarray                   # (n_ch, N) raw uV
    emg_channel_names: list
    emg_selected: list                # indices of real EMG channels
    emg_scores: list                  # [(ch, clip_frac, rep_corr), ...]
    anchor_offset_s: float            # EMG-vs-rep-window residual (0 = well anchored)

    # Force (native rate)
    force_t: np.ndarray               # (M,) elapsed seconds
    f1: np.ndarray
    f2: np.ndarray
    f_combined: np.ndarray            # F1 + F2

    # E-skin (native rate)
    eskin_t: np.ndarray               # (K,) elapsed seconds
    eskin_frames: np.ndarray          # (K, 16, 16)
    rep_rois: list                    # [(rep_no, mask(16,16), peak(16,16)), ...]
    roi_union: np.ndarray             # (16, 16) bool
    eskin_roi: np.ndarray             # (K,) per-frame sum over its rep's ROI

    # Coverage flags
    force_coverage: str
    eskin_coverage: str

    @property
    def sample_rate_hz(self) -> float:
        return emg_txt.DEFAULT_SAMPLE_RATE_HZ


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rep_windows_s(manifest: dict) -> list:
    """Rep windows as (rep_no, start_s, end_s) in seconds from trial start."""
    origin = datetime.fromisoformat(manifest["start_wall_time"])
    windows = []
    for rep in manifest.get("repetitions", []):
        start = (datetime.fromisoformat(rep["start_wall_time"]) - origin).total_seconds()
        end = (datetime.fromisoformat(rep["end_wall_time"]) - origin).total_seconds()
        windows.append((rep["rep"], start, end))
    return windows


def _rep_mask(t: np.ndarray, reps) -> np.ndarray:
    """1.0 while inside a labelled press window, else 0.0."""
    mask = np.zeros_like(t, dtype=float)
    for _, start_s, end_s in reps:
        mask[(t >= start_s) & (t <= end_s)] = 1.0
    return mask


def coverage(t: np.ndarray) -> str:
    if t.size < 2:
        return "n/a"
    return "continuous" if np.diff(t).max() < GAP_THRESHOLD_S else "gappy (reps-only)"


def select_emg_channels(emg: np.ndarray, emg_t: np.ndarray, reps, rate: float):
    """Pick real EMG channels: low clipping AND envelope correlated with the
    press/rest pattern (uses the manifest rep windows as ground truth).
    Returns ``(selected_indices, [(ch, clip_frac, rep_corr), ...])``."""
    mask = _rep_mask(emg_t, reps)
    scores, selected = [], []
    for ch in range(emg.shape[0]):
        clip = float(np.mean(np.abs(emg[ch]) >= EMG_RAIL))
        env = rms_envelope(emg[ch], rate, ENV_WINDOW_S)
        corr = float(np.corrcoef(env, mask)[0, 1]) if env.std() > 0 and mask.std() > 0 else 0.0
        scores.append((ch, clip, corr))
        if clip <= EMG_CLIP_MAX and corr >= EMG_REP_CORR_MIN:
            selected.append(ch)
    if not selected:                              # fallback: best-correlated, still reported
        selected = [max(scores, key=lambda s: s[2])[0]]
    return selected, scores


def anchoring_offset(emg, selected, emg_t, reps, rate, grid_hz=100.0, max_lag_s=6.0):
    """Residual EMG mis-anchoring: cross-correlate the selected channels'
    envelope against the rep on/off mask and return the best lag (s) within
    +-max_lag_s (0 => well anchored). Robust to sparse/gappy force because it
    never looks at force."""
    if not reps or emg.size == 0:
        return float("nan")
    env = rms_envelope(emg[selected].sum(axis=0), rate, ENV_WINDOW_S)
    grid = np.arange(emg_t[0], emg_t[-1], 1.0 / grid_hz)
    a = np.interp(grid, emg_t, env)
    m = _rep_mask(grid, reps)
    if a.std() == 0 or m.std() == 0:
        return float("nan")
    a = (a - a.mean()) / a.std()
    m = (m - m.mean()) / m.std()
    corr = np.correlate(a, m, mode="full")
    lags = np.arange(-len(m) + 1, len(a)) / grid_hz
    win = np.abs(lags) <= max_lag_s
    return float(lags[win][np.argmax(corr[win])])


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def align_trial(trial_dir: Path) -> AlignedTrial:
    """Load and time-align one trial folder's streams. EMG is optional
    (``emg_raw.txt`` absent -> the EMG fields are empty and ``emg_present`` is
    False)."""
    trial_dir = Path(trial_dir)
    manifest = json.loads((trial_dir / "manifest.json").read_text())
    reps = _rep_windows_s(manifest)

    # --- Force (native ~100 Hz) ---
    forces_df = forces_proc.load_forces_csv(trial_dir / "forces.csv")
    force_t = forces_df["elapsed_s"].to_numpy()
    f1 = forces_df["F1_N"].to_numpy()
    f2 = forces_df["F2_N"].to_numpy()
    f_combined = forces_df["F_combined"].to_numpy()

    # --- E-skin (native ~200 Hz) + per-rep ROI ---
    eskin_df = eskin_proc.load_eskin_csv(trial_dir / "eskin.csv")
    eskin_t = eskin_df["elapsed_s"].to_numpy()
    eskin_frames = eskin_proc.frames_array(eskin_df)
    rep_rois = eskin_proc.detect_roi_per_rep(eskin_frames, eskin_t, reps)
    eskin_roi, roi_union = eskin_proc.roi_signal(eskin_frames, eskin_t, reps, rep_rois)

    # --- EMG (native 2000 Hz, optional) ---
    emg_path = trial_dir / "emg_raw.txt"
    if emg_path.exists():
        emg_data = emg_txt.load_emg_txt(emg_path)
        rate = emg_data.sample_rate_hz
        emg = emg_data.signals
        emg_t = np.arange(emg.shape[1]) / rate
        emg_selected, emg_scores = select_emg_channels(emg, emg_t, reps, rate)
        anchor = anchoring_offset(emg, emg_selected, emg_t, reps, rate)
        emg_present = True
        emg_names = emg_data.channel_names
    else:
        emg_present = False
        emg = np.empty((0, 0))
        emg_t = np.empty(0)
        emg_selected, emg_scores, anchor, emg_names = [], [], float("nan"), []

    return AlignedTrial(
        trial_dir=trial_dir, manifest=manifest, reps=reps,
        emg_present=emg_present, emg_t=emg_t, emg=emg, emg_channel_names=emg_names,
        emg_selected=emg_selected, emg_scores=emg_scores, anchor_offset_s=anchor,
        force_t=force_t, f1=f1, f2=f2, f_combined=f_combined,
        eskin_t=eskin_t, eskin_frames=eskin_frames, rep_rois=rep_rois,
        roi_union=roi_union, eskin_roi=eskin_roi,
        force_coverage=coverage(force_t), eskin_coverage=coverage(eskin_t),
    )


def report(trial: AlignedTrial) -> str:
    """Human-readable one-trial summary (returned as a string; caller prints)."""
    m = trial.manifest
    lines = [
        "=" * 68,
        f"TRIAL  {m.get('trial_id')}   [{m.get('task_kind')}]",
        "=" * 68,
        f"subject={m.get('subject_id')}  reps={len(trial.reps)}  "
        f"target={m.get('target_force_n')} N  tol={m.get('tolerance_n')} N",
        f"  force  : {trial.f1.size} samp  "
        f"[{trial.force_t[0]:.2f}..{trial.force_t[-1]:.2f}] s  [{trial.force_coverage}]",
        f"  e-skin : {trial.eskin_roi.size} frames  "
        f"[{trial.eskin_t[0]:.2f}..{trial.eskin_t[-1]:.2f}] s  [{trial.eskin_coverage}]",
    ]
    if trial.emg_present:
        lines.append(
            f"  EMG    : {trial.emg.shape[0]} ch x {trial.emg.shape[1]} samp  "
            f"[{trial.emg_t[0]:.2f}..{trial.emg_t[-1]:.2f}] s  @~{trial.sample_rate_hz:.0f} Hz")
        lines.append("  EMG channel scan (clip% / rep-correlation):")
        for ch, clip, corr in trial.emg_scores:
            tag = "  <-- REAL" if ch in trial.emg_selected else ""
            lines.append(f"      {trial.emg_channel_names[ch]}: clip {clip*100:5.1f}%  "
                         f"corr {corr:+.2f}{tag}")
        ao = trial.anchor_offset_s
        verdict = ("OK, well anchored" if abs(ao) < 0.5 else "CHECK: possible mis-anchor")
        lines.append(f"  EMG anchoring vs rep windows: {ao*1000:+.0f} ms  ({verdict})")
    else:
        lines.append("  EMG    : (no emg_raw.txt in this trial)")
    lines.append("  e-skin per-rep ROI:")
    for rep_no, mask, _ in trial.rep_rois:
        ys, xs = np.where(mask)
        if ys.size:
            lines.append(f"      rep {rep_no}: {int(mask.sum()):3d} cells  "
                         f"rows {ys.min()}-{ys.max()}, cols {xs.min()}-{xs.max()}")
        else:
            lines.append(f"      rep {rep_no}: no ROI (no e-skin data in window)")
    return "\n".join(lines)
