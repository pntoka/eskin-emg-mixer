"""Aligns a trial's e-skin, force, and (optionally) EMG streams onto a
common time axis anchored at the trial's start, using the manifest.json
written by SessionRecorder.stop_trial().

Sample-level sync is not guaranteed -- EMG and the e-skin/force boards are
two independently wall-clock-triggered devices (see the plan's "Out of
scope" note) -- this is for trial-level correlation, not frame-perfect
alignment.
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from . import emg_c3d
from . import eskin as eskin_processing
from . import forces as forces_processing


def load_manifest(session_dir: Path) -> dict:
    with open(Path(session_dir) / "manifest.json") as f:
        return json.load(f)


def build_aligned_frame(session_dir: Path, emg_c3d_path: Optional[Path] = None,
                         resample_hz: float = 100.0) -> pd.DataFrame:
    """Resample forces + e-skin total load (and an EMG envelope, if a
    parsed C3D source is supplied) onto a shared `elapsed_s` axis.

    `emg_c3d_path` is a placeholder: EMG_Eyetracker_Tool saves EMG as .txt,
    not .c3d, so aligning a trial's *own* EMG currently requires converting
    that .txt to .c3d (or adding a native .txt loader) first -- this only
    demonstrates the alignment mechanics against a C3D-formatted EMG source
    such as archive/PT_max_squeeze.c3d. Pass None to align without an EMG
    column.
    """
    manifest = load_manifest(session_dir)

    forces_df = forces_processing.load_forces_csv(Path(manifest["forces_csv"]))
    eskin_df = eskin_processing.load_eskin_csv(Path(manifest["eskin_csv"]))

    duration = min(forces_df["elapsed_s"].max(), eskin_df["elapsed_s"].max())
    t = np.arange(0.0, duration, 1.0 / resample_hz)

    aligned = pd.DataFrame({"elapsed_s": t})
    aligned["F_combined_N"] = np.interp(t, forces_df["elapsed_s"], forces_df["F_combined"]
                                         if "F_combined" in forces_df else forces_df["F1_N"] + forces_df["F2_N"])

    eskin_total = eskin_df[eskin_processing._COLUMNS].sum(axis=1)
    aligned["eskin_total_load"] = np.interp(t, eskin_df["elapsed_s"], eskin_total)

    if emg_c3d_path is not None:
        emg = emg_c3d.process(emg_c3d_path)
        emg_envelope_mean = emg["envelope"].mean(axis=0)
        emg_t = np.arange(len(emg_envelope_mean)) / emg["sample_rate_hz"]
        aligned["emg_envelope_mean"] = np.interp(t, emg_t, emg_envelope_mean, left=np.nan, right=np.nan)

    return aligned
