"""Cross-trial analysis of how e-skin and EMG signals relate to grasp force.

Pools the per-rep and per-sample data from several already-aligned trials
(see :mod:`align`) spanning a range of target forces plus one max-effort
ramp, to (a) correlate e-skin / EMG against force and (b) check whether the
e-skin signal saturates (stops increasing) above some force -- see
PROJECT.md's "e-skin saturates at high force" finding, which this puts on a
firmer, cross-trial footing.

This module only assembles data and computes statistics; see
``force_relationship_plots.py`` for the matplotlib side.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .align import align_trial, AlignedTrial
from .emg_txt import rms_envelope
from . import forces as forces_proc

TRIAL_IDS = [
    "PT_max_squeeze_001_20260722_153104",
    "PT_target_15_001_20260722_153324",
    "PT_target_30_002_20260722_153507",
    "PT_target_45_004_20260722_154404",
    "PT_target_60_005_20260722_154642",
    "PT_target_75_008_20260722_155202",
]


@dataclass
class CorrelationResult:
    x_col: str
    y_col: str
    n: int
    pearson_r: float
    pearson_p: float
    spearman_rho: float
    spearman_p: float


@dataclass
class SaturationKnee:
    metric: str
    bin_width_n: float
    force_bin_centers: np.ndarray
    slopes: np.ndarray
    max_slope: float
    max_slope_force_n: float
    threshold_frac: float
    knee_force_n: Optional[float]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_trials(data_dir: Path, trial_ids: Sequence[str] = TRIAL_IDS) -> dict:
    """Load and align every trial in ``trial_ids``. Returns ``{trial_id:
    AlignedTrial}``."""
    data_dir = Path(data_dir)
    return {tid: align_trial(data_dir / tid) for tid in trial_ids}


def discover_trials(experiment_dir: Path) -> dict:
    """Align every immediate subfolder of ``experiment_dir`` that contains a
    ``manifest.json`` -- one complete experiment (a max_effort trial plus its
    target_force trials, e.g. one subject's session). Returns ``{trial_id:
    AlignedTrial}``, keyed by folder name. No subject_id string-matching --
    the folder itself defines the experiment's scope."""
    experiment_dir = Path(experiment_dir)
    trial_dirs = sorted(p for p in experiment_dir.iterdir()
                        if p.is_dir() and (p / "manifest.json").exists())
    return {p.name: align_trial(p) for p in trial_dirs}


def _emg_envelope_mean(trial: AlignedTrial) -> Optional[np.ndarray]:
    """Mean RMS envelope across the trial's selected ("real") EMG channels,
    on ``trial.emg_t``'s native grid. ``None`` if EMG is absent or no
    channel passed selection."""
    if not trial.emg_present or not trial.emg_selected:
        return None
    env = rms_envelope(trial.emg[trial.emg_selected], trial.sample_rate_hz, 0.10)
    return env.mean(axis=0)


def _smoothed_force(trial: AlignedTrial) -> np.ndarray:
    return forces_proc.smooth(pd.Series(trial.f_combined)).to_numpy()


def _in_any_rep(t: np.ndarray, reps) -> np.ndarray:
    mask = np.zeros(t.shape, dtype=bool)
    for _, start_s, end_s in reps:
        mask |= (t >= start_s) & (t <= end_s)
    return mask


def _rep_no_at(t: np.ndarray, reps) -> np.ndarray:
    out = np.full(t.shape, -1, dtype=int)
    for rep_no, start_s, end_s in reps:
        out[(t >= start_s) & (t <= end_s)] = rep_no
    return out


# ---------------------------------------------------------------------------
# Per-rep summary table
# ---------------------------------------------------------------------------

def build_rep_summary(trials: dict) -> pd.DataFrame:
    """One row per (trial, rep): mean/std/peak force, mean/peak e-skin ROI,
    mean/peak EMG envelope -- each restricted to that rep's window."""
    rows = []
    for trial_id, trial in trials.items():
        m = trial.manifest
        emg_env = _emg_envelope_mean(trial)
        force = _smoothed_force(trial)
        for rep_no, start_s, end_s in trial.reps:
            f_sel = (trial.force_t >= start_s) & (trial.force_t <= end_s)
            e_sel = (trial.eskin_t >= start_s) & (trial.eskin_t <= end_s)
            row = {
                "trial_id": trial_id,
                "task_kind": m.get("task_kind"),
                "target_force_n": m.get("target_force_n"),
                "tolerance_n": m.get("tolerance_n"),
                "rep_no": rep_no,
                "force_mean_n": float(np.mean(force[f_sel])) if f_sel.any() else np.nan,
                "force_std_n": float(np.std(force[f_sel])) if f_sel.any() else np.nan,
                "force_peak_n": float(np.max(force[f_sel])) if f_sel.any() else np.nan,
                "eskin_roi_mean": float(np.mean(trial.eskin_roi[e_sel])) if e_sel.any() else np.nan,
                "eskin_roi_peak": float(np.max(trial.eskin_roi[e_sel])) if e_sel.any() else np.nan,
            }
            if emg_env is not None:
                m_sel = (trial.emg_t >= start_s) & (trial.emg_t <= end_s)
                row["emg_env_mean"] = float(np.mean(emg_env[m_sel])) if m_sel.any() else np.nan
                row["emg_env_peak"] = float(np.max(emg_env[m_sel])) if m_sel.any() else np.nan
            else:
                row["emg_env_mean"] = np.nan
                row["emg_env_peak"] = np.nan
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Pooled per-sample table
# ---------------------------------------------------------------------------

def build_pooled_samples(trials: dict) -> pd.DataFrame:
    """One row per e-skin frame that falls inside any rep window, pooled
    across all trials, with force and EMG envelope interpolated onto the
    e-skin's native timestamps."""
    frames = []
    for trial_id, trial in trials.items():
        m = trial.manifest
        keep = _in_any_rep(trial.eskin_t, trial.reps)
        if not keep.any():
            continue
        t = trial.eskin_t[keep]
        eskin = trial.eskin_roi[keep]
        force = np.interp(t, trial.force_t, _smoothed_force(trial))

        emg_env = _emg_envelope_mean(trial)
        if emg_env is not None:
            emg = np.interp(t, trial.emg_t, emg_env)
        else:
            emg = np.full_like(t, np.nan)

        rep_no = _rep_no_at(t, trial.reps)

        target_force_n = m.get("target_force_n")
        frames.append(pd.DataFrame({
            "trial_id": trial_id,
            "task_kind": m.get("task_kind"),
            "target_force_n": target_force_n if target_force_n is not None else np.nan,
            "rep_no": rep_no,
            "t_s": t,
            "force_n": force,
            "eskin_roi": eskin,
            "emg_env": emg,
        }))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ---------------------------------------------------------------------------
# Binning + correlation + saturation-knee detection
# ---------------------------------------------------------------------------

def bin_by_force(df: pd.DataFrame, value_col: str, force_col: str = "force_n",
                  bin_width_n: float = 5.0, min_n: int = 5) -> pd.DataFrame:
    """Bin ``df`` by ``force_col`` into ``bin_width_n``-wide bins and compute
    mean/std/count of ``value_col`` per bin."""
    sub = df[[force_col, value_col]].dropna()
    if sub.empty:
        return pd.DataFrame(columns=["force_bin_lo", "force_bin_hi", "force_bin_center", "mean", "std", "n"])
    max_force = sub[force_col].max()
    edges = np.arange(0.0, max_force + bin_width_n, bin_width_n)
    bins = pd.cut(sub[force_col], bins=edges, right=False)
    grouped = sub.groupby(bins, observed=True)[value_col].agg(["mean", "std", "count"])
    grouped = grouped.reset_index(names="bin")
    out = pd.DataFrame({
        "force_bin_lo": grouped["bin"].apply(lambda b: b.left).astype(float),
        "force_bin_hi": grouped["bin"].apply(lambda b: b.right).astype(float),
        "mean": grouped["mean"],
        "std": grouped["std"],
        "n": grouped["count"],
    })
    out["force_bin_center"] = (out["force_bin_lo"] + out["force_bin_hi"]) / 2.0
    out["valid"] = out["n"] >= min_n
    return out.sort_values("force_bin_center").reset_index(drop=True)


def compute_correlations(df: pd.DataFrame, x_col: str, y_col: str) -> CorrelationResult:
    sub = df[[x_col, y_col]].dropna()
    x, y = sub[x_col].to_numpy(), sub[y_col].to_numpy()
    if len(sub) < 2:
        return CorrelationResult(x_col, y_col, len(sub), float("nan"), float("nan"),
                                  float("nan"), float("nan"))
    pr, pp = stats.pearsonr(x, y)
    sr, sp = stats.spearmanr(x, y)
    return CorrelationResult(x_col, y_col, len(sub), float(pr), float(pp), float(sr), float(sp))


def detect_saturation_knee(binned: pd.DataFrame, threshold_frac: float = 0.2,
                            metric: str = "value", bin_width_n: float = 5.0) -> SaturationKnee:
    """Estimate the force above which ``binned``'s mean stops rising.

    Computes the slope between consecutive *valid* (sufficiently-sampled)
    bins, then walks forward from the steepest slope looking for the first
    point after which every remaining slope stays at or below
    ``threshold_frac`` of the max slope -- i.e. the rise has durably
    flattened, not just dipped once from noise.
    """
    valid = binned[binned.get("valid", True)].reset_index(drop=True) if "valid" in binned else binned
    centers = valid["force_bin_center"].to_numpy()
    means = valid["mean"].to_numpy()
    if len(centers) < 3:
        return SaturationKnee(metric, bin_width_n, centers, np.array([]), float("nan"),
                               float("nan"), threshold_frac, None)

    slopes = np.diff(means) / np.diff(centers)
    max_idx = int(np.argmax(slopes))
    max_slope = float(slopes[max_idx])
    max_slope_force_n = float(centers[max_idx])

    knee = None
    for i in range(max_idx, len(slopes)):
        if np.all(slopes[i:] <= threshold_frac * max_slope):
            knee = float(centers[i + 1])
            break

    return SaturationKnee(metric, bin_width_n, centers, slopes, max_slope,
                           max_slope_force_n, threshold_frac, knee)


# ---------------------------------------------------------------------------
# Human-readable summary
# ---------------------------------------------------------------------------

def summarize_text(rep_summary: pd.DataFrame, pooled: pd.DataFrame,
                    corr_results: dict, knees: dict) -> str:
    lines = ["=" * 72, "CROSS-TRIAL FORCE / E-SKIN / EMG RELATIONSHIP", "=" * 72]

    lines.append("")
    lines.append(f"Trials: {rep_summary['trial_id'].nunique()}   "
                 f"Reps: {len(rep_summary)}   Pooled samples: {len(pooled)}")
    counts = rep_summary.groupby("target_force_n", dropna=False).size()
    lines.append("Reps per target force (NaN = max-effort):")
    for target, n in counts.items():
        lines.append(f"    {target}: {n}")

    lines.append("")
    lines.append("Correlations (dropping NaN rows):")
    for label, cr in corr_results.items():
        lines.append(f"  {label}: n={cr.n}  "
                     f"Pearson r={cr.pearson_r:+.3f} (p={cr.pearson_p:.2g})  "
                     f"Spearman rho={cr.spearman_rho:+.3f} (p={cr.spearman_p:.2g})")
        if not np.isnan(cr.pearson_r) and not np.isnan(cr.spearman_rho):
            gap = cr.spearman_rho - cr.pearson_r
            if gap > 0.1:
                lines.append(f"      -> Spearman notably exceeds Pearson (gap {gap:+.3f}): "
                             f"consistent with a saturating, non-linear-but-monotonic relationship.")

    lines.append("")
    lines.append("Saturation-knee estimates (binned-slope method):")
    for label, knee in knees.items():
        if knee.knee_force_n is not None:
            lines.append(f"  {label}: knee ~ {knee.knee_force_n:.1f} N  "
                         f"(max slope {knee.max_slope:.3g}/N at {knee.max_slope_force_n:.1f} N, "
                         f"threshold {knee.threshold_frac*100:.0f}% of max, bin width {knee.bin_width_n} N)")
        else:
            lines.append(f"  {label}: no durable knee found in the observed force range "
                         f"(bin width {knee.bin_width_n} N)")

    return "\n".join(lines)
