"""MVC-reference and %MVC-normalization for EMG activity.

Two grasp tasks per subject: ``max_effort`` (max-squeeze, N reps) establishes
a per-subject reference value (MVC-like, though not a true maximum voluntary
contraction test -- it's the average activation across max-squeeze reps).
``target_force`` reps' EMG activity is then expressed as a percentage of that
reference.

Operates only on :class:`~src.processing.align.AlignedTrial` objects (from
``align_trial``) -- never re-parses raw EMG files. Reuses the onset detection
already computed by ``align_trial`` (``trial.rep_onsets``) rather than
redoing it. Follows ``force_relationship.py``'s pattern: thin functions
returning plain ``pandas.DataFrame``s, one row per rep.
"""

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .align import AlignedTrial, coverage
from .emg_txt import combined_channel_envelope, EMG_ACTIVE_CHANNELS


def compute_mvc_reference(trial: AlignedTrial) -> Tuple[float, pd.DataFrame]:
    """max_effort trial only. Computes the combined-channel activation
    envelope once over the whole trial; for each rep, takes the envelope's
    time-mean over ``[onset_s, end_s]`` (onset from ``trial.rep_onsets`` --
    trims the reaction-time lead-in); averages those per-rep means across
    reps into one scalar reference. Returns ``(mvc_reference, per_rep_df)``.
    """
    if trial.manifest.get("task_kind") != "max_effort":
        raise ValueError(
            f"compute_mvc_reference requires a max_effort trial, got "
            f"{trial.manifest.get('task_kind')!r} ({trial.trial_dir})")
    if not trial.emg_present:
        raise ValueError(f"no EMG data in {trial.trial_dir}; cannot compute an MVC reference")

    env = combined_channel_envelope(trial.emg, trial.sample_rate_hz)
    onset_by_rep = {rep_no: (onset_s, detected) for rep_no, onset_s, detected in trial.rep_onsets}

    rows = []
    for rep_no, start_s, end_s in trial.reps:
        onset_s, onset_detected = onset_by_rep.get(rep_no, (start_s, False))
        sel = (trial.emg_t >= onset_s) & (trial.emg_t <= end_s)
        window_mean = float(env[sel].mean()) if sel.any() else float("nan")
        rows.append({
            "trial_id": trial.manifest.get("trial_id"),
            "subject_id": trial.manifest.get("subject_id"),
            "rep_no": rep_no,
            "start_s": start_s,
            "end_s": end_s,
            "onset_s": onset_s,
            "onset_detected": onset_detected,
            "window_mean_uv": window_mean,
            "n_samples": int(sel.sum()),
            "status": "ok",
        })
    per_rep_df = pd.DataFrame(rows)
    mvc_reference = float(per_rep_df["window_mean_uv"].mean())
    return mvc_reference, per_rep_df


def compute_target_force_activity(trial: AlignedTrial, mvc_reference: float) -> pd.DataFrame:
    """target_force trial only. Requires the per-rep ``emg_rep{N}.txt``
    capture shape -- raises if the trial's EMG is a legacy single-file
    (continuous) capture, which this pipeline doesn't support. For each rep,
    slices ``trial.emg`` to that rep's own contiguous raw samples (via the
    ``emg_t`` time-range trick -- each per-rep segment only contains its own
    reconstructed timestamps, so this recovers exactly that rep's data,
    uncontaminated by others), then filters that isolated segment
    independently (never across the artificial concatenation boundary
    between reps). No onset trimming -- capture already starts at hold.
    """
    if trial.manifest.get("task_kind") != "target_force":
        raise ValueError(
            f"compute_target_force_activity requires a target_force trial, got "
            f"{trial.manifest.get('task_kind')!r} ({trial.trial_dir})")
    if trial.emg_present and coverage(trial.emg_t) == "continuous":
        raise ValueError(
            f"{trial.trial_dir} has a legacy single-file (continuous) target_force "
            "EMG capture; only the per-rep emg_rep{N}.txt format is supported")

    rows = []
    for rep_no, start_s, end_s in trial.reps:
        row = {
            "trial_id": trial.manifest.get("trial_id"),
            "subject_id": trial.manifest.get("subject_id"),
            "target_force_n": trial.manifest.get("target_force_n"),
            "tolerance_n": trial.manifest.get("tolerance_n"),
            "rep_no": rep_no,
            "start_s": start_s,
            "end_s": end_s,
            "mvc_reference_uv": mvc_reference,
        }
        if not trial.emg_present:
            rows.append({**row, "n_samples": 0, "window_mean_uv": float("nan"),
                        "percent_mvc": float("nan"), "status": "missing_segment"})
            continue

        sel = (trial.emg_t >= start_s) & (trial.emg_t <= end_s)
        if not sel.any():
            rows.append({**row, "n_samples": 0, "window_mean_uv": float("nan"),
                        "percent_mvc": float("nan"), "status": "missing_segment"})
            continue

        segment = trial.emg[np.ix_(EMG_ACTIVE_CHANNELS, sel)]
        try:
            env = combined_channel_envelope(segment, trial.sample_rate_hz)
        except ValueError:
            rows.append({**row, "n_samples": int(sel.sum()), "window_mean_uv": float("nan"),
                        "percent_mvc": float("nan"), "status": "too_short_to_filter"})
            continue

        window_mean = float(env.mean())
        percent_mvc = (100.0 * window_mean / mvc_reference
                       if mvc_reference and mvc_reference > 0 else float("nan"))
        rows.append({**row, "n_samples": int(sel.sum()), "window_mean_uv": window_mean,
                    "percent_mvc": percent_mvc, "status": "ok"})

    return pd.DataFrame(rows)


def mvc_reference_from_trials(trials: Dict[str, AlignedTrial]) -> Tuple[float, str]:
    """Finds the single max_effort trial among ``trials`` and computes its
    MVC reference via :func:`compute_mvc_reference`. Raises ValueError if
    there isn't exactly one (this pipeline assumes one experiment == one
    max_effort trial). Returns ``(mvc_reference, trial_id_used)``."""
    max_effort = {tid: t for tid, t in trials.items()
                  if t.manifest.get("task_kind") == "max_effort"}
    if len(max_effort) != 1:
        raise ValueError(
            f"expected exactly one max_effort trial among {sorted(trials)}, "
            f"found {sorted(max_effort)}")
    trial_id, trial = next(iter(max_effort.items()))
    mvc_reference, _ = compute_mvc_reference(trial)
    return mvc_reference, trial_id


def percent_mvc_by_rep(trials: Dict[str, AlignedTrial], mvc_reference: float) -> pd.DataFrame:
    """One row per rep across every max_effort/target_force trial in
    ``trials``: ``trial_id``, ``rep_no``, ``percent_mvc``. max_effort rows
    come from :func:`compute_mvc_reference`'s own per-rep table
    (``window_mean_uv / mvc_reference * 100`` -- averages to ~100% exactly,
    by construction); target_force rows from
    :func:`compute_target_force_activity`. A target_force trial with an
    unsupported legacy (continuous) EMG capture is skipped with a printed
    warning rather than aborting the whole run. Trials of any other
    task_kind, or without EMG, contribute no rows."""
    rows = []
    for trial_id, trial in trials.items():
        kind = trial.manifest.get("task_kind")
        if kind == "max_effort":
            _, per_rep = compute_mvc_reference(trial)
            for _, row in per_rep.iterrows():
                rows.append({
                    "trial_id": trial_id,
                    "rep_no": row["rep_no"],
                    "percent_mvc": 100.0 * row["window_mean_uv"] / mvc_reference
                                    if mvc_reference else float("nan"),
                })
        elif kind == "target_force":
            try:
                activity = compute_target_force_activity(trial, mvc_reference)
            except ValueError as exc:
                print(f"[percent_mvc_by_rep] skipping {trial_id}: {exc}")
                continue
            for _, row in activity.iterrows():
                rows.append({"trial_id": trial_id, "rep_no": row["rep_no"],
                            "percent_mvc": row["percent_mvc"]})
    return pd.DataFrame(rows, columns=["trial_id", "rep_no", "percent_mvc"])
