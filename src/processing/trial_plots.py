"""Static (matplotlib) plots for a single aligned trial.

Two figures:
  * :func:`plot_overview`   -- EMG envelope / force / e-skin-ROI on one shared
                               time axis, with the labelled rep windows shaded.
  * :func:`plot_rep_rois`   -- one panel per rep: that rep's peak-pressure map
                               with its ROI outlined (to see whether the grasp
                               region moves between reps).

Uses the non-interactive ``Agg`` backend so it works headless and always saves
a PNG.
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .align import AlignedTrial, ENV_WINDOW_S, GAP_THRESHOLD_S
from .emg_txt import rms_envelope, EMG_ACTIVE_CHANNELS


def _with_gaps(t: np.ndarray, y: np.ndarray, gap_s: float = GAP_THRESHOLD_S):
    """Insert NaNs where samples are >gap_s apart so lines aren't drawn across
    intervals that have no data (e.g. rest gaps in a hold trial)."""
    out_t, out_y = [], []
    for i in range(t.size):
        if i > 0 and (t[i] - t[i - 1]) > gap_s:
            out_t.append(np.nan)
            out_y.append(np.nan)
        out_t.append(t[i])
        out_y.append(y[i])
    return np.array(out_t), np.array(out_y)


def plot_overview(trial: AlignedTrial, out_path: Path) -> Path:
    m = trial.manifest
    fig, (ax_emg, ax_force, ax_eskin) = plt.subplots(3, 1, sharex=True, figsize=(12, 8))

    for ax in (ax_emg, ax_force, ax_eskin):
        for _, start_s, end_s in trial.reps:
            ax.axvspan(start_s, end_s, color="0.88", zorder=0)
    for rep_no, start_s, end_s in trial.reps:
        ax_emg.text((start_s + end_s) / 2, 0.96, f"rep {rep_no}", ha="center",
                    va="top", transform=ax_emg.get_xaxis_transform(),
                    fontsize=8, color="0.4")

    # EMG envelope -- the two known-wired channels (EMG_ACTIVE_CHANNELS), not
    # trial.emg_selected: that heuristic's rep-correlation check always reads
    # 0 for per-rep-captured target_force trials (no in-file rest sample), so
    # it degenerates to an arbitrary single-channel pick for that shape.
    active_channels = [ch for ch in EMG_ACTIVE_CHANNELS if ch < trial.emg.shape[0]]
    if trial.emg_present and active_channels:
        for ch in active_channels:
            env = rms_envelope(trial.emg[ch], trial.sample_rate_hz, ENV_WINDOW_S)
            et, ey = _with_gaps(trial.emg_t, env)
            ax_emg.plot(et, ey, lw=0.8, label=trial.emg_channel_names[ch])
        # Per-rep onset marker: solid = detected (trims the reaction-time
        # lead-in), dotted = fallback to the rep's nominal start (no in-file
        # rest baseline to detect onset against, e.g. target_force reps).
        for i, (_, onset_s, onset_detected) in enumerate(trial.rep_onsets):
            ax_emg.axvline(onset_s, color="tab:red", lw=1.0,
                            ls="-" if onset_detected else ":", alpha=0.8,
                            label="rep onset" if i == 0 else None)
        ax_emg.legend(loc="upper right", fontsize=8)
    else:
        ax_emg.text(0.5, 0.5, "no EMG", ha="center", va="center",
                    transform=ax_emg.transAxes, color="0.5")
    ax_emg.set_ylabel("EMG |env| (uV)")
    ax_emg.set_title(f"{m.get('trial_id')}  [{m.get('task_kind')}]  "
                     f"— aligned streams (shaded = labelled press windows)")

    # Force (gap-broken); F_combined = F1 + F2
    ft, fc = _with_gaps(trial.force_t, trial.f_combined)
    ax_force.plot(ft, fc, "k", lw=1.2, label="F1+F2 (N)")
    _, fy1 = _with_gaps(trial.force_t, trial.f1)
    _, fy2 = _with_gaps(trial.force_t, trial.f2)
    ax_force.plot(ft, fy1, "r", lw=0.7, alpha=0.6, label="F1")
    ax_force.plot(ft, fy2, "b", lw=0.7, alpha=0.6, label="F2")
    tgt, tol = m.get("target_force_n"), m.get("tolerance_n")
    if tgt is not None:
        ax_force.axhspan(tgt - (tol or 0), tgt + (tol or 0), color="green", alpha=0.12)
        ax_force.axhline(tgt, color="green", lw=0.8, ls="--", label=f"target {tgt} N")
    ax_force.set_ylabel("Force (N)")
    ax_force.legend(loc="upper right", fontsize=8)

    # E-skin ROI sum (gap-broken)
    et, ey = _with_gaps(trial.eskin_t, trial.eskin_roi)
    ax_eskin.plot(et, ey, "purple", lw=1.0)
    ax_eskin.set_ylabel(f"E-skin ROI Σ\n(per-rep; ∪={int(trial.roi_union.sum())} cells)")
    ax_eskin.set_xlabel("time (s from trial start)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def plot_rep_rois(trial: AlignedTrial, out_path: Path) -> Path:
    rep_rois = trial.rep_rois
    n = max(1, len(rep_rois))
    fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 3.4), squeeze=False)
    for ax, (rep_no, mask, peak) in zip(axes[0], rep_rois):
        ax.imshow(peak, cmap="plasma")
        if mask.any():
            ax.contour(mask, levels=[0.5], colors="red", linewidths=1.4)
        ax.set_title(f"rep {rep_no} — {int(mask.sum())} cells", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"{trial.manifest.get('trial_id')} — per-rep e-skin ROI "
                 f"(peak pressure + ROI outline)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path
