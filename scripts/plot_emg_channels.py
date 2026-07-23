"""Diagnostic plot: raw + RMS-envelope EMG for specific channel(s) of one
trial, against time, with the labelled rep windows shaded -- for sanity-
checking whether a trial's EMG recording looks right (e.g. whether it
actually spans the whole trial).

Usage:
    python -m scripts.plot_emg_channels data/PT_target_30_002_20260722_153507
    python -m scripts.plot_emg_channels data/PT_target_30_002_20260722_153507 --channels 0,1
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.processing.align import align_trial, ENV_WINDOW_S
from src.processing.emg_txt import rms_envelope

_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]


def main():
    ap = argparse.ArgumentParser(description="Plot raw+envelope EMG for given channels of one trial.")
    ap.add_argument("trial_dir", type=Path)
    ap.add_argument("--channels", default="0,1", help="comma-separated 0-based channel indices")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    trial = align_trial(args.trial_dir)
    if not trial.emg_present:
        sys.exit(f"No EMG data in {args.trial_dir}")

    channels = [int(c) for c in args.channels.split(",")]
    out_path = args.out or (args.trial_dir / "emg_channels_diag.png")

    fig, (ax_raw, ax_env) = plt.subplots(2, 1, sharex=True, figsize=(11, 7))

    for ax in (ax_raw, ax_env):
        for _, start_s, end_s in trial.reps:
            ax.axvspan(start_s, end_s, color="0.88", zorder=0)
    for rep_no, start_s, end_s in trial.reps:
        ax_raw.text((start_s + end_s) / 2, 0.96, f"rep {rep_no}", ha="center", va="top",
                    transform=ax_raw.get_xaxis_transform(), fontsize=8, color="0.4")

    for i, ch in enumerate(channels):
        color = _COLORS[i % len(_COLORS)]
        name = trial.emg_channel_names[ch]
        ax_raw.plot(trial.emg_t, trial.emg[ch], lw=0.5, alpha=0.7, color=color, label=name)
        env = rms_envelope(trial.emg[ch], trial.sample_rate_hz, ENV_WINDOW_S)
        ax_env.plot(trial.emg_t, env, lw=1.2, color=color, label=name)

    ax_raw.set_ylabel("raw EMG (uV)")
    ax_raw.legend(loc="upper right", fontsize=8)
    ax_raw.set_title(f"{trial.manifest.get('trial_id')}  [{trial.manifest.get('task_kind')}, "
                     f"target={trial.manifest.get('target_force_n')} N]  "
                     f"-- EMG span {trial.emg_t[-1]:.2f}s vs trial span {trial.force_t[-1]:.2f}s "
                     f"({len(trial.reps)} reps)")

    ax_env.set_ylabel(f"RMS envelope (uV, {ENV_WINDOW_S*1000:.0f} ms window)")
    ax_env.set_xlabel("time (s from trial start)")
    ax_env.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"[plot] {out_path}")
    print(f"EMG span: 0..{trial.emg_t[-1]:.2f}s   Trial span: 0..{trial.force_t[-1]:.2f}s   "
         f"Reps: {trial.reps}")


if __name__ == "__main__":
    main()
