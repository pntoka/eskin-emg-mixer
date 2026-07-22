"""Align + plot one recorded trial (or a whole data/ folder of trials).

Loads a trial's e-skin, force, and EMG streams, time-aligns them onto the
trial-start origin (keeping native rates -- no fusion), auto-selects the real
EMG channels and per-rep e-skin ROIs, prints a summary, and saves two figures
into the trial folder:

    aligned_overview.png   EMG env / force / e-skin-ROI on one time axis
    eskin_rep_rois.png     per-rep peak-pressure map + ROI outline

Usage:
    python -m scripts.align_trial data/AM_hold_001_20260721_165006   # one trial
    python -m scripts.align_trial data                                # every trial folder
    python -m scripts.align_trial data --no-plot                      # text report only
    python -m scripts.align_trial data --dump                         # also write aligned/*.csv
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.processing import align as align_mod
from src.processing import trial_plots


def find_trials(path: Path):
    if (path / "manifest.json").exists():
        return [path]
    return sorted(p for p in path.iterdir()
                  if p.is_dir() and (p / "manifest.json").exists())


def dump_aligned(trial) -> None:
    """Write each stream's aligned samples to <trial>/aligned/*.csv (streams
    kept separate, native rates, shared t_s axis)."""
    out = trial.trial_dir / "aligned"
    out.mkdir(exist_ok=True)

    with open(out / "force_aligned.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t_s", "F1_N", "F2_N", "F_combined_N"])
        for i in range(trial.force_t.size):
            w.writerow([round(trial.force_t[i], 5), round(trial.f1[i], 4),
                        round(trial.f2[i], 4), round(trial.f_combined[i], 4)])

    with open(out / "eskin_roi_aligned.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t_s", "eskin_roi_sum"])
        for i in range(trial.eskin_t.size):
            w.writerow([round(trial.eskin_t[i], 5), round(trial.eskin_roi[i], 3)])

    if trial.emg_present and trial.emg_selected:
        header = ["t_s"] + [trial.emg_channel_names[ch] for ch in trial.emg_selected]
        sel = trial.emg[trial.emg_selected]
        with open(out / "emg_aligned.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(header)
            for i in range(trial.emg_t.size):
                w.writerow([round(trial.emg_t[i], 5)] + [round(v, 3) for v in sel[:, i]])
    print(f"[dump] {out}/")


def main():
    ap = argparse.ArgumentParser(description="Align + plot recorded trial(s).")
    ap.add_argument("path", type=Path, help="a trial folder or a parent of trial folders")
    ap.add_argument("--no-plot", action="store_true", help="text report only")
    ap.add_argument("--dump", action="store_true", help="also write aligned/*.csv")
    args = ap.parse_args()

    trials = find_trials(args.path)
    if not trials:
        sys.exit(f"No trial folders (with manifest.json) found under {args.path}")

    for folder in trials:
        trial = align_mod.align_trial(folder)
        print(align_mod.report(trial))
        if not args.no_plot:
            print(f"[plot] {trial_plots.plot_overview(trial, folder / 'aligned_overview.png')}")
            print(f"[plot] {trial_plots.plot_rep_rois(trial, folder / 'eskin_rep_rois.png')}")
        if args.dump:
            dump_aligned(trial)
        print()


if __name__ == "__main__":
    main()
