"""Cross-trial analysis: how e-skin ROI and EMG envelope relate to grasp
force across the max-effort + 15/30/45/60/75 N target-force trials, and
whether the e-skin signal saturates above some force.

Loads the fixed set of committed PT_ trials (see
``force_relationship.TRIAL_IDS``), pools per-rep and per-sample data, computes
Pearson/Spearman correlations, estimates a saturation knee, and writes CSVs +
a text report + PNGs to an output directory.

Usage:
    python -m scripts.analyze_force_relationship
    python -m scripts.analyze_force_relationship --data-dir data --out-dir outputs/force_relationship --bin-width 5.0 --knee-threshold 0.2
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.processing import force_relationship as fr
from src.processing import force_relationship_plots as frp


def main():
    ap = argparse.ArgumentParser(description="Cross-trial force/e-skin/EMG relationship analysis.")
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/force_relationship"))
    ap.add_argument("--bin-width", type=float, default=5.0, help="force bin width in N")
    ap.add_argument("--knee-threshold", type=float, default=0.2,
                    help="fraction of max slope below which the rise counts as flattened")
    ap.add_argument("--min-bin-n", type=int, default=5, help="min samples per force bin to trust it")
    args = ap.parse_args()

    print(f"Trials: {fr.TRIAL_IDS}")
    trials = fr.load_trials(args.data_dir)

    rep_summary = fr.build_rep_summary(trials)
    pooled = fr.build_pooled_samples(trials)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rep_summary.to_csv(args.out_dir / "rep_summary.csv", index=False)
    pooled.to_csv(args.out_dir / "pooled_samples.csv", index=False)

    binned_eskin = fr.bin_by_force(pooled, "eskin_roi", bin_width_n=args.bin_width, min_n=args.min_bin_n)
    binned_emg = fr.bin_by_force(pooled, "emg_env", bin_width_n=args.bin_width, min_n=args.min_bin_n)
    binned_eskin.to_csv(args.out_dir / "force_bins_eskin.csv", index=False)
    binned_emg.to_csv(args.out_dir / "force_bins_emg.csv", index=False)

    corr_results = {
        "rep-level  force vs eskin_roi_mean": fr.compute_correlations(rep_summary, "force_mean_n", "eskin_roi_mean"),
        "rep-level  force vs emg_env_mean":   fr.compute_correlations(rep_summary, "force_mean_n", "emg_env_mean"),
        "pooled     force vs eskin_roi":      fr.compute_correlations(pooled, "force_n", "eskin_roi"),
        "pooled     force vs emg_env":        fr.compute_correlations(pooled, "force_n", "emg_env"),
    }

    knee_eskin = fr.detect_saturation_knee(binned_eskin, threshold_frac=args.knee_threshold,
                                            metric="eskin_roi", bin_width_n=args.bin_width)
    knee_emg = fr.detect_saturation_knee(binned_emg, threshold_frac=args.knee_threshold,
                                          metric="emg_env", bin_width_n=args.bin_width)
    knees = {"e-skin ROI": knee_eskin, "EMG envelope": knee_emg}

    report = fr.summarize_text(rep_summary, pooled, corr_results, knees)
    print(report)
    (args.out_dir / "report.txt").write_text(report)

    print(f"[plot] {frp.plot_eskin_vs_force(pooled, binned_eskin, knee_eskin, args.out_dir / 'eskin_vs_force.png')}")
    print(f"[plot] {frp.plot_emg_vs_force(pooled, binned_emg, knee_emg, args.out_dir / 'emg_vs_force.png')}")
    print(f"[plot] {frp.plot_combined(pooled, binned_eskin, binned_emg, knee_eskin, knee_emg, args.out_dir / 'combined_force_relationship.png')}")

    print(f"\nWrote outputs to {args.out_dir}/")


if __name__ == "__main__":
    main()
