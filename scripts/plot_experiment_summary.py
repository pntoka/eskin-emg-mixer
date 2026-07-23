"""Quick-look plots for one complete experiment (a max_effort trial plus its
target_force trials, in one folder): e-skin ROI vs. force, and %MVC-normalized
EMG vs. force, one point per rep.

Usage:
    python -m scripts.plot_experiment_summary data/some_experiment_folder
    python -m scripts.plot_experiment_summary data/some_experiment_folder --out-dir outputs/some_experiment
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.processing import force_relationship as fr
from src.processing import force_relationship_plots as frp
from src.processing import emg_activation as ea


def main():
    ap = argparse.ArgumentParser(
        description="Simple per-rep e-skin-vs-force and %MVC-vs-force plots for one experiment.")
    ap.add_argument("experiment_dir", type=Path,
                    help="folder whose immediate subfolders are one experiment's trials "
                         "(exactly one max_effort trial + one-or-more target_force trials)")
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()
    out_dir = args.out_dir or (args.experiment_dir / "experiment_summary")

    trials = fr.discover_trials(args.experiment_dir)
    if not trials:
        sys.exit(f"No trials (manifest.json) found under {args.experiment_dir}")
    print(f"Trials: {sorted(trials)}")

    try:
        mvc_reference, mvc_trial_id = ea.mvc_reference_from_trials(trials)
    except ValueError as exc:
        sys.exit(f"error: {exc}")
    print(f"MVC reference: {mvc_reference:.1f} uV (from {mvc_trial_id})")

    rep_summary = fr.build_rep_summary(trials)
    percent_mvc = ea.percent_mvc_by_rep(trials, mvc_reference)
    rep_summary = rep_summary.merge(percent_mvc, on=["trial_id", "rep_no"], how="left")

    out_dir.mkdir(parents=True, exist_ok=True)
    rep_summary.to_csv(out_dir / "rep_summary.csv", index=False)

    print(f"[plot] {frp.plot_rep_scatter(rep_summary, 'eskin_roi_mean', 'E-skin ROI Σ (per-rep mean)', 'E-skin ROI vs. grasp force', out_dir / 'eskin_vs_force.png')}")
    print(f"[plot] {frp.plot_rep_scatter(rep_summary, 'percent_mvc', 'EMG activity (% max-squeeze MVC)', 'EMG activity (%MVC) vs. grasp force', out_dir / 'percent_mvc_vs_force.png', aggregate_max_effort=True)}")

    print(f"\nWrote outputs to {out_dir}/")


if __name__ == "__main__":
    main()
