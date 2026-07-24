"""Presentation-ready version of experiments/PT/experiment_summary/eskin_vs_force.png
for slides: PT's per-rep e-skin-ROI-vs-force scatter, with narges' 10 N
target-force trial folded in (PT has no 10 N target trial of its own).

Target-force levels are colored with a high-contrast viridis gradient
(rather than force_relationship_plots' pale Blues ramp, whose lightest
steps wash out on a white/gridded background) so every level -- including
the added 10 N one -- stays clearly distinguishable.

Larger fonts, 300 dpi, for the pretty_plots/ folder.

Run: python -m scripts.plot_pretty_eskin_vs_force  (from repo root)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.processing import force_relationship as fr
from src.processing.align import align_trial
from src.processing.force_relationship_plots import _style_axis, _MAX_EFFORT_COLOR

ROOT = Path(__file__).resolve().parents[1]
PT_DIR = ROOT / "experiments" / "PT"
NARGES_TARGET_10_DIR = ROOT / "experiments" / "narges" / "narges_target_10_002_20260722_173151"
OUT_DIR = ROOT / "pretty_plots"

MARKER_SIZE = 75

FONT_AXIS_LABEL = 16
FONT_TICK = 13
FONT_TITLE = 16
FONT_LEGEND = 12


def _strong_gradient_colors(levels) -> dict:
    """High-contrast, perceptually-uniform gradient (viridis) across the
    given force levels -- every step stays clearly visible, unlike a pale
    sequential-blue ramp whose low end nearly disappears on white."""
    levels = sorted(levels)
    if not levels:
        return {}
    cmap = plt.get_cmap("viridis")
    if len(levels) == 1:
        return {levels[0]: cmap(0.5)}
    fracs = np.linspace(0.05, 0.9, len(levels))
    return {level: cmap(f) for level, f in zip(levels, fracs)}


def main():
    trials = fr.discover_trials(PT_DIR)
    print(f"PT trials: {sorted(trials)}")
    rep_summary = fr.build_rep_summary(trials)

    narges_trials = {NARGES_TARGET_10_DIR.name: align_trial(NARGES_TARGET_10_DIR)}
    narges_rep_summary = fr.build_rep_summary(narges_trials)
    print(f"narges 10N-target reps: {len(narges_rep_summary)}")

    rep_summary = pd.concat([rep_summary, narges_rep_summary], ignore_index=True)

    fig, ax = plt.subplots(figsize=(9, 7))
    _style_axis(ax)
    ax.grid(True, color="#d8d8d8", lw=0.8, zorder=0)

    sub = rep_summary[["force_mean_n", "eskin_roi_mean", "task_kind", "target_force_n"]].dropna(
        subset=["force_mean_n", "eskin_roi_mean"])

    max_effort = sub[sub["task_kind"] == "max_effort"]
    if not max_effort.empty:
        ax.scatter(max_effort["force_mean_n"], max_effort["eskin_roi_mean"], s=MARKER_SIZE,
                   color=_MAX_EFFORT_COLOR, edgecolors="none", zorder=3,
                   label="max effort (per rep)")

    target = sub[sub["task_kind"] == "target_force"]
    colors = _strong_gradient_colors(target["target_force_n"].dropna().unique())
    for level, color in colors.items():
        level_df = target[np.isclose(target["target_force_n"], level)]
        if level_df.empty:
            continue
        ax.scatter(level_df["force_mean_n"], level_df["eskin_roi_mean"], s=MARKER_SIZE,
                  color=color, edgecolors="none", zorder=2,
                  label=f"{level:.0f} N target (per rep)")

    ax.set_xlabel("Force F1+F2 (N)", fontsize=FONT_AXIS_LABEL)
    ax.set_ylabel("E-skin ROI Σ (per-rep mean)", fontsize=FONT_AXIS_LABEL)
    ax.set_title("E-skin ROI vs. grasp force (PT) + narges 10 N target", fontsize=FONT_TITLE)
    ax.tick_params(axis="both", labelsize=FONT_TICK, labelcolor="black")
    ax.legend(loc="best", fontsize=FONT_LEGEND, framealpha=0.9)

    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "eskin_vs_force_pretty.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
