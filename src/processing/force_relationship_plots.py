"""Cross-trial plots: e-skin / EMG vs. force, pooled across trials, with a
binned mean+/-std overlay and an annotated saturation knee.

Uses the non-interactive ``Agg`` backend so it works headless and always
saves a PNG (same convention as ``trial_plots.py``).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .force_relationship import SaturationKnee

# Force-level -> hue (sequential blue ramp, light->dark with increasing force)
# plus a distinct categorical hue for the max-effort ramp trial.
_TARGET_COLORS = {
    15.0: "#9ec5f4",
    30.0: "#5598e7",
    45.0: "#2a78d6",
    60.0: "#1c5cab",
    75.0: "#0d366b",
}
_MAX_EFFORT_COLOR = "#eb6834"
_BINNED_COLOR = "#0b0b0b"
_KNEE_COLOR = "#52514e"
_GRID_COLOR = "#e1e0d9"
_AXIS_COLOR = "#898781"


def _style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(_AXIS_COLOR)
    ax.spines["bottom"].set_color(_AXIS_COLOR)
    ax.tick_params(colors=_AXIS_COLOR)
    ax.grid(True, color=_GRID_COLOR, lw=0.8, zorder=0)
    ax.set_axisbelow(True)


def _scatter_binned_panel(ax, pooled: pd.DataFrame, binned: pd.DataFrame,
                           knee: SaturationKnee, value_col: str, ylabel: str,
                           x_col: str = "force_n"):
    _style_axis(ax)

    sub = pooled[[x_col, value_col, "task_kind", "target_force_n", "trial_id", "rep_no"]].dropna(
        subset=[x_col, value_col])

    max_effort = sub[sub["task_kind"] == "max_effort"]
    if not max_effort.empty:
        for i, (rep_no, rep_df) in enumerate(max_effort.groupby("rep_no")):
            rep_df = rep_df.sort_values(x_col)
            ax.plot(rep_df[x_col], rep_df[value_col], color=_MAX_EFFORT_COLOR,
                    lw=1.0, alpha=0.6, zorder=2,
                    label="max effort (per rep)" if i == 0 else None)

    target = sub[sub["task_kind"] == "target_force"]
    for level in sorted(_TARGET_COLORS):
        level_df = target[np.isclose(target["target_force_n"], level)]
        if level_df.empty:
            continue
        ax.scatter(level_df[x_col], level_df[value_col], s=8, alpha=0.35,
                   color=_TARGET_COLORS[level], edgecolors="none", zorder=2,
                   label=f"{level:.0f} N target")

    if not binned.empty:
        valid = binned[binned.get("valid", True)]
        ax.plot(valid["force_bin_center"], valid["mean"], color=_BINNED_COLOR,
                lw=2.0, zorder=4, label="binned mean")
        ax.fill_between(valid["force_bin_center"],
                        valid["mean"] - valid["std"], valid["mean"] + valid["std"],
                        color=_BINNED_COLOR, alpha=0.12, zorder=1)

    if knee.knee_force_n is not None:
        ax.axvline(knee.knee_force_n, color=_KNEE_COLOR, lw=1.2, ls="--", zorder=3)
        ax.text(knee.knee_force_n, ax.get_ylim()[1], f"  knee ~{knee.knee_force_n:.0f} N",
               color=_KNEE_COLOR, fontsize=8, va="top", ha="left")

    ax.set_ylabel(ylabel)
    ax.set_xlabel("Force F1+F2 (N)")
    ax.legend(loc="upper left", fontsize=7, framealpha=0.9)


def plot_eskin_vs_force(pooled: pd.DataFrame, binned_eskin: pd.DataFrame,
                        knee_eskin: SaturationKnee, out_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 6))
    _scatter_binned_panel(ax, pooled, binned_eskin, knee_eskin, "eskin_roi",
                          "E-skin ROI Σ (per-rep)")
    ax.set_title("E-skin ROI sum vs. grasp force (pooled across trials)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def plot_emg_vs_force(pooled: pd.DataFrame, binned_emg: pd.DataFrame,
                      knee_emg: SaturationKnee, out_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 6.6))
    _scatter_binned_panel(ax, pooled, binned_emg, knee_emg, "emg_env",
                          "EMG RMS envelope (μV, selected channels)")
    fig.suptitle("EMG envelope vs. grasp force (pooled across trials)", fontsize=11, y=0.98)
    fig.text(0.5, 0.925,
             "secondary/low-confidence: ~2 real channels; anticipatory ramp between reps (PROJECT.md)",
             ha="center", va="top", fontsize=8.5, color=_AXIS_COLOR)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def plot_combined(pooled: pd.DataFrame, binned_eskin: pd.DataFrame, binned_emg: pd.DataFrame,
                  knee_eskin: SaturationKnee, knee_emg: SaturationKnee, out_path: Path) -> Path:
    fig, (ax_eskin, ax_emg) = plt.subplots(2, 1, sharex=True, figsize=(9, 10))
    _scatter_binned_panel(ax_eskin, pooled, binned_eskin, knee_eskin, "eskin_roi",
                          "E-skin ROI Σ (per-rep)")
    ax_eskin.set_xlabel("")
    _scatter_binned_panel(ax_emg, pooled, binned_emg, knee_emg, "emg_env",
                          "EMG RMS envelope (μV)")
    fig.suptitle("Force relationship: e-skin (top) and EMG (bottom), pooled across trials")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Simple per-rep scatter (one point per rep, no pooling/binning/knee) --
# the quick-look counterpart to the pooled-samples plots above, for
# scripts.plot_experiment_summary.
# ---------------------------------------------------------------------------

def _target_color_map(levels) -> dict:
    """Maps whatever distinct target_force_n values are present (sorted) to
    a sequential blue ramp (light->dark with increasing force), sampled from
    a colormap -- generalizes _TARGET_COLORS (hardcoded to 15/30/45/60/75 N)
    to arbitrary experiments (e.g. 10/15 N)."""
    levels = sorted(levels)
    if not levels:
        return {}
    cmap = plt.get_cmap("Blues")
    if len(levels) == 1:
        return {levels[0]: cmap(0.7)}
    return {level: cmap(0.35 + 0.55 * i / (len(levels) - 1)) for i, level in enumerate(levels)}


def plot_rep_scatter(rep_df: pd.DataFrame, value_col: str, ylabel: str, title: str,
                     out_path: Path, aggregate_max_effort: bool = False) -> Path:
    """One point per rep, colored by group (max_effort, or each distinct
    target_force_n via _target_color_map). If aggregate_max_effort, the
    max_effort group's reps are collapsed into a single mean point with std
    error bars on both axes instead of individual dots (used for the %MVC
    plot, where max-squeeze reps are trivially ~100% by construction -- the
    MVC reference is their own mean -- so N near-identical dots add clutter,
    not information; the std captures rep-to-rep effort variability). No
    binning, no saturation-knee overlay -- see plot_eskin_vs_force /
    plot_emg_vs_force for that (pooled-samples) style."""
    fig, ax = plt.subplots(figsize=(8, 6))
    _style_axis(ax)

    sub = rep_df[["force_mean_n", value_col, "task_kind", "target_force_n"]].dropna(
        subset=["force_mean_n", value_col])

    max_effort = sub[sub["task_kind"] == "max_effort"]
    if not max_effort.empty:
        if aggregate_max_effort:
            x, y = max_effort["force_mean_n"], max_effort[value_col]
            ax.errorbar(x.mean(), y.mean(), xerr=x.std(), yerr=y.std(),
                        fmt="o", ms=9, color=_MAX_EFFORT_COLOR, ecolor=_MAX_EFFORT_COLOR,
                        capsize=4, zorder=4, label="max effort (mean ± std)")
        else:
            ax.scatter(max_effort["force_mean_n"], max_effort[value_col], s=36,
                      color=_MAX_EFFORT_COLOR, edgecolors="none", zorder=3,
                      label="max effort (per rep)")

    target = sub[sub["task_kind"] == "target_force"]
    colors = _target_color_map(target["target_force_n"].dropna().unique())
    for level, color in colors.items():
        level_df = target[np.isclose(target["target_force_n"], level)]
        if level_df.empty:
            continue
        ax.scatter(level_df["force_mean_n"], level_df[value_col], s=36,
                  color=color, edgecolors="none", zorder=2,
                  label=f"{level:.0f} N target (per rep)")

    ax.set_xlabel("Force F1+F2 (N)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path
