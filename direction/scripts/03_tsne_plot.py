"""Pooled t-SNE (+ PCA companion) of per-attempt averaged e-skin heatmaps,
colored by push/pull direction, to check whether attempts visually cluster
by direction.

Reads direction/reports/dataset.npz (from 02_build_dataset.py). Writes:
  direction/reports/fig_tsne_main.png
  direction/reports/fig_tsne_perplexity_grid.png
  direction/reports/fig_pca.png
  direction/reports/tsne_embedding.csv
  direction/reports/results.md

Run: python direction/scripts/03_tsne_plot.py  (from repo root)
"""
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import common

# palette slots 1/2/3 (blue/orange/aqua) -- fixed categorical order, axis identity
AXIS_COLOR = {"x": "#2a78d6", "y": "#eb6834", "z": "#1baf7a"}
SIGN_MARKER = {1: "o", -1: "^"}   # filled circle = positive, triangle = negative
PRIMARY_PERPLEXITY = 10
GRID_PERPLEXITIES = [5, 10, 15, 30]
RANDOM_STATE = 0


def scatter_by_direction(ax, x, y, axis, sign, title):
    for a in ("x", "y", "z"):
        for s in (1, -1):
            m = (axis == a) & (sign == s)
            if not m.any():
                continue
            ax.scatter(x[m], y[m], color=AXIS_COLOR[a], marker=SIGN_MARKER[s],
                       s=45, alpha=0.85, edgecolors="none",
                       label=common.direction_label(a, s))
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def legend_handles():
    handles = []
    for a in ("x", "y", "z"):
        for s in (1, -1):
            handles.append(Line2D([0], [0], marker=SIGN_MARKER[s], color="none",
                                   markerfacecolor=AXIS_COLOR[a], markersize=8,
                                   label=common.direction_label(a, s)))
    return handles


def main():
    data = np.load(common.REPORTS / "dataset.npz", allow_pickle=True)
    X = data["X"]
    axis = data["axis"]
    sign = data["sign"]
    direction_label = data["direction_label"]
    trial_id = data["trial_id"]
    attempt_idx = data["attempt_idx"]

    n = len(X)
    counts = pd.Series(direction_label).value_counts()
    print(f"Loaded {n} attempts, {X.shape[1]}-dim features")
    print(counts.to_string())
    min_class = int(counts.min())
    max_perp = min(PRIMARY_PERPLEXITY, min_class - 1)
    if max_perp != PRIMARY_PERPLEXITY:
        print(f"NOTE: smallest class has {min_class} attempts, "
              f"capping primary perplexity at {max_perp}")

    # --- primary t-SNE ---
    tsne = TSNE(n_components=2, perplexity=max_perp, init="pca",
                learning_rate="auto", random_state=RANDOM_STATE)
    emb = tsne.fit_transform(X)

    fig, ax = plt.subplots(figsize=(7, 6))
    scatter_by_direction(ax, emb[:, 0], emb[:, 1], axis, sign,
                         f"t-SNE of averaged e-skin heatmaps per attempt (perplexity={max_perp})")
    ax.legend(handles=legend_handles(), loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    fig.tight_layout()
    fig.savefig(common.REPORTS / "fig_tsne_main.png", dpi=130)
    plt.close(fig)

    embedding_rows = pd.DataFrame(dict(
        trial_id=trial_id, attempt_idx=attempt_idx, axis=axis, sign=sign,
        direction_label=direction_label, x=emb[:, 0], y=emb[:, 1], perplexity=max_perp,
    ))

    # --- perplexity grid ---
    valid_perps = [p for p in GRID_PERPLEXITIES if p < n]
    fig, axes = plt.subplots(1, len(valid_perps), figsize=(5 * len(valid_perps), 5))
    if len(valid_perps) == 1:
        axes = [axes]
    for a, p in zip(axes, valid_perps):
        e = TSNE(n_components=2, perplexity=p, init="pca",
                 learning_rate="auto", random_state=RANDOM_STATE).fit_transform(X)
        scatter_by_direction(a, e[:, 0], e[:, 1], axis, sign, f"perplexity={p}")
    axes[-1].legend(handles=legend_handles(), loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    fig.tight_layout()
    fig.savefig(common.REPORTS / "fig_tsne_perplexity_grid.png", dpi=130)
    plt.close(fig)

    # --- PCA companion (deterministic sanity check) ---
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    pemb = pca.fit_transform(X)
    var = pca.explained_variance_ratio_
    fig, ax = plt.subplots(figsize=(7, 6))
    scatter_by_direction(ax, pemb[:, 0], pemb[:, 1], axis, sign,
                         f"PCA of averaged e-skin heatmaps "
                         f"(PC1 {var[0]*100:.0f}%, PC2 {var[1]*100:.0f}%)")
    ax.legend(handles=legend_handles(), loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    fig.tight_layout()
    fig.savefig(common.REPORTS / "fig_pca.png", dpi=130)
    plt.close(fig)

    embedding_rows.to_csv(common.REPORTS / "tsne_embedding.csv", index=False)

    write_results_md(counts, max_perp, valid_perps, var)
    print(f"\nWrote fig_tsne_main.png, fig_tsne_perplexity_grid.png, fig_pca.png, "
          f"tsne_embedding.csv, results.md to {common.REPORTS}")


def write_results_md(counts, primary_perp, grid_perps, pca_var):
    warn_path = common.REPORTS / "segmentation_warnings.txt"
    warnings_text = warn_path.read_text().strip() if warn_path.exists() else ""
    override_trials = sorted(common.OVERRIDES.keys())

    lines = [
        "# Direction-clustering results",
        "",
        "## Attempt counts per direction",
        "",
        "| direction | n attempts |",
        "|---|---|",
    ]
    for k, v in counts.items():
        lines.append(f"| {k} | {v} |")
    lines += [
        "",
        f"Total attempts: {int(counts.sum())} across 6 trials (2 per axis x 3 axes).",
        "",
        "## Segmentation notes",
        "",
        f"- Segmentation warnings: {warnings_text if warnings_text else 'none'}",
        f"- Trials using a per-trial rest_frac/active_frac override: "
        f"{', '.join(override_trials) if override_trials else 'none'}",
        "",
        "## t-SNE / PCA parameters",
        "",
        f"- Primary t-SNE perplexity: {primary_perp} (capped below the smallest class size)",
        f"- Perplexity grid checked: {grid_perps}",
        f"- random_state: {RANDOM_STATE}",
        f"- PCA explained variance: PC1 {pca_var[0]*100:.1f}%, PC2 {pca_var[1]*100:.1f}%",
        "",
        "## Interpretation",
        "",
        "See fig_tsne_main.png, fig_tsne_perplexity_grid.png, and fig_pca.png. "
        "Compare whether the 6 direction classes (color=axis, marker=sign) separate "
        "consistently across perplexities AND agree with the deterministic PCA "
        "projection -- agreement between the two is stronger evidence of real "
        "clustering than either alone, given the small per-class sample sizes here.",
    ]
    (common.REPORTS / "results.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
