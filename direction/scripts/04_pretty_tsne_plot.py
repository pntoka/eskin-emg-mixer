"""Presentation-ready version of fig_tsne_main.png for slides.

Same t-SNE embedding as 03_tsne_plot.py (read from the already-computed
direction/reports/tsne_embedding.csv, so results stay identical -- no
re-running TSNE), but with visible axes + tick numbers, a grid, larger
fonts, and 300 dpi output for the pretty_plots/ folder.

Run: python direction/scripts/04_pretty_tsne_plot.py  (from repo root)
"""
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import common

OUT_DIR = common.ROOT.parents[0] / "pretty_plots"

AXIS_COLOR = {"x": "#2a78d6", "y": "#1baf7a", "z": "#d62728"}
SIGN_MARKER = {1: "o", -1: "^"}

FONT_AXIS_LABEL = 16
FONT_TICK = 13
FONT_TITLE = 17
FONT_LEGEND = 13


def legend_handles():
    handles = []
    for a in ("x", "y", "z"):
        for s in (1, -1):
            handles.append(Line2D([0], [0], marker=SIGN_MARKER[s], color="none",
                                   markerfacecolor=AXIS_COLOR[a], markersize=10,
                                   label=common.direction_label(a, s)))
    return handles


def main():
    emb = pd.read_csv(common.REPORTS / "tsne_embedding.csv")
    perplexity = emb["perplexity"].iloc[0]

    fig, ax = plt.subplots(figsize=(8.5, 7))
    for a in ("x", "y", "z"):
        for s in (1, -1):
            sub = emb[(emb["axis"] == a) & (emb["sign"] == s)]
            if sub.empty:
                continue
            ax.scatter(sub["x"], sub["y"], color=AXIS_COLOR[a], marker=SIGN_MARKER[s],
                       s=70, alpha=0.85, edgecolors="none",
                       label=common.direction_label(a, s))

    ax.set_title(f"t-SNE of averaged e-skin heatmaps per attempt (perplexity={perplexity})",
                 fontsize=FONT_TITLE)
    ax.set_xlabel("t-SNE dimension 1", fontsize=FONT_AXIS_LABEL)
    ax.set_ylabel("t-SNE dimension 2", fontsize=FONT_AXIS_LABEL)
    ax.tick_params(axis="both", labelsize=FONT_TICK)
    ax.grid(True, color="#d8d8d8", lw=0.8, zorder=0)
    ax.set_axisbelow(True)

    ax.legend(handles=legend_handles(), loc="center left", bbox_to_anchor=(1.0, 0.5),
              frameon=False, fontsize=FONT_LEGEND)
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "fig_tsne_main_pretty.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
