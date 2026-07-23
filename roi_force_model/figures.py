"""Three slide-ready figures for the e-skin ROI -> grip-force story.

All numbers come from train.build_and_eval() so titles match the model's
printed metrics. Large fonts + tight layout for projection.

Run:  python roi_force_model/figures.py   ->  outputs/slide_1|2|3_*.png

  slide_1  the e-skin ROI response saturates above ~30 N  (why we cap there)
  slide_2  below the knee, predicted vs measured force is accurate (held-out)
  slide_3  the model in action: tracks force below the knee, flags saturated peaks
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pipeline import ROOT, metrics  # noqa: E402
from train import build_and_eval    # noqa: E402

OUT = HERE / "outputs"

plt.rcParams.update({
    "font.size": 15, "axes.titlesize": 17, "axes.labelsize": 15,
    "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 12,
    "figure.dpi": 150, "savefig.bbox": "tight",
})


def slide_1_saturation(r):
    """Sensor response curve: ROI reading vs force, flattening above the knee."""
    F, roi = r["F"], r["rmean"]
    knee = r["knee_N"]
    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.scatter(F, roi, s=4, alpha=.12, color="0.5", label="samples (whole trial)")

    # binned median response (bin by force)
    edges = np.arange(0, min(F.max(), 85) + 5, 5)
    cx, cy = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (F >= lo) & (F < hi)
        if m.sum() >= 20:
            cx.append((lo + hi) / 2); cy.append(np.median(roi[m]))
    ax.plot(cx, cy, "o-", color="tab:blue", lw=2.5, ms=6, label="median response")

    x_hi = ax.get_xlim()[1]
    ax.axvspan(knee, x_hi, color="tab:red", alpha=.08)
    ax.axvline(knee, color="tab:red", ls="--", lw=2, label=f"saturation knee ≈ {knee:.0f} N")
    ax.text((knee + x_hi) / 2, 0.5 * (roi.max() + roi.min()),
            "saturated\nreading flat →\nforce unresolved", color="tab:red",
            fontsize=12, ha="center", va="center")
    ax.set(xlabel="grip force  F1+F2  (N)", ylabel="e-skin ROI mean pressure",
           title="E-skin ROI saturates above ~30 N")
    ax.legend(loc="lower right", framealpha=.95)
    p = OUT / "slide_1_saturation.png"
    fig.savefig(p); plt.close(fig)
    return p


def slide_2_accuracy(r):
    """Held-out predicted vs measured force, in the resolvable regime (measured
    force below the knee -- the range the e-skin can actually resolve)."""
    F, pred, te = r["F"], r["pred_te"], r["te"]
    mask = F[te] < r["knee_N"]
    ya, yp = F[te][mask], pred[mask]
    s = metrics(ya, yp)
    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    ax.scatter(ya, yp, s=7, alpha=.25, color="tab:green")
    lo, hi = min(ya.min(), yp.min()), max(ya.max(), yp.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.5, label="perfect")
    ax.set(xlabel="measured  F1+F2  (N)", ylabel="predicted  F1+F2  (N)",
           title=f"Below the knee (< {r['knee_N']:.0f} N): predicted vs measured\n"
                 f"(held-out, unseen in time)")
    ax.text(.05, .95, f"RMSE = {s['rmse']:.1f} N\nMAE  = {s['mae']:.1f} N\nR²   = {s['r2']:.2f}",
            transform=ax.transAxes, va="top", fontsize=14,
            bbox=dict(boxstyle="round", fc="white", ec="0.7"))
    ax.legend(loc="lower right")
    ax.set_aspect("equal", adjustable="box")
    p = OUT / "slide_2_accuracy.png"
    fig.savefig(p); plt.close(fig)
    return p


def slide_3_timeseries(r):
    """Model in action on the held-out test block: prediction tracks measured
    force below the knee; saturated peaks are flagged, not guessed."""
    F, t, te, pred, ok = r["F"], r["t"], r["te"], r["pred_te"], r["gate_ok"]
    knee = r["knee_N"]
    tt = t[te] - t[te][0]                      # seconds from start of held-out block
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.plot(tt, F[te], color="black", lw=1.6, label="measured F1+F2")
    # predicted where the sensor is usable; faint where the gate flags saturation
    pred_ok = np.where(ok, pred, np.nan)
    pred_sat = np.where(~ok, pred, np.nan)
    ax.plot(tt, pred_ok, color="tab:green", lw=1.6, label="predicted (usable)")
    ax.plot(tt, pred_sat, color="tab:green", lw=1.2, alpha=.35,
            label="predicted (flagged saturated)")
    ax.axhline(knee, color="tab:red", ls="--", lw=1.5, label=f"saturation knee {knee:.0f} N")
    ax.set(xlabel="time within held-out block (s)", ylabel="F1+F2 (N)",
           title="Model tracks grip force below the knee; flags saturation at the peaks")
    ax.legend(loc="upper right", ncol=2, framealpha=.95)
    p = OUT / "slide_3_timeseries.png"
    fig.savefig(p); plt.close(fig)
    return p


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    r = build_and_eval()
    for fn in (slide_1_saturation, slide_2_accuracy, slide_3_timeseries):
        p = fn(r)
        print(f"wrote {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
