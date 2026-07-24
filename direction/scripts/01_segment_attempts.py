"""Detect individual push/pull attempts within each YL direction recording.

Each of the 6 trials in common.TRIALS is a single continuous "free_form"
recording (no manifest rep windows) containing several repeated attempts,
separated by the handle being let go. Boundaries are detected purely from
e-skin-total dropping back to rest -- force is NOT used for the segmentation
decision (it was found to under-report some attempts: a batch of presses in
YL_negative_y peaked only ~8-9N above baseline, just under a force-based
active threshold, even though e-skin clearly showed a normal press/release
for each one). Force is still loaded and plotted as a passive reference line
for visual cross-checking only.

This script finds e-skin rest gaps, keeps the segments that reach a real
active e-skin level, trims each kept attempt down to its active sub-window,
and writes:

  direction/reports/segments.csv               -- one row per candidate segment
  direction/reports/fig_segmentation_<id>.png   -- per-trial diagnostic plot
  direction/reports/segmentation_warnings.txt   -- any under-segmentation flags

Run: python direction/scripts/01_segment_attempts.py  (from repo root)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common

COLOR_FORCE = "#c3c2b7"   # muted ink -- reference only, not used for segmentation
COLOR_ESKIN = "#eb6834"   # palette slot 2 (orange) -- drives the segmentation
COLOR_ACTIVE = "#1baf7a"  # palette slot 3 (aqua), active-window shading
COLOR_GAP = "#898781"     # muted ink, gap boundary lines


def load_trial(trial_id: str):
    d = common.DATA / trial_id
    force = pd.read_csv(d / "forces.csv")
    eskin = pd.read_csv(d / "eskin.csv")
    return force, eskin


def process_trial(trial_id: str, axis: str, sign: int, warnings: list[str]):
    force, eskin = load_trial(trial_id)
    t = force["elapsed_s"].to_numpy(float)
    F = (force["F1_N"] + force["F2_N"]).to_numpy(float)
    et = eskin["elapsed_s"].to_numpy(float)
    Etot = eskin[common.TAXELS].sum(axis=1).to_numpy(float)

    override = common.OVERRIDES.get(trial_id, {})
    rest_frac = override.get("rest_frac", common.ESKIN_REST_MARGIN_FRAC)
    active_frac = override.get("active_frac", common.ESKIN_ACTIVE_MIN_FRAC)

    segments, baseline, rng = common.find_segments_eskin(
        et, Etot, common.MIN_GAP_S, rest_frac, common.SMOOTH_WINDOW_ESKIN
    )
    qualify_thresh = baseline + active_frac * rng
    window_thresh = baseline + rest_frac * rng  # same bar as "resting", for trimming

    rows = []
    attempt_idx = 0
    for seg in segments:
        e_max_ok = seg["Etot_max"] >= qualify_thresh
        aw = common.active_window(et, Etot, seg["idx_a"], seg["idx_b"], window_thresh) if e_max_ok else None

        f_mask = (t >= seg["t0"]) & (t <= seg["t1"])
        f_max_info = float(F[f_mask].max()) if f_mask.any() else float("nan")

        if not e_max_ok:
            kept, reason = False, "below_active_threshold"
            active_t0 = active_t1 = np.nan
            n_active = 0
        elif aw is None:
            kept, reason = False, "no_active_window"
            active_t0 = active_t1 = np.nan
            n_active = 0
        else:
            active_t0, active_t1, n_active = aw
            if n_active == 0:
                kept, reason = False, "no_eskin_frames_in_active_window"
            else:
                kept, reason = True, ""
                attempt_idx += 1

        rows.append(dict(
            trial_id=trial_id, axis=axis, sign=sign,
            direction_label=common.direction_label(axis, sign),
            attempt_idx=attempt_idx if kept else -1,
            t0=seg["t0"], t1=seg["t1"], duration_s=seg["duration"],
            active_t0=active_t0, active_t1=active_t1,
            n_eskin_frames_active=n_active,
            Etot_max=seg["Etot_max"], F_max=f_max_info,
            kept=kept, reason=reason,
        ))

    n_kept = sum(r["kept"] for r in rows)
    trial_duration = et[-1] - et[0]
    longest_kept = max((r["duration_s"] for r in rows if r["kept"]), default=0.0)

    print(f"{trial_id}: e_baseline={baseline:.1f} e_range={rng:.1f} "
          f"rest_frac={rest_frac} active_frac={active_frac} -> "
          f"{len(segments)} segments, {n_kept} kept attempts")

    if n_kept < 3:
        msg = f"{trial_id}: only {n_kept} attempts kept (< 3) -- check segmentation"
        print(f"  WARNING: {msg}")
        warnings.append(msg)
    if trial_duration > 0 and longest_kept > 0.5 * trial_duration:
        msg = (f"{trial_id}: longest kept attempt is {longest_kept:.1f}s, "
               f">50% of trial duration {trial_duration:.1f}s -- "
               f"e-skin rest-mask likely missed an internal gap")
        print(f"  WARNING: {msg}")
        warnings.append(msg)

    plot_diagnostic(trial_id, t, F, et, Etot, rows)
    return rows


def plot_diagnostic(trial_id, t, F, et, Etot, rows):
    f_norm = (F - F.min()) / (F.max() - F.min() + 1e-9)
    e_norm = (Etot - Etot.min()) / (Etot.max() - Etot.min() + 1e-9)

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(t, f_norm, color=COLOR_FORCE, lw=1.0,
            label="Force (F1+F2), normalized -- reference only")
    ax.plot(et, e_norm, color=COLOR_ESKIN, lw=1.2,
            label="E-skin total, normalized -- drives segmentation")

    for r in rows:
        if r["kept"]:
            ax.axvspan(r["active_t0"], r["active_t1"], color=COLOR_ACTIVE, alpha=0.25, lw=0)
            mid = 0.5 * (r["active_t0"] + r["active_t1"])
            ax.text(mid, 1.05, str(r["attempt_idx"]), ha="center", va="bottom",
                     fontsize=8, color="#52514e", transform=ax.get_xaxis_transform())
        ax.axvline(r["t1"], color=COLOR_GAP, lw=0.6, ls="--", alpha=0.6)

    ax.set_ylim(-0.05, 1.3)
    ax.set_xlabel("elapsed_s")
    ax.set_ylabel("normalized signal")
    ax.set_title(f"Segmentation diagnostic: {trial_id}", pad=14)
    ax.legend(loc="upper right", frameon=False, bbox_to_anchor=(1.0, 1.18))
    ax.grid(True, color="#e1e0d9", lw=0.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    out = common.REPORTS / f"fig_segmentation_{trial_id}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main():
    common.REPORTS.mkdir(parents=True, exist_ok=True)
    all_rows = []
    warnings: list[str] = []
    for trial_id, (axis, sign) in common.TRIALS.items():
        rows = process_trial(trial_id, axis, sign, warnings)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    out_csv = common.REPORTS / "segments.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nWrote {len(df)} segment rows ({df['kept'].sum()} kept) to {out_csv}")

    warn_path = common.REPORTS / "segmentation_warnings.txt"
    warn_path.write_text("\n".join(warnings) + ("\n" if warnings else ""))
    print(f"{len(warnings)} warning(s) written to {warn_path}")


if __name__ == "__main__":
    main()
