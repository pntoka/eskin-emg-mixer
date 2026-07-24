"""Split the single long YL_grasp_dynamic recording into individual trials
by detecting rest gaps (F_combined dropping back to baseline for a
sustained period) between grasps.

Segmentation: smooth F_combined with a short rolling median (denoise only,
not used downstream), find contiguous runs sitting within ARGIN of the
session's 5th-percentile baseline, and treat runs >= MIN_GAP_S as real
inter-trial rests (short in-trial blips from a quick regrip don't qualify --
empirically these two populations are well separated: blips <=0.3s vs real
gaps >=3.5s, see the margin=3.0 exploration). The boundary between two
adjacent trials is placed at the midpoint of the gap between them, which by
construction sits inside a multi-second true rest, so no active squeeze
data is ever cut through.
"""
from pathlib import Path
import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parents[2] / "data" / "YL_grasp_dynamic_002_20260723_155859"
OUT_DATA = Path(__file__).resolve().parents[1] / "data"

REST_MARGIN_N = 3.0
MIN_GAP_S = 2.0
SMOOTH_WINDOW = 21  # ~0.2s at ~100Hz, denoise-only for segmentation


def find_segments(t: np.ndarray, F: np.ndarray):
    s = pd.Series(F).rolling(SMOOTH_WINDOW, center=True, min_periods=1).median().to_numpy()
    baseline = np.percentile(s, 5)
    resting = s <= baseline + REST_MARGIN_N

    n = len(resting)
    runs = []
    i = 0
    while i < n:
        if resting[i]:
            j = i
            while j < n and resting[j]:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1

    gaps = [r for r in runs if (t[r[1] - 1] - t[r[0]]) >= MIN_GAP_S]
    edges = [0] + [(g[0] + g[1]) // 2 for g in gaps] + [n]

    segments = []
    for k in range(len(edges) - 1):
        a, b = edges[k], edges[k + 1]
        if b <= a:
            continue
        seg_F = F[a:b]
        segments.append(dict(idx_a=a, idx_b=b, t0=t[a], t1=t[b - 1],
                              duration=t[b - 1] - t[a], n=b - a,
                              F_max=seg_F.max(), F_min=seg_F.min()))
    return segments, baseline


print(f"Loading {SRC / 'forces.csv'} ...")
force = pd.read_csv(SRC / "forces.csv")
t = force["elapsed_s"].to_numpy(float)
F = (force["F1_N"] + force["F2_N"]).to_numpy(float)

segments, baseline = find_segments(t, F)
print(f"baseline (5th pct, smoothed): {baseline:.2f} N")
print(f"{len(segments)} raw segments found (incl. leading/trailing rest-only):")
for k, seg in enumerate(segments):
    print(f"  [{k}] t=[{seg['t0']:7.2f},{seg['t1']:7.2f}] dur={seg['duration']:6.2f}s "
          f"n={seg['n']:5d} F=[{seg['F_min']:6.1f},{seg['F_max']:6.1f}]")

# Drop segments that never rise meaningfully above baseline -- these are the
# leading/trailing rest-only stretches, not trials.
ACTIVE_MIN_F = baseline + 10.0  # N above baseline to count as "a real trial happened"
trials = [seg for seg in segments if seg["F_max"] >= ACTIVE_MIN_F]
print(f"\n{len(trials)} active trials kept "
      f"(dropped {len(segments) - len(trials)} rest-only segment(s)).")

print(f"\nLoading {SRC / 'eskin.csv'} (this is the big one, ~124MB) ...")
eskin = pd.read_csv(SRC / "eskin.csv")
eskin_t = eskin["elapsed_s"].to_numpy(float)

print()
for i, seg in enumerate(trials, start=1):
    name = f"YL_dynamic_{i:02d}"
    out_dir = OUT_DATA / name
    out_dir.mkdir(parents=True, exist_ok=True)

    f_slice = force.iloc[seg["idx_a"]:seg["idx_b"]]
    f_slice.to_csv(out_dir / "forces.csv", index=False)

    e_mask = (eskin_t >= seg["t0"]) & (eskin_t <= seg["t1"])
    e_slice = eskin.loc[e_mask]
    e_slice.to_csv(out_dir / "eskin.csv", index=False)

    print(f"  {name}: t=[{seg['t0']:.2f},{seg['t1']:.2f}] dur={seg['duration']:.2f}s "
          f"-> {len(f_slice)} force rows, {len(e_slice)} eskin rows")

print(f"\nWrote {len(trials)} trial folders under {OUT_DATA}")
