"""Build the pooled per-attempt feature dataset from direction/reports/segments.csv.

For each trial, computes the whole-trial per-taxel baseline (5th percentile,
before any trimming) once, then for every kept attempt averages the e-skin
frames in its active window and subtracts that baseline -> one 256-dim
feature vector per attempt.

Outputs:
  direction/reports/dataset.npz            -- X (n,256), axis, sign,
                                               direction_label, trial_id, attempt_idx, taxels
  direction/reports/attempts_summary.csv   -- same metadata, no feature vectors

Run: python direction/scripts/02_build_dataset.py  (from repo root)
"""
import numpy as np
import pandas as pd

import common


def main():
    segments = pd.read_csv(common.REPORTS / "segments.csv")
    kept = segments[segments["kept"]].sort_values(["trial_id", "attempt_idx"])

    features, axis, sign, direction_label, trial_id, attempt_idx, n_frames = (
        [], [], [], [], [], [], []
    )

    for tid, group in kept.groupby("trial_id", sort=False):
        eskin = pd.read_csv(common.DATA / tid / "eskin.csv")
        E = eskin[common.TAXELS].to_numpy(float)
        taxel_baseline = np.percentile(E, 5, axis=0)

        for _, row in group.iterrows():
            feat, n = common.attempt_heatmap(
                eskin, taxel_baseline, row["active_t0"], row["active_t1"]
            )
            features.append(feat)
            axis.append(row["axis"])
            sign.append(int(row["sign"]))
            direction_label.append(row["direction_label"])
            trial_id.append(tid)
            attempt_idx.append(int(row["attempt_idx"]))
            n_frames.append(n)

        print(f"{tid}: {len(group)} attempts -> features extracted "
              f"(taxel baseline range [{taxel_baseline.min():.1f}, {taxel_baseline.max():.1f}])")

    X = np.array(features)
    axis = np.array(axis)
    sign = np.array(sign, dtype=np.int8)
    direction_label = np.array(direction_label)
    trial_id = np.array(trial_id)
    attempt_idx = np.array(attempt_idx)
    taxels = np.array(common.TAXELS)

    out_npz = common.REPORTS / "dataset.npz"
    np.savez(out_npz, X=X, axis=axis, sign=sign, direction_label=direction_label,
              trial_id=trial_id, attempt_idx=attempt_idx, taxels=taxels)
    print(f"\nWrote {X.shape} feature matrix to {out_npz}")

    summary = pd.DataFrame(dict(
        trial_id=trial_id, axis=axis, sign=sign, direction_label=direction_label,
        attempt_idx=attempt_idx, n_eskin_frames=n_frames,
        feature_sum=X.sum(axis=1), feature_peak=X.max(axis=1),
    ))
    summary.to_csv(common.REPORTS / "attempts_summary.csv", index=False)

    print("\nPer-direction attempt counts:")
    counts = summary.groupby("direction_label").size().sort_values(ascending=False)
    print(counts.to_string())
    if counts.max() > 2 * counts.min():
        print(f"\nWARNING: class imbalance -- largest class ({counts.max()}) is more than "
              f"2x the smallest ({counts.min()})")


if __name__ == "__main__":
    main()
