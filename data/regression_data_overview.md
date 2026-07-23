# Data overview for e-skin → force regression

## Context / why this doc exists

The original approach recorded grasps at fixed *target* forces and tried to
correlate a summary e-skin reading with that target. The correlation isn't
clean — the e-skin reading varies too much trial-to-trial (and within a
trial) at a given nominal target force.

Because e-skin and force are recorded **simultaneously** at the sample
level, the new direction is to treat this as a **regression problem**:
learn a mapping from the raw 16×16 e-skin frame (256 taxels) to the
combined grip force `F_combined = F1 + F2` at each timestep, using the
existing recordings as `(eskin_frame, force)` training pairs. There is no
target-force label needed for this — every sample in every trial is a
valid pair.

This doc is EMG-free by design: EMG is not relevant to an eskin→force
regression and is omitted entirely, including any notes about which trials
have missing/empty EMG files.

## Data format & sampling rates

| Modality | Native rate | Columns |
|---|---|---|
| E-skin | ~200 Hz (16×16 = 256 FSR taxels; polled over serial with no fixed rate limiter, so actual rate drifts a bit trial to trial) | `wall_time, elapsed_s, R00_C00 … R15_C15` (258 columns total) |
| Force | 100 Hz (on-device block-averaged from a 1000 Hz raw ADC sample rate) | `wall_time, elapsed_s, F1_N, F2_N` |

Notes:
- `F_combined = F1_N + F2_N` is the physically meaningful total grasp force
  and is the natural regression target. F2's calibration is currently a
  placeholder copied from F1's calibration (see `PROJECT.md`) — worth a
  sanity check before treating F1/F2 as independently trustworthy, though
  `F_combined` itself should still be reasonable.
- Both streams carry `elapsed_s` measured from the same per-trial origin
  (`manifest.json`'s `start_wall_time`), so eskin rows and force rows can be
  time-aligned. **The two streams are not on the same clock grid** — eskin
  ~200 Hz vs force 100 Hz — so pairing them requires a resampling/
  interpolation decision (e.g. downsample eskin to the force timestamps,
  or upsample force to eskin timestamps, or resample both to a common
  grid). This repo's `src/processing/align.py` already contains eskin/force
  CSV loaders and elapsed-time handling that can be reused as a starting
  point instead of re-deriving the parsing from scratch.
- `manifest.json` fields relevant here (per trial, in `data/<trial_id>/`):
  `trial_id`, `subject_id`, `task_kind` (`max_effort` | `target_force`),
  `duration_s`, `repetitions_total`, `rest_s`,
  `repetitions[].start_wall_time`/`end_wall_time`, `target_force_n`,
  `tolerance_n`, `aborted`, `start_wall_time`, `stop_wall_time`. The
  manifest also stores absolute file paths for the CSVs from the recording
  machine — ignore these and load `eskin.csv`/`forces.csv` from next to the
  manifest instead.

## Data inventory — `data/` (22 trials)

Trial type is determined by `manifest.json`'s `task_kind` and
`target_force_n`, not by folder name (folder names are inconsistent —
e.g. `AM_hold_001_...` is actually `task_kind=target_force`, and
`AM_max_squeeze_n_5`/`AM_max_squeeze_n_5_old`'s manifests internally say
`subject_id: P1`, not AM — likely mislabeled/copied folders, flag before
trusting subject attribution for those two).

**`max_effort` (max squeeze) — 8 trials.** These deliberately squeeze as
hard as possible and are believed to **saturate the e-skin sensor** (many
taxels pinned at max reading regardless of force) — per your note, probably
not good regression training data, at least not without special handling
(e.g. excluding saturated samples, or using them only as an
out-of-distribution/high-force check rather than for training). Listed for
completeness:

| Folder | eskin rows | force rows | span |
|---|---|---|---|
| `P1_001_20260721_153217` | 1,542 | 503 | ~5.0 s |
| `P1_001_20260721_154040` | 8,515 | 2,522 | ~25.2 s |
| `P1_001_20260721_155335` (aborted) | 3,284 | 1,165 | ~11.7 s |
| `P1_001_20260721_155529` | 7,351 | 2,517 | ~25.2 s |
| `AM_max_squeeze_n_5` (manifest subject_id: P1) | 10,864 | 4,531 | ~45.4 s |
| `AM_max_squeeze_n_5_old` (manifest subject_id: P1) | 11,527 | 4,540 | ~45.4 s |
| `narges_max_squeeze_001_20260722_173017` | 5,268 | 2,515 | ~25.2 s |
| `PT_max_squeeze_001_20260722_153104` | 13,611 | 4,537 | ~45.4 s |

**`target_force` — 14 trials.** Force varies continuously within and across
these trials (ramp up to target, hold, release, repeat), which is exactly
the useful regime for regression — a much better primary training source
than the max-effort trials:

| Folder | Target (N) | eskin rows | force rows | span |
|---|---|---|---|---|
| `P1_002_20260721_155648` | 5 | 7,130 | 2,521 | ~25.2 s |
| `narges_target_10_002_20260722_173151` | 10 | 2,491 | 1,524 | ~73.0 s |
| `P1_003_20260721_155726` (aborted) | 12 | 5,082 | 2,049 | ~20.5 s |
| `narges_target_15_003_20260722_173534` | 15 | 4,315 | 1,520 | ~56.5 s |
| `PT_target_15_001_20260722_153324` | 15 | 5,980 | 2,540 | ~61.9 s |
| `P1_002_20260721_161939` | 20 | 3,745 | 1,517 | ~15.2 s |
| `AM_hold_001_20260721_165006` | 20 | 4,190 | 1,524 | ~54.6 s |
| `PT_target_30_002_20260722_153507` | 30 | 6,158 | 2,545 | ~66.9 s |
| `PT_target_45_003_20260722_153749` (aborted) | 45 | 3,903 | 1,512 | ~356.6 s (long stall before abort) |
| `PT_target_45_004_20260722_154404` (redo of 003) | 45 | 6,463 | 2,550 | ~106.2 s |
| `PT_target_60_005_20260722_154642` | 60 | 6,318 | 2,552 | ~68.3 s |
| `PT_target_75_006_20260722_154945` (aborted) | 75 | 2,411 | 1,017 | ~49.4 s |
| `PT_target_75_007_20260722_155050` (aborted) | 75 | 1,393 | 504 | ~45.6 s |
| `PT_target_75_008_20260722_155202` | 75 | 2,406 | 1,023 | ~44.0 s |

Aborted trials aren't unusable — the eskin/force pairs up to the abort
point are still valid samples — they're just shorter.

Row counts above include a header line each; treat them as approximate
sample counts (off by one).

## Data inventory — `archive/` (legacy, pre-manifest recordings)

Two older recording sessions, from before the manifest/task/rep structure
existed:

| Session | eskin rows | force rows |
|---|---|---|
| `eskin_20260720_165516.csv` / `forces_20260720_165516.csv` | 10,594 | 4,838 |
| `eskin_20260720_173729.csv` / `forces_20260720_173729.csv` | 8,648 | 3,824 |

- **Same CSV schema** as current `data/` trials for both eskin
  (`wall_time, elapsed_s, R00_C00…R15_C15`) and forces
  (`wall_time, elapsed_s, F1_N, F2_N`) — directly loadable with the same
  parsing logic, no format conversion needed.
- **No `manifest.json`** — no task type, no target force, no rep
  boundaries. Still fully usable as unlabeled `(eskin_frame, force)` pairs
  for the regression, since no task label is needed for that mapping; just
  treat each as one continuous unstructured trial.
- `force_bias_calibration.json` in the same folder: `{"BIAS1":
  3.986257791600007, "BIAS2": 3.1104103968000305}` — zero-offset bias (N)
  for F1/F2 from this recording session. Check whether it's already
  applied in the CSV's `F1_N`/`F2_N` values or needs to be subtracted
  before use.
- `PT_max_squeeze.c3d` — a motion-capture recording, different modality
  entirely (not eskin/force CSV data); not usable for this regression
  without a separate C3D reader, and not a priority.
- `summer_school_project 2.py` — the legacy PyQt recorder script that
  produced these two archive sessions (predecessor of
  `src/recording/session.py`); mentioned only for provenance, not needed
  for the regression pipeline itself.

## Combined totals (approximate sample counts)

- `data/` target_force trials only (recommended primary training set):
  **~63,006 eskin rows / ~25,394 force rows** across 14 trials.
- `data/` max_effort trials (likely saturated, use with caution or exclude):
  **~58,682 eskin rows / ~22,832 force rows** across 8 trials.
- `data/` all 22 trials combined: **~121,747 eskin rows / ~44,637 force
  rows**.
- `archive/` legacy sessions: **~19,242 eskin rows / ~8,662 force rows**
  across 2 sessions.
- **Grand total available: ~140,989 eskin rows / ~53,299 force rows**
  before any resampling/pairing to a common time grid, and before deciding
  whether to keep or drop the max_effort trials.

## Open items for the regression-design conversation

These are flagged as decisions for the next step, not resolved here:

1. **Rate mismatch**: eskin (~200 Hz) vs force (100 Hz) need to be paired
   onto a common time base — pick a resampling strategy (downsample eskin,
   upsample force, or interpolate both to a shared grid).
2. **max_effort trials**: decide whether to exclude them from training
   entirely (saturation likely breaks the eskin→force relationship at the
   high end), keep them only for OOD/high-force evaluation, or mask out
   saturated taxels/samples.
3. **Labeled vs unlabeled pooling**: whether to pool `data/` (labeled with
   task/target) and `archive/` (unlabeled, no manifest) trials together for
   training, or hold `archive/` out separately since it lacks rep/task
   structure.
4. **Train/val split**: likely by trial or by subject (AM/P1/PT/narges) to
   avoid leaking near-duplicate frames from the same grasp across splits.
5. **Reuse existing code**: `src/processing/align.py` already has
   eskin/force CSV loading and elapsed-time alignment helpers — reuse
   rather than reimplement, ignoring the EMG-specific parts of that module.
