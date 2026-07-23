# Phase 2 plan: more dynamic data + closing the generalization gap

## 1. Where we are and why this plan looks the way it does

The Hour-1 feasibility test (`HOUR1_ESKIN_FORCE_FEASIBILITY.md`) and the follow-up
archive-only model (`04_archive_model.py`, `results_archive.md`) established:

- The 10 `target_force` "hold" trials are **not usable for training** — they're
  deliberately narrow-band (hold near a fixed target), so there's almost no
  within-trial force variation for a model to learn from. Keep excluding them.
- The 2 `archive/` sessions (unstructured, wide dynamic range) are where the
  real signal lives: within-session R² of 0.63–0.76 on the full 256-taxel
  model.
- But with only **2 sessions**, leave-one-session-out generalization is bad
  for the 256-taxel model (R² −0.22 / −2.36 — it overfits each session's
  specific contact geometry) and only partially works for the simple summed
  signal (R² 0.40 / 0.47 — and even that plateaus at a per-session constant
  during a squeeze rather than tracking graded force, see
  `fig_archive_loso_timeseries.png`).
- **Diagnosis:** 2 sessions is not enough diversity for any model to learn a
  session-invariant mapping — it can only ever overfit one geometry or fall
  back to a crude on/off signal. The fix is more independent sessions with
  deliberately varied contact patterns and speeds, not a fancier model.

Constraints for this round, as given: **4 subjects, ~2–3 minutes of recording
each**, using `archive/summer_school_project 2.py` (the simple Start/Stop
Recording + Calibrate Zero tool — no manifest, no on-screen target force, no
rep structure). The handle is bolted to a table, so grasping is the only
motion available (no repositioning the handle itself).

That's roughly **10–12 minutes of new raw recording across 4 new
sessions**, on top of the existing 2 archive sessions — 6 independent
sessions total. That's enough to move from a meaningless n=2
leave-one-out to a real `GroupKFold`/leave-one-subject-out evaluation, which
is the main thing this round needs to produce.

## 2. Recording protocol (per subject, ~2.5 min, fits the 2–3 min budget)

Do this once per subject. Everything after "hands off the handle" moments
should genuinely have zero contact — those are the baseline references the
whole pipeline depends on.

**Before recording starts:**
1. Let the tool's auto-tare finish (hands off the e-skin/handle) — don't
   start recording until the GUI shows tare is done.
2. Press **Calibrate Zero (1s)** with hands completely off the handle. Do
   this **fresh for every subject**, not just once at the start of the day —
   the existing archive sessions' inconsistent resting force levels are
   likely partly because this wasn't done consistently.
3. Start recording.

**Recording (~2:10, leaves buffer under the 2–3 min budget):**

| Time | Phase | What to do |
|---|---|---|
| 0:00–0:08 | Rest | Hands fully off the handle. Clean zero-force / zero-contact baseline. |
| 0:08–1:10 | Varied squeeze-release cycles (~8–10 reps) | Squeeze and release the handle repeatedly, deliberately varying **both** effort (light / medium / firm — self-judged, no target display) **and** speed (some slow ~3s ramps up-and-down, some quick ~1s pulses). Leave ~1–2s of hands-off rest between reps. This is the core data — it's what gave the archive sessions their signal. |
| 1:10–1:50 | Grip-position variation (~4–5 reps) | Same squeeze motion, but shift hand/finger placement on the handle between reps (different finger spread, higher/lower on the handle, rotated grip) at a couple of effort levels. The handle can't move, but the *contact patch* can — this directly targets the problem of the 256-taxel model overfitting one fixed contact geometry. |
| 1:50–2:10 | Final rest | Hands off again. A second baseline lets us check for zero-drift over the session. |

Notes:
- No need for a target-force display or verbal countdown — "light / medium /
  firm, your judgment" is enough. The point is coverage of the force range,
  not hitting precise numbers (that's what the `target_force` trials were
  for, and they didn't work well for this).
- If a subject has more than 2–3 minutes to spare, more reps in the
  squeeze-release phase are more valuable than a longer rest — that's the
  phase producing the real signal.

## 3. Manual session log (since the archive recorder writes no manifest)

Keep a simple log at **`regression/data_log.csv`** with one row per
recording. Columns:

```
session_id,subject_label,date,duration_s,notes
```

- `session_id`: the timestamp suffix from the recorder's output filenames
  (e.g. `20260724_143012`, matching `eskin_20260724_143012.csv`).
- `subject_label`: an arbitrary short tag (`S1`, `S2`, `S3`, `S4`) — no need
  for real names.
- `date`: recording date.
- `duration_s`: roughly how long the recording ran.
- `notes`: anything unusual — grip style attempted, hand size/dominant hand
  if relevant, any interruption, anything that looked like a glitch on the
  live plot.

Example:

```
session_id,subject_label,date,duration_s,notes
20260724_143012,S1,2026-07-24,148,"right-handed, normal grip, no issues"
20260724_150501,S2,2026-07-24,152,"left-handed, smaller hands, one accidental bump ~1:40"
```

## 4. Getting new recordings into the pipeline

1. Copy each subject's `eskin_<id>.csv` / `forces_<id>.csv` pair into
   `regression/data/<subject_label>_<session_id>/eskin.csv` /
   `forces.csv` — same convention already used for `archive_165516/`,
   `archive_173729/` (no manifest needed).
2. Append the corresponding row(s) to `regression/data_log.csv`.
3. **Code change needed in `common.py`:** replace the hardcoded
   `ARCHIVE_TRIALS = ["archive_165516", "archive_173729"]` list with logic
   that reads `regression/data_log.csv` and builds the trial list (and
   `subject_label`) from it, so adding future sessions never requires
   editing code again — just recording + logging. Keep `DATA_TRIALS` (the
   manifest-based `target_force` list) as-is; it stays excluded from
   training either way.
4. Re-run `01_build_dataset.py` — the existing alignment / baseline /
   idle-trimming logic already handles manifest-less sessions (the
   force-threshold-based activity detector in `trim_idle_mask`), so this
   should just work on the new sessions unchanged.

## 5. Re-evaluation once the new data is in

1. Re-run the archive-style leave-one-*group*-out evaluation
   (`04_archive_model.py`, generalized from 2 archive sessions to all
   sessions), but switch the grouping key from session to **subject**
   (`subject_label`) wherever a subject only has one session — leave-one-
   -subject-out is the more realistic test of deployment generalization
   than leave-one-session-out, now that real subject labels exist.
2. Compare `raw256` vs `total` again. The key question this round is meant
   to answer: **does `raw256` start beating `total` once there are ~6
   independent contact geometries instead of 2?** If yes, that confirms the
   "not enough diversity yet" diagnosis and the model is on the right
   track. If `raw256` still can't beat `total` cross-subject, the plateau
   problem is more fundamental than sample diversity (see §6).
3. Also try the already-implemented-but-untested `scalar_features()` in
   `common.py` (`top16`, `area`, `peak`, `sat_frac`) as a middle ground
   between the too-coarse `total` and the too-overfit `raw256` — cheap to
   test, might generalize better than either extreme.
4. Retrain and save updated final model(s), re-generate
   `fig_archive_loso_timeseries.png` / `fig_archive_scatter.png` equivalents
   for the new evaluation, and update `results_archive.md`.

## 6. What to try if generalization is still poor after this round

Not to be done pre-emptively — only if step 5 shows the plateau problem
persists across subjects, not just across the original 2 sessions:

- **Per-session affine recalibration** (already validated conceptually in
  the Hour-1 test — a short calibration touch at the start of a new
  session, fit gain+offset, apply for the rest). Cheap and already coded
  once (Step 5 of the Hour-1 script) — could be revived as a
  "first-N-seconds-of-each-held-out-session" calibration step here too.
- **Normalize contact pattern from magnitude**: feed the model each frame's
  *shape* (taxel values divided by that frame's own total) separately from
  its *magnitude* (the total), instead of raw taxel values — this could
  let a shape-independent gain live on top of a session-specific pattern,
  rather than asking one linear model to encode both at once.

## 7. Success criteria for this round

| Result | Reading |
|---|---|
| `raw256` leave-one-subject-out R² clearly positive (>0.5) and better than `total` | Diversity was the fix — proceed to collecting more sessions the same way, or start thinking about live deployment. |
| `raw256` still negative/poor, but `total` (or `top16`/`area`) generalizes reasonably (>0.5) across subjects | The spatial detail isn't transferable yet, but a simpler summary signal is — a coarser but real force-sensing capability exists now; worth deciding whether that's good enough for a first live demo while spatial generalization gets more data. |
| Nothing generalizes across subjects, even the simple features | Points at something more structural (e.g. per-subject force-sensor calibration, or grip mechanics varying more than e-skin contact can capture) — worth revisiting the calibration/normalization ideas in §6 before recording much more data. |
