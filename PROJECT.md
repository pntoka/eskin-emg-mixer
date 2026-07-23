# Project Notes & Onboarding

Context and decisions that are **not obvious from the code** — read this first,
then `README.md` (how to run) and `docs/alignment_pipeline.md` (how alignment
works). Keep it current as things are confirmed.

## What this is

Imperial College London summer-school project. We record **grasping trials**
with three sensors and study the relationship between them:

- **E-skin** — a 16×16 FSR (Force-Sensitive Resistor) pressure matrix (256 cells).
- **Force** — two handle-mounted load cells, `F1`/`F2`, in Newtons.
- **EMG** — Cometa **WaveX**, muscle activation (electrode-free effort is the
  eventual aim; see direction below).

Two task types: **max_effort** (squeeze as hard as possible, N reps) and
**target_force** (hold a target, e.g. 20 N ± 2 N, for a set time).

## Project direction (the "why")

The end goal is estimating **grip force / effort** from wearable contact
sensing. Key strategic conclusions reached so far:

- **e-skin (+force) already give grip force directly** — after calibration, an
  FSR array is a force sensor. This is the solid, deployable target.
- **Predicting EMG *from* force is ill-posed** — muscle→force is the natural
  causal arrow; inverting it (force→which-muscles) is underdetermined and
  confounded by fatigue/posture/co-contraction. Only a coarse *effort level*
  tracks force.
- **The intended pivot: replace noisy/unstable EMG with FMG** (Force
  Myography) — the *same FSR-array technology*, but wrapped around the
  **forearm** over the finger muscles. Muscle bulge → pressure → a stable,
  electrode-free activation proxy. "e-skin at the grip + FMG on the forearm"
  is the north-star sensing setup.
- **EMG in the current data is ground-truth / comparison**, not on the critical
  path for the force-estimation product.

## Hardware & sensors

- E-skin + force boards: two Arduino-like USB serial devices (protocols and
  port auto-detection live in `src/sensors/`).
- Force: `F1`,`F2` are **summed** into total grasp force (`F_combined = F1+F2`)
  — the two cells act on opposite sides of the grip and add up.
- EMG (WaveX): driver/SDK is **Windows-only**; the recorder drives
  `EMG_Eyetracker_Tool.exe` via simulated F5/F6 hotkeys (`src/emg/`). Not usable
  from Linux directly. On the recording rig, ~**8 channels** are wired but
  typically **only ~2 carry real EMG** (see below).

## Data format (`data/<trial_id>/`)

Each recorded trial is one folder with:

- `manifest.json` — task kind, subject, `start_wall_time`, and **per-rep
  windows** (`start/end_wall_time`). The rep windows are **labelled ground
  truth** — used to select EMG channels, validate alignment, and segment reps.
- `forces.csv`, `eskin.csv` — `wall_time`, `elapsed_s`, then the values.
  `eskin.csv` has 256 `R00_C00..R15_C15` taxel columns. Both share **one
  `elapsed_s` origin** (trial start).
- `emg_raw.txt` — **optional**, headerless whitespace text. Blank-line-separated
  blocks of 8 channel-major lines; concatenating line *k* across blocks rebuilds
  channel *k* → `(8, N)`. **No timestamps / labels / rate in the file** — rate
  is nominal **2000 Hz**, channels named `Emg_1..Emg_8`.

Gotcha: the manifest's `*_csv`/`emg_txt` paths are absolute Windows paths from
the recording machine — ignore them; load the files next to the manifest.

### Key data facts
- **Only ~2 EMG channels are real** (quiet at rest, active during press, low
  clipping). The rest are floating/disconnected (rail-level noise) or dead.
  `align.py` auto-selects the real ones by correlating each channel with the rep
  on/off pattern.
- **The recording tool pre-synchronizes the streams** (starts them together), so
  anchoring EMG at trial start (t = i/2000) lines up with the rep windows.
  Always *validate* against the rep windows; for **hold tasks** the EMG stays
  active between reps (anticipatory ramp), so trust the rep windows over any
  EMG-derived offset. General rule: never trust raw device timestamps for
  sample-level sync — validate against the shared physical event (the squeeze).

## Processing pipeline (what's built)

Offline alignment — details and a diagram in `docs/alignment_pipeline.md`.
Run: `python -m scripts.align_trial data/<trial_id>` (or `data` for all).

Code map (`src/processing/`):
- `forces.py` — load + `F_combined = F1+F2`, features.
- `eskin.py` — loaders + **per-rep ROI** detection and `roi_signal`.
- `emg_txt.py` — native `emg_raw.txt` loader.
- `align.py` — `align_trial()` → `AlignedTrial`: all three streams on the
  trial-start origin at **native rates (alignment, not fusion)**; EMG channel
  auto-selection + rep-window anchoring check; EMG optional.
- `trial_plots.py` — overview (EMG env / force / e-skin-ROI, rep windows
  shaded, per-rep onset markers) + per-rep ROI figures.
- `emg_activation.py` — MVC-reference (`max_effort`) / %MVC-normalization
  (`target_force`) library functions, built on `AlignedTrial.rep_onsets`.

### E-skin: ROI, not full-grid sum
Summing all 256 taxels drowns the contact in baseline noise. Instead we sum an
auto-detected **contact ROI**:
- Metric = per-cell **peak pressure** (95th percentile); keep cells ≥ 25% of the
  peak, then the **largest connected component** (drops isolated faulty/edge
  cells). Peak — *not* temporal std, which misses steady contact in holds.
- Computed **per rep** — the contact patch can move between grips (confirmed: it
  shifts across reps in the target_force trial). Full-hand max-effort squeezes
  cover ~most of the grid.

## Critical finding: the e-skin saturates

This gates any e-skin→force model:
- The e-skin **saturates at high force** — in max-effort, the ROI sum plateaus
  (~52k) for *all* forces above a knee (~50–60 N in that geometry); it can't
  resolve force there. (Within a rep, force fell to 62% while the e-skin barely
  moved.)
- It's **not** a per-cell digital rail (cells reach higher elsewhere) — it's the
  FSR response flattening, plus creep/hysteresis.
- **"Max force" is not one number**: the e-skin measures *pressure*, and
  force = pressure × contact area. The same force spread wide vs. concentrated
  reads very differently. A force model must use ROI pressure **and** contact
  area/geometry.
- **To characterize it you need a calibration ramp** (not more grasp trials):
  slowly ramp force 0→beyond-max (up *and* down) against a ground-truth load
  cell, with a known contact area, to get the response curve, the saturation
  knee, hysteresis, and creep. Keep predicted grasps **below the knee**.

## Open questions / next steps

- **Fusion stage** — after alignment, resample to a common ~100 Hz grid and
  merge into one rep-labelled table (using `emg_txt` + per-rep ROI). Decide sum
  vs **mean** over ROI (mean is comparable across reps, since ROI sizes differ).
- **Calibration ramp** — collect + analyze it to quantify the saturation
  ceiling and an e-skin→N mapping.
- **FMG experiment** — try the e-skin/FSR array on the forearm as an EMG
  replacement; compare against recorded EMG.
- **Confirm with GTA**: is the `target_force` target on *total* `F1+F2`?
  (rep 1 was ≈12+8 N ≈ 20 N.)

## Caveats & gotchas

- **F2 calibration is a placeholder** (copied from F1 in the original firmware)
  — treat absolute `F2` cautiously until recalibrated.
- EMG capture requires the Windows-only WaveX tool running first (see README).
- `emg_raw.txt` is not self-describing — the 8-channel / 2000 Hz assumptions are
  external; always sanity-check EMG anchoring against the rep windows.
- The offline pipeline was validated without hardware, against the `data/`
  recordings (max_effort, target_force, and a no-EMG trial).
