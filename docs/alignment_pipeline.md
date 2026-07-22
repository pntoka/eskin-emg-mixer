# Trial alignment pipeline

How a recorded trial's three streams — **e-skin**, **force**, and **EMG** — are
loaded and put onto one common time axis for comparison.

This stage does **alignment, not fusion**: every stream is kept at its own
native sample rate and in its own array. There is *no* resampling onto a shared
grid and *no* merged table here (that is what `src/processing/correlate.py`
does). The output is an inspectable, per-stream view (`AlignedTrial`) plus two
plots, so the streams can be eyeballed and validated against the labelled rep
windows before any modelling.

Entry point: `python -m scripts.align_trial <trial_or_parent_folder>`
Core module: [`src/processing/align.py`](../src/processing/align.py)

## Pipeline

```mermaid
flowchart TD
    subgraph IN["Trial folder  data/&lt;trial&gt;/"]
        MAN["manifest.json<br/>task, rep windows, start_wall_time"]
        FCSV["forces.csv<br/>wall_time, elapsed_s, F1_N, F2_N"]
        ECSV["eskin.csv<br/>elapsed_s + 256 R##_C## taxels"]
        ETXT["emg_raw.txt<br/>8 ch, channel-major blocks (optional)"]
    end

    MAN --> ORIGIN{{"Common time origin = trial start<br/>rep windows converted to elapsed s"}}

    subgraph FORCE["Force stream — native ~100 Hz"]
        FCSV --> FLOAD["load_forces_csv"]
        FLOAD --> FSUM["F_combined = F1 + F2<br/>(two load cells add to total grasp force)"]
    end

    subgraph ESKIN["E-skin stream — native ~200 Hz"]
        ECSV --> ELOAD["load_eskin_csv -> frames_array (K x 16 x 16)"]
        ELOAD --> ROI["detect_roi_per_rep<br/>per-cell p95 peak, keep >= 25% of peak,<br/>then largest connected component"]
        ROI --> ESIG["roi_signal<br/>each frame: sum over ITS rep's ROI"]
    end

    subgraph EMG["EMG stream — native 2000 Hz, optional"]
        ETXT --> ELOAD2["load_emg_txt<br/>concat channel-major blocks -> 8 x N"]
        ELOAD2 --> ESEL["select_emg_channels<br/>low clipping AND envelope correlated<br/>with the rep on/off pattern"]
        ESEL --> EANCH["anchoring_offset<br/>xcorr envelope vs rep mask (validation only)"]
    end

    ORIGIN --> FORCE
    ORIGIN --> ESKIN
    ORIGIN --> EMG

    FSUM --> AT["AlignedTrial<br/>streams kept SEPARATE, native rates"]
    ESIG --> AT
    EANCH --> AT

    AT --> PLOT["plot_overview + plot_rep_rois (PNG)"]
    AT --> REP["report (text summary)"]
    AT --> DUMP["--dump: aligned/*.csv"]
```

## Steps in detail

### 1. Common time origin

`forces.csv` and `eskin.csv` both carry an `elapsed_s` column measured from the
**same** trial-start instant (the recorder sets one origin for the trial), so
`elapsed_s` is used directly as the common axis for those two streams.

`emg_raw.txt` has **no timestamps**, so the EMG is **anchored at elapsed 0**
(trial start) and timed off its nominal **2000 Hz** rate: `t[i] = i / 2000`.

The manifest's rep windows are wall-clock; they are converted to elapsed
seconds relative to `start_wall_time` (the same origin) so the windows, force,
e-skin, and EMG all share one axis.

> **Sync caveat.** EMG and the e-skin/force boards are independently triggered,
> so sample-perfect sync is not guaranteed. `anchoring_offset` *reports* the
> residual EMG-vs-rep-window offset as a validation; it does not correct it.

### 2. Force — how the two sensors are merged

The two handle load cells `F1` and `F2` are **summed**:
`F_combined = F1_N + F2_N` (via `forces.load_forces_csv`). They sit on opposite
sides of the grip and their readings **add up to the total grasp force**, so the
sum is the physically meaningful quantity. `F1`/`F2` are also kept individually
for inspection. Native rate (~100 Hz) is preserved.

### 3. E-skin — how the 256 taxels are merged (ROI)

Summing all 256 taxels drowns a real contact in baseline noise from cells that
never touch anything, so instead we isolate the **contact patch (ROI)** and sum
only that.

- **Metric = per-cell peak pressure** (95th percentile over the window's
  frames). Untouched cells read ~0; contact cells reach high pressure. This
  separates contact from no-contact for both transient (squeeze) and sustained
  (hold) grasps. Temporal *std* is deliberately **not** used — steady contact in
  a hold barely varies, so std would pick noisy/flaky cells and miss the real
  contact.
- **Threshold + largest connected component.** Cells whose peak is `>= 25%` of
  the grid peak are candidates; the **largest 4-connected cluster** is kept,
  which drops isolated faulty/edge cells.
- **Per-rep ROI.** Each rep is a fresh grip that can land on a different patch,
  so the ROI is computed **per rep window** (`detect_roi_per_rep`), not once for
  the whole trial. `roi_signal` then aggregates each frame with the ROI of the
  rep it belongs to (rest frames use the union of all rep ROIs; they read ~0
  anyway). The per-frame e-skin scalar is the **sum over that ROI**.

### 4. EMG — channel selection and validation

- **Load** (`load_emg_txt`): `emg_raw.txt` is a sequence of blank-line-separated
  blocks; each block has 8 channel-major lines; concatenating line *k* across
  blocks rebuilds channel *k* → `(8, N)`.
- **Channel selection** (`select_emg_channels`): a channel is "real" if it
  **rarely rails** (clip fraction `<= 15%`) **and** its rectified envelope
  **correlates with the rep on/off pattern** (`>= 0.30`). Disconnected/floating
  channels rail or don't track the reps and are dropped. All channel scores are
  reported (never a silent drop).
- **Anchoring check** (`anchoring_offset`): cross-correlate the selected
  channels' envelope against the rep mask; the best lag (within ±6 s) is the
  residual mis-anchoring. `|offset| < 0.5 s` ⇒ well anchored. This uses the
  ground-truth rep windows, so it is robust even when force is sparse/gappy.

EMG is **optional**: a trial without `emg_raw.txt` still aligns force + e-skin.

### 5. Outputs

- `AlignedTrial` — dataclass holding each stream at its native rate (separate).
- `aligned_overview.png` — EMG envelope / force (`F1+F2`, `F1`, `F2`) / e-skin
  ROI sum on one time axis, rep windows shaded; target±tolerance band drawn for
  `target_force` trials.
- `eskin_rep_rois.png` — one panel per rep: that rep's peak-pressure map with
  its ROI outlined (see whether the grasp region moves between reps).
- `report(...)` — text summary (stream ranges, coverage, EMG channel scan,
  anchoring, per-rep ROI sizes).
- `--dump` — per-stream aligned CSVs under `<trial>/aligned/`.

## Terminology

- **Trial** — one recording of a task (the repo's standard term: `trial_id`,
  `TrialController`, `data/<trial_id>/`).
- **Rep** — one press/hold repetition within a trial (labelled in the manifest
  with start/end wall-clock times).
- **ROI** — region of interest: the contiguous set of e-skin cells under the
  actual contact.
