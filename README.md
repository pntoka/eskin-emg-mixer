# Robotics Summer School: E-skin + Force + EMG Grasp Recording

Records and correlates a 16×16 e-skin pressure matrix, two handle-mounted force
sensors, and EMG during grasping tasks (max-effort grip, and holding a target
force for a set time).

- **New here? Start with [`PROJECT.md`](PROJECT.md)** — project context,
  decisions, and findings — then this file for how to run things.
- Original brief: [`project_overview.md`](project_overview.md).
- Alignment method + pipeline diagram: [`docs/alignment_pipeline.md`](docs/alignment_pipeline.md).

## Directory structure

```
config/                  # force-sensor zero-bias calibration (live config)
data/                    # recorded trials, one folder per trial (see below)
docs/
  alignment_pipeline.md  # how offline alignment works (+ mermaid diagram)
src/
  sensors/               # e-skin + force serial protocols, port autodetection
  emg/                   # drives EMG_Eyetracker_Tool.exe via simulated F5/F6 hotkeys
  recording/             # SessionRecorder + the grasp-trial state machine
  gui/                   # live heatmap/force plot + task-guided recording GUI
  processing/            # offline: loaders/features, alignment, plots, correlation
scripts/
  run_gui.py             # record trials (GUI)
  align_trial.py         # align + plot a recorded trial
project_overview.md      # original one-paragraph brief
PROJECT.md               # project context, decisions, findings
requirements.txt
```

A recorded trial, `data/<trial_id>/`:

```
manifest.json   task, subject, per-rep windows, start/stop timestamps
eskin.csv       wall_time, elapsed_s, 256 taxel columns (R00_C00..R15_C15)
forces.csv      wall_time, elapsed_s, F1_N, F2_N
emg_raw.txt     raw WaveX EMG for the whole trial (max_effort trials, if the
                EMG tool was running)
emg_rep{N}.txt  raw WaveX EMG for one successfully-completed rep (target_force
                trials instead capture EMG per hold attempt, one short file per
                rep, to avoid overrunning the EMG tool's recording buffer on
                long/retry-heavy trials -- see docs/alignment_pipeline.md)
```

## Quickstart

```
# 1. Setup (once)
conda activate eskin
pip install -r requirements.txt

# 2. Record trials  (needs sensor hardware; EMG optional — see prerequisites)
python -m scripts.run_gui

# 3. Align + plot a recorded trial  (offline, no hardware needed)
python -m scripts.align_trial data/<trial_id>
```

## Setup

Use the existing `eskin` conda environment (`conda activate eskin`), which already
has PyQt6/pyqtgraph/pyserial set up correctly for this hardware. Then install the
remaining dependencies into it:

```
conda activate eskin
pip install -r requirements.txt
```

(A shared/base Anaconda environment is *not* recommended: PyQt6 fails to import
there with a `DLL load failed` error, because base's own PyQt5 + bundled Qt/MSVC
runtime DLLs conflict with PyQt6's. `eskin` is a clean env without that conflict.)

## Recording trials (GUI)

### EMG (Cometa WaveX) prerequisites

Only needed if you want EMG captured alongside e-skin/force (otherwise skip — the
recording still works without it).

- No build required: a pre-published .exe is already checked into
  `EMG_Eyetracker_Tool/EMG_Eyetracker_Tool/Publish/EMG_Eyetracker_Tool.exe`.
- Requires the **.NET 8.0 Desktop Runtime** installed on the machine (Windows will
  prompt to install it on first launch if missing).
- Cometa WaveX device drivers (CyUSB) must be installed (already done per project
  notes).
- The Tobii Pro Fusion eye tracker referenced in that tool is **not required** for
  this project -- if none is found it just shows a one-time "Tobii device not
  detected" popup (click OK) and continues on to the EMG connection.
- Before recording any trial: launch `EMG_Eyetracker_Tool.exe`, type a
  participant/session header id, and click its own "Start" button once. This opens
  the WaveX device connection and arms the F5/F6 hotkeys that the Python GUI
  simulates to start/stop each trial's EMG capture.

### Run the GUI

```
python -m scripts.run_gui                 # auto-detect e-skin + force COM ports
python -m scripts.run_gui COM3 COM4        # override ports explicitly
```

The GUI shows a live e-skin heatmap + force plot and guides you through each
task's reps. Recorded trials are written under `data/<trial_id>/` (see the
layout above): `eskin.csv`, `forces.csv`, EMG (`emg_raw.txt` for max_effort,
`emg_rep{N}.txt` per rep for target_force, if the EMG tool was running), and
`manifest.json` tying them together with the task parameters and per-rep
start/stop timestamps.

## Aligning & plotting (offline)

Time-align a trial's e-skin, force, and EMG streams onto one axis (native rates,
no fusion), auto-select the real EMG channels, compute a per-rep e-skin contact
ROI, and save plots — no hardware needed:

```
python -m scripts.align_trial data/<trial_id>     # one trial
python -m scripts.align_trial data                 # every trial folder under data/
python -m scripts.align_trial data --no-plot       # text report only
python -m scripts.align_trial data --dump          # also write <trial>/aligned/*.csv
```

Outputs written into each trial folder:

- `aligned_overview.png` — EMG envelope / force (`F1+F2`, plus `F1`,`F2`) /
  e-skin ROI sum on one time axis, with the labelled rep windows shaded.
- `eskin_rep_rois.png` — one panel per rep: that rep's peak-pressure map with the
  detected contact ROI outlined.

Force is merged as `F1 + F2` (total grasp force); e-skin is summed over an
auto-detected per-rep contact ROI (not the full grid). Full method and a pipeline
diagram: [`docs/alignment_pipeline.md`](docs/alignment_pipeline.md).

## Status

No e-skin/force/EMG hardware was available while building this, so hardware-level
verification is still pending (see the plan's Verification section for that
checklist). Everything that *can* be exercised without hardware has been, in the
`eskin` conda env:
- `src/processing/` (forces, e-skin, EMG C3D loading/filtering/RMS-envelope,
  correlation) run directly against the example recordings in `archive/`.
- `src/emg/hotkey_trigger.py` and `session_linker.py` unit-tested with mocks
  (correct F5/F6 calls; correct newest-file-after-start-time picking).
- `src/recording/tasks.py`'s `TrialController` state machine
  (IDLE→COUNTDOWN→HOLD→DONE→IDLE) runtime-tested end-to-end against a mocked
  `SessionRecorder`.
- `src/recording/session.py`'s `SessionRecorder` runtime-tested end-to-end with
  synthetic e-skin/force frames: tare/calibration, idle vs. recording buffering,
  and `eskin.csv`/`forces.csv`/`manifest.json` output, with the EMG hotkey calls
  mocked out (so no real F5/F6 was sent to the desktop during testing).
- `src/gui/task_feedback.py` (`TaskFeedbackWidget`, `ForceGaugeBar`) and
  `src/gui/main_window.py`'s `HeatmapPanel` constructed and driven with synthetic
  data under a real `QApplication`.
- `src/processing/align.py` + `trial_plots.py` (offline alignment, per-rep ROI,
  plots) run against the `data/` recordings (max_effort, target_force, and a
  no-EMG trial).

Not yet tested: the full `MainWindow` wired to real serial ports, and an actual
EMG_Eyetracker_Tool.exe run triggered by the simulated hotkeys — both need the
real hardware.
