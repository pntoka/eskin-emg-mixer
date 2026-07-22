# Robotics Summer School: E-skin + Force + EMG Grasp Recording

Correlates a 16x16 e-skin pressure matrix, two handle-mounted force sensors, and
EMG signals during grasping tasks (max-effort grip, and holding a target force
level for a set time). See `project_overview.md` for the original brief.

## Project layout

```
config/                 # live config (force-sensor zero-bias calibration)
archive/                # old monolithic script + example recordings, kept for reference
src/
  sensors/               # e-skin + force-sensor serial protocols, port autodetection
  emg/                   # drives EMG_Eyetracker_Tool.exe via simulated F5/F6 hotkeys
  recording/              # SessionRecorder + the grasp-trial state machine
  gui/                    # live heatmap/force plot + task-guided recording GUI
  processing/             # offline: loaders/features (forces, e-skin, EMG c3d + txt),
                          #   per-trial alignment (align.py), plots (trial_plots.py), correlation
scripts/
  run_gui.py               # recording GUI entry point
  align_trial.py           # offline: align + plot a recorded trial (see docs/alignment_pipeline.md)
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

### EMG (Cometa WaveX) prerequisites

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

## Running

```
python -m scripts.run_gui                 # auto-detect e-skin + force COM ports
python -m scripts.run_gui COM3 COM4        # override ports explicitly
```

Recorded trials are written under `data/<trial_id>/`: `eskin.csv`, `forces.csv`,
`emg_raw.txt` (if `EMG_Eyetracker_Tool.exe` was running), and `manifest.json`
tying them together with the task parameters and start/stop timestamps.

### Aligning a recorded trial (offline)

Time-align a trial's e-skin, force, and EMG streams onto one axis (native rates,
no fusion), with per-rep e-skin ROI and EMG channel auto-selection, and save
overview + per-rep-ROI plots:

```
python -m scripts.align_trial data/<trial_id>     # one trial
python -m scripts.align_trial data                 # every trial folder under data/
python -m scripts.align_trial data --no-plot       # text report only
python -m scripts.align_trial data --dump          # also write <trial>/aligned/*.csv
```

Force is merged as `F1 + F2` (total grasp force); e-skin is summed over an
auto-detected per-rep contact ROI. Full method and a pipeline diagram:
[`docs/alignment_pipeline.md`](docs/alignment_pipeline.md).

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

Not yet tested: the full `MainWindow` wired to real serial ports, and an actual
EMG_Eyetracker_Tool.exe run triggered by the simulated hotkeys — both need the
real hardware.
