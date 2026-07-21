"""SessionRecorder: owns the long-lived e-skin + force reader threads,
buffers frames into per-trial CSVs while a trial is armed, and coordinates
the EMG start/stop hotkey + EMG file linking around each trial.
"""

import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from PyQt6 import QtCore

from ..emg import hotkey_trigger, session_linker
from ..sensors.eskin import ESKIN_TARE_SAMPLES, EskinReader, remap
from ..sensors.forces import ForceReader
from .tasks import TaskSpec

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"


class SessionRecorder(QtCore.QObject):
    """
    Owns one EskinReader + one ForceReader (started once, for the life of
    the GUI). While idle, forwards live frames for display and runs the
    one-time startup tare. While a trial is active, also buffers frames and
    fires the EMG start/stop hotkeys, writing everything out on stop_trial().

    Emits:
        eskin_frame(np.ndarray): live (16, 16) post-tare, post-remap frame
        force_frame(t, f1_N, f2_N, wall_time): live force sample
        calibration_ready(): once both e-skin and force tare are complete
    """

    eskin_frame = QtCore.pyqtSignal(np.ndarray)
    force_frame = QtCore.pyqtSignal(float, float, float, str)
    calibration_ready = QtCore.pyqtSignal()

    def __init__(self, eskin_port, forces_port, data_dir: Path = DEFAULT_DATA_DIR, parent=None):
        super().__init__(parent)
        self.data_dir = Path(data_dir)

        self._eskin_tare_sum = np.zeros((16, 16), dtype=np.float64)
        self._eskin_tare_count = 0
        self._eskin_tare_mean: Optional[np.ndarray] = None
        self._force_tared = False

        self._recording = False
        self._buffering = False
        self._segment_eskin_mark = 0
        self._segment_force_mark = 0
        self._trial_id: Optional[str] = None
        self._task: Optional[TaskSpec] = None
        self._subject_id = ""
        self._start_wall_time: Optional[datetime] = None
        self._start_time_monotonic: Optional[float] = None
        self._eskin_rows = []
        self._force_rows = []

        self.eskin_thread = EskinReader(eskin_port)
        self.eskin_thread.data_received.connect(self._on_eskin)
        self.forces_thread = ForceReader(forces_port)
        self.forces_thread.data_received.connect(self._on_forces)
        self.forces_thread.tare_done.connect(self._on_force_tared)

    def start_threads(self):
        self.eskin_thread.start()
        self.forces_thread.start()

    def stop_threads(self):
        self.eskin_thread.stop()
        self.forces_thread.stop()

    @property
    def calibrated(self) -> bool:
        return self._eskin_tare_mean is not None and self._force_tared

    @property
    def recording(self) -> bool:
        return self._recording

    # ------------------------------------------------------------------
    # Trial control
    # ------------------------------------------------------------------

    def start_trial(self, trial_id: str, task: TaskSpec, subject_id: str = "") -> None:
        if self._recording:
            raise RuntimeError("A trial is already recording")
        self._trial_id = trial_id
        self._task = task
        self._subject_id = subject_id
        self._eskin_rows = []
        self._force_rows = []
        self._start_wall_time = datetime.now()
        self._start_time_monotonic = time.time()
        hotkey_trigger.send_start()
        self._recording = True
        self._buffering = True
        self._segment_eskin_mark = 0
        self._segment_force_mark = 0

    def pause_recording(self) -> None:
        """Stop appending frames without ending the trial (e.g. between
        repetitions, or while re-stabilizing on target)."""
        self._buffering = False

    def resume_recording(self) -> None:
        """Resume appending frames, marking the current position so a
        subsequent discard_current_segment() call can roll back to it."""
        self._segment_eskin_mark = len(self._eskin_rows)
        self._segment_force_mark = len(self._force_rows)
        self._buffering = True

    def discard_current_segment(self) -> None:
        """Drop frames appended since the last resume_recording() call
        (e.g. a hold attempt that drifted out of tolerance) and pause."""
        del self._eskin_rows[self._segment_eskin_mark:]
        del self._force_rows[self._segment_force_mark:]
        self._buffering = False

    def stop_trial(self, aborted: bool = False, repetitions: Optional[list] = None) -> dict:
        if not self._recording:
            raise RuntimeError("No trial is currently recording")
        self._recording = False
        hotkey_trigger.send_stop()
        stop_wall_time = datetime.now()

        session_dir = self.data_dir / self._trial_id
        session_dir.mkdir(parents=True, exist_ok=True)

        eskin_path = session_dir / "eskin.csv"
        self._write_eskin_csv(eskin_path)

        forces_path = session_dir / "forces.csv"
        self._write_forces_csv(forces_path)

        emg_path = session_linker.link_emg_file(self._start_time_monotonic, session_dir)

        manifest = {
            "trial_id": self._trial_id,
            "subject_id": self._subject_id,
            "task_kind": self._task.kind.value,
            "duration_s": self._task.duration_s,
            "repetitions_total": self._task.repetitions,
            "rest_s": self._task.rest_s,
            "repetitions": repetitions or [],
            "target_force_n": self._task.target_force_n,
            "tolerance_n": self._task.tolerance_n,
            "aborted": aborted,
            "start_wall_time": self._start_wall_time.isoformat(timespec="milliseconds"),
            "stop_wall_time": stop_wall_time.isoformat(timespec="milliseconds"),
            "eskin_csv": str(eskin_path),
            "forces_csv": str(forces_path),
            "emg_txt": str(emg_path) if emg_path else None,
        }
        with open(session_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        print(f"[SessionRecorder] Trial '{self._trial_id}' saved -> {session_dir}")
        return manifest

    def _write_eskin_csv(self, path: Path) -> None:
        header = ["wall_time", "elapsed_s"] + \
                 [f"R{r:02d}_C{c:02d}" for r in range(16) for c in range(16)]
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(self._eskin_rows)

    def _write_forces_csv(self, path: Path) -> None:
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["wall_time", "elapsed_s", "F1_N", "F2_N"])
            w.writerows(self._force_rows)

    # ------------------------------------------------------------------
    # Data pipeline
    # ------------------------------------------------------------------

    def _on_eskin(self, fsr: np.ndarray) -> None:
        """fsr: (16, 1, 16) uint16, axis layout (col, layer=0, row)."""
        raw = fsr[:, 0, :].astype(np.float32)

        if self._eskin_tare_mean is None:
            self._eskin_tare_sum += raw
            self._eskin_tare_count += 1
            if self._eskin_tare_count == ESKIN_TARE_SAMPLES:
                self._eskin_tare_mean = self._eskin_tare_sum / ESKIN_TARE_SAMPLES
                print("E-skin tare complete")
                self._check_calibration_ready()
            return

        zeroed = np.clip(raw - self._eskin_tare_mean, 0.0, None)
        data = remap(zeroed)
        self.eskin_frame.emit(data)

        if self._recording and self._buffering:
            wall_time = datetime.now().isoformat(timespec="milliseconds")
            elapsed = round(time.time() - self._start_time_monotonic, 4)
            self._eskin_rows.append([wall_time, elapsed] + data.flatten().tolist())

    def _on_forces(self, t: float, f1: float, f2: float, wall_time: str) -> None:
        self.force_frame.emit(t, f1, f2, wall_time)

        if self._recording and self._buffering:
            elapsed = round(time.time() - self._start_time_monotonic, 4)
            self._force_rows.append([wall_time, elapsed, round(f1, 4), round(f2, 4)])

    def _on_force_tared(self, tare1: float, tare2: float) -> None:
        self._force_tared = True
        print(f"Force sensors tare complete: F1_offset={tare1:+.4f} N  F2_offset={tare2:+.4f} N")
        self._check_calibration_ready()

    def _check_calibration_ready(self) -> None:
        if self.calibrated:
            self.calibration_ready.emit()
