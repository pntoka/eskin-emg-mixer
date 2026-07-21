"""Grasping-task definitions and the trial state machine that drives them.

States: IDLE -> COUNTDOWN (3..1) -> [STABILIZING ->] HOLD (running timer)
-> [REST -> [STABILIZING ->] HOLD]* -> DONE (or ABORTED if cancelled during
COUNTDOWN/STABILIZING/HOLD/REST).

For TARGET_FORCE tasks, a STABILIZING phase precedes every HOLD attempt: it
waits for the combined force to sit continuously within tolerance of the
target for STABILIZE_S seconds before the hold timer starts, resetting to
zero any time the force drifts back out of band. Once HOLD is running, the
combined force is monitored on every sample too: if it drifts out of
tolerance before the full duration elapses, that attempt is aborted (its
buffered frames are discarded and it does not count as a repetition) and
the state machine drops back to STABILIZING to reacquire, retrying the same
repetition until it completes a full in-band hold. `repetitions` therefore
counts *successful* holds, not attempts.

Recording (SessionRecorder.start_trial) is armed once, the moment the
first hold attempt of the trial begins (rep 1 STABILIZING succeeding).
Within that armed trial, frames are only actually buffered while a HOLD
attempt is in progress: buffering pauses during STABILIZING/REST and is
discarded outright for aborted attempts, so the saved CSVs only contain
the successful, fully in-band hold windows. MAX_EFFORT tasks have no
target/tolerance concept and skip straight from COUNTDOWN/REST to HOLD as
before, with recording starting at rep 1's HOLD and running continuously
(including REST) like today.
"""

import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import List, Optional

from PyQt6 import QtCore


class TaskKind(Enum):
    MAX_EFFORT = "max_effort"
    TARGET_FORCE = "target_force"


@dataclass
class TaskSpec:
    kind: TaskKind
    duration_s: float
    repetitions: int = 1
    rest_s: float = 0.0
    target_force_n: Optional[float] = None
    tolerance_n: Optional[float] = None


class TrialState(Enum):
    IDLE = "idle"
    COUNTDOWN = "countdown"
    STABILIZING = "stabilizing"
    HOLD = "hold"
    REST = "rest"
    DONE = "done"
    ABORTED = "aborted"


class TrialController(QtCore.QObject):
    """Drives one grasping trial end-to-end against a SessionRecorder.

    Emits:
        state_changed(str): new TrialState value
        countdown_tick(int): whole seconds remaining in COUNTDOWN
        hold_tick(float): seconds remaining in the current HOLD repetition
        rest_tick(float): seconds remaining in the current REST phase
        rep_changed(int, int): (current repetition, total repetitions),
            emitted whenever a new HOLD repetition begins
        stabilize_tick(float, bool, float): (seconds continuously in-band,
            currently in-band, current combined force N), emitted on every
            force sample while in STABILIZING (TARGET_FORCE tasks only)
        trial_finished(dict): the manifest from SessionRecorder.stop_trial
    """

    state_changed = QtCore.pyqtSignal(str)
    countdown_tick = QtCore.pyqtSignal(int)
    hold_tick = QtCore.pyqtSignal(float)
    rest_tick = QtCore.pyqtSignal(float)
    rep_changed = QtCore.pyqtSignal(int, int)
    stabilize_tick = QtCore.pyqtSignal(float, bool, float)
    trial_finished = QtCore.pyqtSignal(dict)

    COUNTDOWN_S = 3
    STABILIZE_S = 2.0
    TICK_MS = 100

    def __init__(self, recorder, parent=None):
        super().__init__(parent)
        self._recorder = recorder
        self._recorder.force_frame.connect(self._on_force_frame)
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(self.TICK_MS)
        self._timer.timeout.connect(self._tick)

        self.state = TrialState.IDLE
        self._task: Optional[TaskSpec] = None
        self._trial_id: Optional[str] = None
        self._subject_id = ""
        self._phase_start = 0.0
        self._countdown_remaining = 0
        self._rep_index = 0
        self._rep_start_wall: Optional[datetime] = None
        self._rep_marks: List[dict] = []
        self._inband_since: Optional[float] = None

    def start(self, task: TaskSpec, trial_id: str, subject_id: str = ""):
        if self.state != TrialState.IDLE:
            return
        self._task = task
        self._trial_id = trial_id
        self._subject_id = subject_id
        self._rep_index = 0
        self._rep_marks = []
        self._inband_since = None
        self._countdown_remaining = self.COUNTDOWN_S
        self._set_state(TrialState.COUNTDOWN)
        self.countdown_tick.emit(self._countdown_remaining)
        self._phase_start = time.monotonic()
        self._timer.start()

    def abort(self):
        if self.state in (TrialState.COUNTDOWN, TrialState.STABILIZING,
                           TrialState.HOLD, TrialState.REST):
            self._finish(aborted=True)

    def _tick(self):
        elapsed = time.monotonic() - self._phase_start
        if self.state == TrialState.COUNTDOWN:
            remaining = self.COUNTDOWN_S - int(elapsed)
            if remaining != self._countdown_remaining:
                self._countdown_remaining = remaining
                self.countdown_tick.emit(max(remaining, 0))
            if elapsed >= self.COUNTDOWN_S:
                self._enter_hold_or_stabilize()
        elif self.state == TrialState.HOLD:
            remaining = max(self._task.duration_s - elapsed, 0.0)
            self.hold_tick.emit(remaining)
            if elapsed >= self._task.duration_s:
                self._end_hold()
        elif self.state == TrialState.REST:
            remaining = max(self._task.rest_s - elapsed, 0.0)
            self.rest_tick.emit(remaining)
            if elapsed >= self._task.rest_s:
                self._enter_hold_or_stabilize()
        # STABILIZING has no time-based expiry; it's driven entirely by
        # _on_force_frame reacting to live force samples.

    def _enter_hold_or_stabilize(self):
        self._rep_index += 1
        self.rep_changed.emit(self._rep_index, self._task.repetitions)
        if self._task.kind == TaskKind.TARGET_FORCE:
            self._inband_since = None
            self._set_state(TrialState.STABILIZING)
            self.stabilize_tick.emit(0.0, False, 0.0)
        else:
            self._begin_hold()

    def _on_force_frame(self, t, f1_N, f2_N, wall_time):
        if self._task is None or self._task.kind != TaskKind.TARGET_FORCE:
            return
        if self.state not in (TrialState.STABILIZING, TrialState.HOLD):
            return
        combined = f1_N + f2_N
        in_band = abs(combined - self._task.target_force_n) <= self._task.tolerance_n

        if self.state == TrialState.HOLD:
            if not in_band:
                self._abort_hold_attempt(combined)
            return

        if in_band:
            now = time.monotonic()
            if self._inband_since is None:
                self._inband_since = now
            elapsed_in_band = now - self._inband_since
            self.stabilize_tick.emit(elapsed_in_band, True, combined)
            if elapsed_in_band >= self.STABILIZE_S:
                self._begin_hold()
        else:
            self._inband_since = None
            self.stabilize_tick.emit(0.0, False, combined)

    def _begin_hold(self):
        self._set_state(TrialState.HOLD)
        self._phase_start = time.monotonic()
        self._rep_start_wall = datetime.now()
        if not self._recorder.recording:
            self._recorder.start_trial(self._trial_id, self._task, self._subject_id)
        if self._task.kind == TaskKind.TARGET_FORCE:
            self._recorder.resume_recording()

    def _abort_hold_attempt(self, combined_force_n: float):
        """Force drifted out of tolerance mid-HOLD: discard this attempt's
        buffered frames and retry the same repetition from STABILIZING."""
        self._recorder.discard_current_segment()
        self._inband_since = None
        self._set_state(TrialState.STABILIZING)
        self.stabilize_tick.emit(0.0, False, combined_force_n)

    def _end_hold(self):
        if self._task.kind == TaskKind.TARGET_FORCE:
            self._recorder.pause_recording()
        rep_end_wall = datetime.now()
        self._rep_marks.append({
            "rep": self._rep_index,
            "start_wall_time": self._rep_start_wall.isoformat(timespec="milliseconds"),
            "end_wall_time": rep_end_wall.isoformat(timespec="milliseconds"),
        })
        if self._rep_index < self._task.repetitions:
            self._set_state(TrialState.REST)
            self._phase_start = time.monotonic()
        else:
            self._finish(aborted=False)

    def _finish(self, aborted: bool):
        self._timer.stop()
        manifest = None
        if self._recorder.recording:
            manifest = self._recorder.stop_trial(aborted=aborted, repetitions=self._rep_marks)
        self._set_state(TrialState.ABORTED if aborted else TrialState.DONE)
        if manifest is not None:
            self.trial_finished.emit(manifest)
        self._set_state(TrialState.IDLE)

    def _set_state(self, state: TrialState):
        self.state = state
        self.state_changed.emit(state.value)
