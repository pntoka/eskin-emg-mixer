"""Main GUI window: live e-skin heatmap + force plot, plus task-guided
grasping-trial recording (Max Effort / Target Force) with visual feedback.
"""

from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtWidgets
from PyQt6.QtGui import QFont

from ..recording.session import SessionRecorder
from ..recording.tasks import TaskKind, TaskSpec, TrialController, TrialState
from ..sensors.forces import FORCE_PLOT_WINDOW_S, READ_RATE_HZ
from .task_feedback import TaskFeedbackWidget

HEATMAP_VMIN = 0.0
HEATMAP_VMAX = 1200.0


class HeatmapPanel:
    """
    Renders a 16x16 FSR pressure map with a numeric value label overlaid on
    every cell, onto a pyqtgraph PlotItem that already lives inside a shared
    GraphicsLayoutWidget (so it can sit next to the force subplot in one
    figure).

    Colour map : plasma (dark = low pressure, bright yellow = high)
    Value range: HEATMAP_VMIN ... HEATMAP_VMAX (fixed, not auto-scaled)
    """

    _CELL_FONT = None

    def __init__(self, plot_item, vmin=HEATMAP_VMIN, vmax=HEATMAP_VMAX):
        if HeatmapPanel._CELL_FONT is None:
            f = QFont()
            f.setPointSize(6)
            HeatmapPanel._CELL_FONT = f

        self._plot = plot_item
        self._N = 16

        self._plot.setAspectLocked(True)
        self._plot.invertX(True)
        self._plot.invertY(True)
        self._plot.hideAxis("left")
        self._plot.hideAxis("bottom")
        self._plot.setRange(xRange=[0, self._N], yRange=[0, self._N], padding=0)
        self._plot.setMouseEnabled(x=False, y=False)

        self._img = pg.ImageItem()
        self._plot.addItem(self._img)
        cmap = pg.colormap.get("plasma")
        self._img.setColorMap(cmap)
        self._img.setLevels([vmin, vmax])

        self._texts = []
        for row in range(self._N):
            row_texts = []
            for col in range(self._N):
                t = pg.TextItem(text="", anchor=(0.5, 0.5), color=(255, 255, 255))
                t.setFont(HeatmapPanel._CELL_FONT)
                self._plot.addItem(t)
                t.setPos(row + 0.5, col + 0.5)
                row_texts.append(t)
            self._texts.append(row_texts)

        self.update_data(np.zeros((self._N, self._N), dtype=np.float32))

    def update_data(self, data):
        """data: (16, 16) float32 array, post-tare, post-remap."""
        self._img.setImage(data, autoLevels=False)
        for row in range(self._N):
            for col in range(self._N):
                self._texts[row][col].setText(f"{data[row, col]:.1f}")


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, eskin_port, forces_port, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Robotics Summer School - Grasp Recording")
        self.resize(1300, 800)

        self.recorder = SessionRecorder(eskin_port, forces_port)
        self.recorder.eskin_frame.connect(self._update_heatmap)
        self.recorder.force_frame.connect(self._update_forces)
        self.recorder.calibration_ready.connect(self._on_calibration_ready)

        self.controller = TrialController(self.recorder)
        self.controller.state_changed.connect(self._on_state_changed)
        self.controller.countdown_tick.connect(self._on_countdown_tick)
        self.controller.hold_tick.connect(self._on_hold_tick)
        self.controller.rest_tick.connect(self._on_rest_tick)
        self.controller.rep_changed.connect(self._on_rep_changed)
        self.controller.stabilize_tick.connect(self._on_stabilize_tick)
        self.controller.trial_finished.connect(self._on_trial_finished)

        self._hold_remaining = 0.0
        self._trial_counter = 0

        hist_len = max(2, int(FORCE_PLOT_WINDOW_S * READ_RATE_HZ))
        self._force_t = deque(maxlen=hist_len)
        self._force_f1 = deque(maxlen=hist_len)
        self._force_f2 = deque(maxlen=hist_len)

        self._build_ui()
        self.recorder.start_threads()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)

        # --- Left: live sensor visualization ---------------------------------
        left = QtWidgets.QVBoxLayout()

        self._status = QtWidgets.QLabel(
            "Calibrating - please do NOT touch the e-skin or force sensors...")
        self._status.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        left.addWidget(self._status)

        self._glw = pg.GraphicsLayoutWidget()
        left.addWidget(self._glw)

        heatmap_plot = self._glw.addPlot(row=0, col=0, title="E-Skin Pressure (16x16)")
        self._hmap = HeatmapPanel(heatmap_plot)

        force_plot = self._glw.addPlot(row=0, col=1, title="Force Sensors")
        force_plot.setLabel("bottom", "t (s)")
        force_plot.setLabel("left", "Force (N)")
        force_plot.showGrid(x=True, y=True, alpha=0.3)
        force_plot.addLegend()
        self._curve1 = force_plot.plot(pen=pg.mkPen("r", width=2), name="F1 (N)")
        self._curve2 = force_plot.plot(pen=pg.mkPen("b", width=2), name="F2 (N)")

        root.addLayout(left, 3)

        # --- Right: trial controls + task feedback ----------------------------
        right = QtWidgets.QVBoxLayout()

        form = QtWidgets.QFormLayout()
        self._subject_edit = QtWidgets.QLineEdit("P1")
        form.addRow("Subject/Participant ID:", self._subject_edit)

        self._task_combo = QtWidgets.QComboBox()
        self._task_combo.addItem("Max Effort", TaskKind.MAX_EFFORT)
        self._task_combo.addItem("Target Force", TaskKind.TARGET_FORCE)
        self._task_combo.currentIndexChanged.connect(self._on_task_kind_changed)
        form.addRow("Task:", self._task_combo)

        self._duration_spin = QtWidgets.QDoubleSpinBox()
        self._duration_spin.setRange(1.0, 120.0)
        self._duration_spin.setValue(5.0)
        self._duration_spin.setSuffix(" s")
        form.addRow("Hold duration:", self._duration_spin)

        self._repetitions_spin = QtWidgets.QSpinBox()
        self._repetitions_spin.setRange(1, 50)
        self._repetitions_spin.setValue(1)
        self._repetitions_spin.valueChanged.connect(self._on_repetitions_changed)
        form.addRow("Repetitions:", self._repetitions_spin)

        self._rest_spin = QtWidgets.QDoubleSpinBox()
        self._rest_spin.setRange(0.0, 120.0)
        self._rest_spin.setValue(5.0)
        self._rest_spin.setSuffix(" s")
        self._rest_spin.setEnabled(False)
        form.addRow("Rest duration:", self._rest_spin)

        self._target_spin = QtWidgets.QDoubleSpinBox()
        self._target_spin.setRange(0.1, 200.0)
        self._target_spin.setValue(5.0)
        self._target_spin.setSuffix(" N")
        form.addRow("Target force:", self._target_spin)

        self._tolerance_spin = QtWidgets.QDoubleSpinBox()
        self._tolerance_spin.setRange(0.1, 50.0)
        self._tolerance_spin.setValue(1.0)
        self._tolerance_spin.setSuffix(" N")
        form.addRow("Tolerance (±):", self._tolerance_spin)

        right.addLayout(form)

        self._start_btn = QtWidgets.QPushButton("Start Trial")
        self._start_btn.setEnabled(False)
        self._start_btn.clicked.connect(self._on_start_clicked)
        right.addWidget(self._start_btn)

        self._feedback = TaskFeedbackWidget()
        right.addWidget(self._feedback)

        right.addStretch(1)
        root.addLayout(right, 2)

        self._on_task_kind_changed()

    # ------------------------------------------------------------------
    # Calibration / live data
    # ------------------------------------------------------------------

    def _on_calibration_ready(self):
        self._start_btn.setEnabled(True)
        self._status.setText("Live")

    def _update_heatmap(self, data):
        self._hmap.update_data(data)

    def _update_forces(self, t, f1, f2, wall_time):
        self._force_t.append(t)
        self._force_f1.append(f1)
        self._force_f2.append(f2)
        self._curve1.setData(list(self._force_t), list(self._force_f1))
        self._curve2.setData(list(self._force_t), list(self._force_f2))

        if self.controller.state == TrialState.HOLD:
            self._feedback.set_hold(self._hold_remaining, current_force=f1 + f2)

    # ------------------------------------------------------------------
    # Trial controls
    # ------------------------------------------------------------------

    def _on_task_kind_changed(self):
        is_target = self._task_combo.currentData() == TaskKind.TARGET_FORCE
        self._target_spin.setEnabled(is_target)
        self._tolerance_spin.setEnabled(is_target)

    def _on_repetitions_changed(self, value):
        self._rest_spin.setEnabled(value > 1)

    def _on_start_clicked(self):
        if self.controller.state != TrialState.IDLE:
            self.controller.abort()
            return

        kind = self._task_combo.currentData()
        task = TaskSpec(
            kind=kind,
            duration_s=self._duration_spin.value(),
            repetitions=self._repetitions_spin.value(),
            rest_s=self._rest_spin.value(),
            target_force_n=self._target_spin.value() if kind == TaskKind.TARGET_FORCE else None,
            tolerance_n=self._tolerance_spin.value() if kind == TaskKind.TARGET_FORCE else None,
        )
        self._trial_counter += 1
        subject = self._subject_edit.text().strip() or "subject"
        trial_id = f"{subject}_{self._trial_counter:03d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self._feedback.set_task(task)
        self.controller.start(task, trial_id, subject_id=subject)

    def _on_state_changed(self, state_value):
        state = TrialState(state_value)
        active = state in (TrialState.COUNTDOWN, TrialState.STABILIZING,
                           TrialState.HOLD, TrialState.REST)
        self._start_btn.setText("Abort Trial" if active else "Start Trial")
        self._start_btn.setEnabled(active or self.recorder.calibrated)
        self._set_form_enabled(state == TrialState.IDLE)

    def _set_form_enabled(self, enabled):
        self._subject_edit.setEnabled(enabled)
        self._task_combo.setEnabled(enabled)
        self._duration_spin.setEnabled(enabled)
        self._repetitions_spin.setEnabled(enabled)
        self._rest_spin.setEnabled(enabled and self._repetitions_spin.value() > 1)
        is_target = self._task_combo.currentData() == TaskKind.TARGET_FORCE
        self._target_spin.setEnabled(enabled and is_target)
        self._tolerance_spin.setEnabled(enabled and is_target)

    def _on_countdown_tick(self, seconds_remaining):
        self._feedback.set_countdown(seconds_remaining)

    def _on_hold_tick(self, seconds_remaining):
        self._hold_remaining = seconds_remaining
        self._feedback.set_hold(seconds_remaining)

    def _on_rest_tick(self, seconds_remaining):
        self._feedback.set_rest(seconds_remaining)

    def _on_rep_changed(self, current_rep, total_reps):
        self._feedback.set_rep(current_rep, total_reps)

    def _on_stabilize_tick(self, elapsed_in_band_s, in_band, combined_force_n):
        self._feedback.set_stabilizing(
            elapsed_in_band_s, in_band, required_s=self.controller.STABILIZE_S,
            current_force=combined_force_n)

    def _on_trial_finished(self, manifest):
        aborted = manifest.get("aborted", False)
        self._feedback.set_done(aborted=aborted)
        session_dir = Path(manifest["eskin_csv"]).parent
        self._status.setText(
            f"{'Aborted' if aborted else 'Saved'} trial '{manifest['trial_id']}' -> {session_dir}")

    # ------------------------------------------------------------------
    # Window lifecycle
    # ------------------------------------------------------------------

    def closeEvent(self, event):
        if self.controller.state != TrialState.IDLE:
            self.controller.abort()
        self.recorder.stop_threads()
        event.accept()
