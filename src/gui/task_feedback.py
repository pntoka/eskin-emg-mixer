"""Visual feedback widget for a grasping trial: a big status banner plus,
for TARGET_FORCE trials, a gauge bar showing current combined force against
the target +/- tolerance band; for MAX_EFFORT trials, a live peak-force
readout.
"""

from typing import Optional

from PyQt6 import QtCore, QtGui, QtWidgets

from ..recording.tasks import TaskKind, TaskSpec

_BANNER_COLORS = {
    "idle": "#424242",
    "ready": "#c99b00",
    "stabilize_in_band": "#558b2f",
    "stabilize_out_of_band": "#b71c1c",
    "hold_in_band": "#2e7d32",
    "hold_out_of_band": "#c62828",
    "rest": "#5c6bc0",
    "done": "#1565c0",
    "aborted": "#616161",
}


class ForceGaugeBar(QtWidgets.QWidget):
    """Horizontal bar: shaded tolerance band around the target, a white
    target line, and a needle for the current combined force."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(70)
        self._target = 0.0
        self._tolerance = 0.0
        self._max_scale = 10.0
        self._value = 0.0

    def set_target(self, target_n: float, tolerance_n: float) -> None:
        self._target = target_n or 0.0
        self._tolerance = tolerance_n or 0.0
        self._max_scale = max(self._target * 1.5, self._target + self._tolerance + 2.0, 1.0)
        self._value = 0.0
        self.update()

    def update_value(self, value_n: float) -> bool:
        """Update the needle and return whether value_n is within tolerance."""
        self._value = value_n
        self.update()
        return abs(value_n - self._target) <= self._tolerance

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        rect = QtCore.QRectF(self.rect().adjusted(10, 10, -10, -30))
        painter.fillRect(rect, QtGui.QColor("#2b2b2b"))

        def x_for(v):
            frac = max(0.0, min(1.0, v / self._max_scale))
            return rect.left() + frac * rect.width()

        lo, hi = x_for(self._target - self._tolerance), x_for(self._target + self._tolerance)
        painter.fillRect(QtCore.QRectF(lo, rect.top(), hi - lo, rect.height()),
                          QtGui.QColor(46, 125, 50, 110))

        tx = x_for(self._target)
        painter.setPen(QtGui.QPen(QtGui.QColor("white"), 2))
        painter.drawLine(QtCore.QPointF(tx, rect.top()), QtCore.QPointF(tx, rect.bottom()))

        vx = x_for(self._value)
        in_band = abs(self._value - self._target) <= self._tolerance
        needle_color = QtGui.QColor("#66ff66") if in_band else QtGui.QColor("#ff6666")
        painter.setPen(QtGui.QPen(needle_color, 4))
        painter.drawLine(QtCore.QPointF(vx, rect.top()), QtCore.QPointF(vx, rect.bottom()))

        painter.setPen(QtGui.QPen(QtGui.QColor("white")))
        text_rect = QtCore.QRectF(rect.left(), rect.bottom() + 2, rect.width(), 20)
        painter.drawText(text_rect, QtCore.Qt.AlignmentFlag.AlignLeft,
                          f"{self._value:.1f} N")
        painter.drawText(text_rect, QtCore.Qt.AlignmentFlag.AlignRight,
                          f"target {self._target:.1f} ± {self._tolerance:.1f} N")


class TaskFeedbackWidget(QtWidgets.QWidget):
    """Owns the banner + (gauge or peak-force readout), switched based on
    the active TaskSpec's kind."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)

        self._rep_label = QtWidgets.QLabel("")
        self._rep_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        rep_font = QtGui.QFont()
        rep_font.setPointSize(12)
        self._rep_label.setFont(rep_font)
        self._rep_label.hide()
        layout.addWidget(self._rep_label)

        self._banner = QtWidgets.QLabel("IDLE")
        self._banner.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        banner_font = QtGui.QFont()
        banner_font.setPointSize(28)
        banner_font.setBold(True)
        self._banner.setFont(banner_font)
        self._banner.setMinimumHeight(80)
        layout.addWidget(self._banner)

        self._timer_label = QtWidgets.QLabel("")
        self._timer_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        timer_font = QtGui.QFont()
        timer_font.setPointSize(18)
        self._timer_label.setFont(timer_font)
        layout.addWidget(self._timer_label)

        self._gauge = ForceGaugeBar()
        self._gauge.hide()
        layout.addWidget(self._gauge)

        self._peak_label = QtWidgets.QLabel("")
        self._peak_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._peak_label.hide()
        layout.addWidget(self._peak_label)

        self._task: Optional[TaskSpec] = None
        self._peak_force = 0.0
        self._set_banner_color(_BANNER_COLORS["idle"])

    def set_task(self, task: TaskSpec) -> None:
        self._task = task
        self._peak_force = 0.0
        if task.repetitions > 1:
            self._rep_label.setText(f"Repetition 1 of {task.repetitions}")
            self._rep_label.show()
        else:
            self._rep_label.hide()
        if task.kind == TaskKind.TARGET_FORCE:
            self._gauge.set_target(task.target_force_n, task.tolerance_n)
            self._gauge.show()
            self._peak_label.hide()
        else:
            self._gauge.hide()
            self._peak_label.setText("Peak: 0.0 N")
            self._peak_label.show()

    def set_rep(self, current: int, total: int) -> None:
        if total > 1:
            self._rep_label.setText(f"Repetition {current} of {total}")
            self._rep_label.show()
        else:
            self._rep_label.hide()

    def set_idle(self) -> None:
        self._banner.setText("IDLE")
        self._set_banner_color(_BANNER_COLORS["idle"])
        self._timer_label.setText("")
        self._rep_label.hide()

    def set_countdown(self, seconds_remaining: int) -> None:
        if seconds_remaining > 0:
            self._banner.setText("GET READY")
            self._timer_label.setText(str(seconds_remaining))
        else:
            self._banner.setText("SQUEEZE NOW!")
            self._timer_label.setText("")
        self._set_banner_color(_BANNER_COLORS["ready"])

    def set_stabilizing(self, elapsed_in_band_s: float, in_band: bool, required_s: float,
                        current_force: Optional[float] = None) -> None:
        clamped = min(max(elapsed_in_band_s, 0.0), required_s)
        if in_band:
            self._banner.setText("HOLD STEADY...")
            self._timer_label.setText(f"in range - {clamped:0.1f} / {required_s:0.1f}s")
            self._set_banner_color(_BANNER_COLORS["stabilize_in_band"])
        else:
            self._banner.setText("GET INTO RANGE")
            self._timer_label.setText(f"out of range - 0.0 / {required_s:0.1f}s")
            self._set_banner_color(_BANNER_COLORS["stabilize_out_of_band"])

        if current_force is not None:
            self._gauge.update_value(current_force)

    def set_hold(self, seconds_remaining: float, current_force: Optional[float] = None) -> None:
        self._banner.setText("HOLD")
        self._timer_label.setText(f"{seconds_remaining:0.1f} s remaining")

        if current_force is None:
            return

        if self._task is not None and self._task.kind == TaskKind.TARGET_FORCE:
            in_band = self._gauge.update_value(current_force)
            self._set_banner_color(_BANNER_COLORS["hold_in_band" if in_band else "hold_out_of_band"])
        else:
            self._peak_force = max(self._peak_force, current_force)
            self._peak_label.setText(f"Peak: {self._peak_force:.1f} N")
            self._set_banner_color(_BANNER_COLORS["hold_in_band"])

    def set_rest(self, seconds_remaining: float) -> None:
        self._banner.setText("REST")
        self._timer_label.setText(f"{seconds_remaining:0.1f} s remaining")
        self._set_banner_color(_BANNER_COLORS["rest"])

    def set_done(self, aborted: bool = False) -> None:
        self._banner.setText("ABORTED" if aborted else "DONE – Saved")
        self._set_banner_color(_BANNER_COLORS["aborted" if aborted else "done"])
        self._timer_label.setText("")

    def _set_banner_color(self, hex_color: str) -> None:
        self._banner.setStyleSheet(
            f"color: white; background-color: {hex_color}; padding: 14px; border-radius: 8px;"
        )
