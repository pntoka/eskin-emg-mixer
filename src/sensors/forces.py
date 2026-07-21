"""Dual load-cell force-sensor serial protocol, calibration, and reader thread.

Protocol (continuous stream, 10-byte frames):
    [0xAA] [0x55] [counter (uint32)] [raw1 (uint16)] [raw2 (uint16)]
"""

import json
import struct
import time
from datetime import datetime
from pathlib import Path

import serial
from PyQt6 import QtCore

FORCES_BAUD = 250000
READ_RATE_HZ = 100  # effective output rate after block-averaging
ARDUINO_RATE_HZ = 1000  # must match SAMPLE_RATE_HZ on the Arduino sketch

# Calibration:  mass_g = a * raw + b,   force_N = mass_g * G
A1, B1 = 19.96, 10.32  # sensor 1 (A0)
A2, B2 = 19.96, 10.32  # sensor 2 (A1)  <-- update after calibrating!
G = 0.00981  # grams -> Newtons

# Manual zero-calibration offset (N), subtracted after calibration.
# Loaded from BIAS_CALIBRATION_FILE at startup if that file exists (falls
# back to these hardcoded values otherwise). Re-measured live at any time
# via the "Calibrate Zero" button in the GUI, which overwrites both the
# globals below and the saved file.
BIAS1 = 0.0  # sensor 1 (A0)
BIAS2 = 0.0  # sensor 2 (A1)

BIAS_CALIBRATION_FILE = Path(__file__).resolve().parents[2] / "config" / "force_bias_calibration.json"
BIAS_CALIBRATION_DURATION_S = 1.0  # hold the sensors still for this long

# Auto-tare (averages the first TARE_SAMPLES samples, after TARE_WARMUP
# warmup samples, into a zero offset). Runs automatically at startup as
# part of the "don't touch the sensors" calibration phase; BIAS above is
# still applied on top for any residual fine-tuning.
USE_TARE = True
TARE_SAMPLES = 1000
TARE_WARMUP = 500

FILTER_ALPHA = 1.0  # EMA low-pass on the emitted forces, 1.0 = off
FORCE_PLOT_WINDOW_S = 10.0  # seconds of history shown on the force plot

FORCE_HEADER0 = 0xAA
FORCE_HEADER1 = 0x55
FORCE_FRAME_SIZE = 10
_UNPACK = struct.Struct("<IHH").unpack_from  # counter, raw1, raw2


def stream_force_frames(ser: serial.Serial):
    """Yield (counter, raw1, raw2) tuples. Resyncs byte-by-byte on misalign."""
    buf = bytearray()
    while True:
        want = max(FORCE_FRAME_SIZE, ser.in_waiting)
        chunk = ser.read(want)
        if chunk:
            buf.extend(chunk)

        while len(buf) >= FORCE_FRAME_SIZE:
            if buf[0] == FORCE_HEADER0 and buf[1] == FORCE_HEADER1:
                yield _UNPACK(buf, 2)
                del buf[:FORCE_FRAME_SIZE]
            else:
                del buf[0]


def raw_to_newton(raw: int, a: float, b: float, bias: float = 0.0) -> float:
    return (a * raw + b) * G - bias


def load_saved_force_bias() -> None:
    """Populate the global BIAS1/BIAS2 from BIAS_CALIBRATION_FILE if a
    previous "Calibrate Zero" run saved one; otherwise leave the hardcoded
    defaults in place."""
    global BIAS1, BIAS2
    try:
        with open(BIAS_CALIBRATION_FILE) as f:
            saved = json.load(f)
        BIAS1, BIAS2 = saved["BIAS1"], saved["BIAS2"]
        print(f"Loaded saved force-sensor bias: BIAS1={BIAS1:+.4f} N  "
              f"BIAS2={BIAS2:+.4f} N (from {BIAS_CALIBRATION_FILE})")
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        print(f"No saved bias calibration found at {BIAS_CALIBRATION_FILE}; "
              f"using defaults BIAS1={BIAS1:+.4f} N  BIAS2={BIAS2:+.4f} N. "
              "Use the 'Calibrate Zero' button once connected.")


def save_force_bias(bias1: float, bias2: float) -> None:
    with open(BIAS_CALIBRATION_FILE, "w") as f:
        json.dump({"BIAS1": bias1, "BIAS2": bias2}, f)


class ForceReader(QtCore.QThread):
    """
    Streams and block-averages the two load-cell channels down to
    READ_RATE_HZ, applies calibration + manual bias (+ optional tare), and
    emits one averaged sample at a time.

    Emits:
        data_received(t, f1_N, f2_N, wall_time_iso)
        tare_done(tare1_N, tare2_N) -- once, when the zero baseline is set
        bias_calibrated(bias1_N, bias2_N) -- once per on-demand "Calibrate
            Zero" request, when the fresh measurement completes
    """

    data_received = QtCore.pyqtSignal(float, float, float, str)
    tare_done = QtCore.pyqtSignal(float, float)
    bias_calibrated = QtCore.pyqtSignal(float, float)

    def __init__(self, port, parent=None):
        super().__init__(parent)
        self.port = port
        self.running = True
        self._calib_active = False
        self._calib_target_count = 0
        self._calib_count = 0
        self._calib_sum1 = 0.0
        self._calib_sum2 = 0.0

    def request_bias_calibration(self, duration_s=BIAS_CALIBRATION_DURATION_S):
        """Ask the running thread to measure a fresh zero-bias over the next
        `duration_s` seconds of incoming samples. Safe to call from the GUI
        thread; picked up on the next sample in run()."""
        self._calib_sum1 = 0.0
        self._calib_sum2 = 0.0
        self._calib_count = 0
        self._calib_target_count = max(1, int(duration_s * ARDUINO_RATE_HZ))
        self._calib_active = True

    def run(self):
        try:
            ser = serial.Serial(self.port, FORCES_BAUD, timeout=1)
        except Exception as e:
            print("Force sensor serial connection error:", e)
            return

        with ser:
            time.sleep(2.0)
            ser.reset_input_buffer()

            read_rate_hz = min(READ_RATE_HZ, ARDUINO_RATE_HZ)
            samples_per_output = ARDUINO_RATE_HZ / read_rate_hz
            sample_period_s = 1.0 / ARDUINO_RATE_HZ

            tare_buf1, tare_buf2 = [], []
            tare_warmup_left = TARE_WARMUP
            tare1 = tare2 = None
            filt1 = filt2 = None
            first_counter = None
            acc_f1 = acc_f2 = 0.0
            n_in_block = 0
            block_start_counter = None
            n_samples_since_start = 0
            next_emit_at = samples_per_output

            try:
                for counter, raw1, raw2 in stream_force_frames(ser):
                    if not self.running:
                        break

                    f1_N = raw_to_newton(raw1, A1, B1, BIAS1)
                    f2_N = raw_to_newton(raw2, A2, B2, BIAS2)

                    # --- On-demand zero-bias calibration ----------------------
                    if self._calib_active:
                        # f1_N/f2_N already have the *old* BIAS1/BIAS2
                        # subtracted; add it back to get the bias-free level.
                        self._calib_sum1 += f1_N + BIAS1
                        self._calib_sum2 += f2_N + BIAS2
                        self._calib_count += 1
                        if self._calib_count >= self._calib_target_count:
                            self._calib_active = False
                            self.bias_calibrated.emit(
                                self._calib_sum1 / self._calib_count,
                                self._calib_sum2 / self._calib_count)

                    # --- Tare phase ------------------------------------------
                    if tare1 is None:
                        if tare_warmup_left > 0:
                            tare_warmup_left -= 1
                            continue

                        if not USE_TARE:
                            tare1 = tare2 = 0.0
                            first_counter = counter
                            self.tare_done.emit(tare1, tare2)
                            continue

                        tare_buf1.append(f1_N)
                        tare_buf2.append(f2_N)
                        if len(tare_buf1) >= TARE_SAMPLES:
                            tare1 = sum(tare_buf1) / len(tare_buf1)
                            tare2 = sum(tare_buf2) / len(tare_buf2)
                            first_counter = counter + 1
                            print(f"Force sensors tared: "
                                  f"F1={tare1:+.4f} N  F2={tare2:+.4f} N")
                            self.tare_done.emit(tare1, tare2)
                        continue

                    # --- Accumulate into current block ------------------------
                    if n_in_block == 0:
                        block_start_counter = counter
                    acc_f1 += f1_N
                    acc_f2 += f2_N
                    n_in_block += 1
                    n_samples_since_start += 1

                    if n_samples_since_start < next_emit_at:
                        continue

                    # --- Emit one averaged output -----------------------------
                    next_emit_at += samples_per_output
                    f1_avg = acc_f1 / n_in_block - tare1
                    f2_avg = acc_f2 / n_in_block - tare2

                    center_ctr = block_start_counter + (n_in_block - 1) / 2.0
                    t = (center_ctr - first_counter) * sample_period_s

                    acc_f1 = acc_f2 = 0.0
                    n_in_block = 0

                    if filt1 is None:
                        filt1, filt2 = f1_avg, f2_avg
                    else:
                        filt1 += FILTER_ALPHA * (f1_avg - filt1)
                        filt2 += FILTER_ALPHA * (f2_avg - filt2)

                    wall_time = datetime.now().isoformat(timespec="milliseconds")
                    self.data_received.emit(t, filt1, filt2, wall_time)

            except Exception as e:
                print("Force sensor read error:", e)

    def stop(self):
        self.running = False
        self.wait()
