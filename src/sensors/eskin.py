"""E-skin (16x16 FSR pressure matrix) serial protocol and reader thread.

Protocol (request/response):
    PC sends 1 byte: bits[7:6] = mode (01 = FSR-only), bits[5:0] = grid size (16)
    Board replies 520 bytes: 8-byte header + 256 x uint16 FSR readings.
"""

import time

import numpy as np
import serial
from PyQt6 import QtCore

ESKIN_BAUD = 500000
ESKIN_TARE_SAMPLES = 100  # frames averaged for the e-skin baseline

ESKIN_MODE = 0b01  # FSR-only mode (no accelerometer)
ESKIN_GRID = 16
ESKIN_DATA_LEN = 8 + 512  # 8-byte header + 256 x uint16


def remap(data):
    """Correct the physical sensor layout to match the glove's spatial orientation."""
    remapped = data.T
    remapped = remapped[:, ::-1]
    row_order = [0, 1, 2, 3, 4, 5, 6, 7, 15, 14, 13, 12, 11, 10, 9, 8]
    return remapped[row_order, :]


class EskinReader(QtCore.QThread):
    """
    Polls the e-skin board on a dedicated thread so the GUI never blocks
    waiting for serial data.

    Emits:
        data_received(np.ndarray): shaped (16, 1, 16), uint16 FSR values,
            already reordered so rows 0-7 are the first half of the sensor
            array and rows 8-15 are the second half flipped.
    """

    data_received = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, port, parent=None):
        super().__init__(parent)
        self.port = port
        self.running = True
        try:
            self.ser = serial.Serial(port, ESKIN_BAUD, timeout=3)
            time.sleep(1)
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()
        except Exception as e:
            print("E-skin serial connection error:", e)
            self.running = False

    def run(self):
        cmd_bytes = ((ESKIN_MODE << 6) | ESKIN_GRID).to_bytes(1, "big")

        while self.running:
            try:
                self.ser.write(cmd_bytes)
                d = self.ser.read(ESKIN_DATA_LEN)

                if len(d) != ESKIN_DATA_LEN:
                    continue

                fsr = np.frombuffer(d[8:], dtype=np.uint16).reshape((16, 1, 16))
                # Rows 8-15 are wired in reverse order on the PCB.
                fsr = np.concatenate((fsr[:8], np.flip(fsr[8:], axis=0)))
                # (row, layer, col) -> (col, layer, row)
                fsr = fsr.transpose((2, 1, 0))

                self.data_received.emit(fsr)

            except Exception as e:
                print("E-skin serial read error:", e)
                self.running = False

    def stop(self):
        self.running = False
        self.wait()
        if hasattr(self, "ser") and self.ser.is_open:
            self.ser.close()
