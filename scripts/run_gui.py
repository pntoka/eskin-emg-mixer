"""Entry point for the grasp-recording GUI.

Usage:
    python -m scripts.run_gui                   # auto-detect both sensor ports
    python -m scripts.run_gui COM3 COM4          # override: eskin, forces

Prerequisites:
    - E-skin + force-sensor Arduino boards connected via USB.
    - EMG_Eyetracker_Tool.exe already running with its own "Start" button
      clicked (see EMG_Eyetracker_Tool/README.md) if EMG recording is needed.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PyQt6 import QtWidgets

from src.gui.main_window import MainWindow
from src.sensors.forces import load_saved_force_bias
from src.sensors.ports import find_two_ports


def main():
    if len(sys.argv) >= 3:
        eskin_port, forces_port = sys.argv[1], sys.argv[2]
        print(f"Using manually specified ports: eskin={eskin_port}  forces={forces_port}")
    else:
        eskin_port, forces_port = find_two_ports()

    load_saved_force_bias()

    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow(eskin_port, forces_port)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
