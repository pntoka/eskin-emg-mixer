"""Drives the existing EMG_Eyetracker_Tool.exe (Cometa WaveX EMG capture) by
simulating the same global F5/F6 keypresses that its README documents Unity
sending as start/stop triggers.

EMG_Eyetracker_Tool.exe must already be running with its own "Start" button
clicked (this opens the WaveX device connection and arms the F5/F6 global
hotkey handling registered via RegisterHotKey/WM_HOTKEY in its Form1.cs)
before send_start()/send_stop() are called here.

`keyboard.send()` injects the keypress via the Windows SendInput API, which
is what actually fires a registered global hotkey -- posting a window
message directly would not.
"""

import keyboard

START_KEY = "f5"
STOP_KEY = "f6"


def send_start() -> None:
    keyboard.send(START_KEY)


def send_stop() -> None:
    keyboard.send(STOP_KEY)
