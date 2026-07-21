"""Finds the EMG .txt file that EMG_Eyetracker_Tool.exe just saved for a
trial and links (copies) it into that trial's session folder.

The tool writes CustomData\\EMG\\{header}_{fileCount}.txt synchronously
inside its own 1s CheckTimer_Tick after the F6 stop trigger, so the file
may not exist the instant send_stop() returns -- poll with a short retry
window instead of assuming it's immediate.
"""

import shutil
import time
from pathlib import Path
from typing import Optional

# EMG_Eyetracker_Tool writes to Application.StartupPath/CustomData/EMG, i.e.
# relative to wherever its .exe is run from. Default assumes the sibling
# repo layout on this machine; override via `emg_dir=` if run elsewhere.
DEFAULT_EMG_DATA_DIR = (
    Path(__file__).resolve().parents[3]
    / "EMG_Eyetracker_Tool" / "EMG_Eyetracker_Tool" / "Publish" / "CustomData" / "EMG"
)

POLL_INTERVAL_S = 0.2
POLL_TIMEOUT_S = 3.0


def find_latest_emg_file(after_time: float, emg_dir: Path = DEFAULT_EMG_DATA_DIR,
                          timeout_s: float = POLL_TIMEOUT_S) -> Optional[Path]:
    """Poll `emg_dir` for the newest .txt file modified after `after_time`
    (a time.time()-style timestamp, normally the trial's start time).
    Returns None if nothing shows up within `timeout_s`."""
    deadline = time.time() + timeout_s
    best: Optional[Path] = None
    while time.time() < deadline:
        candidates = [p for p in emg_dir.glob("*.txt") if p.stat().st_mtime >= after_time]
        if candidates:
            best = max(candidates, key=lambda p: p.stat().st_mtime)
            break
        time.sleep(POLL_INTERVAL_S)
    return best


def link_emg_file(trial_start_time: float, session_dir: Path,
                   dest_name: str = "emg_raw.txt",
                   emg_dir: Path = DEFAULT_EMG_DATA_DIR) -> Optional[Path]:
    """Find the EMG file for a just-finished trial and copy it into
    `session_dir` as `dest_name`. Returns the destination path, or None if
    no matching file was found (e.g. EMG_Eyetracker_Tool wasn't running)."""
    src = find_latest_emg_file(trial_start_time, emg_dir=emg_dir)
    if src is None:
        print(f"[session_linker] No EMG file found in {emg_dir} "
              f"(EMG_Eyetracker_Tool running and started?)")
        return None

    session_dir.mkdir(parents=True, exist_ok=True)
    dest = session_dir / dest_name
    shutil.copy2(src, dest)
    return dest
