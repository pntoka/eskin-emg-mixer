"""Native loader + filtering utilities for EMG_Eyetracker_Tool's ``emg_raw.txt``
dump (the project does not use C3D files).

The WaveX EMG stream is saved by the recorder as whitespace-separated text.
Layout, reverse-engineered from the data:

    * The file is a sequence of BLOCKS separated by blank lines.
    * Each block has ``n_channels`` lines (channel-major).
    * Within a block, line ``k`` is a chunk of consecutive samples for
      channel ``k`` (a streaming packet, ~0.1 s worth).
    * Concatenating line ``k`` across all blocks rebuilds channel ``k``.

The file carries NO header, channel labels, sample rate, or timestamps, so
the rate is supplied externally (nominal 2000 Hz for the WaveX) and the
channels are named ``Emg_1..Emg_n``. This module also holds the shared EMG
filtering helpers (bandpass/lowpass/rms-envelope/activation-envelope) used by
``align.py`` and ``emg_activation.py``.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

DEFAULT_N_CHANNELS = 8
DEFAULT_SAMPLE_RATE_HZ = 2000.0

EMG_ACTIVE_CHANNELS = (0, 1)   # Emg_1/Emg_2 -- only these 2 of 8 channels are
                                # physically wired on the recording rig (see
                                # PROJECT.md). Single source of truth for any
                                # MVC/onset work; never rely on align.py's
                                # auto-selected emg_selected for this -- that
                                # heuristic is documented as unreliable for
                                # per-rep-captured target_force trials.


@dataclass
class EmgData:
    channel_names: list
    sample_rate_hz: float
    signals: np.ndarray  # (n_channels, n_samples)


def load_emg_txt(path: Path, n_channels: int = DEFAULT_N_CHANNELS,
                 sample_rate_hz: float = DEFAULT_SAMPLE_RATE_HZ) -> EmgData:
    """Parse ``emg_raw.txt`` into an :class:`EmgData` (signals shaped
    ``(n_channels, n_samples)``).

    Blocks whose line count differs from ``n_channels`` or whose lines have
    unequal length (malformed/partial packets) are skipped; the count of
    skipped blocks is attached as ``EmgData`` has no field for it, so callers
    that care can compare ``signals.shape`` against expectations.
    """
    blocks, current = [], []
    with open(path) as handle:
        for line in handle:
            stripped = line.strip()
            if stripped == "":
                if current:
                    blocks.append(current)
                    current = []
            else:
                current.append(np.fromstring(stripped, sep=" "))
        if current:
            blocks.append(current)

    good = [b for b in blocks
            if len(b) == n_channels and len({len(x) for x in b}) == 1]
    if not good:
        raise ValueError(
            f"No well-formed {n_channels}-channel blocks found in {path}; "
            "check n_channels or the file format.")

    signals = np.array([np.concatenate([block[k] for block in good])
                        for k in range(n_channels)])
    names = [f"Emg_{i + 1}" for i in range(n_channels)]
    return EmgData(channel_names=names, sample_rate_hz=sample_rate_hz, signals=signals)


def bandpass_filter(signals: np.ndarray, sample_rate_hz: float,
                     low_hz: float = 20.0, high_hz: float = 450.0, order: int = 4) -> np.ndarray:
    nyquist = sample_rate_hz / 2.0
    sos = butter(order, [low_hz / nyquist, min(high_hz, nyquist * 0.99) / nyquist],
                 btype="bandpass", output="sos")
    return sosfiltfilt(sos, signals, axis=-1)


def lowpass_filter(signals: np.ndarray, sample_rate_hz: float,
                    cutoff_hz: float = 5.0, order: int = 4) -> np.ndarray:
    nyquist = sample_rate_hz / 2.0
    sos = butter(order, min(cutoff_hz, nyquist * 0.99) / nyquist, btype="low", output="sos")
    return sosfiltfilt(sos, signals, axis=-1)


def rms_envelope(signals: np.ndarray, sample_rate_hz: float, window_s: float = 0.1) -> np.ndarray:
    window = max(1, int(window_s * sample_rate_hz))
    kernel = np.ones(window) / window
    squared = signals ** 2
    smoothed = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode="same"), axis=-1, arr=squared)
    return np.sqrt(smoothed)


def activation_envelope(signals: np.ndarray, sample_rate_hz: float,
                         bp_low: float = 50.0, bp_high: float = 400.0,
                         lp_cutoff: float = 5.0, order: int = 4) -> np.ndarray:
    """project_overview.md's activation-envelope recipe: bandpass -> center
    (subtract mean) -> rectify (abs) -> lowpass. Works on (n_samples,) or
    (n_channels, n_samples). Distinct from :func:`rms_envelope` (windowed RMS,
    used for channel QC/diagnostics) -- both stay, serving different consumers.
    """
    filtered = bandpass_filter(signals, sample_rate_hz, low_hz=bp_low, high_hz=bp_high, order=order)
    centered = filtered - np.mean(filtered, axis=-1, keepdims=True)
    rectified = np.abs(centered)
    return lowpass_filter(rectified, sample_rate_hz, cutoff_hz=lp_cutoff, order=order)


def combined_channel_envelope(emg: np.ndarray, sample_rate_hz: float,
                               channels: tuple = EMG_ACTIVE_CHANNELS) -> np.ndarray:
    """activation_envelope on emg[channels, :], averaged channel-wise into one
    (n_samples,) envelope. Shared by align.py (onset detection) and
    emg_activation.py (MVC/percent computation)."""
    return activation_envelope(emg[list(channels)], sample_rate_hz).mean(axis=0)
