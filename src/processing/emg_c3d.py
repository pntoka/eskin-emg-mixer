"""Loads EMG analog channels from a C3D file (see archive/PT_max_squeeze.c3d:
18 channels @ 2000 Hz, 0 markers) and computes filtered/rectified RMS
envelopes per channel.

Requires the pure-Python `c3d` package (pip install c3d) -- no compiler or
conda needed, unlike `ezc3d`.
"""

from dataclasses import dataclass
from pathlib import Path

import c3d
import numpy as np
from scipy.signal import butter, sosfiltfilt


@dataclass
class EmgData:
    channel_names: list
    sample_rate_hz: float
    signals: np.ndarray  # (n_channels, n_samples)


def load_c3d_emg(path: Path) -> EmgData:
    with open(path, "rb") as handle:
        reader = c3d.Reader(handle)
        labels = [label.strip() for label in reader.analog_labels]
        rate = float(reader.analog_rate)

        blocks = [np.asarray(analog, dtype=np.float64) for _, _, analog in reader.read_frames()]

    # Each block is (n_channels, samples_per_frame); concatenate along time.
    signals = np.concatenate(blocks, axis=1)
    return EmgData(channel_names=labels, sample_rate_hz=rate, signals=signals)


def bandpass_filter(signals: np.ndarray, sample_rate_hz: float,
                     low_hz: float = 20.0, high_hz: float = 450.0, order: int = 4) -> np.ndarray:
    nyquist = sample_rate_hz / 2.0
    sos = butter(order, [low_hz / nyquist, min(high_hz, nyquist * 0.99) / nyquist],
                 btype="bandpass", output="sos")
    return sosfiltfilt(sos, signals, axis=-1)


def rms_envelope(signals: np.ndarray, sample_rate_hz: float, window_s: float = 0.1) -> np.ndarray:
    window = max(1, int(window_s * sample_rate_hz))
    kernel = np.ones(window) / window
    squared = signals ** 2
    smoothed = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode="same"), axis=-1, arr=squared)
    return np.sqrt(smoothed)


def process(path: Path) -> dict:
    """Full pipeline: load -> bandpass -> rectify -> RMS envelope.

    Returns a dict with channel_names, sample_rate_hz, raw, filtered, envelope
    (each of the latter three shaped (n_channels, n_samples)).
    """
    emg = load_c3d_emg(path)
    filtered = bandpass_filter(emg.signals, emg.sample_rate_hz)
    envelope = rms_envelope(np.abs(filtered), emg.sample_rate_hz)
    return {
        "channel_names": emg.channel_names,
        "sample_rate_hz": emg.sample_rate_hz,
        "raw": emg.signals,
        "filtered": filtered,
        "envelope": envelope,
    }
