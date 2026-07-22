"""Native loader for EMG_Eyetracker_Tool's ``emg_raw.txt`` dump.

The WaveX EMG stream is saved by the recorder as whitespace-separated text
(NOT c3d, unlike ``emg_c3d.py``). Layout, reverse-engineered from the data:

    * The file is a sequence of BLOCKS separated by blank lines.
    * Each block has ``n_channels`` lines (channel-major).
    * Within a block, line ``k`` is a chunk of consecutive samples for
      channel ``k`` (a streaming packet, ~0.1 s worth).
    * Concatenating line ``k`` across all blocks rebuilds channel ``k``.

The file carries NO header, channel labels, sample rate, or timestamps, so
the rate is supplied externally (nominal 2000 Hz for the WaveX) and the
channels are named ``Emg_1..Emg_n``. Filtering / RMS-envelope helpers are
reused from :mod:`emg_c3d` so txt- and c3d-sourced EMG share one pipeline.
"""

from pathlib import Path

import numpy as np

from .emg_c3d import EmgData  # reuse the dataclass (+ bandpass_filter/rms_envelope live there)

DEFAULT_N_CHANNELS = 8
DEFAULT_SAMPLE_RATE_HZ = 2000.0


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
