"""Synchronized force + EMG video for a continuous (rep-less) dynamic-grasp
trial: draws the whole trial's F_combined and 4 EMG channel envelopes once as
a static two-panel figure (same layout convention as
src/processing/trial_plots.py's plot_overview / scripts/plot_emg_channels.py),
then composites a moving vertical playhead across both panels per frame.

Rendering follows the same "render once, composite cheaply" approach as
videos/make_eskin_video.py: a full matplotlib redraw per frame would be far
too slow for a 60fps video of a multi-minute trial, so the background is
rasterized a single time and each frame is just a numpy array copy + a thin
vertical line, piped straight into ffmpeg.

EMG channel choice is passed explicitly, not read from align.py's
auto-selected trial.emg_selected: that heuristic correlates each channel
against the labelled rep-window mask, and this trial has no reps
(task_kind=free_form, repetitions=[]), so it degenerates to picking channel 0
regardless of which channels are actually live. Verified directly for this
trial: channels 0-3 (Emg_1..Emg_4) are dead (constant, rail-clipped at
-3300); channels 4-7 (Emg_5..Emg_8) carry real signal.

Usage:
    python -m videos.make_force_emg_video data/YL_grasp_dynamic_002_20260723_155859
    python -m videos.make_force_emg_video data/YL_grasp_dynamic_002_20260723_155859 --channels 4,5,6,7
"""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import imageio_ffmpeg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.processing import forces as forces_proc
from src.processing import emg_txt

DEFAULT_CHANNELS = [4, 5, 6, 7]   # Emg_5..Emg_8 -- the 4 live channels in this trial
ENV_WINDOW_S = 0.10                # matches src/processing/align.py's ENV_WINDOW_S
_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # scripts/plot_emg_channels.py palette

DEFAULT_FPS = 60.0
FIGSIZE = (12.8, 7.2)   # inches
DPI = 100                # figsize*dpi = 1280x720 (even, required for yuv420p)
PLAYHEAD_WIDTH = 3        # px
PLAYHEAD_COLOR = np.array([220, 20, 20], dtype=np.uint8)


def render_background(t_force, f_combined, t_emg, envelopes, channels, channel_names, trial_id):
    """Draw the whole-trial static figure once; return (rgb (H,W,3) uint8, ax_force for transData)."""
    fig, (ax_emg, ax_force) = plt.subplots(2, 1, sharex=True, figsize=FIGSIZE, dpi=DPI)

    for ch, env, color in zip(channels, envelopes, _COLORS):
        ax_emg.plot(t_emg, env, lw=1.0, color=color, label=channel_names[ch])
    ax_emg.set_ylabel(f"RMS envelope (uV, {ENV_WINDOW_S*1000:.0f} ms window)")
    ax_emg.legend(loc="upper right", fontsize=8)
    ax_emg.set_title(f"{trial_id} -- force vs. EMG")

    ax_force.plot(t_force, f_combined, color="k", lw=1.0)
    ax_force.set_ylabel("F_combined (N)")
    ax_force.set_xlabel("time (s from trial start)")

    # Pin the x-axis to the real data span -- matplotlib's default ~5% autoscale
    # margin would otherwise leak into the time<->pixel mapping below and make
    # the video several seconds longer than the actual trial.
    t0 = min(t_force[0], t_emg[0])
    t1 = max(t_force[-1], t_emg[-1])
    ax_force.set_xlim(t0, t1)

    fig.tight_layout()
    fig.canvas.draw()
    rgb = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()

    x0_px = ax_force.transData.transform((t0, 0))[0]
    x1_px = ax_force.transData.transform((t1, 0))[0]

    plt.close(fig)
    return rgb, t0, t1, x0_px, x1_px


def write_mp4(background: np.ndarray, t0: float, t1: float, x0_px: float, x1_px: float,
              fps: float, out_path: Path) -> None:
    h, w, _ = background.shape
    n_frames = max(int(round((t1 - t0) * fps)) + 1, 2)

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ffmpeg, "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
           "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps),
           "-i", "-", "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p", str(out_path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    half = PLAYHEAD_WIDTH // 2
    for i in range(n_frames):
        t = t0 + (t1 - t0) * i / (n_frames - 1)
        x = int(round(np.interp(t, [t0, t1], [x0_px, x1_px])))
        lo, hi = max(0, x - half), min(w, x + half + 1)
        frame = background.copy()
        frame[:, lo:hi] = PLAYHEAD_COLOR
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {proc.returncode}")
    return n_frames


def main():
    ap = argparse.ArgumentParser(
        description="Render a synchronized force + EMG video for a rep-less dynamic-grasp trial.")
    ap.add_argument("trial_dir", type=Path, help="trial folder containing forces.csv and emg_raw.txt")
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent,
                     help="where to write the .mp4 (default: this videos/ folder)")
    ap.add_argument("--channels", default=",".join(str(c) for c in DEFAULT_CHANNELS),
                     help=f"comma-separated 0-based EMG channel indices (default: live channels "
                          f"for this trial, {DEFAULT_CHANNELS})")
    ap.add_argument("--fps", type=float, default=DEFAULT_FPS)
    args = ap.parse_args()
    channels = [int(c) for c in args.channels.split(",")]
    if len(channels) > len(_COLORS):
        sys.exit(f"Only {len(_COLORS)} colors defined; got {len(channels)} channels.")

    force_df = forces_proc.load_forces_csv(args.trial_dir / "forces.csv")
    t_force = force_df["elapsed_s"].to_numpy(dtype=float)
    f_combined = force_df["F_combined"].to_numpy(dtype=float)

    emg = emg_txt.load_emg_txt(args.trial_dir / "emg_raw.txt")
    t_emg = np.arange(emg.signals.shape[1]) / emg.sample_rate_hz
    envelopes = [emg_txt.rms_envelope(emg.signals[ch], emg.sample_rate_hz, ENV_WINDOW_S)
                 for ch in channels]

    trial_id = args.trial_dir.resolve().name
    print(f"Trial: {args.trial_dir}")
    print(f"Force: {len(t_force)} samples, 0..{t_force[-1]:.1f}s")
    print(f"EMG: {emg.signals.shape[0]} nominal channels @ {emg.sample_rate_hz:.0f} Hz, "
          f"{t_emg[-1]:.1f}s -- using {[emg.channel_names[c] for c in channels]}")

    background, t0, t1, x0_px, x1_px = render_background(
        t_force, f_combined, t_emg, envelopes, channels, emg.channel_names, trial_id)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"{trial_id}_force_emg.mp4"
    n_frames = write_mp4(background, t0, t1, x0_px, x1_px, args.fps, out_path)

    print(f"Output: {n_frames} frames @ {args.fps:.0f} fps ({(t1 - t0):.1f}s)")
    print(f"[write] {out_path}")


if __name__ == "__main__":
    main()
