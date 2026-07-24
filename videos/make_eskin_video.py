"""Render a trial's eskin.csv as a heatmap video, trimmed to the actual
contact window (skips the baseline time before the handle is picked up and
after it's put down).

Colour style matches the live GUI heatmap (src/gui/main_window.py,
HeatmapPanel): plasma colormap, fixed levels (not auto-scaled). No per-cell
numeric labels (too slow / cluttered at video speed).

Rendering bypasses matplotlib's per-frame figure draw (the slow part of the
first version of this script -- a full Agg canvas redraw per frame, ~10
frames/sec): the colormap lookup is a single vectorized numpy call over all
frames at once, frames are upscaled and stamped with a colorbar by plain
array concatenation, and raw RGB bytes are piped straight into ffmpeg. Also
subsamples the active window to a fixed output fps (native rate is ~230 fps
here, far more than needed for playback) which cuts frame count and so
render time roughly in proportion.

Usage:
    python -m videos.make_eskin_video data/demo_PT
    python -m videos.make_eskin_video data/demo_PT --formats mp4 --vmax 800 --fps 30
    python -m videos.make_eskin_video data/some_trial --start 0 --end 304.6164 --fps 60
"""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import imageio_ffmpeg
import matplotlib
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from src.processing import eskin as eskin_proc

HEATMAP_VMIN = 0.0
DEFAULT_VMAX = 800.0
DEFAULT_FPS = 30.0

ZOOM = 24          # 16x16 taxels -> ZOOM*16 px per side
BAR_WIDTH = 48     # colorbar strip width, px
MARGIN = 8         # gap between grid and colorbar, px

# Mirrors direction/scripts/common.py's percentile-baseline segmentation,
# simplified to a single active window (this trial is one hold, not
# multiple attempts, so there's no need to split into segments).
SMOOTH_WINDOW = 41
ACTIVE_MIN_FRAC = 0.15


def find_active_window(t: np.ndarray, total: np.ndarray):
    """Return (idx_start, idx_end, threshold) for the contiguous span where
    the smoothed frame-total signal clears baseline + ACTIVE_MIN_FRAC * range.
    """
    smoothed = pd.Series(total).rolling(
        SMOOTH_WINDOW, center=True, min_periods=1).median().to_numpy()
    baseline = float(np.percentile(smoothed, 5))
    rng = float(np.percentile(smoothed, 95) - baseline)
    threshold = baseline + ACTIVE_MIN_FRAC * rng

    active = np.where(smoothed > threshold)[0]
    if active.size == 0:
        raise SystemExit("No frames cleared the contact threshold -- "
                          "handle never appears to be held in this trial.")
    return int(active[0]), int(active[-1]), threshold


def subsample_to_fps(t: np.ndarray, t0: float, t1: float, fps: float) -> np.ndarray:
    """Pick the nearest-available frame index for each tick of a uniform
    ``fps`` grid spanning [t0, t1] -- downsamples the native (irregular, much
    higher) rate without interpolating pixel data. The grid itself always
    spans exactly [t0, t1] regardless of where the nearest real samples fall,
    so frame count/duration exactly match a reference (e.g. another video
    being synced to), with only sub-frame nearest-neighbor snap error."""
    n = max(int(round((t1 - t0) * fps)) + 1, 2)
    targets = np.linspace(t0, t1, n)

    pos = np.clip(np.searchsorted(t, targets), 0, len(t) - 1)
    left = np.clip(pos - 1, 0, len(t) - 1)
    use_left = np.abs(t[left] - targets) < np.abs(t[pos] - targets)
    return np.where(use_left, left, pos)


def colorbar_strip(height: int, vmin: float, vmax: float, cmap_name: str = "plasma") -> np.ndarray:
    """(height, BAR_WIDTH, 3) uint8 gradient strip with min/max text ticks,
    built once and reused for every frame (plain array ops, not a matplotlib draw)."""
    cmap = matplotlib.colormaps[cmap_name]
    grad = np.linspace(1, 0, height)  # top = vmax, bottom = vmin
    colors = (cmap(grad)[:, :3] * 255).astype(np.uint8)
    bar = np.repeat(colors[:, None, :], BAR_WIDTH, axis=1)
    img = Image.fromarray(bar)
    draw = ImageDraw.Draw(img)
    draw.text((4, 2), f"{vmax:.0f}", fill=(255, 255, 255))
    draw.text((4, height - 14), f"{vmin:.0f}", fill=(255, 255, 255))
    return np.array(img)


def frames_to_rgb(frames: np.ndarray, vmin: float, vmax: float, cmap_name: str = "plasma") -> np.ndarray:
    """(n, 16, 16) float -> (n, 16*ZOOM, 16*ZOOM, 3) uint8, one vectorized
    colormap lookup over the whole batch -- no per-frame figure/canvas draw."""
    norm = np.clip((frames - vmin) / (vmax - vmin), 0.0, 1.0)
    cmap = matplotlib.colormaps[cmap_name]
    rgb = (cmap(norm)[..., :3] * 255).astype(np.uint8)          # (n, 16, 16, 3)
    rgb = np.repeat(np.repeat(rgb, ZOOM, axis=1), ZOOM, axis=2)  # (n, 16*ZOOM, 16*ZOOM, 3)
    return rgb


def write_mp4(rgb_frames: np.ndarray, fps: float, vmin: float, vmax: float, out_path: Path,
              cmap_name: str = "plasma") -> None:
    n, h, w, _ = rgb_frames.shape
    bar = colorbar_strip(h, vmin, vmax, cmap_name)
    margin = np.zeros((h, MARGIN, 3), dtype=np.uint8)
    total_w = w + MARGIN + BAR_WIDTH

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ffmpeg, "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
           "-pix_fmt", "rgb24", "-s", f"{total_w}x{h}", "-r", str(fps),
           "-i", "-", "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p", str(out_path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for i in range(n):
        frame = np.concatenate([rgb_frames[i], margin, bar], axis=1)
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {proc.returncode}")


def write_gif(rgb_frames: np.ndarray, fps: float, vmin: float, vmax: float, out_path: Path,
              cmap_name: str = "plasma") -> None:
    n, h, w, _ = rgb_frames.shape
    bar = colorbar_strip(h, vmin, vmax, cmap_name)
    margin = np.zeros((h, MARGIN, 3), dtype=np.uint8)
    images = [Image.fromarray(np.concatenate([rgb_frames[i], margin, bar], axis=1))
              for i in range(n)]
    images[0].save(str(out_path), save_all=True, append_images=images[1:],
                   duration=1000 / fps, loop=0)


def main():
    ap = argparse.ArgumentParser(
        description="Render a trial's eskin.csv as a heatmap video, trimmed to the contact window.")
    ap.add_argument("trial_dir", type=Path, help="trial folder containing eskin.csv")
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent,
                     help="where to write the .gif/.mp4 (default: this videos/ folder)")
    ap.add_argument("--formats", default="mp4",
                     help="comma-separated subset of gif,mp4 (default: mp4)")
    ap.add_argument("--vmax", type=float, default=DEFAULT_VMAX,
                     help=f"colour scale upper bound (default {DEFAULT_VMAX}; "
                          "lower than the GUI's 1200 makes mid-range signal changes more visible)")
    ap.add_argument("--fps", type=float, default=DEFAULT_FPS,
                     help=f"output frame rate (default {DEFAULT_FPS}; native rate is far higher, "
                          "so this subsamples frames -- fewer frames means faster rendering)")
    ap.add_argument("--mean-subtract", action="store_true",
                     help="subtract each cell's own temporal mean (over the active window) from "
                          "every frame instead of showing raw pressure -- switches to a coolwarm "
                          "diverging scale, symmetric around zero, auto-picked from the data "
                          "(ignores --vmax)")
    ap.add_argument("--start", type=float, default=None,
                     help="explicit window start (s); skips onset/offset detection when given "
                          "together with --end -- use to sync this video's time span exactly to "
                          "another already-rendered video (e.g. a force/EMG video covering the "
                          "whole trial, not just the auto-detected contact window)")
    ap.add_argument("--end", type=float, default=None, help="explicit window end (s); see --start")
    args = ap.parse_args()
    formats = {f.strip() for f in args.formats.split(",") if f.strip()}

    df = eskin_proc.load_eskin_csv(args.trial_dir / "eskin.csv")
    frames = eskin_proc.frames_array(df)
    t = df["elapsed_s"].to_numpy(dtype=float)
    total = frames.reshape(frames.shape[0], -1).sum(axis=1)

    print(f"Trial: {args.trial_dir}")
    print(f"Full trial: {len(frames)} frames, {t[-1] - t[0]:.1f}s")

    if args.start is not None and args.end is not None:
        t_start, t_end = args.start, args.end
        idx_start = int(np.searchsorted(t, t_start))
        idx_end = min(int(np.searchsorted(t, t_end, side="right")) - 1, len(t) - 1)
        print(f"Explicit window: [{t_start:.2f}s, {t_end:.2f}s] (onset/offset detection skipped)")
    else:
        idx_start, idx_end, threshold = find_active_window(t, total)
        t_start, t_end = t[idx_start], t[idx_end]
        print(f"Active window: [{t_start:.2f}s, {t_end:.2f}s] "
              f"({t_end - t_start:.1f}s, threshold={threshold:.0f})")

    sub_idx = subsample_to_fps(t, t_start, t_end, args.fps)
    active_frames = frames[sub_idx]

    if args.mean_subtract:
        mean_frame = frames[idx_start:idx_end + 1].mean(axis=0)
        active_frames = active_frames - mean_frame
        vmax = float(np.percentile(np.abs(active_frames), 99))
        vmin = -vmax
        cmap_name = "coolwarm"
        suffix = "_meandev"
        print(f"Output: {len(active_frames)} frames @ {args.fps:.1f} fps, "
              f"mean-subtracted, scale [{vmin:.0f}, {vmax:.0f}] (auto, 99th pct |deviation|)")
    else:
        vmin, vmax = HEATMAP_VMIN, args.vmax
        cmap_name = "plasma"
        suffix = ""
        print(f"Output: {len(active_frames)} frames @ {args.fps:.1f} fps, "
              f"scale [{vmin:.0f}, {vmax:.0f}]")

    rgb_frames = frames_to_rgb(active_frames, vmin, vmax, cmap_name)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    trial_name = args.trial_dir.resolve().name
    if "mp4" in formats:
        mp4_path = args.out_dir / f"{trial_name}_eskin_heatmap{suffix}.mp4"
        write_mp4(rgb_frames, args.fps, vmin, vmax, mp4_path, cmap_name)
        print(f"[write] {mp4_path}")
    if "gif" in formats:
        gif_path = args.out_dir / f"{trial_name}_eskin_heatmap{suffix}.gif"
        write_gif(rgb_frames, args.fps, vmin, vmax, gif_path, cmap_name)
        print(f"[write] {gif_path}")


if __name__ == "__main__":
    main()
