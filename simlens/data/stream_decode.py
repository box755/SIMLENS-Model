"""Video frame sampling with Decord and audio decoding with FFmpeg."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

# Audio sampling rate in Hz
AUDIO_SR = 16000


def find_ffmpeg() -> str:
    """Locate ffmpeg: `SIMLENS_FFMPEG`, then `Library/bin` of the Python environment, then `PATH`."""
    env = os.environ.get("SIMLENS_FFMPEG")
    if env and Path(env).exists():
        return env
    cand = Path(sys.prefix) / "Library" / "bin" / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
    if cand.exists():
        return str(cand)
    found = shutil.which("ffmpeg")
    if found:
        return found
    raise FileNotFoundError("ffmpeg not found (set SIMLENS_FFMPEG or activate the conda env)")


_FFMPEG = None
def _ffmpeg() -> str:
    """Return the ffmpeg path, located on first use."""
    global _FFMPEG
    if _FFMPEG is None:
        _FFMPEG = find_ffmpeg()
    return _FFMPEG


def decode_audio_16k(path: str | Path, *, max_sec: float | None = None) -> np.ndarray:
    """
    Decode the audio track to a mono waveform at 16 kHz.

    Args:
        path (`str` or `os.PathLike`):
            Video or audio file.
        max_sec (`float`, *optional*):
            Decode only the first `max_sec` seconds.

    Returns:
        `np.ndarray`: The waveform as `float32` in `[-1, 1]`. It is empty when the file has no audio track.
    """
    cmd = [_ffmpeg(), "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(AUDIO_SR)]
    if max_sec is not None:
        cmd += ["-t", str(float(max_sec))]
    cmd += ["-f", "s16le", "-"]
    raw = subprocess.run(cmd, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def decode_video_fps(
    path: str | Path, *, px: int = 112, fps: float = 1.0, max_sec: float | None = None
) -> np.ndarray:
    """
    Sample frames at a fixed rate and resize them to `px` x `px` pixels.

    Args:
        path (`str` or `os.PathLike`):
            Video file.
        px (`int`, *optional*, defaults to 112):
            Side length in pixels of the returned frames.
        fps (`float`, *optional*, defaults to 1.0):
            Sampling rate in frames per second.
        max_sec (`float`, *optional*):
            Decode only the first `max_sec` seconds.

    Returns:
        `np.ndarray` of shape `(num_frames, px, px, 3)`: The frames as `uint8`.
    """
    import decord
    vr = decord.VideoReader(str(path), width=px, height=px, num_threads=1)
    vfps = vr.get_avg_fps() or 30.0
    n = len(vr)
    dur = n / vfps
    if max_sec is not None:
        dur = min(dur, max_sec)
    step = 1.0 / fps
    times = np.arange(0.0, dur, step)
    # Index of the frame closest to each sampling time, from the average frame rate of the file
    idx = [min(int(round(t * vfps)), n - 1) for t in times]
    if not idx:
        return np.zeros((0, px, px, 3), dtype=np.uint8)
    return vr.get_batch(idx).asnumpy().astype(np.uint8)
