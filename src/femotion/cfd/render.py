"""Flow-field rendering: smoothing, upscaling, piping grey-scale frames straight into a video (separate from the solver).
Needs Pillow (upscale) and ffmpeg on the PATH (GrayVideoWriter)."""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np


def binomial3(a: np.ndarray, passes: int = 1) -> np.ndarray:
    """3×3 binomial smoothing ([1,2,1]/4 once in each direction), edges replicated.
    BGK at low τ excites lattice-scale checkerboarding near sharp corners that shows in vorticity plots; one pass before
    rendering removes it without touching structures at visible scales."""
    a = np.asarray(a, np.float32)
    for _ in range(passes):
        p = np.pad(a, 1, mode="edge")
        a = 0.25 * (p[:, :-2] + 2 * p[:, 1:-1] + p[:, 2:])[1:-1]
        p = np.pad(a, ((1, 1), (0, 0)), mode="edge")
        a = 0.25 * (p[:-2] + 2 * p[1:-1] + p[2:])
    return a


def upscale(a: np.ndarray, w: int, h: int) -> np.ndarray:
    """bilinear upscaling of a float field to (h, w), pixel centres aligned."""
    from PIL import Image
    return np.asarray(Image.fromarray(np.ascontiguousarray(a, dtype=np.float32), mode="F")
                      .resize((w, h), Image.BILINEAR))


class GrayVideoWriter:
    """pipe grey-scale frames (uint8, (h, w)) to ffmpeg as H.264 yuv420p (TV range; pure grey, chroma fixed at 128)."""

    def __init__(self, path: str | Path, w: int, h: int, fps: int = 30, crf: int = 16, preset: str = "slow"):
        self.w, self.h = w, h
        cmd = ["ffmpeg", "-y", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
               # grey → yuv420p with ffmpeg's default conversion (full-range grey → TV-range luma, chroma 128, lossless round trip).
               # Tag TV range only, not bt709: tagging bt709 (or an explicit scale=out_color_matrix) decodes mid-grey 128 as 126
               "-vf", "format=yuv420p",
               "-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p",
               "-color_range", "tv", "-movflags", "+faststart", str(path)]
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, frame: np.ndarray):
        assert frame.shape == (self.h, self.w) and frame.dtype == np.uint8
        self.p.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self):
        self.p.stdin.close()
        if self.p.wait() != 0:
            raise RuntimeError("ffmpeg encoding failed")
