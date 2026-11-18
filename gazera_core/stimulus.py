"""Stimulus-aware RGB exclusion filter.

Research stimuli are rarely pure content: a slide has a black background, a
poster has white margins, a video has a watermark plate.  Gaze that lands on
those flat areas is usually not what the study is about, so GAZERA can remove
it before fixation/saccade classification, AOI metrics, statistics and the
heatmap — and it reports exactly how much it removed.

The stimulus may be a still image (read with Pillow) or a video (read with
OpenCV when it is installed, otherwise with an ``ffmpeg`` subprocess).  For a
video the gaze timestamp decides which frame is inspected::

    stimulus_time_ms = sample.time_ms + offset_ms
    frame_ms         = floor(stimulus_time_ms / bucket_ms) * bucket_ms

``bucket_ms`` trades accuracy for speed: one decoded frame serves every sample
that falls inside the same bucket.

Nothing here depends on a vendor format: a PNG/JPG/MP4 the researcher already
has is enough.
"""

from __future__ import annotations

import math
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "IMAGE_SUFFIXES",
    "VIDEO_SUFFIXES",
    "RGBFilterConfig",
    "RGBFilterReport",
    "Stimulus",
    "apply_rgb_exclusion",
    "open_stimulus",
    "parse_color",
    "parse_colors",
]

IMAGE_SUFFIXES: frozenset[str] = frozenset(
    {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".webp", ".avif"}
)
VIDEO_SUFFIXES: frozenset[str] = frozenset({".mp4", ".mov", ".avi", ".webm", ".mkv", ".m4v", ".ogv"})

_HEX = re.compile(r"^#?([0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")
_NUMBERS = re.compile(r"-?\d+(?:\.\d+)?")
_NAMED = {
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "gray": (128, 128, 128),
    "grey": (128, 128, 128),
    "red": (255, 0, 0),
    "green": (0, 128, 0),
    "blue": (0, 0, 255),
}

#: Largest frame kept in memory while decoding a video (long side, px).
_MAX_FRAME_SIDE = 1920


# ---------------------------------------------------------------------------
# colour parsing
# ---------------------------------------------------------------------------


def parse_color(value: Any) -> tuple[int, int, int]:
    """Parse one colour into ``(r, g, b)``.

    Accepted: ``"#000000"``, ``"#fff"``, ``"0,0,0"``, ``"0 0 0"``,
    ``"rgb(0, 0, 0)"``, ``"black"``, ``(0, 0, 0)`` and ``[0, 0, 0]``.
    """
    if isinstance(value, (tuple, list, np.ndarray)):
        parts = list(value)
        if len(parts) < 3:
            raise ValueError(f"RGB colour needs three values, got {value!r}")
        rgb = tuple(int(round(float(parts[i]))) for i in range(3))
    else:
        text = str(value).strip().lower()
        if not text:
            raise ValueError("Empty colour value")
        if text in _NAMED:
            return _NAMED[text]
        match = _HEX.match(text)
        if match:
            digits = match.group(1)
            if len(digits) == 3:
                digits = "".join(ch * 2 for ch in digits)
            return (int(digits[0:2], 16), int(digits[2:4], 16), int(digits[4:6], 16))
        numbers = _NUMBERS.findall(text)
        if len(numbers) < 3:
            raise ValueError(f"Cannot read a colour from {value!r}")
        rgb = tuple(int(round(float(numbers[i]))) for i in range(3))
    if not all(0 <= channel <= 255 for channel in rgb):
        raise ValueError(f"Colour channels must be 0-255, got {rgb}")
    return rgb  # type: ignore[return-value]


def parse_colors(values: Any) -> list[tuple[int, int, int]]:
    """Parse one colour or a sequence of colours, dropping duplicates."""
    if values is None:
        return []
    if isinstance(values, tuple) and len(values) == 3 and all(
        isinstance(v, (int, float, np.number)) for v in values
    ):
        return [parse_color(values)]  # a single (r, g, b) triple
    if isinstance(values, str):
        chunks = [c for c in re.split(r"[;|]", values) if c.strip()]
        return _dedupe(parse_color(chunk) for chunk in chunks)
    out: list[tuple[int, int, int]] = []
    for item in values:
        if isinstance(item, str) and re.search(r"[;|]", item):
            out.extend(parse_color(c) for c in item.split(";") if c.strip())
        else:
            out.append(parse_color(item))
    return _dedupe(out)


def _dedupe(colors: Iterable[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    seen: set[tuple[int, int, int]] = set()
    out: list[tuple[int, int, int]] = []
    for color in colors:
        if color not in seen:
            seen.add(color)
            out.append(color)
    return out


# ---------------------------------------------------------------------------
# configuration and report
# ---------------------------------------------------------------------------


@dataclass
class RGBFilterConfig:
    """What to exclude, and how the stimulus timeline is sampled."""

    colors: list[tuple[int, int, int]] = field(default_factory=list)
    tolerance: float = 15.0
    offset_ms: float = 0.0
    bucket_ms: float = 200.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "exclude_rgb": [list(c) for c in self.colors],
            "tolerance": self.tolerance,
            "offset_ms": self.offset_ms,
            "frame_bucket_ms": self.bucket_ms,
        }


@dataclass
class RGBFilterReport:
    """Everything a reader needs to judge what the filter removed."""

    stimulus: str
    kind: str
    resolution: tuple[int, int] | None
    colors: list[list[int]]
    tolerance: float
    offset_ms: float
    frame_bucket_ms: float
    samples_total: int
    samples_sampled: int
    excluded_samples: int
    excluded_ratio: float
    excluded_by_color: list[dict[str, Any]] = field(default_factory=list)
    frames_read: int = 0
    duration_ms: float | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stimulus": self.stimulus,
            "kind": self.kind,
            "resolution": list(self.resolution) if self.resolution else None,
            "duration_ms": None if self.duration_ms is None else round(self.duration_ms, 1),
            "exclude_rgb": self.colors,
            "tolerance": self.tolerance,
            "offset_ms": self.offset_ms,
            "frame_bucket_ms": self.frame_bucket_ms,
            "samples_total": self.samples_total,
            "samples_sampled": self.samples_sampled,
            "excluded_samples": self.excluded_samples,
            "excluded_ratio": round(self.excluded_ratio, 6),
            "excluded_by_color": self.excluded_by_color,
            "frames_read": self.frames_read,
            "warnings": self.warnings,
        }


# ---------------------------------------------------------------------------
# stimulus reading
# ---------------------------------------------------------------------------


class Stimulus:
    """A still image or video that gaze coordinates can be sampled from."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Stimulus not found: {self.path}")
        self.suffix = self.path.suffix.lower()
        if self.suffix in IMAGE_SUFFIXES:
            self.kind = "image"
        elif self.suffix in VIDEO_SUFFIXES:
            self.kind = "video"
        else:
            raise ValueError(
                f"Unsupported stimulus type {self.suffix or '(none)'}: "
                f"use an image ({', '.join(sorted(IMAGE_SUFFIXES))}) "
                f"or a video ({', '.join(sorted(VIDEO_SUFFIXES))})."
            )
        self.width = 0
        self.height = 0
        self.duration_ms: float | None = None
        self._array: np.ndarray | None = None
        self._cap: Any = None
        self._cache: dict[int, np.ndarray] = {}
        self._cache_order: list[int] = []
        self.frames_read = 0
        self._ffmpeg: str | None = None
        if self.kind == "image":
            self._open_image()
        else:
            self._open_video()

    # -- images ----------------------------------------------------------
    def _open_image(self) -> None:
        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - Pillow is a dependency
            raise RuntimeError(
                "Reading a stimulus image needs Pillow: python -m pip install Pillow"
            ) from exc
        with Image.open(self.path) as img:
            try:
                rgb = img.convert("RGB")
            except Exception as exc:  # pragma: no cover - corrupted image
                raise ValueError(f"Could not decode the stimulus image: {self.path}") from exc
            self.width, self.height = rgb.size
            self._array = np.asarray(rgb, dtype=np.uint8)

    # -- videos ----------------------------------------------------------
    def _open_video(self) -> None:
        try:  # OpenCV is optional but the preferred, fastest decoder
            import cv2  # type: ignore

            cap = cv2.VideoCapture(str(self.path))
            if cap.isOpened():
                self._cap = cap
                self._cv2 = cv2
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
                frames = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
                if width and height:
                    self.width, self.height = width, height
                if fps > 0 and frames > 0:
                    self.duration_ms = frames / fps * 1000.0
                return
        except ImportError:
            pass
        except Exception:  # pragma: no cover - a broken codec should fall through
            pass
        self._cap = None
        self._open_video_ffmpeg()

    def _open_video_ffmpeg(self) -> None:
        executable = shutil.which("ffmpeg")
        if not executable:
            raise RuntimeError(
                "Reading a stimulus video needs OpenCV (python -m pip install opencv-python-headless) "
                "or an ffmpeg executable on PATH."
            )
        self._ffmpeg = executable
        probe = shutil.which("ffprobe")
        width = height = 0
        duration = None
        if probe:
            try:
                out = subprocess.run(
                    [
                        probe, "-v", "error", "-select_streams", "v:0",
                        "-show_entries", "stream=width,height:format=duration",
                        "-of", "default=nokey=1:noprint_wrappers=1", str(self.path),
                    ],
                    capture_output=True, text=True, timeout=30, check=False,
                )
                numbers = [line for line in out.stdout.split() if line.strip()]
                if len(numbers) >= 2:
                    width, height = int(float(numbers[0])), int(float(numbers[1]))
                if len(numbers) >= 3:
                    duration = float(numbers[2]) * 1000.0
            except Exception:  # pragma: no cover
                pass
        if not width or not height:
            raise RuntimeError("Could not read the video size; install OpenCV or keep ffprobe on PATH.")
        self.width, self.height = _fit(width, height)
        self.duration_ms = duration

    # -- sampling --------------------------------------------------------
    def close(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:  # pragma: no cover
                pass
            self._cap = None
        self._cache.clear()
        self._cache_order.clear()

    def __enter__(self) -> "Stimulus":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def _scale(self) -> tuple[int, int]:
        return _fit(self.width, self.height)

    @property
    def resolution(self) -> tuple[int, int]:
        """Pixel size actually used for sampling (long side capped)."""
        return self._scale()

    def _pixel_index(self, xs: np.ndarray, ys: np.ndarray, screen: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
        width, height = self._scale()
        sw, sh = screen
        u = np.clip(np.asarray(xs, dtype=float) / (sw or 1.0), 0.0, 1.0)
        v = np.clip(np.asarray(ys, dtype=float) / (sh or 1.0), 0.0, 1.0)
        px = np.clip((u * (width - 1) + 0.5).astype(int), 0, width - 1)
        py = np.clip((v * (height - 1) + 0.5).astype(int), 0, height - 1)
        return px, py

    def colors_at(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        times_ms: np.ndarray | None,
        screen: tuple[float, float],
        *,
        offset_ms: float = 0.0,
        bucket_ms: float = 200.0,
    ) -> np.ndarray:
        """Return an ``(n, 3)`` uint8 array of stimulus colours, NaN-free."""
        xs = np.asarray(xs, dtype=float)
        ys = np.asarray(ys, dtype=float)
        out = np.full((len(xs), 3), np.nan, dtype=float)
        px, py = self._pixel_index(xs, ys, screen)
        if self.kind == "image":
            array = self._array
            if array is None:  # pragma: no cover
                return out
            out[:, :] = array[py, px, :]
            return out

        if times_ms is None:
            times = np.zeros(len(xs))
        else:
            times = np.asarray(times_ms, dtype=float)
        stimulus_time = times + float(offset_ms)
        if self.duration_ms:
            stimulus_time = np.clip(stimulus_time, 0.0, max(self.duration_ms - 1.0, 0.0))
        else:
            stimulus_time = np.clip(stimulus_time, 0.0, None)
        bucket_ms = max(float(bucket_ms), 1.0)
        buckets = np.floor(stimulus_time / bucket_ms).astype(int)
        for bucket in np.unique(buckets):
            frame = self._frame_for_bucket(int(bucket), bucket_ms)
            if frame is None:
                continue
            sel = buckets == bucket
            out[sel, :] = frame[py[sel], px[sel], :]
        return out

    # -- video decoding ---------------------------------------------------
    def _frame_for_bucket(self, bucket: int, bucket_ms: float) -> np.ndarray | None:
        if bucket in self._cache:
            return self._cache[bucket]
        target_ms = bucket * bucket_ms
        frame = self._read_video_frame(target_ms)
        if frame is None:
            return None
        self.frames_read += 1
        self._cache[bucket] = frame
        self._cache_order.append(bucket)
        while len(self._cache_order) > 2:  # keep memory flat on long videos
            self._cache.pop(self._cache_order.pop(0), None)
        return frame

    def _read_video_frame(self, target_ms: float) -> np.ndarray | None:
        if self._cap is not None:
            cap = self._cap
            cap.set(self._cv2.CAP_PROP_POS_MSEC, float(max(target_ms, 0.0)))
            ok, frame = cap.read()
            if not ok or frame is None:
                cap.set(self._cv2.CAP_PROP_POS_MSEC, 0.0)
                ok, frame = cap.read()
                if not ok or frame is None:
                    return None
            width, height = self._scale()
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = self._cv2.resize(frame, (width, height), interpolation=self._cv2.INTER_AREA)
            return frame[:, :, ::-1].copy()  # BGR -> RGB

        # ffmpeg fallback: one frame per bucket, raw RGB on stdout
        width, height = self._scale()
        assert self._ffmpeg is not None
        command = [
            self._ffmpeg, "-v", "error", "-ss", f"{max(target_ms, 0.0) / 1000.0:.3f}",
            "-i", str(self.path), "-frames:v", "1", "-vf", f"scale={width}:{height}",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
        ]
        try:
            completed = subprocess.run(command, capture_output=True, timeout=60, check=False)
        except Exception:  # pragma: no cover
            return None
        expected = width * height * 3
        if completed.returncode != 0 or len(completed.stdout) < expected:
            return None
        array = np.frombuffer(completed.stdout[:expected], dtype=np.uint8)
        return array.reshape(height, width, 3).copy()


def _fit(width: int, height: int) -> tuple[int, int]:
    """Keep the aspect ratio but cap the long side so memory stays bounded."""
    if not width or not height:
        return width or 1, height or 1
    longest = max(width, height)
    if longest <= _MAX_FRAME_SIDE:
        return int(width), int(height)
    factor = _MAX_FRAME_SIDE / longest
    return max(1, int(round(width * factor))), max(1, int(round(height * factor)))


def open_stimulus(path: str | Path) -> Stimulus:
    """Open an image or video stimulus (the caller closes it)."""
    try:
        return Stimulus(path)
    except ValueError:
        raise
    except Exception as exc:  # unreadable file -> a clear 4xx, never a 500
        raise ValueError(f"Could not open the stimulus {path}: {exc}") from exc


# ---------------------------------------------------------------------------
# frame-level exclusion
# ---------------------------------------------------------------------------


def _screen_size(frame: pd.DataFrame, screen: tuple[float, float] | None) -> tuple[float, float]:
    if screen:
        width, height = screen
    else:
        width, height = frame.attrs.get("screen", (np.nan, np.nan))
    if (width is None or not np.isfinite(width)) and "media_width" in frame and frame["media_width"].notna().any():
        width = float(frame["media_width"].dropna().median())
    if (height is None or not np.isfinite(height)) and "media_height" in frame and frame["media_height"].notna().any():
        height = float(frame["media_height"].dropna().median())
    if not width or not height or not np.isfinite(width) or not np.isfinite(height):
        raise ValueError("Screen size is required to map gaze onto the stimulus; pass screen=(width, height).")
    return float(width), float(height)


def distance_to_colors(rgb: np.ndarray, colors: Sequence[tuple[int, int, int]]) -> tuple[np.ndarray, list[np.ndarray]]:
    """Euclidean RGB distance from every sample to every excluded colour."""
    distances = []
    for color in colors:
        delta = np.asarray(rgb, dtype=float) - np.asarray(color, dtype=float)
        distances.append(np.sqrt(np.nansum(delta**2, axis=1)))
    if not distances:
        return np.full(len(rgb), np.inf), []
    return np.min(np.vstack(distances), axis=0), distances


def apply_rgb_exclusion(
    frame: pd.DataFrame,
    stimulus: str | Path | Stimulus,
    colors: Sequence[Any],
    *,
    tolerance: float = 15.0,
    offset_ms: float = 0.0,
    bucket_ms: float = 200.0,
    screen: tuple[float, float] | None = None,
    keep_open: bool = False,
) -> tuple[pd.DataFrame, RGBFilterReport]:
    """Mark gaze samples that fall on an excluded stimulus colour.

    Excluded samples keep their coordinates but are flagged
    (``rgb_excluded``) and set to ``valid = False`` so I-VT, AOI metrics,
    condition statistics and the heatmap all skip them.  Nothing is deleted:
    the original rows stay in the frame.
    """
    parsed = parse_colors(colors)
    if not parsed:
        raise ValueError("No RGB colour to exclude was given.")
    own = not isinstance(stimulus, Stimulus)
    stim = open_stimulus(stimulus) if own else stimulus
    try:
        width, height = _screen_size(frame, screen)
        xs = pd.to_numeric(frame.get("gaze_x"), errors="coerce").to_numpy(dtype=float)
        ys = pd.to_numeric(frame.get("gaze_y"), errors="coerce").to_numpy(dtype=float)
        times = (
            pd.to_numeric(frame["time_ms"], errors="coerce").to_numpy(dtype=float)
            if "time_ms" in frame
            else np.zeros(len(frame))
        )
        sampled_mask = np.isfinite(xs) & np.isfinite(ys)
        rgb = np.full((len(frame), 3), np.nan, dtype=float)
        if sampled_mask.any():
            rgb[sampled_mask, :] = stim.colors_at(
                xs[sampled_mask],
                ys[sampled_mask],
                times[sampled_mask],
                (width, height),
                offset_ms=offset_ms,
                bucket_ms=bucket_ms,
            )

        excluded = np.zeros(len(frame), dtype=bool)
        per_color: list[dict[str, Any]] = []
        for color in parsed:
            delta = rgb - np.asarray(color, dtype=float)
            distance = np.sqrt(np.nansum(delta**2, axis=1))
            hit = sampled_mask & (distance <= float(tolerance))
            excluded |= hit
            per_color.append({"rgb": list(color), "samples": int(hit.sum())})

        out = frame.copy()
        out.attrs = dict(frame.attrs)
        out["stimulus_rgb"] = [
            _hex(row) if row[0] == row[0] else None for row in rgb  # NaN -> None
        ]
        out["rgb_excluded"] = excluded
        if "valid" in out:
            out.loc[excluded, "valid"] = False
        else:  # pragma: no cover - the loader always provides valid
            out["valid"] = ~excluded

        total = int(len(out))
        sampled = int(sampled_mask.sum())
        removed = int(excluded.sum())
        warnings: list[str] = []
        if total and removed / total >= 0.5:
            warnings.append(
                f"RGB filter removed {removed / total:.0%} of the samples; "
                "check the tolerance and the excluded colours."
            )
        if sampled < total * 0.9:
            warnings.append("Some samples have no coordinates, so no stimulus colour could be read for them.")

        report = RGBFilterReport(
            stimulus=str(stim.path),
            kind=stim.kind,
            resolution=stim._scale(),
            colors=[list(c) for c in parsed],
            tolerance=float(tolerance),
            offset_ms=float(offset_ms),
            frame_bucket_ms=float(bucket_ms),
            samples_total=total,
            samples_sampled=sampled,
            excluded_samples=removed,
            excluded_ratio=(removed / total) if total else 0.0,
            excluded_by_color=per_color,
            frames_read=int(stim.frames_read),
            duration_ms=stim.duration_ms,
            warnings=warnings,
        )
        return out, report
    finally:
        if own and not keep_open:
            stim.close()


def _hex(row: Iterable[float]) -> str:
    r, g, b = (int(round(float(channel))) for channel in list(row)[:3])
    return f"#{r:02X}{g:02X}{b:02X}"


def haversine_rgb(a: tuple[int, int, int], b: tuple[int, int, int]) -> float:  # pragma: no cover - helper
    return math.sqrt(sum((int(x) - int(y)) ** 2 for x, y in zip(a, b)))
