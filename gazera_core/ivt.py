"""Clean-room I-VT (Identification by Velocity Threshold) classifier.

Why this module exists
----------------------
An export may contain only raw gaze coordinates.  Fixation and saccade then do
not exist as columns yet: they have to be derived.  GAZERA derives them with its
own implementation of the classical velocity-threshold algorithm, written from
the published literature (Salvucci & Goldberg, 2000), not from any vendor's
code.  No vendor library, binary or reverse-engineered format is used.

Pipeline per recording:
    samples -> smoothed velocity in deg/s -> threshold -> runs
    -> merge runs shorter than the minimum durations -> segments

Angles are computed from a pixel-per-degree estimate derived from screen
geometry, so thresholds are in visual degrees per second and independent of the
screen resolution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
import pandas as pd

__all__ = [
    "IVTConfig",
    "IVTResult",
    "classify_ivt",
    "estimate_px_per_deg",
    "assign_events",
    "compare_with_export_events",
]


@dataclass
class IVTConfig:
    """Parameters for velocity-threshold classification."""

    velocity_threshold_deg_s: float = 30.0
    min_fixation_ms: float = 60.0
    min_saccade_ms: float = 20.0
    max_sample_gap_ms: float = 75.0
    smoothing_samples: int = 1
    px_per_deg: float | None = None
    screen_width_px: float = 1920.0
    screen_width_mm: float = 530.0
    viewing_distance_mm: float = 650.0
    include_unclassified: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "velocity_threshold_deg_s": self.velocity_threshold_deg_s,
            "min_fixation_ms": self.min_fixation_ms,
            "min_saccade_ms": self.min_saccade_ms,
            "max_sample_gap_ms": self.max_sample_gap_ms,
            "smoothing_samples": self.smoothing_samples,
            "px_per_deg": self.px_per_deg,
            "screen_width_px": self.screen_width_px,
            "screen_width_mm": self.screen_width_mm,
            "viewing_distance_mm": self.viewing_distance_mm,
        }


@dataclass
class IVTResult:
    segments: pd.DataFrame
    samples: pd.DataFrame
    px_per_deg: float
    config: dict[str, Any] = field(default_factory=dict)
    agreement: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "px_per_deg": round(self.px_per_deg, 3),
            "config": self.config,
            "segments": _records(self.segments),
            "agreement_with_export": self.agreement,
        }


def _records(df: pd.DataFrame, limit: int | None = None) -> list[dict[str, Any]]:
    if df is None or df.empty:
        return []
    out = df.replace({np.nan: None}).to_dict(orient="records")
    if limit:
        return out[:limit]
    return out


def estimate_px_per_deg(
    screen_width_px: float = 1920.0,
    screen_width_mm: float = 530.0,
    viewing_distance_mm: float = 650.0,
) -> float:
    """Pixels subtended by one visual degree at the screen centre.

    Assumes a flat screen viewed perpendicular to its centre.
    """
    if screen_width_mm <= 0 or viewing_distance_mm <= 0:
        return 1.0
    width_deg = 2.0 * math.degrees(math.atan((screen_width_mm / 2.0) / viewing_distance_mm))
    if width_deg <= 0:
        return 1.0
    return float(screen_width_px) / width_deg


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window is None or window <= 1 or len(values) < window:
        return values
    kernel = np.ones(window) / window
    padded = np.concatenate([np.full(window // 2, values[0]), values, np.full(window // 2, values[-1])])
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed[: len(values)]


def _segments_for_run(
    times: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    labels: np.ndarray,
    px_per_deg: float,
    meta: dict[str, Any],
) -> list[dict[str, Any]]:
    """Turn per-sample labels into segment records."""
    rows: list[dict[str, Any]] = []
    if len(labels) == 0:
        return rows
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            end = i - 1
            kind = "fixation" if labels[start] == 1 else "saccade"
            seg_x = xs[start : end + 1]
            seg_y = ys[start : end + 1]
            seg_t = times[start : end + 1]
            duration = float(seg_t[-1] - seg_t[0]) if len(seg_t) > 1 else 0.0
            dt = np.diff(seg_t) / 1000.0
            if len(dt) and np.any(dt > 0):
                dx = np.diff(seg_x)
                dy = np.diff(seg_y)
                dist_deg = np.hypot(dx, dy) / px_per_deg
                with np.errstate(divide="ignore", invalid="ignore"):
                    vel = np.where(dt > 0, dist_deg / dt, 0.0)
                mean_v = float(np.nanmean(vel)) if len(vel) else 0.0
                peak_v = float(np.nanmax(vel)) if len(vel) else 0.0
            else:
                mean_v = peak_v = 0.0
            if kind == "saccade" and len(seg_x) >= 2:
                head = int(min(3, len(seg_x)))
                x0, y0 = float(np.mean(seg_x[:head])), float(np.mean(seg_y[:head]))
                x1, y1 = float(np.mean(seg_x[-head:])), float(np.mean(seg_y[-head:]))
            else:
                x0, y0 = float(np.mean(seg_x)), float(np.mean(seg_y))
                x1, y1 = x0, y0
            amplitude = float(math.hypot(x1 - x0, y1 - y0) / px_per_deg) if kind == "saccade" else 0.0
            rows.append(
                {
                    **meta,
                    "type": kind,
                    "start_ms": float(seg_t[0]),
                    "end_ms": float(seg_t[-1]),
                    "duration_ms": duration,
                    "start_x": x0,
                    "start_y": y0,
                    "end_x": x1,
                    "end_y": y1,
                    "center_x": float(np.mean(seg_x)),
                    "center_y": float(np.mean(seg_y)),
                    "amplitude_deg": amplitude,
                    "mean_velocity_deg_s": mean_v,
                    "peak_velocity_deg_s": peak_v,
                    "n_samples": int(len(seg_x)),
                }
            )
            start = i
    return rows


def _runs(labels: np.ndarray) -> list[tuple[int, int, int]]:
    out = []
    if len(labels) == 0:
        return out
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            out.append((start, i - 1, int(labels[start])))
            start = i
    return out


def _flip_short_runs(
    labels: np.ndarray,
    times: np.ndarray,
    kind: int,
    min_ms: float,
    *,
    between_only: bool = False,
) -> np.ndarray:
    """Flip runs of ``kind`` shorter than ``min_ms`` to the other label.

    Flipping is monotone: a run never becomes shorter by absorbing its
    neighbour, so repeated passes converge instead of oscillating.
    """
    out = labels.copy()
    for start, end, label in _runs(out):
        if label != kind:
            continue
        duration = float(times[end] - times[start]) if end > start else 0.0
        if duration >= min_ms:
            continue
        if between_only and (start == 0 or end == len(out) - 1):
            continue
        out[start : end + 1] = 1 - kind
    return out


def _relabel(
    labels: np.ndarray,
    times: np.ndarray,
    config: IVTConfig,
) -> np.ndarray:
    """Enforce minimum fixation and saccade durations (fixation=1, saccade=0).

    Short fixations are absorbed into the surrounding saccade, short saccades
    into the surrounding fixation.  A final fixation pass guarantees the
    reported fixation durations never fall below ``min_fixation_ms``.
    """
    labels = labels.copy()
    if len(labels) == 0:
        return labels
    for _ in range(4):
        labels = _flip_short_runs(labels, times, 1, config.min_fixation_ms)
        labels = _flip_short_runs(labels, times, 0, config.min_saccade_ms, between_only=True)
    labels = _flip_short_runs(labels, times, 1, config.min_fixation_ms)
    return labels


def classify_ivt(
    frame: pd.DataFrame,
    config: IVTConfig | None = None,
    *,
    min_valid_ratio: float = 0.2,
) -> IVTResult:
    """Classify raw gaze samples into fixation and saccade segments.

    ``frame`` must be a canonical GAZERA frame (``gaze_x``, ``gaze_y``,
    ``time_ms``, ``valid``, optional ``subject``/``trial``/``condition``).
    """
    config = config or IVTConfig()
    required = {"gaze_x", "gaze_y", "time_ms"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"classify_ivt needs columns: {sorted(missing)}")

    width, height = frame.attrs.get("screen", (np.nan, np.nan))
    if "media_width" in frame and frame["media_width"].notna().any():
        width = float(frame["media_width"].dropna().median())
    px_per_deg = config.px_per_deg or estimate_px_per_deg(
        width if width and not np.isnan(width) else config.screen_width_px,
        config.screen_width_mm,
        config.viewing_distance_mm,
    )

    group_cols = [c for c in ("subject", "trial") if c in frame and frame[c].notna().any()]
    groups: Iterable[tuple[Any, pd.DataFrame]]
    if group_cols:
        groups = frame.groupby(group_cols, dropna=False, sort=False)
    else:  # pragma: no cover - single recording path
        groups = [((None,) * 0, frame)]

    sample_frames = []
    segment_rows: list[dict[str, Any]] = []

    for key, sub in groups:
        if not isinstance(key, tuple):
            key = (key,)
        meta = {col: (None if pd.isna(val) else val) for col, val in zip(group_cols, key)}
        sub = sub.sort_values("time_ms").reset_index(drop=True)
        if "condition" in sub and sub["condition"].notna().any():
            meta["condition"] = sub["condition"].dropna().iloc[0]
        valid = sub["valid"].fillna(False).astype(bool).to_numpy() if "valid" in sub else np.ones(len(sub), bool)
        times = sub["time_ms"].to_numpy(dtype=float)
        xs = sub["gaze_x"].to_numpy(dtype=float)
        ys = sub["gaze_y"].to_numpy(dtype=float)

        dt = np.diff(times, prepend=times[0] - np.nanmedian(np.diff(times)) if len(times) > 1 else 0.0)
        # split the recording into continuous runs
        breaks = (~valid) | (dt > config.max_sample_gap_ms) | np.isnan(xs) | np.isnan(ys)
        breaks[0] = True  # every recording starts a new continuous run

        run_start = None
        labels_all = np.full(len(sub), -1, dtype=int)
        for i in range(len(sub) + 1):
            is_break = i == len(sub) or breaks[i]
            if not is_break and run_start is None:
                run_start = i
            if is_break and run_start is not None:
                rs, re_ = run_start, i  # [rs, re_)
                run_start = None
                if re_ - rs < 3:
                    continue
                rt = times[rs:re_]
                rx = _smooth(xs[rs:re_], config.smoothing_samples)
                ry = _smooth(ys[rs:re_], config.smoothing_samples)
                dt_s = np.diff(rt) / 1000.0
                dist = np.hypot(np.diff(rx), np.diff(ry)) / px_per_deg
                with np.errstate(divide="ignore", invalid="ignore"):
                    velocity = np.where(dt_s > 0, dist / dt_s, 0.0)
                velocity = np.concatenate([[velocity[0] if len(velocity) else 0.0], velocity])
                labels = (velocity <= config.velocity_threshold_deg_s).astype(int)
                labels = _relabel(labels, rt, config)
                labels_all[rs:re_] = labels
                segment_rows.extend(
                    _segments_for_run(rt, rx, ry, labels, px_per_deg, meta)
                )

        sub_out = sub.copy()
        sub_out["ivt_label"] = np.where(labels_all == 1, "fixation", np.where(labels_all == 0, "saccade", "unclassified"))
        sub_out["velocity_deg_s"] = np.nan
        for i in range(1, len(sub_out)):
            if labels_all[i] >= 0 and labels_all[i - 1] >= 0:
                dt_s = (times[i] - times[i - 1]) / 1000.0
                if dt_s > 0:
                    sub_out.loc[i, "velocity_deg_s"] = (
                        math.hypot(xs[i] - xs[i - 1], ys[i] - ys[i - 1]) / px_per_deg / dt_s
                    )
        sample_frames.append(sub_out)

    segments = pd.DataFrame(segment_rows)
    if not segments.empty:
        segments["duration_s"] = segments["duration_ms"] / 1000.0
        seg_groups = [c for c in ("subject", "trial") if c in segments]
        segments["index"] = segments.groupby(seg_groups).cumcount() if seg_groups else np.arange(len(segments))
    samples = pd.concat(sample_frames, ignore_index=True) if sample_frames else frame.copy()

    result = IVTResult(
        segments=segments,
        samples=samples,
        px_per_deg=px_per_deg,
        config=config.to_dict(),
    )
    if "event" in samples and samples["event"].notna().any():
        result.agreement = compare_with_export_events(samples)
    return result


def assign_events(segments: pd.DataFrame, samples: pd.DataFrame) -> pd.DataFrame:
    """Attach a segment id to every sample (``segment_id`` column)."""
    out = samples.copy()
    out["segment_id"] = -1
    if segments.empty:
        return out
    group_cols = [c for c in ("subject", "trial") if c in segments and c in out]
    segs = segments.reset_index(drop=True)
    if group_cols:
        for key, group in segs.groupby(group_cols, dropna=False):
            key = key if isinstance(key, tuple) else (key,)
            mask = pd.Series(True, index=out.index)
            for col, val in zip(group_cols, key):
                mask &= out[col] == val
            idx = out.index[mask]
            for sid, row in group.iterrows():
                inside = (out.loc[idx, "time_ms"] >= row["start_ms"] - 1e-6) & (
                    out.loc[idx, "time_ms"] <= row["end_ms"] + 1e-6
                )
                out.loc[idx[inside], "segment_id"] = sid
    else:  # pragma: no cover
        for sid, row in segs.iterrows():
            inside = (out["time_ms"] >= row["start_ms"] - 1e-6) & (out["time_ms"] <= row["end_ms"] + 1e-6)
            out.loc[inside, "segment_id"] = sid
    return out


def compare_with_export_events(samples: pd.DataFrame) -> dict[str, Any]:
    """Sample-level agreement with an event column from the export.

    Offered as a sanity check when the export carries its own event labels
    (for example a pre-computed ``Eye movement type`` column).  GAZERA never
    depends on it: the classification above stands on its own.
    """
    if "event" not in samples:
        return {}
    truth = samples["event"].astype(str).str.strip().str.lower()
    pred = samples["ivt_label"].astype(str)
    mask = truth.isin(["fixation", "saccade", "eyes not found", "unclassified"]) & pred.isin(
        ["fixation", "saccade"]
    )
    truth_norm = truth.where(truth.isin(["fixation", "saccade"]))
    subset = mask & truth_norm.notna()
    if not subset.any():
        return {"comparable_samples": 0}
    agree = float((truth_norm[subset] == pred[subset]).mean())
    by_type = {}
    for kind in ("fixation", "saccade"):
        m = subset & (truth_norm == kind)
        if m.any():
            by_type[kind] = round(float((pred[m] == kind).mean()), 4)
    return {"comparable_samples": int(subset.sum()), "agreement": round(agree, 4), "by_type": by_type}
