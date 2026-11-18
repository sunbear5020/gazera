"""Data quality inspection and light cleaning.

The cleaner never silently deletes a researcher's data: every action it takes is
returned as a list of ``CleaningAction`` records so it can be reported next to
the results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from .loader import GazeData

__all__ = ["QualityReport", "CleaningAction", "quality_report", "clean_gaze", "sample_rate_hz"]


@dataclass
class CleaningAction:
    action: str
    detail: str
    rows_affected: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {"action": self.action, "detail": self.detail, "rows_affected": self.rows_affected}


@dataclass
class QualityReport:
    """Per-recording and pooled data quality summary."""

    rows: int
    recordings: int
    duration_s: float
    sample_rate_hz: float | None
    sample_rate_sd: float | None
    valid_ratio: float
    missing_gaze_ratio: float
    missing_pupil_ratio: float
    missing_cell_ratio: float
    non_monotonic: int
    duplicate_timestamps: int
    out_of_screen_ratio: float
    subjects: list[str] = field(default_factory=list)
    conditions: list[str] = field(default_factory=list)
    per_recording: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    screen: tuple[float | None, float | None] = (None, None)
    timestamp_unit: str | None = None
    mapping: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "recordings": self.recordings,
            "duration_s": round(self.duration_s, 3),
            "sample_rate_hz": None if self.sample_rate_hz is None else round(self.sample_rate_hz, 2),
            "sample_rate_sd": None if self.sample_rate_sd is None else round(self.sample_rate_sd, 3),
            "valid_ratio": round(self.valid_ratio, 4),
            "missing_gaze_ratio": round(self.missing_gaze_ratio, 4),
            "missing_pupil_ratio": round(self.missing_pupil_ratio, 4),
            "missing_cell_ratio": round(self.missing_cell_ratio, 4),
            "non_monotonic_timestamps": self.non_monotonic,
            "duplicate_timestamps": self.duplicate_timestamps,
            "out_of_screen_ratio": round(self.out_of_screen_ratio, 4),
            "subjects": self.subjects,
            "conditions": self.conditions,
            "per_recording": self.per_recording,
            "warnings": self.warnings,
            "screen": {"width": self.screen[0], "height": self.screen[1]},
            "timestamp_unit": self.timestamp_unit,
            "mapping": self.mapping,
        }

    def summary_markdown(self) -> str:
        lines = [
            "# GAZERA — Data quality report",
            "",
            f"- Rows: {self.rows:,}",
            f"- Recordings: {self.recordings}",
            f"- Total duration: {self.duration_s:.1f} s",
            f"- Estimated sampling rate: "
            + (f"{self.sample_rate_hz:.1f} Hz" if self.sample_rate_hz else "unknown"),
            f"- Valid gaze samples: {self.valid_ratio * 100:.1f}%",
            f"- Missing gaze ratio: {self.missing_gaze_ratio * 100:.1f}%",
            f"- Missing pupil ratio: {self.missing_pupil_ratio * 100:.1f}%",
            f"- Timestamps out of order: {self.non_monotonic}",
            f"- Duplicate timestamps: {self.duplicate_timestamps}",
        ]
        if self.warnings:
            lines.append("")
            lines.append("## Warnings")
            lines.extend(f"- {w}" for w in self.warnings)
        return "\n".join(lines)


def _recording_masks(frame: pd.DataFrame) -> list[tuple[str, pd.Series]]:
    if "subject" not in frame and "trial" not in frame:
        return [("all", pd.Series(True, index=frame.index))]
    key = pd.Series("", index=frame.index)
    for col in ("subject", "trial"):
        if col in frame:
            key = key + frame[col].fillna("-").astype(str) + "|"
    masks = []
    for value in key.drop_duplicates():
        masks.append((str(value).strip("|"), key == value))
    return masks


def sample_rate_hz(frame: pd.DataFrame, *, drop_non_positive: bool = True) -> tuple[float | None, float | None]:
    """Estimate sampling rate from the median inter-sample interval."""
    if "time_ms" not in frame:
        return None, None
    dt = frame["time_ms"].astype(float).diff().dropna()
    if drop_non_positive:
        dt = dt[dt > 0]
    if dt.empty:
        return None, None
    median = float(dt.median())
    if median <= 0:
        return None, None
    return 1000.0 / median, float(dt.std(ddof=0))


def quality_report(
    data: GazeData | pd.DataFrame,
    *,
    screen: tuple[float, float] | None = None,
) -> QualityReport:
    """Inspect a canonical frame and report quality without changing it."""
    frame = data.frame if isinstance(data, GazeData) else data
    meta = data.meta if isinstance(data, GazeData) else {}
    width, height = screen or frame.attrs.get("screen", (np.nan, np.nan))
    if (width is None or np.isnan(width)) and "media_width" in frame:
        width = float(frame["media_width"].dropna().median())
    if (height is None or np.isnan(height)) and "media_height" in frame:
        height = float(frame["media_height"].dropna().median())

    n = int(len(frame))
    valid = frame.get("valid", pd.Series(True, index=frame.index)).fillna(False).astype(bool)
    valid_ratio = float(valid.mean()) if n else 0.0
    missing_gaze = frame[["gaze_x", "gaze_y"]].isna().any(axis=1) if {"gaze_x", "gaze_y"} <= set(frame) else pd.Series(False, index=frame.index)
    missing_pupil = frame["pupil"].isna() if "pupil" in frame else pd.Series(True, index=frame.index)

    non_monotonic = duplicate = 0
    rate, rate_sd = sample_rate_hz(frame)
    per_recording: list[dict[str, Any]] = []
    for name, mask in _recording_masks(frame):
        sub = frame.loc[mask]
        dt = sub["time_ms"].diff().dropna() if "time_ms" in sub else pd.Series(dtype=float)
        nm = int((dt < 0).sum())
        dup = int(sub["time_ms"].duplicated().sum()) if "time_ms" in sub else 0
        non_monotonic += nm
        duplicate += dup
        sub_rate, _ = sample_rate_hz(sub)
        sub_valid = sub.get("valid", pd.Series(True, index=sub.index)).fillna(False)
        per_recording.append(
            {
                "recording": name,
                "rows": int(len(sub)),
                "duration_s": round(float(sub["time_ms"].max() - sub["time_ms"].min()) / 1000.0, 3)
                if "time_ms" in sub and len(sub)
                else 0.0,
                "sample_rate_hz": None if sub_rate is None else round(sub_rate, 2),
                "valid_ratio": round(float(sub_valid.mean()), 4) if len(sub) else 0.0,
                "non_monotonic": nm,
                "duplicates": dup,
            }
        )

    out_of_screen = 0.0
    if {"gaze_x", "gaze_y"} <= set(frame) and width and height and not (np.isnan(width) or np.isnan(height)):
        outside = (
            (frame["gaze_x"] < 0)
            | (frame["gaze_y"] < 0)
            | (frame["gaze_x"] > width)
            | (frame["gaze_y"] > height)
        )
        out_of_screen = float(outside.mean())

    warnings: list[str] = []
    if valid_ratio < 0.7:
        warnings.append(f"Valid gaze ratio is low ({valid_ratio * 100:.1f}%). Check calibration and lighting.")
    if rate is None:
        warnings.append("No usable timestamp column; a nominal 60 Hz time base was assumed.")
    elif rate < 30:
        warnings.append(f"Sampling rate looks low ({rate:.1f} Hz); short fixations may merge.")
    if non_monotonic:
        warnings.append(f"{non_monotonic} timestamps go backwards; recordings were rebased per subject/trial.")
    if missing_pupil.mean() > 0.3 and "pupil" in frame:
        warnings.append(f"Pupil diameter missing in {missing_pupil.mean() * 100:.0f}% of rows.")
    if out_of_screen > 0.02:
        warnings.append(f"{out_of_screen * 100:.1f}% of gaze coordinates fall outside the screen area.")

    duration = 0.0
    if "time_ms" in frame and len(frame):
        duration = float(
            sum(
                (sub["time_ms"].max() - sub["time_ms"].min()) / 1000.0
                for _, sub in frame.groupby(
                    [c for c in ("subject", "trial") if c in frame] or None, dropna=False
                )
                if len(sub)
            )
        )

    missing_cells = float(frame.drop(columns=[c for c in ("valid",) if c in frame]).isna().to_numpy().mean()) if n else 0.0

    return QualityReport(
        rows=n,
        recordings=len(per_recording),
        duration_s=duration,
        sample_rate_hz=rate,
        sample_rate_sd=rate_sd,
        valid_ratio=valid_ratio,
        missing_gaze_ratio=float(missing_gaze.mean()) if n else 0.0,
        missing_pupil_ratio=float(missing_pupil.mean()) if n else 0.0,
        missing_cell_ratio=missing_cells,
        non_monotonic=non_monotonic,
        duplicate_timestamps=duplicate,
        out_of_screen_ratio=out_of_screen,
        subjects=(data.subjects() if isinstance(data, GazeData) else []),
        conditions=(data.conditions() if isinstance(data, GazeData) else []),
        per_recording=per_recording,
        warnings=warnings,
        screen=(
            None if width is None or (isinstance(width, float) and np.isnan(width)) else float(width),
            None if height is None or (isinstance(height, float) and np.isnan(height)) else float(height),
        ),
        timestamp_unit=meta.get("timestamp_unit"),
        mapping=dict(data.mapping) if isinstance(data, GazeData) else {},
    )


def clean_gaze(
    frame: pd.DataFrame,
    *,
    sort_timestamps: bool = True,
    interpolate_gap_ms: float = 40.0,
    drop_invalid: bool = False,
    clip_to_screen: bool = True,
    screen: tuple[float, float] | None = None,
) -> tuple[pd.DataFrame, list[CleaningAction]]:
    """Return a cleaned copy of a canonical frame plus the actions taken."""
    out = frame.copy()
    actions: list[CleaningAction] = []
    group_cols = [c for c in ("subject", "trial") if c in out]

    if sort_timestamps and "time_ms" in out:
        before = out["time_ms"].copy()
        if group_cols:
            out = out.sort_values(group_cols + ["time_ms"], kind="stable")
        else:
            out = out.sort_values("time_ms", kind="stable")
        out = out.reset_index(drop=True)
        moved = int((before.to_numpy() != out["time_ms"].to_numpy()).sum()) if len(out) == len(before) else 0
        if moved:
            actions.append(CleaningAction("sort", "Rows sorted by subject/trial and timestamp.", moved))

    if "valid" not in out:
        out["valid"] = out[["gaze_x", "gaze_y"]].notna().all(axis=1) if {"gaze_x", "gaze_y"} <= set(out) else True

    if clip_to_screen and {"gaze_x", "gaze_y"} <= set(out):
        width, height = screen or out.attrs.get("screen", (np.nan, np.nan))
        if width and height and not (np.isnan(width) or np.isnan(height)):
            outside = (
                (out["gaze_x"] < 0)
                | (out["gaze_y"] < 0)
                | (out["gaze_x"] > width)
                | (out["gaze_y"] > height)
            )
            outside &= out["gaze_x"].notna()
            if outside.any():
                out.loc[outside, ["gaze_x", "gaze_y"]] = np.nan
                out.loc[outside, "valid"] = False
                actions.append(
                    CleaningAction(
                        "clip",
                        f"Gaze coordinates outside 0-{width:.0f} x 0-{height:.0f} px set to missing.",
                        int(outside.sum()),
                    )
                )

    if interpolate_gap_ms and interpolate_gap_ms > 0 and {"gaze_x", "gaze_y"} <= set(out):
        filled = 0
        groups = out.groupby(group_cols, dropna=False) if group_cols else [(None, out)]
        for _, sub in groups:
            idx = sub.index
            gaps = sub["time_ms"].diff().to_numpy() if "time_ms" in sub else np.array([])
            missing = sub["gaze_x"].isna().to_numpy()
            if missing.any() and len(gaps) == len(sub):
                run_start = None
                for pos, flag in enumerate(missing):
                    if flag and run_start is None:
                        run_start = pos
                    if (not flag or pos == len(missing) - 1) and run_start is not None:
                        run_end = pos if not flag else pos + 1
                        length = run_end - run_start
                        span = float(np.nansum(gaps[run_start : run_end + 1]))
                        if 0 < length <= 3 and span <= interpolate_gap_ms:
                            for col in ("gaze_x", "gaze_y"):
                                out.loc[idx[run_start:run_end], col] = (
                                    sub[col].interpolate(limit_direction="both").to_numpy()[run_start:run_end]
                                )
                            out.loc[idx[run_start:run_end], "valid"] = True
                            filled += length
                        run_start = None
        if filled:
            actions.append(
                CleaningAction(
                    "interpolate",
                    f"Short gaps (<= {interpolate_gap_ms:.0f} ms, <= 3 samples) linearly interpolated.",
                    filled,
                )
            )

    if drop_invalid:
        before = len(out)
        out = out[out["valid"].fillna(False)].reset_index(drop=True)
        dropped = before - len(out)
        if dropped:
            actions.append(CleaningAction("drop-invalid", "Invalid samples removed.", dropped))

    return out, actions
