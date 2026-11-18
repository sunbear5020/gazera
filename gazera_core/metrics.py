"""Descriptive eye-tracking metrics: fixation, saccade and AOI.

Every metric is defined from the canonical frame and the GAZERA I-VT segments,
so results do not depend on which software produced the export.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "AOI",
    "DEFAULT_DVS",
    "aoi_metrics",
    "condition_summary",
    "fixation_summary",
    "normalise_aois",
    "saccade_summary",
    "subject_metrics",
]

#: Dependent variables reported per recording and compared across conditions.
DEFAULT_DVS: tuple[str, ...] = (
    "fixation_count",
    "fixation_mean_ms",
    "fixation_median_ms",
    "fixation_rate_per_s",
    "saccade_count",
    "saccade_amplitude_mean_deg",
    "saccade_peak_velocity_mean_deg_s",
)


@dataclass
class AOI:
    """A rectangular area of interest in normalised (0-1) media coordinates."""

    name: str
    x: float
    y: float
    w: float
    h: float
    color: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "x": self.x, "y": self.y, "w": self.w, "h": self.h, **(
            {"color": self.color} if self.color else {}
        )}

    def contains(self, xs: np.ndarray, ys: np.ndarray, width: float, height: float) -> np.ndarray:
        nx = xs / width
        ny = ys / height
        return (nx >= self.x) & (nx <= self.x + self.w) & (ny >= self.y) & (ny <= self.y + self.h)


def default_aois(bands: int = 3) -> list[AOI]:
    """Even horizontal bands across the stimulus (a neutral starting point)."""
    labels = ["상단", "중앙", "하단"] if bands == 3 else [f"영역 {i + 1}" for i in range(bands)]
    height = 1.0 / bands
    return [AOI(labels[i], 0.0, round(i * height, 4), 1.0, round(height, 4)) for i in range(bands)]


def normalise_aois(
    aois: Sequence[AOI | Mapping[str, Any]],
    screen: tuple[float, float] | None = None,
) -> list[AOI]:
    """Accept normalised or pixel rectangles and return normalised AOIs."""
    width, height = (screen or (1.0, 1.0))
    out: list[AOI] = []
    for idx, item in enumerate(aois):
        if isinstance(item, AOI):
            out.append(item)
            continue
        name = str(item.get("name") or f"AOI {idx + 1}")
        x, y = float(item.get("x", 0)), float(item.get("y", 0))
        w, h = float(item.get("w", 0)), float(item.get("h", 0))
        if max(x, y, w, h) > 1.0 and width and height and not math.isnan(width):
            x, w = x / width, w / width
            y, h = y / height, h / height
        out.append(AOI(name, x, y, w, h, item.get("color")))
    return out


def _recording_key(frame: pd.DataFrame) -> pd.Series | None:
    cols = [c for c in ("subject", "trial") if c in frame and frame[c].notna().any()]
    if not cols:
        return None
    return frame[cols].astype(str).agg(" | ".join, axis=1)


def fixation_summary(segments: pd.DataFrame) -> dict[str, Any]:
    """Fixation count and duration statistics (per pooled set of segments)."""
    if segments is None or segments.empty:
        return {
            "fixation_count": 0,
            "fixation_total_ms": 0.0,
            "fixation_mean_ms": 0.0,
            "fixation_median_ms": 0.0,
            "fixation_min_ms": 0.0,
            "fixation_max_ms": 0.0,
            "fixation_rate_per_s": 0.0,
        }
    fix = segments[segments["type"] == "fixation"]
    total = float(fix["duration_ms"].sum()) if not fix.empty else 0.0
    span = float(segments["duration_ms"].sum()) / 1000.0
    return {
        "fixation_count": int(len(fix)),
        "fixation_total_ms": total,
        "fixation_mean_ms": float(fix["duration_ms"].mean()) if not fix.empty else 0.0,
        "fixation_median_ms": float(fix["duration_ms"].median()) if not fix.empty else 0.0,
        "fixation_min_ms": float(fix["duration_ms"].min()) if not fix.empty else 0.0,
        "fixation_max_ms": float(fix["duration_ms"].max()) if not fix.empty else 0.0,
        "fixation_rate_per_s": float(len(fix) / span) if span > 0 else 0.0,
    }


def saccade_summary(segments: pd.DataFrame) -> dict[str, Any]:
    """Saccade frequency, amplitude and velocity statistics."""
    if segments is None or segments.empty:
        return {
            "saccade_count": 0,
            "saccade_rate_per_s": 0.0,
            "saccade_amplitude_mean_deg": 0.0,
            "saccade_amplitude_median_deg": 0.0,
            "saccade_amplitude_max_deg": 0.0,
            "saccade_mean_velocity_deg_s": 0.0,
            "saccade_peak_velocity_mean_deg_s": 0.0,
        }
    sac = segments[segments["type"] == "saccade"]
    span = float(segments["duration_ms"].sum()) / 1000.0
    return {
        "saccade_count": int(len(sac)),
        "saccade_rate_per_s": float(len(sac) / span) if span > 0 else 0.0,
        "saccade_amplitude_mean_deg": float(sac["amplitude_deg"].mean()) if not sac.empty else 0.0,
        "saccade_amplitude_median_deg": float(sac["amplitude_deg"].median()) if not sac.empty else 0.0,
        "saccade_amplitude_max_deg": float(sac["amplitude_deg"].max()) if not sac.empty else 0.0,
        "saccade_mean_velocity_deg_s": float(sac["mean_velocity_deg_s"].mean()) if not sac.empty else 0.0,
        "saccade_peak_velocity_mean_deg_s": float(sac["peak_velocity_deg_s"].mean()) if not sac.empty else 0.0,
    }


def aoi_metrics(
    frame: pd.DataFrame,
    aois: Sequence[AOI | Mapping[str, Any]],
    *,
    segments: pd.DataFrame | None = None,
    screen: tuple[float, float] | None = None,
    min_visit_ms: float = 40.0,
) -> pd.DataFrame:
    """Per recording x AOI metrics: dwell, dwell ratio, visits, TTFF.

    Dwell and visits are computed on fixation segments when they are supplied
    (the standard AOI definition), and on valid samples otherwise.
    """
    width, height = screen or frame.attrs.get("screen", (np.nan, np.nan))
    if (width is None or np.isnan(width)) and "media_width" in frame:
        width = float(frame["media_width"].dropna().median())
    if (height is None or np.isnan(height)) and "media_height" in frame:
        height = float(frame["media_height"].dropna().median())
    if not width or not height or np.isnan(width) or np.isnan(height):
        raise ValueError("Screen size is required for AOI metrics; pass screen=(width, height).")
    aoi_list = normalise_aois(aois, (float(width), float(height)))

    group_cols = [c for c in ("subject", "trial") if c in frame and frame[c].notna().any()]
    rows: list[dict[str, Any]] = []

    for key, sub in frame.groupby(group_cols, dropna=False, sort=False) if group_cols else [((), frame)]:
        if not isinstance(key, tuple):
            key = (key,)
        meta = {col: (None if pd.isna(v) else v) for col, v in zip(group_cols, key)}
        cond = sub["condition"].iloc[0] if "condition" in sub and len(sub) else None
        valid = sub["valid"].fillna(False).astype(bool) if "valid" in sub else pd.Series(True, index=sub.index)
        base = float(sub["time_ms"].min()) if len(sub) else 0.0

        seg_sub = None
        times = sub["time_ms"].to_numpy(float) if len(sub) else np.array([])
        dts = np.diff(times, append=times[-1] if len(times) else 0.0) if len(times) else np.array([])
        if len(dts):
            dts = np.clip(dts, 0, float(np.nanmedian(dts)) * 2)
        ok = valid & sub["gaze_x"].notna() & sub["gaze_y"].notna() if len(sub) else pd.Series(dtype=bool)
        total_valid_ms = float(np.nansum(dts[ok.to_numpy()])) if len(dts) else 0.0
        if segments is not None and not segments.empty:
            seg_sub = segments
            for col, val in meta.items():
                seg_sub = seg_sub[seg_sub[col] == val]
            seg_sub = seg_sub[seg_sub["type"] == "fixation"]
            total_fix = float(seg_sub["duration_ms"].sum())
        else:
            total_fix = total_valid_ms

        for aoi in aoi_list:
            if seg_sub is not None and not seg_sub.empty:
                inside = aoi.contains(
                    seg_sub["center_x"].to_numpy(float), seg_sub["center_y"].to_numpy(float), float(width), float(height)
                )
                dwell = float(seg_sub.loc[inside, "duration_ms"].sum())
                visits = int(inside.sum())
                first = float(seg_sub.loc[inside, "start_ms"].min()) if inside.any() else math.nan
                durations = seg_sub.loc[inside, "duration_ms"]
                mean_fix = float(durations.mean()) if len(durations) else 0.0
            else:
                inside_mask = pd.Series(False, index=sub.index)
                if ok.any():
                    inside_mask.loc[ok] = aoi.contains(
                        sub.loc[ok, "gaze_x"].to_numpy(float),
                        sub.loc[ok, "gaze_y"].to_numpy(float),
                        float(width),
                        float(height),
                    )
                flag = inside_mask.to_numpy()
                dwell = float(np.nansum(dts[flag])) if len(dts) else 0.0
                visits = 0
                run = 0.0
                run_len = 0
                best = 0.0
                for i, is_in in enumerate(flag):
                    if is_in:
                        run_len += 1
                        run += float(dts[i]) if i < len(dts) else 0.0
                    else:
                        if run_len and run >= min_visit_ms:
                            visits += 1
                            best = max(best, run)
                        run, run_len = 0.0, 0
                if run_len and run >= min_visit_ms:
                    visits += 1
                    best = max(best, run)
                first = float(times[int(np.argmax(flag))]) if flag.any() else math.nan
                mean_fix = best
            rows.append(
                {
                    **meta,
                    "condition": cond,
                    "aoi": aoi.name,
                    "aoi_x": aoi.x,
                    "aoi_y": aoi.y,
                    "aoi_w": aoi.w,
                    "aoi_h": aoi.h,
                    "dwell_ms": dwell,
                    "dwell_ratio": (dwell / total_fix) if total_fix > 0 else 0.0,
                    "visits": visits,
                    "ttff_ms": (first - base) if first == first else math.nan,
                    "mean_dwell_ms": mean_fix,
                }
            )
    return pd.DataFrame(rows)


def subject_metrics(
    frame: pd.DataFrame,
    segments: pd.DataFrame,
    aois: Sequence[AOI | Mapping[str, Any]] | None = None,
    *,
    screen: tuple[float, float] | None = None,
) -> pd.DataFrame:
    """One row per (subject, trial) recording with every headline metric."""
    group_cols = [c for c in ("subject", "trial") if c in frame and frame[c].notna().any()]
    width, height = screen or frame.attrs.get("screen", (np.nan, np.nan))
    rows: list[dict[str, Any]] = []

    for key, sub in frame.groupby(group_cols, dropna=False, sort=False) if group_cols else [((), frame)]:
        if not isinstance(key, tuple):
            key = (key,)
        meta = {col: (None if pd.isna(v) else v) for col, v in zip(group_cols, key)}
        seg = segments
        if seg is not None and not seg.empty:
            for col, val in meta.items():
                seg = seg[seg[col] == val]
        fix = fixation_summary(seg)
        sac = saccade_summary(seg)
        duration_s = float(sub["time_ms"].max() - sub["time_ms"].min()) / 1000.0 if len(sub) > 1 else 0.0
        valid = float(sub["valid"].fillna(False).mean()) if "valid" in sub and len(sub) else 0.0
        row = {
            **meta,
            "condition": sub["condition"].iloc[0] if "condition" in sub and len(sub) else None,
            "n_samples": int(len(sub)),
            "duration_s": duration_s,
            "valid_ratio": valid,
            **fix,
            **sac,
        }
        if aois:
            tab = aoi_metrics(sub, aois, segments=seg, screen=(float(width), float(height))) if len(sub) else pd.DataFrame()
            for _, aoi_row in tab.iterrows():
                key_name = str(aoi_row["aoi"]).replace(" ", "_")
                row[f"aoi_{key_name}_dwell_ms"] = float(aoi_row["dwell_ms"])
                row[f"aoi_{key_name}_dwell_ratio"] = float(aoi_row["dwell_ratio"])
                row[f"aoi_{key_name}_visits"] = int(aoi_row["visits"])
                row[f"aoi_{key_name}_ttff_ms"] = float(aoi_row["ttff_ms"]) if aoi_row["ttff_ms"] == aoi_row["ttff_ms"] else None
        rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty and "subject" not in out and "trial" not in out:
        out.insert(0, "recording", "all")
    return out


def condition_summary(subject_df: pd.DataFrame, dvs: Iterable[str] | None = None) -> pd.DataFrame:
    """Long-format descriptive statistics per condition for each DV."""
    if subject_df is None or subject_df.empty:
        return pd.DataFrame(columns=["metric", "condition", "n", "mean", "sd", "median", "iqr"])
    dvs = list(dvs or DEFAULT_DVS)
    dvs += [c for c in subject_df.columns if c.startswith("aoi_") and c.endswith("_dwell_ratio") and c not in dvs]
    dvs = [d for d in dvs if d in subject_df.columns]
    group = "condition" if "condition" in subject_df and subject_df["condition"].notna().any() else None
    rows = []
    for dv in dvs:
        series_all = pd.to_numeric(subject_df[dv], errors="coerce")
        if group is None:
            values = series_all.dropna()
            rows.append(
                {
                    "metric": dv,
                    "condition": "all",
                    "n": int(values.count()),
                    "mean": float(values.mean()) if len(values) else math.nan,
                    "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "median": float(values.median()) if len(values) else math.nan,
                    "iqr": float(values.quantile(0.75) - values.quantile(0.25)) if len(values) else math.nan,
                }
            )
            continue
        for cond, sub in subject_df.groupby(group, sort=True):
            values = pd.to_numeric(sub[dv], errors="coerce").dropna()
            rows.append(
                {
                    "metric": dv,
                    "condition": str(cond),
                    "n": int(values.count()),
                    "mean": float(values.mean()) if len(values) else math.nan,
                    "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "median": float(values.median()) if len(values) else math.nan,
                    "iqr": float(values.quantile(0.75) - values.quantile(0.25)) if len(values) else math.nan,
                }
            )
    return pd.DataFrame(rows)
