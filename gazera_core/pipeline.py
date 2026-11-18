"""End-to-end GAZERA Core analysis pipeline.

``analyze()`` is the single entry point used by the CLI, the local API and the
tests: load -> quality -> I-VT -> metrics -> statistics -> exportable tables.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .cleaner import QualityReport, clean_gaze, quality_report
from .ivt import IVTConfig, IVTResult, classify_ivt
from .loader import GazeData, load_gaze
from .metrics import (
    AOI,
    DEFAULT_DVS,
    aoi_metrics,
    condition_summary,
    default_aois,
    fixation_summary,
    saccade_summary,
    subject_metrics,
)
from .stats import analyze_comparisons

__all__ = ["AnalysisConfig", "AnalysisResult", "analyze", "analyze_file"]


@dataclass
class AnalysisConfig:
    """Everything a researcher may want to change about a run."""

    ivt: IVTConfig = field(default_factory=IVTConfig)
    aois: Sequence[AOI | Mapping[str, Any]] | None = None
    dvs: Sequence[str] | None = None
    group: str = "condition"
    paired: str | None = "condition"
    clean: bool = True
    interpolate_gap_ms: float = 40.0
    screen: tuple[float, float] | None = None
    decimal_comma: bool = False
    zero_is_invalid: bool | None = None
    include_segments: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "ivt": self.ivt.to_dict(),
            "aois": [a.as_dict() if isinstance(a, AOI) else dict(a) for a in self.aois] if self.aois else None,
            "dvs": list(self.dvs) if self.dvs else None,
            "group": self.group,
            "paired": self.paired,
            "clean": self.clean,
            "interpolate_gap_ms": self.interpolate_gap_ms,
            "screen": list(self.screen) if self.screen else None,
        }


@dataclass
class AnalysisResult:
    data: GazeData
    config: AnalysisConfig
    quality: QualityReport
    ivt: IVTResult
    subject_table: pd.DataFrame
    aoi_table: pd.DataFrame
    condition_table: pd.DataFrame
    comparisons: pd.DataFrame
    fixation_by_condition: pd.DataFrame
    saccade_by_condition: pd.DataFrame
    cleaning_actions: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    # -- convenience -------------------------------------------------------
    @property
    def pooled_fixation(self) -> dict[str, Any]:
        return fixation_summary(self.ivt.segments)

    @property
    def pooled_saccade(self) -> dict[str, Any]:
        return saccade_summary(self.ivt.segments)

    def summary(self) -> dict[str, Any]:
        return {
            "rows": int(len(self.data.frame)),
            "recordings": int(len(self.subject_table)),
            "subjects": self.data.subjects(),
            "conditions": self.data.conditions(),
            "px_per_deg": round(self.ivt.px_per_deg, 3),
            "fixation": {k: _round(v) for k, v in self.pooled_fixation.items()},
            "saccade": {k: _round(v) for k, v in self.pooled_saccade.items()},
            "quality": self.quality.to_dict(),
            "warnings": self.warnings,
            "mapping": self.data.mapping,
        }

    def to_dict(self, *, include_segments: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": "gazera.result.v1",
            "summary": self.summary(),
            "config": self.config.to_dict(),
            "quality": self.quality.to_dict(),
            "ivt": {
                "px_per_deg": round(self.ivt.px_per_deg, 3),
                "config": self.ivt.config,
                "agreement_with_export": self.ivt.agreement,
                "fixation": _round_dict(self.pooled_fixation),
                "saccade": _round_dict(self.pooled_saccade),
            },
            "subjects": _records(self.subject_table),
            "aoi": _records(self.aoi_table),
            "conditions": _records(self.condition_table),
            "fixation_by_condition": _records(self.fixation_by_condition),
            "saccade_by_condition": _records(self.saccade_by_condition),
            "comparisons": _records(self.comparisons),
            "cleaning_actions": self.cleaning_actions,
            "notices": {
                "license": "MIT License · Copyright (c) 2026 FEI · HEL Lab, Gyeongsang National University",
                "affiliation": (
                    "GAZERA is an independent open-source project. It is not affiliated with, "
                    "endorsed by, or sponsored by Tobii AB."
                ),
            },
        }
        if include_segments or self.config.include_segments:
            payload["segments"] = _records(self.ivt.segments)
        return payload

    def sheets(self) -> dict[str, pd.DataFrame]:
        quality_row = pd.DataFrame(
            [{"metric": k, "value": v} for k, v in self.quality.to_dict().items() if not isinstance(v, (dict, list))]
        )
        pooled = pd.DataFrame(
            [
                {"measure": k, "value": _round(v)} for k, v in self.pooled_fixation.items()
            ]
            + [{"measure": k, "value": _round(v)} for k, v in self.pooled_saccade.items()]
        )
        return {
            "Summary": pooled,
            "Quality": quality_row,
            "Quality by recording": pd.DataFrame(self.quality.per_recording),
            "Participants": self.subject_table,
            "Conditions": self.condition_table,
            "Statistics": self.comparisons,
            "AOI": self.aoi_table,
            "Fixation by condition": self.fixation_by_condition,
            "Saccade by condition": self.saccade_by_condition,
        }


def _round(value: Any, digits: int = 4) -> Any:
    if isinstance(value, (float, np.floating)):
        return None if np.isnan(value) else round(float(value), digits)
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def _round_dict(values: Mapping[str, Any]) -> dict[str, Any]:
    return {k: _round(v) for k, v in values.items()}


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df is None or df.empty:
        return []
    clean = df.replace({np.nan: None})
    return clean.to_dict(orient="records")


def _per_condition_segments(segments: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows_fix, rows_sac = [], []
    if segments is None or segments.empty:
        return pd.DataFrame(), pd.DataFrame()
    groups = segments.groupby("condition", dropna=False, sort=True) if "condition" in segments else [("all", segments)]
    for name, sub in groups:
        rows_fix.append({"condition": None if pd.isna(name) else str(name), **fixation_summary(sub)})
        rows_sac.append({"condition": None if pd.isna(name) else str(name), **saccade_summary(sub)})
    return pd.DataFrame(rows_fix), pd.DataFrame(rows_sac)


def analyze(data: GazeData, config: AnalysisConfig | None = None) -> AnalysisResult:
    """Run the full core analysis on already-loaded data."""
    config = config or AnalysisConfig()
    frame = data.frame
    actions: list[dict[str, Any]] = []

    width, height = config.screen or frame.attrs.get("screen", (np.nan, np.nan))
    if (width is None or np.isnan(width)) and "media_width" in frame and frame["media_width"].notna().any():
        width = float(frame["media_width"].dropna().median())
    if (height is None or np.isnan(height)) and "media_height" in frame and frame["media_height"].notna().any():
        height = float(frame["media_height"].dropna().median())
    frame.attrs["screen"] = (width, height)

    if config.clean:
        frame, cleaning = clean_gaze(
            frame,
            interpolate_gap_ms=config.interpolate_gap_ms,
            screen=(width, height) if width and height else None,
        )
        actions = [a.to_dict() for a in cleaning]

    quality = quality_report(frame, screen=(width, height) if width and height else None)
    ivt = classify_ivt(frame, config.ivt)

    aois = list(config.aois) if config.aois else default_aois(3)
    aoi_tab = aoi_metrics(frame, aois, segments=ivt.segments, screen=(width, height))
    subjects = subject_metrics(frame, ivt.segments, aois, screen=(width, height))
    conditions = condition_summary(subjects, config.dvs)
    dvs = list(config.dvs) if config.dvs else list(DEFAULT_DVS)
    dvs += [c for c in subjects.columns if c.startswith("aoi_") and c.endswith("_dwell_ratio") and c not in dvs]
    comparisons = analyze_comparisons(subjects, dvs, group=config.group, paired=config.paired)
    fix_cond, sac_cond = _per_condition_segments(ivt.segments)

    warnings = list(quality.warnings)
    if config.ivt.px_per_deg is None and (not width or np.isnan(width)):
        warnings.append("Screen width unknown: default 1920 px / 530 mm / 650 mm assumed for the deg/s conversion.")
    if comparisons is not None and not comparisons.empty and "p" in comparisons:
        low = comparisons[(comparisons["n_a"] + comparisons["n_b"] < 6)] if {"n_a", "n_b"} <= set(comparisons) else None
        if low is not None and not low.empty:
            warnings.append("Some comparisons rest on fewer than 6 recordings per condition; treat p-values as exploratory.")

    return AnalysisResult(
        data=data,
        config=config,
        quality=quality,
        ivt=ivt,
        subject_table=subjects,
        aoi_table=aoi_tab,
        condition_table=conditions,
        comparisons=comparisons,
        fixation_by_condition=fix_cond,
        saccade_by_condition=sac_cond,
        cleaning_actions=actions,
        warnings=warnings,
    )


def analyze_file(
    path: str | Path,
    *,
    mapping: Mapping[str, str] | None = None,
    config: AnalysisConfig | None = None,
    delimiter: str | None = None,
) -> AnalysisResult:
    """Load a CSV/TSV gaze export and run the core analysis."""
    config = config or AnalysisConfig()
    data = load_gaze(
        path,
        mapping=mapping,
        delimiter=delimiter,
        decimal_comma=config.decimal_comma,
        zero_is_invalid=config.zero_is_invalid,
        screen=config.screen,
    )
    return analyze(data, config)
