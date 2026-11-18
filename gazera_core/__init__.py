"""GAZERA Core — clean-room, vendor-neutral eye-tracking analytics.

MIT License · Copyright (c) 2026 FEI · HEL Lab, Gyeongsang National University.
GAZERA is independent and is not affiliated with, endorsed by, or sponsored by
Tobii AB.  All product names and trademarks belong to their respective owners.

The package reads ordinary delimited text (CSV/TSV) exported by any
eye-tracking software.  No vendor SDK, DLL or binary format is used.
"""

from __future__ import annotations

from .cleaner import CleaningAction, QualityReport, clean_gaze, quality_report, sample_rate_hz
from .exporter import export_csv, export_excel, export_json, to_jsonable
from .ivt import IVTConfig, IVTResult, classify_ivt, compare_with_export_events, estimate_px_per_deg
from .loader import (
    CANONICAL_FIELDS,
    GazeData,
    detect_mapping,
    find_aoi_columns,
    load_gaze,
    load_gaze_bytes,
)
from .metrics import (
    AOI,
    DEFAULT_DVS,
    aoi_metrics,
    condition_summary,
    default_aois,
    fixation_summary,
    normalise_aois,
    saccade_summary,
    subject_metrics,
)
from .pipeline import AnalysisConfig, AnalysisResult, analyze, analyze_file
from .stats import (
    analyze_comparisons,
    compare_many_groups,
    compare_paired,
    compare_two_groups,
    describe_groups,
    holm_correction,
    normality_table,
)
from .synthetic import CONDITIONS, SyntheticConfig, generate_dataset, generate_rows, write_dataset

__version__ = "0.1.0"
__license__ = "MIT"

from . import cli  # noqa: E402  (CLI entry: python -m gazera_core)

__all__ = [
    "AOI",
    "AnalysisConfig",
    "AnalysisResult",
    "CANONICAL_FIELDS",
    "CONDITIONS",
    "CleaningAction",
    "DEFAULT_DVS",
    "GazeData",
    "IVTConfig",
    "IVTResult",
    "QualityReport",
    "SyntheticConfig",
    "analyze",
    "analyze_comparisons",
    "analyze_file",
    "aoi_metrics",
    "classify_ivt",
    "clean_gaze",
    "compare_many_groups",
    "compare_paired",
    "compare_two_groups",
    "compare_with_export_events",
    "condition_summary",
    "default_aois",
    "describe_groups",
    "detect_mapping",
    "estimate_px_per_deg",
    "export_csv",
    "export_excel",
    "export_json",
    "find_aoi_columns",
    "fixation_summary",
    "generate_dataset",
    "generate_rows",
    "holm_correction",
    "load_gaze",
    "load_gaze_bytes",
    "normalise_aois",
    "normality_table",
    "quality_report",
    "saccade_summary",
    "sample_rate_hz",
    "subject_metrics",
    "to_jsonable",
    "write_dataset",
    "__version__",
]
