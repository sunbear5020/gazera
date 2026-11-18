"""Command line interface: ``gazera analyze …`` / ``gazera demo …``.

Examples
--------
    python -m gazera_core demo --out demo.tsv
    python -m gazera_core analyze demo.tsv --out results
    python -m gazera_core analyze export.tsv --gaze-x "Gaze point X" --gaze-y "Gaze point Y"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .exporter import export_csv, export_excel, export_json
from .pipeline import AnalysisConfig, analyze_file
from .synthetic import SyntheticConfig, write_dataset
from .ivt import IVTConfig
from . import __version__


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gazera",
        description="GAZERA Core — clean-room eye-tracking data analytics (MIT).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="write a 100%% synthetic demo dataset")
    demo.add_argument("--out", type=Path, default=Path("gazera_demo.tsv"))
    demo.add_argument("--subjects", type=int, default=10)
    demo.add_argument("--duration", type=float, default=10.0)
    demo.add_argument("--seed", type=int, default=20260520)
    demo.add_argument("--no-eye-movement-type", action="store_true")

    analyze = sub.add_parser("analyze", help="analyze a CSV/TSV gaze export")
    analyze.add_argument("file", type=Path)
    analyze.add_argument("--out", type=Path, default=Path("gazera_results"))
    analyze.add_argument("--threshold", type=float, default=30.0, help="I-VT velocity threshold in deg/s")
    analyze.add_argument("--min-fixation", type=float, default=60.0, help="minimum fixation duration (ms)")
    analyze.add_argument("--delimiter", default=None, help="force a delimiter (default: auto-detect)")
    analyze.add_argument("--screen", nargs=2, type=float, metavar=("W", "H"), default=None)
    analyze.add_argument("--gaze-x", default=None, help="explicit column for gaze x")
    analyze.add_argument("--gaze-y", default=None, help="explicit column for gaze y")
    analyze.add_argument("--timestamp", default=None, help="explicit column for the timestamp")
    analyze.add_argument("--subject", default=None)
    analyze.add_argument("--condition", default=None)
    analyze.add_argument("--decimal-comma", action="store_true")
    analyze.add_argument("--json", action="store_true", help="also print the JSON report to stdout")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "demo":
        path = write_dataset(
            args.out,
            SyntheticConfig(
                n_subjects=args.subjects,
                duration_s=args.duration,
                seed=args.seed,
                include_eye_movement_type=not args.no_eye_movement_type,
            ),
        )
        print(f"Wrote synthetic demo data: {path}")
        return 0

    if args.command == "analyze":
        mapping = {}
        if args.gaze_x:
            mapping["gaze_x"] = args.gaze_x
        if args.gaze_y:
            mapping["gaze_y"] = args.gaze_y
        if args.timestamp:
            mapping["timestamp"] = args.timestamp
        if args.subject:
            mapping["subject"] = args.subject
        if args.condition:
            mapping["condition"] = args.condition

        config = AnalysisConfig(
            ivt=IVTConfig(velocity_threshold_deg_s=args.threshold, min_fixation_ms=args.min_fixation),
            screen=tuple(args.screen) if args.screen else None,
            decimal_comma=args.decimal_comma,
        )
        result = analyze_file(
            args.file,
            mapping=mapping or None,
            config=config,
            delimiter=args.delimiter,
        )

        out = args.out
        out.mkdir(parents=True, exist_ok=True)
        workbook = export_excel(
            out / f"{args.file.stem}_gazera.xlsx",
            result.sheets(),
            metadata={
                "Source": str(args.file),
                "Version": __version__,
                "License": "MIT · FEI · HEL Lab, Gyeongsang National University",
            },
        )
        export_json(out / f"{args.file.stem}_gazera.json", result.to_dict())
        export_csv(out, result.sheets(), prefix=args.file.stem)

        summary = result.summary()
        print(f"Rows: {summary['rows']:,} · recordings: {summary['recordings']} · subjects: {len(summary['subjects'])}")
        print(
            "Fixations: {count} (mean {mean:.0f} ms) · Saccades: {s_count} (mean amplitude {amp:.2f}°)".format(
                count=summary["fixation"]["fixation_count"],
                mean=summary["fixation"]["fixation_mean_ms"],
                s_count=summary["saccade"]["saccade_count"],
                amp=summary["saccade"]["saccade_amplitude_mean_deg"],
            )
        )
        for warning in summary["warnings"]:
            print(f"  ! {warning}")
        print(f"Excel: {workbook}")
        if args.json:
            print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2, default=str))
        return 0

    parser.print_help()  # pragma: no cover
    return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
