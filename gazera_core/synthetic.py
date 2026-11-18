"""100 % synthetic eye-tracking data generator (no real recordings involved).

The generator simulates a two-condition study — an urban-forest scene versus an
urban-street scene — with known ground-truth fixations and saccades.  It exists
so GAZERA can be demonstrated, tested and taught without exposing any
participant data and without redistributing any vendor sample file.

The output is a plain TSV that mimics the *generic* shape of a raw gaze export:
one row per sample, a recording timestamp, gaze coordinates, pupil diameter,
validity flags and (optionally) an eye-movement-type column and AOI hit columns.
Column names are deliberately ordinary so the loader's auto-mapping is exercised
for real.
"""

from __future__ import annotations

import csv
import io
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .metrics import AOI, default_aois

__all__ = [
    "CONDITIONS",
    "SyntheticConfig",
    "generate_dataset",
    "generate_rows",
    "write_dataset",
    "AOI_BANDS",
]

AOI_BANDS: tuple[AOI, ...] = tuple(default_aois(3))

#: Attention weights per band for each simulated condition.
CONDITIONS: dict[str, dict[str, Any]] = {
    "도시숲": {
        "zone_weights": [0.50, 0.25, 0.25],
        "x_bias": {"상단": 0.5, "중앙": 0.5, "하단": 0.5},
        "x_spread": 0.85,
        "fixation_ms": 265.0,
        "saccade_every": 1.0,
    },
    "도심 가로": {
        "zone_weights": [0.12, 0.30, 0.58],
        "x_bias": {"상단": 0.45, "중앙": 0.74, "하단": 0.5},
        "x_spread": 0.55,
        "fixation_ms": 228.0,
        "saccade_every": 1.0,
    },
}


@dataclass
class SyntheticConfig:
    """Knobs for the generator.  Defaults give 10 participants x 2 conditions."""

    n_subjects: int = 10
    duration_s: float = 10.0
    sample_rate_hz: float = 60.0
    screen_width: int = 1920
    screen_height: int = 1080
    seed: int = 20260520
    include_eye_movement_type: bool = True
    include_aoi_hits: bool = False
    invalid_ratio: float = 0.02
    noise_sd_px: float = 7.0
    conditions: Sequence[str] = tuple(CONDITIONS.keys())
    include_preamble: bool = True


def _sample_target(rng: np.random.Generator, condition: str) -> tuple[float, float, str]:
    spec = CONDITIONS[condition]
    weights = np.asarray(spec["zone_weights"], dtype=float)
    weights = weights / weights.sum()
    zone_idx = int(rng.choice(len(AOI_BANDS), p=weights))
    band = AOI_BANDS[zone_idx]
    bias = float(spec["x_bias"].get(band.name, 0.5))
    spread = float(spec["x_spread"])
    x_norm = float(np.clip(rng.normal(bias, spread / 4.0), band.x + 0.05, band.x + band.w - 0.05))
    y_norm = float(rng.uniform(band.y + 0.02, band.y + band.h - 0.02))
    return x_norm, y_norm, band.name


def generate_rows(config: SyntheticConfig | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generate synthetic samples.  Returns ``(rows, truth)``.

    ``truth`` carries the ground-truth fixation/saccade counts used by tests and
    by the demo documentation.
    """
    config = config or SyntheticConfig()
    rng = np.random.default_rng(config.seed)
    dt_ms = 1000.0 / config.sample_rate_hz
    rows: list[dict[str, Any]] = []
    truth: dict[str, Any] = {"conditions": {}, "config": config.__dict__.copy()}

    for condition in config.conditions:
        spec = CONDITIONS[condition]
        n_fix_total = 0
        n_sac_total = 0
        for s in range(config.n_subjects):
            subject = f"P{s + 1:02d}"
            recording = f"{condition}_{subject}"
            t = 0.0
            prev_zone = None
            prev_x, prev_y = None, None
            pupil = 3.2 + rng.normal(0, 0.05)
            while t < config.duration_s * 1000.0:
                x_norm, y_norm, zone = _sample_target(rng, condition)
                if zone == prev_zone:
                    # consecutive fixations in the same band: shorter dwell
                    dur = float(np.clip(rng.lognormal(math.log(spec["fixation_ms"] * 0.7), 0.3), 110, 800))
                else:
                    dur = float(np.clip(rng.lognormal(math.log(spec["fixation_ms"]), 0.3), 140, 900))
                prev_zone = zone
                target_x = x_norm * config.screen_width
                target_y = y_norm * config.screen_height

                # --- saccade into the target -----------------------------
                if prev_x is not None:
                    sac_dur = float(np.clip(rng.normal(42, 9), 18, 90))
                    steps = max(1, int(round(sac_dur / dt_ms)))
                    for k in range(1, steps + 1):
                        frac = k / (steps + 1)
                        ease = 0.5 - 0.5 * math.cos(math.pi * frac)
                        x = prev_x + (target_x - prev_x) * ease
                        y = prev_y + (target_y - prev_y) * ease
                        rows.append(
                            _row(
                                config,
                                t,
                                x,
                                y,
                                pupil,
                                rng,
                                condition,
                                subject,
                                recording,
                                "Saccade",
                                None,
                            )
                        )
                        t += dt_ms
                    n_sac_total += 1

                # --- fixation -------------------------------------------
                n_frames = max(1, int(round(dur / dt_ms)))
                drift_x, drift_y = 0.0, 0.0
                for k in range(n_frames):
                    drift_x = 0.92 * drift_x + rng.normal(0, config.noise_sd_px * 0.35)
                    drift_y = 0.92 * drift_y + rng.normal(0, config.noise_sd_px * 0.35)
                    x = target_x + drift_x + rng.normal(0, config.noise_sd_px * 0.25)
                    y = target_y + drift_y + rng.normal(0, config.noise_sd_px * 0.25)
                    pupil += rng.normal(0, 0.004)
                    rows.append(
                        _row(
                            config,
                            t,
                            x,
                            y,
                            float(np.clip(pupil, 2.0, 5.0)),
                            rng,
                            condition,
                            subject,
                            recording,
                            "Fixation",
                            zone,
                        )
                    )
                    t += dt_ms
                n_fix_total += 1
                prev_x, prev_y = target_x, target_y

        truth["conditions"][condition] = {
            "fixations": n_fix_total,
            "saccades": n_sac_total,
            "subjects": config.n_subjects,
            "duration_s": config.duration_s,
        }
    return rows, truth


def _row(
    config: SyntheticConfig,
    t_ms: float,
    x: float,
    y: float,
    pupil: float,
    rng: np.random.Generator,
    condition: str,
    subject: str,
    recording: str,
    event: str,
    zone: str | None,
) -> dict[str, Any]:
    invalid = rng.random() < config.invalid_ratio
    validity = 4 if invalid else 0
    row: dict[str, Any] = {
        "Recording timestamp": int(round(t_ms * 1000.0)),  # microseconds, like a raw export
        "Gaze point X": round(x, 2) if not invalid else "",
        "Gaze point Y": round(y, 2) if not invalid else "",
        "Pupil diameter left": round(pupil + 0.01, 4) if not invalid else "",
        "Pupil diameter right": round(pupil - 0.01, 4) if not invalid else "",
        "Validity left": validity,
        "Validity right": validity,
        "Participant name": subject,
        "Recording name": recording,
        "Presented stimulus name": condition,
        "Media width": config.screen_width,
        "Media height": config.screen_height,
    }
    if config.include_eye_movement_type:
        row["Eye movement type"] = "Eyes not found" if invalid else event
    if config.include_aoi_hits:
        for band in AOI_BANDS:
            inside = (
                zone == band.name
                and band.x <= x / config.screen_width <= band.x + band.w
                and band.y <= y / config.screen_height <= band.y + band.h
                and not invalid
            )
            row[f"AOI hit [{band.name}]"] = 1 if inside else 0
    return row


def generate_dataset(config: SyntheticConfig | None = None, *, delimiter: str = "\t") -> str:
    """Return the synthetic dataset as delimited text (TSV by default)."""
    config = config or SyntheticConfig()
    rows, _truth = generate_rows(config)
    columns = list(rows[0].keys())
    buffer = io.StringIO()
    if config.include_preamble:
        buffer.write("# GAZERA synthetic demo data — 100% generated, contains no participant data\n")
        buffer.write("# Study: urban forest vs urban street scene viewing\n")
        buffer.write(f"# Generated by gazera_core.synthetic (seed={config.seed})\n")
        buffer.write("\n")
    writer = csv.DictWriter(buffer, fieldnames=columns, delimiter=delimiter, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def write_dataset(path: str | Path, config: SyntheticConfig | None = None, *, delimiter: str | None = None) -> Path:
    path = Path(path)
    if delimiter is None:
        delimiter = "\t" if path.suffix.lower() in (".tsv", ".txt") else ","
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(generate_dataset(config, delimiter=delimiter), encoding="utf-8")
    return path
