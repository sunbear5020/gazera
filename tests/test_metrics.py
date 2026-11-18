"""Metric definition tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import gazera_core as gz
from gazera_core.metrics import AOI


def _segments() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"subject": "P1", "trial": "T1", "condition": "A", "type": "fixation", "start_ms": 0, "end_ms": 200, "duration_ms": 200, "center_x": 100, "center_y": 100, "amplitude_deg": 0, "mean_velocity_deg_s": 1, "peak_velocity_deg_s": 2},
            {"subject": "P1", "trial": "T1", "condition": "A", "type": "saccade", "start_ms": 200, "end_ms": 240, "duration_ms": 40, "center_x": 500, "center_y": 500, "amplitude_deg": 5, "mean_velocity_deg_s": 200, "peak_velocity_deg_s": 300},
            {"subject": "P1", "trial": "T1", "condition": "A", "type": "fixation", "start_ms": 240, "end_ms": 540, "duration_ms": 300, "center_x": 900, "center_y": 900, "amplitude_deg": 0, "mean_velocity_deg_s": 1, "peak_velocity_deg_s": 2},
        ]
    )


def test_fixation_summary_math():
    summary = gz.fixation_summary(_segments())
    assert summary["fixation_count"] == 2
    assert summary["fixation_total_ms"] == 500
    assert summary["fixation_mean_ms"] == 250
    assert summary["fixation_median_ms"] == 250


def test_saccade_summary_math():
    summary = gz.saccade_summary(_segments())
    assert summary["saccade_count"] == 1
    assert summary["saccade_amplitude_mean_deg"] == pytest.approx(5)
    assert summary["saccade_peak_velocity_mean_deg_s"] == pytest.approx(300)


def test_empty_segments_are_safe():
    empty = pd.DataFrame(columns=["type", "duration_ms"])
    assert gz.fixation_summary(empty)["fixation_count"] == 0
    assert gz.saccade_summary(empty)["saccade_count"] == 0


def test_aoi_dwell_ratios_sum_to_one(demo_data, demo_segments):
    aois = gz.default_aois(3)
    table = gz.aoi_metrics(demo_data.frame, aois, segments=demo_segments.segments)
    grouped = table.groupby(["subject", "trial"])["dwell_ratio"].sum()
    assert ((grouped - 1).abs() < 0.02).all()


def test_aoi_ttff_is_relative_and_non_negative(demo_data, demo_segments):
    table = gz.aoi_metrics(demo_data.frame, gz.default_aois(3), segments=demo_segments.segments)
    assert (table["ttff_ms"].dropna() >= -1e-6).all()
    first = table.groupby(["subject", "trial"])["ttff_ms"].min()
    assert first.min() < 500  # something is looked at almost immediately


def test_aoi_visits_count_fixations(demo_data, demo_segments):
    table = gz.aoi_metrics(demo_data.frame, gz.default_aois(3), segments=demo_segments.segments)
    assert table["visits"].sum() == len(demo_segments.segments[demo_segments.segments["type"] == "fixation"])
    assert (table["visits"] >= 0).all()


def test_aoi_accepts_pixel_rectangles(demo_data):
    aois = [{"name": "left", "x": 0, "y": 0, "w": 960, "h": 1080}, {"name": "right", "x": 960, "y": 0, "w": 960, "h": 1080}]
    norm = gz.normalise_aois(aois, (1920, 1080))
    assert norm[1].x == pytest.approx(0.5)
    assert norm[1].w == pytest.approx(0.5)


def test_condition_summary_shape(demo_data, demo_segments):
    subjects = gz.subject_metrics(demo_data.frame, demo_segments.segments, gz.default_aois(3))
    assert len(subjects) == demo_data.frame[["subject", "trial"]].drop_duplicates().shape[0]
    summary = gz.condition_summary(subjects)
    assert set(summary["condition"]) == {"도시숲", "도심 가로"}
    assert (summary["n"] > 0).all()


def test_subject_metrics_include_aoi_columns(demo_data, demo_segments):
    subjects = gz.subject_metrics(demo_data.frame, demo_segments.segments, gz.default_aois(3))
    assert any(c.startswith("aoi_상단_dwell_ratio") for c in subjects.columns)
    assert subjects["fixation_count"].sum() == len(
        demo_segments.segments[demo_segments.segments["type"] == "fixation"]
    )
