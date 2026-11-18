"""I-VT classifier tests, including recovery of known synthetic fixations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import gazera_core as gz
from gazera_core.ivt import IVTConfig, estimate_px_per_deg


def test_px_per_deg_is_plausible():
    ppd = estimate_px_per_deg(1920, 530, 650)
    assert 35 < ppd < 50


def test_segments_have_required_columns(demo_segments):
    seg = demo_segments.segments
    for col in (
        "subject",
        "trial",
        "condition",
        "type",
        "start_ms",
        "end_ms",
        "duration_ms",
        "amplitude_deg",
        "mean_velocity_deg_s",
        "peak_velocity_deg_s",
        "center_x",
        "center_y",
    ):
        assert col in seg.columns
    assert set(seg["type"].unique()) <= {"fixation", "saccade"}
    assert (seg["duration_ms"] >= 0).all()


def test_fixation_durations_respect_minimum(demo_segments):
    fix = demo_segments.segments[demo_segments.segments["type"] == "fixation"]
    assert len(fix) > 0
    # one sample of tolerance for the minimum-duration rule
    basic_dt = 1000 / 60
    assert fix["duration_ms"].min() >= 60 - basic_dt


def test_recovers_known_fixation_count(demo_data, demo_segments, demo_config):
    truth = gz.synthetic.generate_rows(demo_config)[1]["conditions"]
    detected = demo_segments.segments.groupby("condition")["type"].apply(lambda s: (s == "fixation").sum())
    for condition, stats in truth.items():
        expected = stats["fixations"]
        assert detected[condition] > expected * 0.7
        assert detected[condition] < expected * 1.3


def test_saccade_amplitudes_are_larger_than_fixations(demo_segments):
    seg = demo_segments.segments
    sac = seg[seg["type"] == "saccade"]
    assert sac["amplitude_deg"].mean() > 0.5
    assert sac["peak_velocity_deg_s"].mean() > 30


def test_agreement_with_export_labels(demo_segments):
    assert demo_segments.agreement is not None
    assert demo_segments.agreement["comparable_samples"] > 1000
    assert demo_segments.agreement["agreement"] > 0.75


def test_higher_threshold_finds_more_fixations(demo_data):
    low = gz.classify_ivt(demo_data.frame, IVTConfig(velocity_threshold_deg_s=10))
    high = gz.classify_ivt(demo_data.frame, IVTConfig(velocity_threshold_deg_s=80))
    low_fix = (low.samples["ivt_label"] == "fixation").sum()
    high_fix = (high.samples["ivt_label"] == "fixation").sum()
    assert high_fix > low_fix


def test_invalid_samples_are_unclassified(demo_segments):
    samples = demo_segments.samples
    invalid = samples[~samples["valid"]]
    assert (invalid["ivt_label"] == "unclassified").all()


def test_single_recording_frame_without_subject():
    # 1.5 s at x=500, a 6-sample (~100 ms) saccade, then 1.5 s at x=1500
    xs = np.r_[np.full(90, 500.0), np.linspace(500, 1500, 6), np.full(90, 1500.0)]
    frame = pd.DataFrame(
        {
            "time_ms": np.arange(len(xs)) * (1000 / 60),
            "gaze_x": xs,
            "gaze_y": np.full(len(xs), 500.0),
            "valid": True,
            "pupil": 3.0,
        }
    )
    frame.attrs["screen"] = (1920.0, 1080.0)
    result = gz.classify_ivt(frame)
    assert not result.segments.empty
    assert (result.segments["type"] == "fixation").sum() >= 2
    assert (result.segments["type"] == "saccade").sum() >= 1


def test_assign_events_links_samples_to_segments(demo_segments, demo_data):
    assigned = gz.ivt.assign_events(demo_segments.segments, demo_data.frame)
    assert "segment_id" in assigned.columns
    assert (assigned["segment_id"] >= 0).sum() > 0


def test_missing_columns_raise():
    with pytest.raises(ValueError):
        gz.classify_ivt(pd.DataFrame({"gaze_x": [1.0, 2.0]}))
