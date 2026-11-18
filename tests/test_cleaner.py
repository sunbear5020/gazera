"""Quality inspection and cleaning tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

import gazera_core as gz


def test_quality_report_basics(demo_data, demo_config):
    report = gz.quality_report(demo_data)
    assert report.rows == len(demo_data.frame)
    assert report.recordings == demo_config.n_subjects * len(demo_config.conditions)
    assert abs(report.sample_rate_hz - 60) < 2
    assert 0.9 < report.valid_ratio < 1.0
    assert report.missing_gaze_ratio == demo_data.frame["gaze_x"].isna().mean()
    assert report.non_monotonic == 0
    assert report.screen[0] == 1920


def test_quality_flags_low_validity(demo_data):
    frame = demo_data.frame.copy()
    frame.loc[frame.index[: int(len(frame) * 0.5)], "valid"] = False
    report = gz.quality_report(frame)
    assert report.valid_ratio < 0.6
    assert any("Valid gaze ratio" in w for w in report.warnings)


def test_quality_detects_non_monotonic_timestamps(demo_data):
    frame = demo_data.frame.copy()
    # move one sample's timestamp before its predecessor
    frame.loc[5, "time_ms"] = max(0.0, frame.loc[5, "time_ms"] - 500)
    report = gz.quality_report(frame)
    assert report.non_monotonic >= 1


def test_quality_markdown(demo_data):
    text = gz.quality_report(demo_data).summary_markdown()
    assert "GAZERA" in text and "sampling rate" in text


def test_clean_gaze_interpolates_short_gap(demo_data):
    frame = demo_data.frame.copy()
    idx = frame.index[frame["valid"]][10]
    frame.loc[idx, ["gaze_x", "gaze_y"]] = np.nan
    frame.loc[idx, "valid"] = False
    cleaned, actions = gz.clean_gaze(frame, interpolate_gap_ms=80)
    assert cleaned.loc[idx, "gaze_x"] == cleaned.loc[idx, "gaze_x"]  # not NaN
    assert any(a.action == "interpolate" for a in actions)


def test_clean_gaze_clips_out_of_screen(demo_data):
    frame = demo_data.frame.copy()
    idx = frame.index[frame["valid"]][0]
    frame.loc[idx, "gaze_x"] = 5000.0
    cleaned, actions = gz.clean_gaze(frame, sort_timestamps=False, interpolate_gap_ms=0)
    assert pd.isna(cleaned.loc[idx, "gaze_x"])
    assert any(a.action == "clip" for a in actions)


def test_clean_gaze_sort_reports_action(demo_data):
    frame = demo_data.frame.copy()
    shuffled = frame.sample(frac=1.0, random_state=3).reset_index(drop=True)
    cleaned, actions = gz.clean_gaze(shuffled)
    for _, sub in cleaned.groupby(["subject", "trial"]):
        assert sub["time_ms"].is_monotonic_increasing
    assert any(a.action == "sort" for a in actions)
