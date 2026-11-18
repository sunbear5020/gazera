"""Loader tests: encoding, delimiter, header detection and column mapping."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import gazera_core as gz
from gazera_core.loader import detect_mapping, find_aoi_columns, parse_text


def test_detects_tab_delimiter_and_preamble(demo_data, demo_config):
    assert demo_data.meta["delimiter"] == "\t"
    assert demo_data.meta["header_row_index"] >= 3
    assert len(demo_data.frame) > 1000
    assert demo_data.meta["encoding"] in ("utf-8", "utf-8-sig")


def test_auto_mapping_matches_canonical_fields(demo_data):
    mapping = demo_data.mapping
    for field in ("timestamp", "gaze_x", "gaze_y", "subject", "trial", "condition"):
        assert field in mapping, f"{field} was not mapped"
    assert mapping["gaze_x"] == "Gaze point X"
    assert mapping["gaze_y"] == "Gaze point Y"
    assert mapping["event"] == "Eye movement type"
    assert mapping["pupil_left"] == "Pupil diameter left"


def test_timestamp_microseconds_converted_to_ms(demo_data, demo_config):
    assert demo_data.meta["timestamp_unit"] == "us"
    span = demo_data.frame.groupby(["subject", "trial"])["time_ms"].max().max()
    assert 5000 < span < 7000  # ~6 s recording, in ms


def test_comma_delimited_csv_is_detected(demo_config):
    text = gz.generate_dataset(demo_config, delimiter=",")
    data = gz.load_gaze_bytes(text.encode("utf-8"))
    assert data.meta["delimiter"] == ","
    assert data.frame["gaze_x"].notna().mean() > 0.9


def test_semicolon_delimiter_is_detected(demo_config):
    text = gz.generate_dataset(demo_config, delimiter=";")
    data = gz.load_gaze_bytes(text.encode("utf-8"))
    assert data.meta["delimiter"] == ";"
    assert len(data.frame) > 1000


def test_cp949_encoded_korean_headers():
    text = (
        "시각\t시선X\t시선Y\t피험자\n"
        "0\t100\t200\tP01\n"
        "17\t110\t210\tP01\n"
        "33\t120\t220\tP01\n"
    )
    data = gz.load_gaze_bytes(
        text.encode("cp949"),
        mapping={"timestamp": "시각", "gaze_x": "시선X", "gaze_y": "시선Y", "subject": "피험자"},
    )
    assert data.meta["encoding"] in ("cp949", "euc-kr")
    assert data.frame["gaze_x"].tolist() == [100, 110, 120]


def test_custom_mapping_overrides_detection(demo_text):
    data = gz.load_gaze_bytes(
        demo_text.encode("utf-8"),
        mapping={
            "gaze_x": "Gaze point Y",  # deliberately swapped
            "gaze_y": "Gaze point X",
            "timestamp": "Recording timestamp",
        },
    )
    assert data.mapping["gaze_x"] == "Gaze point Y"
    expected = pd.to_numeric(data.raw["Gaze point Y"], errors="coerce").iloc[0]
    assert data.frame["gaze_x"].iloc[0] == pytest.approx(expected, abs=0.01)


def test_missing_gaze_columns_raises():
    text = "a,b,c\n1,2,3\n4,5,6\n"
    with pytest.raises(ValueError, match="gaze"):
        gz.load_gaze_bytes(text.encode())


def test_aoi_hit_columns_are_found():
    columns = ["Recording timestamp", "Gaze point X", "AOI hit [canopy]", "AOI hit [sky]", "Validity left"]
    mapping = find_aoi_columns(columns)
    assert mapping == {"canopy": "AOI hit [canopy]", "sky": "AOI hit [sky]"}


def test_generic_export_names_map_without_vendor_wording():
    columns = ["time_ms", "gaze_x", "gaze_y", "pupil", "validity", "subject", "condition"]
    mapping = detect_mapping(columns)
    assert mapping["gaze_x"] == "gaze_x"
    assert mapping["gaze_y"] == "gaze_y"
    assert mapping["subject"] == "subject"
    assert mapping["validity"] == "validity"


def test_decimal_comma_parsing():
    text = "timestamp;gaze_x;gaze_y;subject\n0;123,5;200,5;P1\n17;124,5;201,5;P1\n34;125,5;202,5;P1\n"
    data = gz.load_gaze_bytes(text.encode("utf-8"), decimal_comma=True)
    assert data.frame["gaze_x"].iloc[0] == pytest.approx(123.5)


def test_validity_convention_detection(demo_data):
    assert demo_data.meta["validity_convention"] == "zero-is-valid"
    # invalid samples carry Validity 4
    invalid = demo_data.frame.loc[~demo_data.frame["valid"], "validity_left"]
    assert set(invalid.dropna().unique()) <= {4.0}


def test_recordings_are_split_by_condition_without_a_trial_column(demo_text):
    """A block-per-stimulus export has no trial column; condition must split it."""
    data = gz.load_gaze_bytes(
        demo_text.encode("utf-8"),
        mapping={
            "gaze_x": "Gaze point X",
            "gaze_y": "Gaze point Y",
            "timestamp": "Recording timestamp",
            "subject": "Participant name",
            "condition": "Presented stimulus name",
        },
    )
    n_subjects = data.frame["subject"].nunique()
    n_conditions = data.frame["condition"].nunique()
    assert data.frame.groupby(["subject", "condition"]).ngroups == n_subjects * n_conditions
    spans = data.frame.groupby(["subject", "condition"])["time_ms"].max()
    assert (spans < 9000).all()


def test_no_condition_or_trial_makes_one_recording(demo_text):
    data = gz.load_gaze_bytes(
        demo_text.encode("utf-8"),
        mapping={"gaze_x": "Gaze point X", "gaze_y": "Gaze point Y"},
    )
    assert len(data.groups()) == 1
