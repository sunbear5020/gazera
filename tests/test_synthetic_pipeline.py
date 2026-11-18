"""Synthetic generator and pipeline/export integration tests."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import gazera_core as gz


def test_generator_is_deterministic():
    cfg = gz.SyntheticConfig(n_subjects=2, duration_s=2.0, seed=1)
    a = gz.generate_dataset(cfg)
    b = gz.generate_dataset(cfg)
    assert a == b
    c = gz.generate_dataset(gz.SyntheticConfig(n_subjects=2, duration_s=2.0, seed=2))
    assert a != c


def test_generator_has_two_conditions_and_no_real_data(demo_text):
    assert "도시숲" in demo_text and "도심 가로" in demo_text
    assert "synthetic" in demo_text.lower()


def test_generator_row_and_column_shape(demo_data, demo_config):
    frame = demo_data.frame
    assert len(frame) == len(demo_data.raw)
    assert set(frame["condition"].dropna().unique()) == {"도시숲", "도심 가로"}
    assert frame["subject"].nunique() == demo_config.n_subjects
    assert frame["valid"].mean() > 0.9


def test_aoi_hit_variant_maps_columns():
    cfg = gz.SyntheticConfig(n_subjects=1, duration_s=2.0, include_aoi_hits=True)
    text = gz.generate_dataset(cfg)
    data = gz.load_gaze_bytes(text.encode())
    aois = data.aoi_columns
    assert set(aois) == {"상단", "중앙", "하단"}


def test_write_dataset(tmp_path):
    target = gz.write_dataset(tmp_path / "demo.tsv", gz.SyntheticConfig(n_subjects=1, duration_s=1.0))
    assert target.exists()
    assert target.read_text(encoding="utf-8").startswith("#")


def test_pipeline_end_to_end(demo_data):
    result = gz.analyze(demo_data)
    payload = result.to_dict()
    assert payload["schema"] == "gazera.result.v1"
    assert payload["summary"]["rows"] > 0
    assert payload["ivt"]["fixation"]["fixation_count"] > 0
    assert len(payload["subjects"]) == payload["summary"]["recordings"]
    assert payload["notices"]["license"].startswith("MIT")
    # JSON-serialisable, no NaN tokens
    text = json.dumps(payload, ensure_ascii=False)
    assert "NaN" not in text


def test_pipeline_sheets_are_dataframes(demo_data):
    result = gz.analyze(demo_data)
    sheets = result.sheets()
    for name, frame in sheets.items():
        assert isinstance(frame, pd.DataFrame), name
    assert "Statistics" in sheets and "Participants" in sheets


def test_excel_export_roundtrip(tmp_path, demo_data):
    from openpyxl import load_workbook

    result = gz.analyze(demo_data)
    path = gz.export_excel(tmp_path / "out.xlsx", result.sheets(), metadata={"Source": "synthetic"})
    workbook = load_workbook(path)
    assert "README" in workbook.sheetnames
    assert "Participants" in workbook.sheetnames
    ws = workbook["Participants"]
    assert ws.max_row > 1
    assert ws.freeze_panes == "A2"


def test_json_and_csv_export(tmp_path, demo_data):
    result = gz.analyze(demo_data)
    json_path = gz.export_json(tmp_path / "out.json", result.to_dict())
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["summary"]["rows"] > 0
    csv_paths = gz.export_csv(tmp_path / "csv", result.sheets())
    assert csv_paths["Participants"].exists()
    assert csv_paths["Participants"].read_bytes().startswith(b"\xef\xbb\xbf")


def test_jsonable_converts_nan():
    payload = {"a": float("nan"), "b": np.float64(2.5), "c": np.int64(3), "d": pd.DataFrame({"x": [np.nan, 1]})}
    out = gz.to_jsonable(payload)
    assert out["a"] is None
    assert out["b"] == 2.5
    assert out["c"] == 3
    assert out["d"][0]["x"] is None


def test_cli_demo_and_analyze(tmp_path, capsys):
    demo = tmp_path / "demo.tsv"
    gz.cli.main(["demo", "--out", str(demo), "--subjects", "2", "--duration", "2"])
    assert demo.exists()
    gz.cli.main(["analyze", str(demo), "--out", str(tmp_path / "results")])
    captured = capsys.readouterr().out
    assert "Fixations" in captured
    assert (tmp_path / "results" / "demo_gazera.xlsx").exists()
