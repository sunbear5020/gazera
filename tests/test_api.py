"""Local API tests (FastAPI TestClient)."""

from __future__ import annotations

import json

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from api.server import app  # noqa: E402

client = TestClient(app)


def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["license"] == "MIT"
    assert "Tobii" in body["notices"]["affiliation"]


def test_sample_endpoint_returns_tsv():
    response = client.get("/api/sample?subjects=2&duration=2")
    assert response.status_code == 200
    text = response.text
    assert "Gaze point X" in text
    assert "도시숲" in text


def test_analyze_endpoint(demo_text):
    files = {"file": ("demo.tsv", demo_text.encode("utf-8"), "text/tab-separated-values")}
    response = client.post("/api/analyze", files=files)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["schema"] == "gazera.result.v1"
    assert body["summary"]["rows"] > 0
    assert len(body["subjects"]) == body["summary"]["recordings"]


def test_analyze_with_custom_mapping_and_config(demo_text):
    files = {"file": ("demo.tsv", demo_text.encode("utf-8"), "text/tab-separated-values")}
    mapping = json.dumps({"gaze_x": "Gaze point X", "gaze_y": "Gaze point Y", "subject": "Participant name"})
    config = json.dumps({"ivt": {"velocity_threshold_deg_s": 45.0}, "dvs": ["fixation_count"]})
    response = client.post("/api/analyze", files=files, data={"mapping": mapping, "config": config})
    assert response.status_code == 200
    body = response.json()
    assert body["config"]["ivt"]["velocity_threshold_deg_s"] == 45.0


def test_export_xlsx(demo_text):
    files = {"file": ("demo.tsv", demo_text.encode("utf-8"), "text/tab-separated-values")}
    response = client.post("/api/export", files=files)
    assert response.status_code == 200
    assert response.content[:2] == b"PK"
    assert "attachment" in response.headers["content-disposition"]


def test_export_json(demo_text):
    files = {"file": ("demo.tsv", demo_text.encode("utf-8"), "text/tab-separated-values")}
    response = client.post("/api/export?fmt=json", files=files)
    assert response.status_code == 200
    assert json.loads(response.text)["schema"] == "gazera.result.v1"


def test_unmappable_file_returns_422():
    files = {"file": ("bad.csv", b"a,b\n1,2\n3,4\n", "text/csv")}
    response = client.post("/api/analyze", files=files)
    assert response.status_code == 422
    assert "gaze" in response.json()["detail"].lower()
