"""Tests for the stimulus RGB exclusion filter (images and videos).

All stimuli here are generated on the fly: no vendor sample and no real
participant data is involved.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import gazera_core as gz
from gazera_core.stimulus import (
    apply_rgb_exclusion,
    open_stimulus,
    parse_color,
    parse_colors,
)

pytest.importorskip("PIL")
from PIL import Image  # noqa: E402


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def split_image(path: Path, left: tuple[int, int, int], right: tuple[int, int, int], size=(200, 100)) -> Path:
    width, height = size
    array = np.zeros((height, width, 3), dtype=np.uint8)
    array[:, : width // 2] = left
    array[:, width // 2 :] = right
    Image.fromarray(array).save(path)
    return path


def banded_image(path: Path, colors: list[tuple[int, int, int]], size=(300, 100)) -> Path:
    width, height = size
    array = np.zeros((height, width, 3), dtype=np.uint8)
    step = width // len(colors)
    for i, color in enumerate(colors):
        array[:, i * step : (i + 1) * step] = color
    Image.fromarray(array).save(path)
    return path


def gaze_frame(xs, ys, *, screen=(200, 100), times=None, subjects=None) -> pd.DataFrame:
    n = len(xs)
    frame = pd.DataFrame(
        {
            "time_ms": np.asarray(times if times is not None else np.arange(n) * 16.7, dtype=float),
            "gaze_x": np.asarray(xs, dtype=float),
            "gaze_y": np.asarray(ys, dtype=float),
            "valid": np.ones(n, dtype=bool),
            "subject": np.asarray(subjects if subjects is not None else ["P01"] * n, dtype=object),
            "condition": np.asarray(["A"] * n, dtype=object),
        }
    )
    frame.attrs["screen"] = screen
    return frame


# ---------------------------------------------------------------------------
# colour parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        ("#000000", (0, 0, 0)),
        ("#fff", (255, 255, 255)),
        ("0,0,0", (0, 0, 0)),
        ("255 255 255", (255, 255, 255)),
        ("rgb(217, 119, 87)", (217, 119, 87)),
        ((20, 30, 40), (20, 30, 40)),
        ("black", (0, 0, 0)),
        ("#D97757", (217, 119, 87)),
    ],
)
def test_parse_color(value, expected):
    assert parse_color(value) == expected


def test_parse_color_rejects_bad_values():
    with pytest.raises(ValueError):
        parse_color("#12")
    with pytest.raises(ValueError):
        parse_color("300,0,0")


def test_parse_colors_dedupes_and_accepts_mixed_input():
    colors = parse_colors(["#000000", "0,0,0", "255,255,255;0,0,255", (0, 0, 0)])
    assert colors == [(0, 0, 0), (255, 255, 255), (0, 0, 255)]


# ---------------------------------------------------------------------------
# still-image exclusion
# ---------------------------------------------------------------------------


def test_exact_colour_match_is_excluded(tmp_path):
    stimulus = split_image(tmp_path / "stim.png", (0, 0, 0), (255, 255, 255))
    frame = gaze_frame([10, 190, 190], [50, 50, 50], screen=(200, 100))

    out, report = apply_rgb_exclusion(frame, stimulus, [(0, 0, 0)], tolerance=0)

    assert out["rgb_excluded"].tolist() == [True, False, False]
    assert out["stimulus_rgb"].tolist()[0] == "#000000"
    assert report.excluded_samples == 1
    assert report.samples_total == 3
    assert report.excluded_ratio == pytest.approx(1 / 3)
    assert report.kind == "image"
    assert out["valid"].tolist() == [False, True, True]


def test_tolerance_boundary_follows_euclidean_distance(tmp_path):
    # middle band is (0, 0, 20): distance 20 from black
    stimulus = banded_image(
        tmp_path / "bands.png",
        [(0, 0, 0), (0, 0, 20), (255, 255, 255)],
        size=(300, 100),
    )
    xs = [50, 150, 250]  # one sample per band
    frame = gaze_frame(xs, [50, 50, 50], screen=(300, 100))

    _, inside = apply_rgb_exclusion(frame, stimulus, [(0, 0, 0)], tolerance=20)
    out_frame, outside = apply_rgb_exclusion(frame, stimulus, [(0, 0, 0)], tolerance=19)

    assert inside.excluded_samples == 2  # distance 0 and distance 20 both <= 20
    assert outside.excluded_samples == 1  # distance 20 > 19
    assert out_frame["rgb_excluded"].tolist() == [True, False, False]


def test_multiple_colours_are_counted_separately(tmp_path):
    stimulus = split_image(tmp_path / "stim.png", (12, 18, 24), (255, 255, 255))
    frame = gaze_frame([10, 190, 10], [50, 50, 50])

    _, report = apply_rgb_exclusion(
        frame, stimulus, [(12, 18, 24), (255, 255, 255)], tolerance=5
    )

    assert report.excluded_samples == 3
    by_colour = {tuple(item["rgb"]): item["samples"] for item in report.excluded_by_color}
    assert by_colour[(12, 18, 24)] == 2
    assert by_colour[(255, 255, 255)] == 1


def test_missing_coordinates_are_not_sampled(tmp_path):
    stimulus = split_image(tmp_path / "stim.png", (0, 0, 0), (255, 255, 255))
    frame = gaze_frame([np.nan, 10, 190], [50, 50, 50])

    out, report = apply_rgb_exclusion(frame, stimulus, [(0, 0, 0)], tolerance=5)

    assert report.samples_sampled == 2
    assert report.samples_total == 3
    assert out["rgb_excluded"].tolist() == [False, True, False]
    assert out.loc[0, "stimulus_rgb"] is None or pd.isna(out.loc[0, "stimulus_rgb"])


def test_colour_tuple_config_and_hex_strings_agree(tmp_path):
    stimulus = split_image(tmp_path / "stim.png", (0, 0, 0), (255, 255, 255))
    frame = gaze_frame([10, 190], [50, 50])

    _, tuple_report = apply_rgb_exclusion(frame, stimulus, [(0, 0, 0)], tolerance=3)
    _, hex_report = apply_rgb_exclusion(frame, stimulus, "#000000", tolerance=3)

    assert tuple_report.excluded_samples == hex_report.excluded_samples == 1


def test_open_stimulus_rejects_unknown_extension(tmp_path):
    path = tmp_path / "document.pdf"
    path.write_bytes(b"%PDF-1.4")
    with pytest.raises(ValueError):
        open_stimulus(path)


# ---------------------------------------------------------------------------
# pipeline integration
# ---------------------------------------------------------------------------


def test_pipeline_excludes_gaze_on_black_half(demo_text, tmp_path):
    stimulus = split_image(tmp_path / "half.png", (0, 0, 0), (255, 255, 255))
    data = gz.load_gaze_bytes(demo_text.encode("utf-8"), filename="demo.tsv")

    baseline = gz.analyze(data, gz.AnalysisConfig())
    filtered = gz.analyze(
        data,
        gz.AnalysisConfig(
            stimulus=stimulus,
            exclude_rgb=[(0, 0, 0)],
            rgb_tolerance=8,
        ),
    )

    report = filtered.rgb_filter
    assert report is not None
    assert report.kind == "image"
    assert report.excluded_samples > 0
    assert report.samples_total == len(data.frame)
    assert 0 < report.excluded_ratio < 0.75

    # rows are kept, only flagged: nothing is silently deleted
    assert len(filtered.data.frame) == len(data.frame)
    assert "rgb_excluded" in filtered.data.frame
    assert filtered.data.frame["rgb_excluded"].sum() == report.excluded_samples

    # no fixation survives inside the excluded (left, black) half
    fixations = filtered.ivt.segments[filtered.ivt.segments["type"] == "fixation"]
    width = filtered.data.frame.attrs["screen"][0]
    assert not fixations.empty
    assert (fixations["center_x"] > width / 2).all()

    # fewer fixations remain after the filter
    assert filtered.pooled_fixation["fixation_count"] < baseline.pooled_fixation["fixation_count"]

    payload = filtered.to_dict()
    assert payload["rgb_filter"]["excluded_samples"] == report.excluded_samples
    assert payload["config"]["exclude_rgb"] == [[0, 0, 0]]
    assert "RGB filter" in filtered.sheets()


def test_pipeline_without_stimulus_reports_no_filter(demo_text):
    data = gz.load_gaze_bytes(demo_text.encode("utf-8"), filename="demo.tsv")
    result = gz.analyze(data, gz.AnalysisConfig())
    assert result.rgb_filter is None
    assert result.to_dict()["rgb_filter"] is None
    assert "RGB filter" not in result.sheets()


# ---------------------------------------------------------------------------
# video stimulus
# ---------------------------------------------------------------------------


def _write_test_video(path: Path, *, fps: int = 10, size=(160, 90), black_frames: int = 10, total: int = 20) -> Path:
    cv2 = pytest.importorskip("cv2")
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():  # pragma: no cover - codec missing in the build
        pytest.skip("OpenCV video writer is unavailable in this build")
    for i in range(total):
        value = 0 if i < black_frames else 255
        frame = np.full((size[1], size[0], 3), value, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    assert path.exists() and path.stat().st_size > 0
    return path


def test_video_frames_follow_the_gaze_timeline(tmp_path):
    video = _write_test_video(tmp_path / "stim.mp4")
    stimulus = open_stimulus(video)
    try:
        assert stimulus.kind == "video"
        times = np.array([0.0, 200.0, 400.0, 1200.0, 1600.0, 1900.0])
        frame = gaze_frame([80] * len(times), [45] * len(times), screen=(160, 90), times=times)
        out, report = apply_rgb_exclusion(frame, stimulus, [(0, 0, 0)], tolerance=10, bucket_ms=100)
    finally:
        stimulus.close()

    # black for the first second, white afterwards
    assert out["rgb_excluded"].tolist() == [True, True, True, False, False, False]
    assert report.kind == "video"
    assert report.frames_read >= 2
    assert report.duration_ms is not None and report.duration_ms > 1000


def test_video_offset_shifts_the_lookup(tmp_path):
    video = _write_test_video(tmp_path / "stim.mp4")
    times = np.array([0.0, 800.0, 950.0, 1100.0])
    frame = gaze_frame([80] * len(times), [45] * len(times), screen=(160, 90), times=times)

    with open_stimulus(video) as stimulus:
        _, plain = apply_rgb_exclusion(frame, stimulus, [(0, 0, 0)], tolerance=10, bucket_ms=100)
    with open_stimulus(video) as stimulus:
        _, shifted = apply_rgb_exclusion(
            frame, stimulus, [(0, 0, 0)], tolerance=10, bucket_ms=100, offset_ms=500
        )

    assert plain.excluded_samples == 3  # 0, 800, 950 ms are black (black ends at 1000 ms)
    assert shifted.excluded_samples == 1  # +500 ms: only 0 -> 500 ms is still black


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


def test_api_accepts_stimulus_upload(demo_text, tmp_path):
    fastapi = pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from api.server import app

    stimulus = split_image(tmp_path / "half.png", (0, 0, 0), (255, 255, 255))
    client = TestClient(app)
    files = {
        "file": ("demo.tsv", demo_text.encode("utf-8"), "text/tab-separated-values"),
        "stimulus": ("half.png", stimulus.read_bytes(), "image/png"),
    }
    config = json.dumps({"exclude_rgb": ["#000000"], "rgb_tolerance": 8})
    response = client.post("/api/analyze", files=files, data={"config": config})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["rgb_filter"]["excluded_samples"] > 0
    assert body["rgb_filter"]["stimulus"] == "half.png"
    assert body["config"]["exclude_rgb"] == [[0, 0, 0]]


def test_api_rejects_broken_stimulus(demo_text):
    fastapi = pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from api.server import app

    client = TestClient(app)
    files = {
        "file": ("demo.tsv", demo_text.encode("utf-8"), "text/tab-separated-values"),
        "stimulus": ("half.png", b"not an image", "image/png"),
    }
    config = json.dumps({"exclude_rgb": [[0, 0, 0]]})
    response = client.post("/api/analyze", files=files, data={"config": config})
    assert response.status_code == 400
