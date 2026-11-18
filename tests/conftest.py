"""Shared fixtures for the GAZERA test suite (all data is synthetic)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gazera_core as gz  # noqa: E402


@pytest.fixture(scope="session")
def demo_config() -> gz.SyntheticConfig:
    return gz.SyntheticConfig(n_subjects=4, duration_s=6.0, seed=7)


@pytest.fixture(scope="session")
def demo_text(demo_config) -> str:
    return gz.generate_dataset(demo_config)


@pytest.fixture(scope="session")
def demo_data(demo_text) -> gz.GazeData:
    return gz.load_gaze_bytes(demo_text.encode("utf-8"), filename="demo.tsv")


@pytest.fixture(scope="session")
def demo_segments(demo_data):
    return gz.classify_ivt(demo_data.frame)
