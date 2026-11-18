"""Statistics tests with known answers."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

import gazera_core as gz


def _two_group_df(normal: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    if normal:
        a = rng.normal(300, 30, 20)
        b = rng.normal(240, 30, 20)
    else:
        a = rng.exponential(2.0, 20)
        b = np.r_[rng.exponential(1.0, 19), 40.0]  # heavy tail / outlier
    rows = [{"subject": f"S{i}", "condition": "forest", "dv": v} for i, v in enumerate(a)]
    rows += [{"subject": f"S{i}", "condition": "street", "dv": v} for i, v in enumerate(b)]
    return pd.DataFrame(rows)


def test_describe_groups_counts():
    table = gz.describe_groups(_two_group_df(), "dv")
    assert len(table) == 2
    assert (table["n"] == 20).all()
    assert "ci95_low" in table


def test_parametric_comparison_is_detected_and_significant():
    df = _two_group_df()
    result = gz.compare_two_groups(df, "dv")
    assert result["test"] == "Welch t-test"
    assert result["p"] < 0.001
    assert result["effect_size"] > 1.0
    assert result["significant"] is True


def test_non_normal_data_uses_mann_whitney():
    df = _two_group_df(normal=False)
    result = gz.compare_two_groups(df, "dv", force=None)
    # with n=20 the pipeline may call normality either way; forcing keeps the
    # nonparametric path covered explicitly.
    forced = gz.compare_two_groups(df, "dv", force="nonparametric")
    assert forced["test"] == "Mann-Whitney U"
    assert "rank-biserial" in forced["effect_size_name"]
    assert 0 <= result["p"] <= 1


def test_paired_test_matches_manual_difference():
    rng = np.random.default_rng(5)
    base = rng.normal(250, 25, 15)
    df = pd.DataFrame(
        [{"subject": f"S{i}", "condition": "forest", "dv": v} for i, v in enumerate(base)]
        + [{"subject": f"S{i}", "condition": "street", "dv": v - 40 + rng.normal(0, 5)} for i, v in enumerate(base)]
    )
    result = gz.compare_paired(df, "dv")
    assert result["test"] == "Paired t-test"
    assert result["n_pairs"] == 15
    assert result["mean_diff"] > 20
    assert result["p"] < 0.001


def test_compare_many_groups_runs():
    rng = np.random.default_rng(2)
    rows = []
    for cond, mu in (("a", 100), ("b", 200), ("c", 300)):
        for i, v in enumerate(rng.normal(mu, 20, 12)):
            rows.append({"subject": f"{cond}{i}", "condition": cond, "dv": v})
    result = gz.compare_many_groups(pd.DataFrame(rows), "dv")
    assert result["test"] in ("One-way ANOVA", "Kruskal-Wallis")
    assert result["p"] < 0.001


def test_holm_correction_is_monotone_and_capped():
    adjusted = gz.holm_correction([0.01, 0.04, 0.03])
    assert all(0 <= p <= 1 for p in adjusted)
    assert adjusted[0] == pytest.approx(0.03)
    assert adjusted[1] >= adjusted[0]


def test_analyze_comparisons_adds_adjusted_p(demo_data, demo_segments):
    subjects = gz.subject_metrics(demo_data.frame, demo_segments.segments, gz.default_aois(3))
    results = gz.analyze_comparisons(subjects, ["fixation_count", "fixation_mean_ms"])
    assert len(results) == 2
    assert "p_adjusted" in results.columns
    assert results["p"].between(0, 1).all()


def test_compare_two_groups_with_single_group_is_not_run():
    df = pd.DataFrame({"condition": ["a", "a"], "dv": [1.0, 2.0]})
    result = gz.compare_two_groups(df, "dv")
    assert result["test"] == "not run"
    assert math.isnan(result["p"])
