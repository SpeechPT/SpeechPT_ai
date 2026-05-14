"""Tests for the coverage → user score mapping."""
from __future__ import annotations

import pytest

from speechpt.report.score_mapping import (
    DEFAULT_MAPPING,
    map_coverage_score,
    map_raw_coverage,
    resolve_mapping,
)


def test_default_endpoints_clamp():
    assert map_raw_coverage(-0.5) == 0.0
    assert map_raw_coverage(0.0) == 0.0
    assert map_raw_coverage(1.0) == 100.0
    assert map_raw_coverage(1.5) == 100.0


def test_default_control_points_are_exact():
    for raw, expected in DEFAULT_MAPPING:
        assert map_raw_coverage(raw) == pytest.approx(expected)


def test_default_interpolation_is_piecewise_linear():
    # midway between (0.30, 65) and (0.50, 82) → (0.40, 73.5)
    assert map_raw_coverage(0.40) == pytest.approx(73.5)
    # midway between (0.50, 82) and (0.70, 95) → (0.60, 88.5)
    assert map_raw_coverage(0.60) == pytest.approx(88.5)


def test_default_curve_lifts_observed_cases():
    # Observed E2E cases (raw averages) — beta-target user score ranges.
    controlv_raw = 0.281  # → ~62
    speechpt_mid_raw = 0.417  # → ~75
    assert 60.0 <= map_raw_coverage(controlv_raw) <= 65.0
    assert 72.0 <= map_raw_coverage(speechpt_mid_raw) <= 78.0


def test_invalid_entries_use_default_curve_value():
    # Curve fallback gives the default-curve value for raw 0.30.
    assert map_raw_coverage(0.30, []) == pytest.approx(65.0)


def test_map_coverage_score_accepts_percent():
    # 28.1 (pct) is equivalent to raw 0.281
    assert map_coverage_score(28.1) == pytest.approx(map_raw_coverage(0.281))
    assert map_coverage_score(100.0) == 100.0
    assert map_coverage_score(0.0) == 0.0


def test_custom_mapping_from_dicts():
    mapping = [
        {"raw": 0.0, "score": 0},
        {"raw": 0.5, "score": 50},
        {"raw": 1.0, "score": 100},
    ]
    assert map_raw_coverage(0.25, mapping) == pytest.approx(25.0)
    assert map_raw_coverage(0.75, mapping) == pytest.approx(75.0)


def test_custom_mapping_from_tuples():
    mapping = [(0.0, 0.0), (1.0, 100.0)]
    assert map_raw_coverage(0.4, mapping) == pytest.approx(40.0)


def test_unsorted_mapping_is_normalized():
    mapping = [
        {"raw": 1.0, "score": 100},
        {"raw": 0.0, "score": 0},
        {"raw": 0.5, "score": 50},
    ]
    assert map_raw_coverage(0.25, mapping) == pytest.approx(25.0)


def test_invalid_entries_are_skipped():
    mapping = [None, {"raw": None, "score": 10}, (0.0, 0.0), (1.0, 100.0)]
    assert map_raw_coverage(0.5, mapping) == pytest.approx(50.0)


def test_resolve_mapping_uses_config_when_present():
    cfg = {
        "content_coverage_user_mapping": [
            {"raw": 0.0, "score": 0},
            {"raw": 1.0, "score": 100},
        ]
    }
    points = resolve_mapping(cfg)
    assert points[0].raw == 0.0 and points[0].score == 0.0
    assert points[-1].raw == 1.0 and points[-1].score == 100.0


def test_resolve_mapping_falls_back_to_default():
    points_none = resolve_mapping(None)
    points_missing = resolve_mapping({})
    assert [p.raw for p in points_none] == [p.raw for p in points_missing]
    assert points_none[0].score == 0.0
    assert points_none[-1].score == 100.0


def test_monotonic_non_decreasing_over_range():
    prev = -1.0
    for i in range(0, 101):
        value = map_raw_coverage(i / 100.0)
        assert value >= prev
        prev = value
