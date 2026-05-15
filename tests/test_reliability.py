"""Tests for the alignment reliability classifier."""
from __future__ import annotations

import pytest

from speechpt.report.reliability import (
    DEFAULT_LOW_MAX,
    DEFAULT_MEDIUM_MAX,
    compute_alignment_reliability,
)


def test_high_confidence_returns_high():
    result = compute_alignment_reliability(0.10, [])
    assert result["alignment_level"] == "high"
    assert result["content_coverage_shown"] is True
    assert result["note"] == ""


def test_medium_band():
    # default thresholds: low<0.03, medium<0.05, else high
    result = compute_alignment_reliability(0.04, [])
    assert result["alignment_level"] == "medium"
    assert result["content_coverage_shown"] is True
    assert result["note"]


def test_low_confidence_returns_low():
    result = compute_alignment_reliability(0.01, [])
    assert result["alignment_level"] == "low"
    assert result["content_coverage_shown"] is False


def test_force_low_warning_overrides_high():
    result = compute_alignment_reliability(0.10, ["low_alignment_confidence"])
    assert result["alignment_level"] == "low"
    assert result["content_coverage_shown"] is False


def test_demote_warning_steps_level_down():
    high = compute_alignment_reliability(0.10, ["robust_fallback_applied"])
    assert high["alignment_level"] == "medium"
    medium = compute_alignment_reliability(0.04, ["robust_fallback_applied"])
    assert medium["alignment_level"] == "low"


def test_demote_does_not_override_force_low():
    # Both warnings present — force_low wins.
    result = compute_alignment_reliability(
        0.10, ["low_alignment_confidence", "robust_fallback_applied"]
    )
    assert result["alignment_level"] == "low"


def test_observed_cases_land_at_expected_levels():
    # Three real cases from E2E runs:
    # - controlv: confidence 0.057 → high
    # - speechpt_mid: confidence 0.043 → medium
    # - desa: confidence 0.021 + warnings → low
    assert compute_alignment_reliability(0.057, []).get("alignment_level") == "high"
    assert compute_alignment_reliability(0.043, []).get("alignment_level") == "medium"
    assert (
        compute_alignment_reliability(
            0.021,
            ["robust_fallback_applied", "low_alignment_confidence"],
        ).get("alignment_level")
        == "low"
    )


def test_custom_thresholds_via_config():
    cfg = {"low_max": 0.1, "medium_max": 0.2}
    assert compute_alignment_reliability(0.05, [], cfg)["alignment_level"] == "low"
    assert compute_alignment_reliability(0.15, [], cfg)["alignment_level"] == "medium"
    assert compute_alignment_reliability(0.25, [], cfg)["alignment_level"] == "high"


def test_none_confidence_treated_as_zero():
    result = compute_alignment_reliability(None, None)
    assert result["alignment_level"] == "low"
    assert result["confidence_raw"] == 0.0
    assert result["warnings"] == []


def test_defaults_are_stable():
    # Sanity: confidence equal to threshold falls into the upper band.
    assert (
        compute_alignment_reliability(DEFAULT_LOW_MAX, [])["alignment_level"]
        == "medium"
    )
    assert (
        compute_alignment_reliability(DEFAULT_MEDIUM_MAX, [])["alignment_level"]
        == "high"
    )
