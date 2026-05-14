"""Map raw CE coverage values to user-facing scores via a piecewise-linear curve.

CE coverage values are cosine-similarity based and rarely exceed ~0.7 even on
well-delivered slides. Exposing the raw value × 100 as a score makes a "normal"
delivery read as failing (28~42 점). This module applies a calibrated, monotonic
curve so the user-facing score lands in an intuitive range.

The curve is data-driven via config (``report.scoring.content_coverage_user_mapping``).
Default control points are the "soft" curve agreed for v0.1 beta.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

DEFAULT_MAPPING: tuple[tuple[float, float], ...] = (
    (0.00, 0.0),
    (0.15, 40.0),
    (0.30, 65.0),
    (0.50, 82.0),
    (0.70, 95.0),
    (1.00, 100.0),
)


@dataclass(frozen=True)
class _Point:
    raw: float
    score: float


def _normalize_mapping(mapping: Iterable | None) -> List[_Point]:
    if mapping is None:
        points = [_Point(raw=r, score=s) for r, s in DEFAULT_MAPPING]
    else:
        points = []
        for entry in mapping:
            if isinstance(entry, dict):
                raw = entry.get("raw")
                score = entry.get("score")
            elif isinstance(entry, (list, tuple)) and len(entry) == 2:
                raw, score = entry
            else:
                continue
            if raw is None or score is None:
                continue
            points.append(_Point(raw=float(raw), score=float(score)))
    if not points:
        points = [_Point(raw=r, score=s) for r, s in DEFAULT_MAPPING]
    points.sort(key=lambda p: p.raw)
    return points


def map_raw_coverage(raw: float, mapping: Iterable | None = None) -> float:
    """Map a raw coverage value (0~1) to a user-facing score (0~100).

    Values outside the mapping range are clamped to the nearest endpoint.
    Between control points the function is piecewise linear.
    """
    points = _normalize_mapping(mapping)
    value = float(raw)
    if value <= points[0].raw:
        return points[0].score
    if value >= points[-1].raw:
        return points[-1].score
    for left, right in zip(points, points[1:]):
        if left.raw <= value <= right.raw:
            span = right.raw - left.raw
            if span <= 0:
                return right.score
            ratio = (value - left.raw) / span
            return left.score + ratio * (right.score - left.score)
    return points[-1].score


def map_coverage_score(score_pct: float, mapping: Iterable | None = None) -> float:
    """Map a coverage score expressed as a percentage (0~100) to user score.

    Convenience wrapper for callers that already multiplied by 100.
    """
    return map_raw_coverage(float(score_pct) / 100.0, mapping)


def resolve_mapping(scoring_cfg: dict | None) -> List[_Point]:
    """Return the configured mapping (or default) as a normalized list of points."""
    if not scoring_cfg:
        return _normalize_mapping(None)
    return _normalize_mapping(scoring_cfg.get("content_coverage_user_mapping"))


__all__ = ["map_raw_coverage", "map_coverage_score", "resolve_mapping", "DEFAULT_MAPPING"]
