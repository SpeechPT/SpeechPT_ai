"""Compute user-facing reliability level from raw alignment confidence.

Raw `alignment.confidence` is a cosine-similarity-derived internal signal that
is not calibrated to a 0~1 user score. Exposing the raw value to the UI or to
the LLM lets downstream consumers misread it as a percentage. This module
converts it into a 3-level category (low / medium / high) plus a boolean
``content_coverage_shown`` flag that the UI uses to decide whether to display
the content coverage score at all.

The thresholds live in config (``report.scoring.alignment_reliability``) so the
curve can be tuned without code changes once a larger evaluation set exists.
"""
from __future__ import annotations

from typing import Dict, Iterable

DEFAULT_LOW_MAX = 0.03
DEFAULT_MEDIUM_MAX = 0.05
DEFAULT_FORCE_LOW_WARNINGS: tuple[str, ...] = ("low_alignment_confidence",)
DEFAULT_DEMOTE_WARNINGS: tuple[str, ...] = ("robust_fallback_applied",)

LEVEL_NOTES = {
    "low": "분석 신뢰도가 낮아 내용 전달 점수는 참고용으로만 보세요.",
    "medium": "분석 신뢰도가 보통입니다. 점수는 대략의 추정으로 보세요.",
    "high": "",
}


def _demote(level: str) -> str:
    if level == "high":
        return "medium"
    if level == "medium":
        return "low"
    return "low"


def compute_alignment_reliability(
    confidence: float | None,
    warnings: Iterable[str] | None = None,
    config: Dict | None = None,
) -> Dict:
    """Return reliability metadata for a single run.

    Returns a dict with:
      - ``alignment_level``: "low" | "medium" | "high"
      - ``content_coverage_shown``: bool — UI hint
      - ``confidence_raw``: float — kept for debugging, not for user display
      - ``warnings``: list[str] — pass-through of alignment warnings
      - ``note``: short Korean string suitable for UI/LLM (empty for "high")
    """
    cfg = config or {}
    low_max = float(cfg.get("low_max", DEFAULT_LOW_MAX))
    medium_max = float(cfg.get("medium_max", DEFAULT_MEDIUM_MAX))
    force_low = set(cfg.get("force_low_warnings", DEFAULT_FORCE_LOW_WARNINGS))
    demote = set(cfg.get("demote_warnings", DEFAULT_DEMOTE_WARNINGS))

    warning_list = list(warnings or [])
    warning_set = set(warning_list)
    raw_confidence = float(confidence) if confidence is not None else 0.0

    if raw_confidence < low_max:
        level = "low"
    elif raw_confidence < medium_max:
        level = "medium"
    else:
        level = "high"

    if warning_set & force_low:
        level = "low"
    elif warning_set & demote:
        level = _demote(level)

    return {
        "alignment_level": level,
        "content_coverage_shown": level != "low",
        "confidence_raw": raw_confidence,
        "warnings": warning_list,
        "note": LEVEL_NOTES.get(level, ""),
    }


__all__ = [
    "compute_alignment_reliability",
    "LEVEL_NOTES",
    "DEFAULT_LOW_MAX",
    "DEFAULT_MEDIUM_MAX",
]
