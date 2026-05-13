"""Create compact calibration snapshots from SpeechPT report JSON files."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable


SNAPSHOT_SCHEMA_VERSION = "report_calibration_snapshot_v1"
CE_ISSUE_IDS = {"content_gap", "title_missing", "bullet_missing", "visual_not_explained"}


def _round_float(value: Any, ndigits: int = 4) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), ndigits)
    except (TypeError, ValueError):
        return None


def _count(items: Iterable[str]) -> Dict[str, int]:
    return dict(sorted(Counter(items).items()))


def _feedback_severity_counts(highlights: list[Dict[str, Any]]) -> Dict[str, int]:
    severities = []
    for highlight in highlights:
        for feedback in highlight.get("feedback", []):
            severity = feedback.get("severity")
            if severity:
                severities.append(str(severity))
    return _count(severities)


def _low_confidence_feedback_count(highlights: list[Dict[str, Any]]) -> int:
    count = 0
    for highlight in highlights:
        for feedback in highlight.get("feedback", []):
            text = str(feedback.get("text", ""))
            if "자동 매칭" in text or "신뢰도" in text:
                count += 1
    return count


def _issue_summary(highlights: list[Dict[str, Any]]) -> Dict[str, Any]:
    all_issues: list[str] = []
    ce_issues: list[str] = []
    ce_high_sections = 0
    for highlight in highlights:
        issues = [str(issue) for issue in highlight.get("issues", [])]
        all_issues.extend(issues)
        slide_ce_issues = [issue for issue in issues if issue in CE_ISSUE_IDS]
        ce_issues.extend(slide_ce_issues)
        if slide_ce_issues and int(highlight.get("severity", 0)) >= 6:
            ce_high_sections += 1
    return {
        "total": len(all_issues),
        "by_issue": _count(all_issues),
        "ce_total": len(ce_issues),
        "ce_by_issue": _count(ce_issues),
        "ce_high_section_count": ce_high_sections,
        "feedback_by_severity": _feedback_severity_counts(highlights),
        "low_confidence_feedback_count": _low_confidence_feedback_count(highlights),
    }


def _slide_summary(per_slide_detail: list[Dict[str, Any]]) -> Dict[str, Any]:
    role_counts: Counter[str] = Counter()
    coverage_values = []
    content_coverage_values = []
    per_slide = []
    for item in per_slide_detail:
        role = str(item.get("slide_role", "content"))
        weight = float(item.get("coverage_weight", 1.0) or 0.0)
        coverage = _round_float(item.get("coverage"))
        role_counts[role] += 1
        if coverage is not None:
            coverage_values.append(coverage)
            if weight > 0:
                content_coverage_values.append(coverage)
        per_slide.append(
            {
                "slide_id": item.get("slide_id"),
                "role": role,
                "coverage_weight": _round_float(weight),
                "coverage": coverage,
                "semantic_coverage": _round_float(item.get("semantic_coverage")),
                "keypoint_coverage": _round_float(item.get("keypoint_coverage")),
            }
        )
    return {
        "total": len(per_slide_detail),
        "roles": dict(sorted(role_counts.items())),
        "coverage_min": min(coverage_values) if coverage_values else None,
        "coverage_max": max(coverage_values) if coverage_values else None,
        "content_slide_count": len(content_coverage_values),
        "per_slide": per_slide,
    }


def _alignment_summary(alignment: Dict[str, Any]) -> Dict[str, Any]:
    boundaries = alignment.get("final_boundaries") or alignment.get("boundaries") or []
    warnings = alignment.get("warnings") or []
    return {
        "mode": alignment.get("mode"),
        "strategy_used": alignment.get("strategy_used"),
        "confidence": _round_float(alignment.get("confidence")),
        "warning_count": len(warnings),
        "warnings": warnings,
        "boundary_count": len(boundaries),
        "boundaries": [_round_float(value, 2) for value in boundaries],
    }


def _delta(new_value: Any, old_value: Any) -> float | None:
    new_num = _round_float(new_value)
    old_num = _round_float(old_value)
    if new_num is None or old_num is None:
        return None
    return round(new_num - old_num, 4)


def _nested_get(payload: Dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def compare_snapshots(current: Dict[str, Any], baseline: Dict[str, Any]) -> Dict[str, Any]:
    score_keys = sorted(set(current.get("overall_scores", {})) | set(baseline.get("overall_scores", {})))
    score_delta = {
        key: _delta(
            _nested_get(current, ("overall_scores", key)),
            _nested_get(baseline, ("overall_scores", key)),
        )
        for key in score_keys
    }
    issue_keys = sorted(
        set(current.get("issues", {}).get("by_issue", {}))
        | set(baseline.get("issues", {}).get("by_issue", {}))
    )
    issue_delta = {
        key: int(current.get("issues", {}).get("by_issue", {}).get(key, 0))
        - int(baseline.get("issues", {}).get("by_issue", {}).get(key, 0))
        for key in issue_keys
    }
    return {
        "baseline_case": baseline.get("case"),
        "overall_score_delta": score_delta,
        "issue_count_delta": issue_delta,
        "ce_total_delta": int(current.get("issues", {}).get("ce_total", 0))
        - int(baseline.get("issues", {}).get("ce_total", 0)),
        "ce_high_section_delta": int(current.get("issues", {}).get("ce_high_section_count", 0))
        - int(baseline.get("issues", {}).get("ce_high_section_count", 0)),
        "alignment_confidence_delta": _delta(
            _nested_get(current, ("alignment", "confidence")),
            _nested_get(baseline, ("alignment", "confidence")),
        ),
    }


def build_report_snapshot(
    report: Dict[str, Any],
    *,
    case: str | None = None,
    baseline: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    highlights = list(report.get("highlight_sections", []))
    per_slide_detail = list(report.get("per_slide_detail", []))
    overall_scores = report.get("overall_scores", {})
    snapshot = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "case": case or report.get("case") or "unnamed_case",
        "overall_scores": {key: _round_float(value, 2) for key, value in sorted(overall_scores.items())},
        "alignment": _alignment_summary(report.get("alignment", {})),
        "issues": _issue_summary(highlights),
        "slides": _slide_summary(per_slide_detail),
    }
    if baseline is not None:
        snapshot["comparison"] = compare_snapshots(snapshot, baseline)
    return snapshot


def load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a compact SpeechPT report calibration snapshot.")
    parser.add_argument("--report", required=True, help="Path to report.json produced by speechpt.pipeline")
    parser.add_argument("--case", default=None, help="Optional case name to store in the snapshot")
    parser.add_argument("--baseline", default=None, help="Optional previous snapshot JSON for delta comparison")
    parser.add_argument("--out", default=None, help="Output path. If omitted, prints to stdout.")
    args = parser.parse_args()

    baseline = load_json(args.baseline) if args.baseline else None
    snapshot = build_report_snapshot(load_json(args.report), case=args.case, baseline=baseline)
    text = json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
