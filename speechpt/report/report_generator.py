"""CE/AE 결과를 종합해 최종 SpeechPT 리포트를 생성한다."""
from __future__ import annotations

import argparse
import math
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import yaml

from speechpt.attitude.attitude_scorer import SegmentAttitude
from speechpt.coherence.coherence_scorer import SlideCoherenceResult
from speechpt.coherence.slide_role_classifier import SlideRole
from speechpt.report.reliability import compute_alignment_reliability
from speechpt.report.score_mapping import map_coverage_score

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class SpeechReport:
    version: str
    overall_scores: Dict
    highlight_sections: List[Dict]
    per_slide_detail: List[Dict]
    global_summary: Dict
    alignment: Dict = field(default_factory=dict)
    transcript_segments: List[Dict] = field(default_factory=list)
    llm_feedback: Dict = field(default_factory=dict)
    reliability: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


ISSUE_WEIGHT = {
    "content_gap": 3,
    "title_missing": 3,
    "bullet_missing": 2,
    "speed_drop": 2,
    "silence_excess": 1,
    "visual_not_explained": 2,
    "dwell_short": 1,
    "dwell_long": 1,
}
DEFAULT_TEMPLATE_TEXT = "슬라이드 {slide_id} 구간에 개선 포인트가 있습니다."
CE_ISSUE_IDS = {"content_gap", "title_missing", "bullet_missing", "visual_not_explained"}
CONTENT_ISSUE_IDS = CE_ISSUE_IDS
PUBLIC_AE_PROBE_FIELDS = {
    "ae_probe_speech_rate",
    "ae_probe_silence_ratio",
    "ae_probe_overall_delivery",
}
GENERIC_VISUAL_PATTERN = re.compile(
    r"^(chart_candidate|image|table|diagram|screenshot|bar_chart|line_chart|flowchart|none)(\s*x?\d+)?$",
    re.IGNORECASE,
)
GENERIC_TITLE_TERMS = {
    "개요",
    "목차",
    "소개",
    "팀소개",
    "팀 소개",
    "프로젝트소개",
    "프로젝트 소개",
    "진행현황",
    "진행 현황",
    "실험결과",
    "실험 결과",
    "결론",
    "향후계획",
    "향후 계획",
    "기대효과",
    "기대 효과",
    "감사합니다",
}


def _load_templates(path: Path) -> Dict:
    return yaml.safe_load(path.read_text())


def _select_template(template_id: str, templates: List[Dict]) -> Dict | None:
    for item in templates:
        if item.get("id") == template_id:
            return item
    return None


def _coverage_score_all(ce_results: Sequence[SlideCoherenceResult]) -> float:
    if not ce_results:
        return 0.0
    return float(np.mean([c.coverage for c in ce_results]) * 100)


def _coverage_weight(slide_id: int, slide_roles: Dict[int, SlideRole] | None, scoring_cfg: Dict | None = None) -> float:
    if not slide_roles or slide_id not in slide_roles:
        return 1.0
    role = slide_roles[slide_id]
    role_weights = (scoring_cfg or {}).get("role_coverage_weights", {})
    return float(role_weights.get(role.role, role.coverage_weight))


def _coverage_score(
    ce_results: Sequence[SlideCoherenceResult],
    slide_roles: Dict[int, SlideRole] | None = None,
    scoring_cfg: Dict | None = None,
) -> float:
    if not ce_results:
        return 0.0
    weighted_values = []
    weights = []
    for result in ce_results:
        weight = _coverage_weight(result.slide_id, slide_roles, scoring_cfg)
        if weight <= 0:
            continue
        weighted_values.append(float(result.coverage) * weight)
        weights.append(weight)
    if not weights:
        return _coverage_score_all(ce_results)
    return float((sum(weighted_values) / sum(weights)) * 100)


def _coverage_scored_slide_count(
    ce_results: Sequence[SlideCoherenceResult],
    slide_roles: Dict[int, SlideRole] | None = None,
    scoring_cfg: Dict | None = None,
) -> int:
    return sum(1 for result in ce_results if _coverage_weight(result.slide_id, slide_roles, scoring_cfg) > 0)


def _keypoint_coverage(ce: SlideCoherenceResult | None) -> float:
    if ce is None:
        return 0.0
    value = getattr(ce, "keypoint_coverage", None)
    if value is None:
        return float(ce.coverage)
    return float(value)


def _semantic_coverage(ce: SlideCoherenceResult | None) -> float:
    if ce is None:
        return 0.0
    value = getattr(ce, "semantic_coverage", None)
    if value is None:
        return float(ce.coverage)
    return float(value)


def _range_plateau_score(value: float, zero_min: float, full_min: float, full_max: float, zero_max: float) -> float:
    if full_min <= value <= full_max:
        return 1.0
    if value < full_min:
        if full_min <= zero_min:
            return 0.0
        return float(np.clip((value - zero_min) / (full_min - zero_min), 0.0, 1.0))
    if zero_max <= full_max:
        return 0.0
    return float(np.clip((zero_max - value) / (zero_max - full_max), 0.0, 1.0))


def _density_score(value: float, full_max: float, zero_at: float) -> float:
    if value <= full_max:
        return 1.0
    if zero_at <= full_max:
        return 0.0
    return float(np.clip((zero_at - value) / (zero_at - full_max), 0.0, 1.0))


def _delivery_stability_components(
    ae_results: Sequence[SegmentAttitude],
    scoring_cfg: Dict | None = None,
) -> Dict[str, float]:
    if not ae_results:
        return {}
    cfg = (scoring_cfg or {}).get("delivery_stability", {})
    silence_cfg = cfg.get("silence", {})
    rate_cfg = cfg.get("speech_rate", {})
    filler_cfg = cfg.get("filler", {})
    dwell_cfg = cfg.get("dwell", {})

    silence_scores = [
        _range_plateau_score(
            float(seg.features.get("silence_ratio", 0.0)),
            float(silence_cfg.get("zero_min", 0.0)),
            float(silence_cfg.get("full_min", 0.05)),
            float(silence_cfg.get("full_max", 0.40)),
            float(silence_cfg.get("zero_max", 0.70)),
        )
        for seg in ae_results
        if seg.features
    ]
    rate_scores = []
    for seg in ae_results:
        if not seg.features:
            continue
        rate = seg.features.get("words_per_sec", seg.features.get("avg_speech_rate", 0.0))
        rate_scores.append(
            _range_plateau_score(
                float(rate),
                float(rate_cfg.get("zero_min", 0.5)),
                float(rate_cfg.get("full_min", 2.0)),
                float(rate_cfg.get("full_max", 5.0)),
                float(rate_cfg.get("zero_max", 7.0)),
            )
        )

    filler_scores = []
    for seg in ae_results:
        if not seg.features:
            continue
        word_count = float(seg.features.get("word_count", 0.0))
        if word_count <= 0:
            filler_scores.append(1.0)
            continue
        density = float(seg.features.get("filler_count", 0.0)) / max(word_count, 1.0)
        filler_scores.append(
            _density_score(
                density,
                float(filler_cfg.get("full_max", 0.02)),
                float(filler_cfg.get("zero_at", 0.05)),
            )
        )

    dwell_scores = [
        _density_score(
            abs(float(seg.features.get("dwell_z", 0.0))),
            float(dwell_cfg.get("full_abs_z", 1.5)),
            float(dwell_cfg.get("zero_abs_z", 3.0)),
        )
        for seg in ae_results
        if seg.features
    ]
    probe_scores = [
        float(seg.features["ae_probe_overall_delivery"])
        for seg in ae_results
        if seg.features and "ae_probe_overall_delivery" in seg.features
    ]
    anomaly_scores = [
        1.0 - min(1.0, len(seg.anomaly_flags) / 3.0)
        for seg in ae_results
    ]

    components: Dict[str, float] = {}
    if silence_scores:
        components["silence"] = float(np.mean(silence_scores))
    if rate_scores:
        components["speech_rate"] = float(np.mean(rate_scores))
    if filler_scores:
        components["filler"] = float(np.mean(filler_scores))
    if dwell_scores:
        components["dwell"] = float(np.mean(dwell_scores))
    if probe_scores:
        components["probe"] = float(np.mean(probe_scores))
    if anomaly_scores:
        components["anomaly_fallback"] = float(np.mean(anomaly_scores))
    return components


def _delivery_stability(ae_results: Sequence[SegmentAttitude], scoring_cfg: Dict | None = None) -> float:
    if not ae_results:
        return 50.0
    cfg = (scoring_cfg or {}).get("delivery_stability", {})
    weights = {
        "silence": float(cfg.get("silence_weight", 0.35)),
        "speech_rate": float(cfg.get("speech_rate_weight", 0.25)),
        "filler": float(cfg.get("filler_weight", 0.15)),
        "dwell": float(cfg.get("dwell_weight", 0.10)),
        "probe": float(cfg.get("probe_weight", 0.15)),
    }
    components = _delivery_stability_components(ae_results, scoring_cfg)
    weighted_sum = 0.0
    weight_sum = 0.0
    for key, weight in weights.items():
        if weight <= 0 or key not in components:
            continue
        weighted_sum += components[key] * weight
        weight_sum += weight

    if weight_sum <= 0 and "anomaly_fallback" in components:
        return float(max(0.0, min(100.0, components["anomaly_fallback"] * 100.0)))
    if weight_sum <= 0:
        return 50.0
    return float(max(0.0, min(100.0, (weighted_sum / weight_sum) * 100.0)))


def _pacing_score(ae_results: Sequence[SegmentAttitude]) -> float:
    rates = [seg.features.get("avg_speech_rate", 0.0) for seg in ae_results if seg.features]
    if len(rates) < 2:
        return 70.0
    mean_rate = np.mean(rates)
    if mean_rate <= 1e-6:
        return 50.0
    cv = np.std(rates) / (mean_rate + 1e-8)
    return float(max(0.0, 100.0 * (1.0 - min(cv, 1.0))))


def _split_missed_keypoints(ce: SlideCoherenceResult | None) -> tuple[List[str], List[str]]:
    if ce is None:
        return [], []
    visual_missed = []
    text_missed = []
    for keypoint in ce.missed_keypoints:
        if keypoint.startswith("VISUAL: "):
            visual_missed.append(keypoint.replace("VISUAL: ", "", 1))
        else:
            text_missed.append(keypoint)
    return text_missed, visual_missed


def _is_generic_visual_miss(item: str) -> bool:
    normalized = item.replace("VISUAL: ", "", 1).strip()
    return bool(GENERIC_VISUAL_PATTERN.fullmatch(normalized))


def _is_actionable_title_miss(item: str) -> bool:
    normalized = " ".join(item.strip().split())
    compact = normalized.replace(" ", "")
    if not compact:
        return False
    if ":" in normalized or any(ch.isascii() and ch.isalnum() for ch in normalized):
        return True
    if normalized in GENERIC_TITLE_TERMS or compact in {term.replace(" ", "") for term in GENERIC_TITLE_TERMS}:
        return False
    if any(term.replace(" ", "") in compact for term in GENERIC_TITLE_TERMS) and len(compact) <= 12:
        return False
    return len(compact) >= 12


def _is_low_alignment_confidence(alignment: Dict | None, scoring_cfg: Dict) -> bool:
    if not alignment:
        return False
    confidence = alignment.get("confidence")
    if confidence is None:
        return False
    threshold = float(scoring_cfg.get("low_alignment_confidence_threshold", 0.06))
    return float(confidence) < threshold


def _mark_low_confidence(issue: Dict, low_confidence: bool) -> Dict:
    if low_confidence and issue.get("id") in CE_ISSUE_IDS:
        issue = dict(issue)
        issue["low_confidence"] = True
    return issue


def _pitch_change_score(ae: SegmentAttitude) -> float:
    score = 0.0
    for cp in ae.change_points:
        if cp.type.startswith("pitch"):
            score += abs(float(getattr(cp, "magnitude", 1.0) or 1.0))
    return score


def _pitch_issue_slide_ids(ae_results: Sequence[SegmentAttitude], scoring_cfg: Dict) -> set[int]:
    scores = {ae.slide_id: _pitch_change_score(ae) for ae in ae_results}
    scores = {slide_id: score for slide_id, score in scores.items() if score > 0}
    if not scores:
        return set()
    max_fraction = float(scoring_cfg.get("pitch_issue_max_fraction", 0.30))
    max_count = max(1, math.ceil(len(ae_results) * max_fraction))
    min_score = float(scoring_cfg.get("pitch_issue_min_score", 0.0))
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return {slide_id for slide_id, score in ranked[:max_count] if score > min_score}


def _public_ae_probe_features(features: Dict) -> Dict:
    return {key: value for key, value in features.items() if key in PUBLIC_AE_PROBE_FIELDS}


def _issue_from_slide(
    ce: SlideCoherenceResult | None,
    ae: SegmentAttitude | None,
    config: Dict | None = None,
    alignment: Dict | None = None,
    slide_role: SlideRole | None = None,
    pitch_issue_slide_ids: set[int] | None = None,
) -> List[Dict]:
    cfg = config or {}
    scoring_cfg = cfg.get("scoring", cfg)
    dwell_short_z = float(scoring_cfg.get("dwell_short_z_threshold", -1.5))
    dwell_long_z = float(scoring_cfg.get("dwell_long_z_threshold", 2.0))
    ce_semantic_threshold = float(scoring_cfg.get("ce_issue_semantic_threshold", 0.50))
    ce_keypoint_threshold = float(scoring_cfg.get("ce_issue_keypoint_threshold", 0.50))
    low_alignment_confidence = _is_low_alignment_confidence(alignment, scoring_cfg)
    issues: List[Dict] = []
    text_missed, visual_missed = _split_missed_keypoints(ce)
    source_missed = ce.source_missed_keypoints if ce and ce.source_missed_keypoints else {}
    title_missed = source_missed.get("title", [])
    bullet_missed = source_missed.get("bullet", [])
    actionable_title_missed = [item for item in title_missed if _is_actionable_title_miss(item)]
    non_actionable_title_missed = set(title_missed) - set(actionable_title_missed)
    text_missed_for_content = [item for item in text_missed if item not in non_actionable_title_missed]
    content_issue_allowed = bool(
        ce
        and _semantic_coverage(ce) < ce_semantic_threshold
        and _keypoint_coverage(ce) <= ce_keypoint_threshold
    )
    suppress_title_missing = bool(slide_role and slide_role.suppress_title_missing)
    suppress_content_issues = bool(slide_role and slide_role.suppress_content_issues)

    if actionable_title_missed and content_issue_allowed and not suppress_title_missing:
        issues.append(_mark_low_confidence({"id": "title_missing", "missed": actionable_title_missed}, low_alignment_confidence))
    if content_issue_allowed and bullet_missed and not suppress_content_issues:
        issues.append(_mark_low_confidence({"id": "bullet_missing", "missed": bullet_missed}, low_alignment_confidence))
    elif content_issue_allowed and text_missed_for_content and not title_missed and not suppress_content_issues:
        issues.append(_mark_low_confidence({"id": "content_gap", "missed": text_missed_for_content}, low_alignment_confidence))
    actionable_visual_missed = [item for item in visual_missed if not _is_generic_visual_miss(item)]
    if actionable_visual_missed and not suppress_content_issues:
        issues.append(
            _mark_low_confidence(
                {"id": "visual_not_explained", "visual_missed": actionable_visual_missed},
                low_alignment_confidence,
            )
        )

    if ae:
        silence = ae.features.get("silence_ratio", 0.0)
        if silence > 0.3:
            issues.append({"id": "silence_excess", "silence_duration": f"{silence:.2f}"})
        if ae.trend_label in {"decreasing_speed", "speed_drop"}:
            issues.append({"id": "speed_drop", "rate_change": ae.trend_label})
        if ae.trend_label in {"increasing_speed", "speed_rise"}:
            issues.append({"id": "speed_rise", "rate_change": ae.trend_label})
        if ae.features.get("filler_count", 0) >= 3:
            issues.append({"id": "filler_many", "filler_count": ae.features.get("filler_count", 0)})
        if any(cp.type.startswith("pitch") for cp in ae.change_points) and (
            pitch_issue_slide_ids is None or ae.slide_id in pitch_issue_slide_ids
        ):
            issues.append({"id": "pitch_shift"})
        dwell_z = float(ae.features.get("dwell_z", 0.0))
        dwell_ratio_pct = f"{float(ae.features.get('dwell_ratio', 0.0)) * 100:.1f}"
        if dwell_z < dwell_short_z:
            issues.append({"id": "dwell_short", "dwell_ratio_pct": dwell_ratio_pct})
        elif dwell_z > dwell_long_z:
            issues.append({"id": "dwell_long", "dwell_ratio_pct": dwell_ratio_pct})
    return issues


def _severity_score(issues: List[Dict]) -> int:
    score = 0
    for issue in issues:
        issue_id = issue["id"]
        weight = ISSUE_WEIGHT.get(issue_id, 1)
        if issue.get("low_confidence") and issue_id in CE_ISSUE_IDS:
            weight = max(1, weight - 2)
        score += weight
    return score


def _template_id_for_issue(issue_id: str, ce: SlideCoherenceResult | None) -> str:
    if issue_id == "content_gap":
        coverage = ce.coverage if ce else 1.0
        if coverage < 0.4:
            return "content_gap_high"
        if coverage < 0.6:
            return "content_gap_mid"
        return "content_gap_low"
    mapping = {
        "title_missing": "title_missing",
        "bullet_missing": "bullet_missing",
        "speed_drop": "speed_drop",
        "speed_rise": "speed_rise",
        "silence_excess": "silence_excess",
        "filler_many": "filler_many",
        "pitch_shift": "pitch_shift",
        "visual_not_explained": "visual_not_explained",
        "dwell_short": "dwell_short",
        "dwell_long": "dwell_long",
    }
    return mapping.get(issue_id, "pacing_inconsistent")


def _render_feedback(template: Dict | None, slide_id: int, issue_payload: Dict) -> Dict:
    template_id = issue_payload["id"]
    safe_payload = {k: v for k, v in issue_payload.items() if k != "id"}
    text_template = template["text"] if template and "text" in template else DEFAULT_TEMPLATE_TEXT
    return {
        "template_id": template_id,
        "text": text_template.format(slide_id=slide_id, **safe_payload),
        "severity": template.get("severity", "low") if template else "low",
    }


def generate_report(
    ce_results: Sequence[SlideCoherenceResult],
    ae_results: Sequence[SegmentAttitude],
    template_path: str | Path,
    version: str = "0.3.0",
    alignment: Dict | None = None,
    attitude_config: Dict | None = None,
    report_config: Dict | None = None,
    transcript_segments: Sequence[Dict] | None = None,
    slide_roles: Dict[int, SlideRole] | None = None,
) -> SpeechReport:
    templates = _load_templates(Path(template_path))
    issue_templates = templates.get("issue_templates", [])
    summary_templates = templates.get("summary_templates", [])
    report_scoring_cfg = {
        **((attitude_config or {}).get("scoring", {})),
        **((report_config or {}).get("scoring", {})),
    }

    ae_map = {item.slide_id: item for item in ae_results}
    ce_map = {item.slide_id: item for item in ce_results}
    role_map = slide_roles or {}
    pitch_issue_ids = _pitch_issue_slide_ids(ae_results, report_scoring_cfg)

    per_slide: List[Dict] = []
    highlights: List[Dict] = []

    all_rates: List[float] = []
    all_silence: List[float] = []
    total_change_points = 0

    for slide_id in sorted(set(list(ae_map.keys()) + list(ce_map.keys()))):
        ce = ce_map.get(slide_id)
        ae = ae_map.get(slide_id)
        slide_role = role_map.get(slide_id)
        issues = _issue_from_slide(
            ce,
            ae,
            config=report_scoring_cfg,
            alignment=alignment,
            slide_role=slide_role,
            pitch_issue_slide_ids=pitch_issue_ids,
        )
        issue_ids = [issue["id"] for issue in issues]
        severity = _severity_score(issues)

        feedbacks: List[Dict] = []
        for issue in issues:
            tid = _template_id_for_issue(issue["id"], ce)
            if issue.get("low_confidence"):
                low_confidence_tid = f"{tid}_low_confidence"
                if _select_template(low_confidence_tid, issue_templates):
                    tid = low_confidence_tid
            feedbacks.append(_render_feedback(_select_template(tid, issue_templates), slide_id, issue))

        if severity > 0:
            highlights.append(
                {
                    "slide_id": slide_id,
                    "severity": severity,
                    "issues": issue_ids,
                    "feedback": feedbacks,
                }
            )

        if ae:
            all_rates.append(ae.features.get("avg_speech_rate", 0.0))
            all_silence.append(ae.features.get("silence_ratio", 0.0))
            total_change_points += len(ae.change_points)

        text_missed, visual_missed = _split_missed_keypoints(ce)
        per_slide.append(
            {
                "slide_id": slide_id,
                "slide_role": slide_role.role if slide_role else "content",
                "slide_role_source": slide_role.source if slide_role else "default",
                "slide_role_reason": slide_role.reason if slide_role else "no_role_metadata",
                "coverage_weight": _coverage_weight(slide_id, role_map, report_scoring_cfg),
                "coverage": ce.coverage if ce else None,
                "semantic_coverage": ce.semantic_coverage if ce else None,
                "soft_keypoint_coverage": ce.soft_keypoint_coverage if ce else None,
                "keypoint_coverage": ce.keypoint_coverage if ce else None,
                "source_coverage": ce.source_coverage if ce and ce.source_coverage else {},
                "missed": text_missed,
                "visual_missed": visual_missed,
                "source_missed": ce.source_missed_keypoints if ce and ce.source_missed_keypoints else {},
                "speech_rate": ae.features.get("avg_speech_rate", None) if ae else None,
                "silence_ratio": ae.features.get("silence_ratio", None) if ae else None,
                "dwell_sec": ae.features.get("dwell_sec", None) if ae else None,
                "dwell_ratio": ae.features.get("dwell_ratio", None) if ae else None,
                "dwell_z": ae.features.get("dwell_z", None) if ae else None,
                "word_count": ae.features.get("word_count", None) if ae else None,
                "words_per_sec": ae.features.get("words_per_sec", None) if ae else None,
                "delivery_stability": round(_delivery_stability([ae], report_scoring_cfg), 2) if ae else None,
                "ae_probe": _public_ae_probe_features(ae.features) if ae else {},
                "trend": ae.trend_label if ae else None,
                "change_points": [cp.time_sec for cp in (ae.change_points if ae else [])],
                "anomalies": ae.anomaly_flags if ae else [],
            }
        )

    highlights = sorted(highlights, key=lambda item: item["severity"], reverse=True)

    coverage_pct = _coverage_score(ce_results, role_map, report_scoring_cfg)
    coverage_pct_all = _coverage_score_all(ce_results)
    mapping_cfg = report_scoring_cfg.get("content_coverage_user_mapping")
    alignment_obj = alignment or {}
    reliability = compute_alignment_reliability(
        alignment_obj.get("confidence"),
        alignment_obj.get("warnings", []),
        report_scoring_cfg.get("alignment_reliability"),
    )
    overall_scores = {
        "content_coverage": round(coverage_pct, 2),
        "content_coverage_all": round(coverage_pct_all, 2),
        "content_coverage_user": round(map_coverage_score(coverage_pct, mapping_cfg), 2),
        "content_coverage_user_all": round(map_coverage_score(coverage_pct_all, mapping_cfg), 2),
        "content_scored_slide_count": _coverage_scored_slide_count(ce_results, role_map, report_scoring_cfg),
        "delivery_stability": round(_delivery_stability(ae_results, report_scoring_cfg), 2),
        "pacing_score": round(_pacing_score(ae_results), 2),
    }

    summary_choice = "summary_mid"
    if overall_scores["content_coverage"] > 80 and overall_scores["delivery_stability"] > 70:
        summary_choice = "summary_high"
    elif overall_scores["content_coverage"] < 60 or overall_scores["delivery_stability"] < 50:
        summary_choice = "summary_low"

    summary_template = _select_template(summary_choice, summary_templates) or {"text": "요약 템플릿이 없습니다."}

    global_summary = {
        "total_slides": len(per_slide),
        "avg_coverage": overall_scores["content_coverage"],
        "avg_coverage_all": overall_scores["content_coverage_all"],
        "avg_coverage_user": overall_scores["content_coverage_user"],
        "avg_coverage_user_all": overall_scores["content_coverage_user_all"],
        "content_scored_slide_count": overall_scores["content_scored_slide_count"],
        "mean_speech_rate": float(np.mean(all_rates)) if all_rates else None,
        "mean_silence_ratio": float(np.mean(all_silence)) if all_silence else None,
        "total_change_points": int(total_change_points),
        "delivery_stability_components": {
            key: round(value * 100.0, 2)
            for key, value in _delivery_stability_components(ae_results, report_scoring_cfg).items()
            if key != "anomaly_fallback"
        },
        "summary_text": summary_template.get("text", ""),
        "alignment_level": reliability.get("alignment_level"),
        "content_coverage_shown": reliability.get("content_coverage_shown"),
    }

    return SpeechReport(
        version=version,
        overall_scores=overall_scores,
        highlight_sections=highlights,
        per_slide_detail=per_slide,
        global_summary=global_summary,
        alignment=alignment or {},
        transcript_segments=list(transcript_segments or []),
        reliability=reliability,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ce", default="examples/example_ce_output.json")
    parser.add_argument("--ae", default="examples/example_ae_output.json")
    parser.add_argument("--tpl", default="speechpt/report/templates/feedback_ko.yaml")
    args = parser.parse_args()

    ce_res: List[SlideCoherenceResult] = []
    ae_res: List[SegmentAttitude] = []

    try:
        ce_data = json.loads(Path(args.ce).read_text())
        for item in ce_data:
            ce_res.append(SlideCoherenceResult(**item))
    except Exception as exc:
        logger.warning("CE load failed: %s", exc)

    try:
        from speechpt.attitude.change_point_detector import ChangePoint

        ae_data = json.loads(Path(args.ae).read_text())
        for item in ae_data:
            cps = [ChangePoint(**cp) for cp in item.get("change_points", [])]
            ae_res.append(
                SegmentAttitude(
                    slide_id=item["slide_id"],
                    start_sec=item["start_sec"],
                    end_sec=item["end_sec"],
                    features=item["features"],
                    change_points=cps,
                    trend_label=item["trend_label"],
                    anomaly_flags=item.get("anomaly_flags", []),
                    fillers=item.get("fillers", []),
                )
            )
    except Exception as exc:
        logger.warning("AE load failed: %s", exc)

    report = generate_report(ce_res, ae_res, args.tpl)
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
