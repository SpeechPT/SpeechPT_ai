import json
import subprocess
import sys
from pathlib import Path

from speechpt.report.snapshot import build_report_snapshot, compare_snapshots


def _sample_report():
    return {
        "overall_scores": {
            "content_coverage": 28.11,
            "content_coverage_all": 24.98,
            "delivery_stability": 74.0,
            "pacing_score": 91.93,
        },
        "alignment": {
            "mode": "hybrid",
            "strategy_used": "auto_without_manual_override",
            "confidence": 0.05174408786891734,
            "warnings": ["low_confidence"],
            "final_boundaries": [0.0, 10.77, 22.27],
        },
        "highlight_sections": [
            {
                "slide_id": 2,
                "severity": 3,
                "issues": ["content_gap", "pitch_shift"],
                "feedback": [
                    {
                        "template_id": "content_gap",
                        "text": "자동 매칭 신뢰도가 낮아 일부 포인트가 약하게 측정되었습니다.",
                        "severity": "low",
                    },
                    {"template_id": "pitch_shift", "text": "음 높이 변화가 큽니다.", "severity": "low"},
                ],
            },
            {
                "slide_id": 3,
                "severity": 6,
                "issues": ["title_missing", "visual_not_explained"],
                "feedback": [
                    {"template_id": "title_missing", "text": "제목이 약하게 잡혔습니다.", "severity": "medium"},
                    {"template_id": "visual_not_explained", "text": "시각자료가 약합니다.", "severity": "medium"},
                ],
            },
        ],
        "per_slide_detail": [
            {
                "slide_id": 1,
                "slide_role": "cover",
                "coverage_weight": 0.0,
                "coverage": 0.20,
                "semantic_coverage": 0.15,
                "keypoint_coverage": 0.0,
            },
            {
                "slide_id": 2,
                "slide_role": "content",
                "coverage_weight": 1.0,
                "coverage": 0.42,
                "semantic_coverage": 0.49,
                "keypoint_coverage": 0.16,
            },
            {
                "slide_id": 3,
                "slide_role": "thanks",
                "coverage_weight": 0.0,
                "coverage": 0.08,
                "semantic_coverage": 0.03,
                "keypoint_coverage": 0.0,
            },
        ],
    }


def test_build_report_snapshot_extracts_calibration_metrics():
    snapshot = build_report_snapshot(_sample_report(), case="controlv")

    assert snapshot["schema_version"] == "report_calibration_snapshot_v1"
    assert snapshot["case"] == "controlv"
    assert snapshot["overall_scores"]["content_coverage"] == 28.11
    assert snapshot["alignment"]["confidence"] == 0.0517
    assert snapshot["alignment"]["boundary_count"] == 3
    assert snapshot["issues"]["ce_total"] == 3
    assert snapshot["issues"]["ce_by_issue"] == {
        "content_gap": 1,
        "title_missing": 1,
        "visual_not_explained": 1,
    }
    assert snapshot["issues"]["ce_high_section_count"] == 1
    assert snapshot["issues"]["low_confidence_feedback_count"] == 1
    assert snapshot["slides"]["roles"] == {"content": 1, "cover": 1, "thanks": 1}
    assert snapshot["slides"]["content_slide_count"] == 1


def test_compare_snapshots_reports_deltas():
    baseline = build_report_snapshot(_sample_report(), case="baseline")
    current_report = _sample_report()
    current_report["overall_scores"]["content_coverage"] = 35.0
    current_report["highlight_sections"] = current_report["highlight_sections"][:1]
    current = build_report_snapshot(current_report, case="current")

    comparison = compare_snapshots(current, baseline)

    assert comparison["overall_score_delta"]["content_coverage"] == 6.89
    assert comparison["issue_count_delta"]["title_missing"] == -1
    assert comparison["ce_total_delta"] == -2


def test_snapshot_cli_writes_output(tmp_path: Path):
    report_path = tmp_path / "report.json"
    output_path = tmp_path / "snapshot.json"
    report_path.write_text(json.dumps(_sample_report(), ensure_ascii=False), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "speechpt.report.snapshot",
            "--report",
            str(report_path),
            "--case",
            "cli_case",
            "--out",
            str(output_path),
        ],
        check=True,
    )

    snapshot = json.loads(output_path.read_text(encoding="utf-8"))
    assert snapshot["case"] == "cli_case"
    assert snapshot["issues"]["total"] == 4
