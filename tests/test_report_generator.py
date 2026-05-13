from pathlib import Path

from speechpt.attitude.attitude_scorer import SegmentAttitude
from speechpt.attitude.change_point_detector import ChangePoint
from speechpt.coherence.coherence_scorer import SlideCoherenceResult
from speechpt.coherence.slide_role_classifier import SlideRole
from speechpt.report.report_generator import generate_report


def test_generate_report_has_expected_sections():
    ce_results = [
        SlideCoherenceResult(
            slide_id=1,
            coverage=0.8,
            missed_keypoints=["리스크"],
            evidence_spans=[],
            source_coverage={"title": 1.0, "bullet": 0.5},
            source_missed_keypoints={"title": [], "bullet": ["리스크"]},
        ),
        SlideCoherenceResult(
            slide_id=2,
            coverage=0.5,
            missed_keypoints=["Alpha 제어 핵심", "VISUAL: 매출 차트"],
            evidence_spans=[],
            source_coverage={"title": 0.0, "visual": 0.0},
            source_missed_keypoints={"title": ["Alpha 제어 핵심"], "visual": ["VISUAL: 매출 차트"]},
            semantic_coverage=0.2,
            keypoint_coverage=0.2,
        ),
    ]
    ae_results = [
        SegmentAttitude(
            slide_id=1,
            start_sec=0.0,
            end_sec=30.0,
            features={"avg_speech_rate": 3.0, "silence_ratio": 0.1, "pitch_mean": 210.0, "energy_mean": -15.0, "filler_count": 0},
            change_points=[ChangePoint(time_sec=12.0, type="energy_drop", magnitude=1.2)],
            trend_label="stable",
            anomaly_flags=[],
            fillers=[],
        ),
        SegmentAttitude(
            slide_id=2,
            start_sec=30.0,
            end_sec=60.0,
            features={"avg_speech_rate": 2.0, "silence_ratio": 0.4, "pitch_mean": 180.0, "energy_mean": -20.0, "filler_count": 3},
            change_points=[ChangePoint(time_sec=44.0, type="speed_drop", magnitude=2.0)],
            trend_label="decreasing_speed",
            anomaly_flags=["speech_rate_per_sec_z>1.5"],
            fillers=[{"word": "음", "time_sec": 40.0}],
        ),
    ]

    report = generate_report(
        ce_results=ce_results,
        ae_results=ae_results,
        template_path=Path("speechpt/report/templates/feedback_ko.yaml"),
        version="0.3.0",
        transcript_segments=[
            {
                "slide_id": 1,
                "start_sec": 0.0,
                "end_sec": 30.0,
                "text": "발표 도입입니다.",
                "words": [{"word": "발표", "start": 0.0, "end": 0.2}],
                "warning_flags": [],
            }
        ],
    )

    payload = report.to_dict()
    assert payload["version"] == "0.3.0"
    assert "overall_scores" in payload
    assert "highlight_sections" in payload
    assert payload["transcript_segments"][0]["slide_id"] == 1
    assert payload["transcript_segments"][0]["words"][0]["word"] == "발표"
    assert len(payload["per_slide_detail"]) == 2
    slide2 = [x for x in payload["per_slide_detail"] if x["slide_id"] == 2][0]
    assert "매출 차트" in " ".join(slide2["visual_missed"])
    assert slide2["source_coverage"]["visual"] == 0.0
    assert "VISUAL: 매출 차트" in slide2["source_missed"]["visual"]
    highlight2 = [x for x in payload["highlight_sections"] if x["slide_id"] == 2][0]
    assert "title_missing" in highlight2["issues"]
    assert any("핵심 제목/주제" in item["text"] for item in highlight2["feedback"])


def test_generate_report_adds_dwell_feedback():
    ce_results = [
        SlideCoherenceResult(slide_id=1, coverage=0.9, missed_keypoints=[], evidence_spans=[]),
        SlideCoherenceResult(slide_id=2, coverage=0.9, missed_keypoints=[], evidence_spans=[]),
    ]
    ae_results = [
        SegmentAttitude(
            slide_id=1,
            start_sec=0.0,
            end_sec=5.0,
            features={
                "avg_speech_rate": 3.0,
                "silence_ratio": 0.0,
                "pitch_mean": 200.0,
                "energy_mean": -10.0,
                "filler_count": 0,
                "dwell_sec": 5.0,
                "dwell_ratio": 0.1,
                "dwell_z": -2.0,
            },
            change_points=[],
            trend_label="stable",
            anomaly_flags=[],
            fillers=[],
        ),
        SegmentAttitude(
            slide_id=2,
            start_sec=5.0,
            end_sec=50.0,
            features={
                "avg_speech_rate": 3.0,
                "silence_ratio": 0.0,
                "pitch_mean": 200.0,
                "energy_mean": -10.0,
                "filler_count": 0,
                "dwell_sec": 45.0,
                "dwell_ratio": 0.9,
                "dwell_z": 2.5,
            },
            change_points=[],
            trend_label="stable",
            anomaly_flags=[],
            fillers=[],
        ),
    ]

    report = generate_report(
        ce_results=ce_results,
        ae_results=ae_results,
        template_path=Path("speechpt/report/templates/feedback_ko.yaml"),
        attitude_config={"scoring": {"dwell_short_z_threshold": -1.5, "dwell_long_z_threshold": 2.0}},
    )

    payload = report.to_dict()
    slide1 = [x for x in payload["highlight_sections"] if x["slide_id"] == 1][0]
    slide2 = [x for x in payload["highlight_sections"] if x["slide_id"] == 2][0]
    assert "dwell_short" in slide1["issues"]
    assert "dwell_long" in slide2["issues"]
    assert any("발표 전체의 10.0%" in item["text"] for item in slide1["feedback"])
    detail1 = [x for x in payload["per_slide_detail"] if x["slide_id"] == 1][0]
    assert detail1["dwell_sec"] == 5.0
    assert detail1["dwell_ratio"] == 0.1


def test_generate_report_uses_ae_probe_overall_delivery_for_delivery_stability():
    ce_results = [SlideCoherenceResult(slide_id=1, coverage=0.9, missed_keypoints=[], evidence_spans=[])]
    ae_results = [
        SegmentAttitude(
            slide_id=1,
            start_sec=0.0,
            end_sec=10.0,
            features={
                "avg_speech_rate": 3.0,
                "silence_ratio": 0.0,
                "filler_count": 0,
                "ae_probe_overall_delivery": 0.64,
            },
            change_points=[],
            trend_label="stable",
            anomaly_flags=[],
            fillers=[],
        ),
        SegmentAttitude(
            slide_id=2,
            start_sec=10.0,
            end_sec=20.0,
            features={
                "avg_speech_rate": 3.0,
                "silence_ratio": 0.0,
                "filler_count": 0,
                "ae_probe_overall_delivery": 0.84,
            },
            change_points=[],
            trend_label="stable",
            anomaly_flags=[],
            fillers=[],
        ),
    ]

    report = generate_report(
        ce_results=ce_results,
        ae_results=ae_results,
        template_path=Path("speechpt/report/templates/feedback_ko.yaml"),
    )

    assert report.to_dict()["overall_scores"]["delivery_stability"] == 74.0


def test_generate_report_softens_ce_feedback_when_alignment_confidence_is_low():
    ce_results = [
        SlideCoherenceResult(
            slide_id=1,
            coverage=0.2,
            missed_keypoints=["Alpha 핵심 주제"],
            evidence_spans=[],
            source_coverage={"title": 0.0},
            source_missed_keypoints={"title": ["Alpha 핵심 주제"]},
            keypoint_coverage=0.2,
        )
    ]

    report = generate_report(
        ce_results=ce_results,
        ae_results=[],
        template_path=Path("speechpt/report/templates/feedback_ko.yaml"),
        alignment={"confidence": 0.04},
        report_config={"scoring": {"low_alignment_confidence_threshold": 0.05}},
    )

    payload = report.to_dict()
    feedback = payload["highlight_sections"][0]["feedback"][0]
    assert feedback["severity"] == "low"
    assert "자동 매칭" in feedback["text"]


def test_generate_report_does_not_create_ce_issue_for_semantic_paraphrase_match():
    ce_results = [
        SlideCoherenceResult(
            slide_id=1,
            coverage=0.50,
            missed_keypoints=["표현이 다른 핵심 주장"],
            evidence_spans=[],
            source_coverage={"title": 0.0},
            source_missed_keypoints={"title": ["표현이 다른 핵심 주장"]},
            semantic_coverage=0.90,
            soft_keypoint_coverage=0.20,
            keypoint_coverage=0.20,
        )
    ]

    report = generate_report(
        ce_results=ce_results,
        ae_results=[],
        template_path=Path("speechpt/report/templates/feedback_ko.yaml"),
        report_config={"scoring": {"ce_issue_semantic_threshold": 0.50, "ce_issue_keypoint_threshold": 0.50}},
    )

    assert report.to_dict()["highlight_sections"] == []


def test_generate_report_ignores_generic_visual_miss():
    ce_results = [
        SlideCoherenceResult(
            slide_id=1,
            coverage=0.9,
            missed_keypoints=["VISUAL: chart_candidate x1", "VISUAL: image x2"],
            evidence_spans=[],
            source_coverage={"visual": 0.0},
            source_missed_keypoints={"visual": ["VISUAL: chart_candidate x1", "VISUAL: image x2"]},
            keypoint_coverage=0.9,
        )
    ]

    report = generate_report(
        ce_results=ce_results,
        ae_results=[],
        template_path=Path("speechpt/report/templates/feedback_ko.yaml"),
    )

    assert report.to_dict()["highlight_sections"] == []


def test_generate_report_suppresses_title_issue_for_non_content_slide_role():
    ce_results = [
        SlideCoherenceResult(
            slide_id=1,
            coverage=0.1,
            missed_keypoints=["표지 제목"],
            evidence_spans=[],
            source_coverage={"title": 0.0},
            source_missed_keypoints={"title": ["표지 제목"]},
            semantic_coverage=0.1,
            keypoint_coverage=0.1,
        ),
        SlideCoherenceResult(
            slide_id=2,
            coverage=0.1,
            missed_keypoints=["Alpha 내용 제목"],
            evidence_spans=[],
            source_coverage={"title": 0.0},
            source_missed_keypoints={"title": ["Alpha 내용 제목"]},
            semantic_coverage=0.1,
            keypoint_coverage=0.1,
        ),
    ]

    report = generate_report(
        ce_results=ce_results,
        ae_results=[],
        template_path=Path("speechpt/report/templates/feedback_ko.yaml"),
        slide_roles={
            1: SlideRole(
                slide_id=1,
                role="cover",
                source="test",
                reason="cover",
                coverage_weight=0.0,
                suppress_title_missing=True,
                suppress_content_issues=True,
            ),
            2: SlideRole(slide_id=2, role="content", source="test", reason="content"),
        },
    )

    payload = report.to_dict()
    highlights = {item["slide_id"]: item for item in payload["highlight_sections"]}
    assert 1 not in highlights
    assert "title_missing" in highlights[2]["issues"]
    assert payload["per_slide_detail"][0]["slide_role"] == "cover"
    assert payload["per_slide_detail"][0]["coverage_weight"] == 0.0


def test_generate_report_suppresses_generic_title_missing():
    ce_results = [
        SlideCoherenceResult(
            slide_id=1,
            coverage=0.1,
            missed_keypoints=["실험 결과"],
            evidence_spans=[],
            source_coverage={"title": 0.0},
            source_missed_keypoints={"title": ["실험 결과"]},
            semantic_coverage=0.1,
            keypoint_coverage=0.1,
        )
    ]

    report = generate_report(
        ce_results=ce_results,
        ae_results=[],
        template_path=Path("speechpt/report/templates/feedback_ko.yaml"),
    )

    assert report.to_dict()["highlight_sections"] == []


def test_generate_report_exposes_content_and_all_coverage():
    ce_results = [
        SlideCoherenceResult(slide_id=1, coverage=0.1, missed_keypoints=[], evidence_spans=[]),
        SlideCoherenceResult(slide_id=2, coverage=0.7, missed_keypoints=[], evidence_spans=[]),
        SlideCoherenceResult(slide_id=3, coverage=0.2, missed_keypoints=[], evidence_spans=[]),
    ]

    report = generate_report(
        ce_results=ce_results,
        ae_results=[],
        template_path=Path("speechpt/report/templates/feedback_ko.yaml"),
        slide_roles={
            1: SlideRole(slide_id=1, role="cover", source="test", reason="cover", coverage_weight=0.0),
            2: SlideRole(slide_id=2, role="content", source="test", reason="content", coverage_weight=1.0),
            3: SlideRole(slide_id=3, role="thanks", source="test", reason="thanks", coverage_weight=0.0),
        },
    )

    payload = report.to_dict()
    assert payload["overall_scores"]["content_coverage"] == 70.0
    assert payload["overall_scores"]["content_coverage_all"] == 33.33
    assert payload["overall_scores"]["content_scored_slide_count"] == 1
    assert payload["global_summary"]["avg_coverage"] == 70.0
    assert payload["global_summary"]["avg_coverage_all"] == 33.33
