from pathlib import Path

from speechpt.attitude.attitude_scorer import SegmentAttitude
from speechpt.attitude.change_point_detector import ChangePoint
from speechpt.coherence.coherence_scorer import SlideCoherenceResult
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
            missed_keypoints=["결론", "VISUAL: 매출 차트"],
            evidence_spans=[],
            source_coverage={"title": 0.0, "visual": 0.0},
            source_missed_keypoints={"title": ["결론"], "visual": ["VISUAL: 매출 차트"]},
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
    )

    payload = report.to_dict()
    assert payload["version"] == "0.3.0"
    assert "overall_scores" in payload
    assert "highlight_sections" in payload
    assert len(payload["per_slide_detail"]) == 2
    slide2 = [x for x in payload["per_slide_detail"] if x["slide_id"] == 2][0]
    assert "매출 차트" in " ".join(slide2["visual_missed"])
    assert slide2["source_coverage"]["visual"] == 0.0
    assert "VISUAL: 매출 차트" in slide2["source_missed"]["visual"]
    highlight2 = [x for x in payload["highlight_sections"] if x["slide_id"] == 2][0]
    assert "title_missing" in highlight2["issues"]
    assert any("핵심 제목/주제" in item["text"] for item in highlight2["feedback"])
