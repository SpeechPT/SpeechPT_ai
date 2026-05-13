import json
from pathlib import Path

import numpy as np
import pytest

from speechpt.attitude.attitude_scorer import SegmentAttitude
from speechpt.attitude.audio_feature_extractor import AudioFeatures
from speechpt.attitude.change_point_detector import ChangePoint
from speechpt.coherence.auto_aligner import AlignmentResult
from speechpt.coherence.coherence_scorer import SlideCoherenceResult
from speechpt.coherence.document_parser import SlideContent
from speechpt.coherence.keypoint_extractor import Keypoint
from speechpt.coherence.transcript_aligner import TranscriptSegment
from speechpt.coherence.vlm_caption import VlmCaptionResult, VlmPresentationCaption, VlmSection, VlmSlideCaption
from speechpt.pipeline import SpeechPTPipeline


def _patch_core(monkeypatch):
    monkeypatch.setattr(
        "speechpt.pipeline.document_parser.parse_document",
        lambda _: [SlideContent(slide_id=1, text="내용", title="제목", bullet_points=["포인트"])],
    )
    monkeypatch.setattr(
        "speechpt.pipeline.keypoint_extractor.extract_keypoints",
        lambda _: [Keypoint(text="포인트", importance=1.0, source="title")],
    )
    monkeypatch.setattr(
        "speechpt.pipeline.transcript_aligner.align_transcript",
        lambda words, times: [TranscriptSegment(slide_id=1, start_sec=0.0, end_sec=5.0, text="포인트", words=words)],
    )
    monkeypatch.setattr(
        "speechpt.pipeline.coherence_scorer.score_slide",
        lambda *args, **kwargs: SlideCoherenceResult(slide_id=1, coverage=0.9, missed_keypoints=[], evidence_spans=[]),
    )
    monkeypatch.setattr(
        "speechpt.pipeline.extract_audio_features",
        lambda *args, **kwargs: AudioFeatures(
            duration_sec=5.0,
            pitch=[100.0, 110.0],
            energy=[-10.0, -11.0],
            speech_rate_per_sec=[3.0, 3.1],
            silence_mask=[False, False],
            frame_times=[0.0, 5.0],
        ),
    )
    monkeypatch.setattr(
        "speechpt.pipeline.detect_change_points",
        lambda *args, **kwargs: [ChangePoint(time_sec=2.0, type="energy_drop", magnitude=1.0)],
    )
    monkeypatch.setattr(
        "speechpt.pipeline.score_attitude",
        lambda *args, **kwargs: [
            SegmentAttitude(
                slide_id=1,
                start_sec=0.0,
                end_sec=5.0,
                features={
                    "avg_speech_rate": 3.0,
                    "silence_ratio": 0.1,
                    "pitch_mean": 105.0,
                    "energy_mean": -10.5,
                    "filler_count": 0,
                },
                change_points=[ChangePoint(time_sec=2.0, type="energy_drop", magnitude=1.0)],
                trend_label="stable",
                anomaly_flags=[],
                fillers=[],
            )
        ],
    )


def test_pipeline_smoke_with_whisper_json(monkeypatch, tmp_path: Path):
    cfg = {
        "version": "0.3.0",
        "coherence": {"model_name": "dummy", "threshold": 0.55},
        "attitude": {"wav2vec2": {"use_probe": False}},
        "stt": {"enabled": False},
        "report": {"template": "speechpt/report/templates/feedback_ko.yaml"},
    }
    cfg_path = tmp_path / "pipeline.yaml"
    cfg_path.write_text(json.dumps(cfg))

    _patch_core(monkeypatch)

    pipeline = SpeechPTPipeline(str(cfg_path))
    report = pipeline.analyze(
        document_path="dummy.pdf",
        audio_path="dummy.wav",
        slide_timestamps=[0.0, 5.0],
        whisper_result={"words": [{"word": "포인트", "start": 0.0, "end": 0.2}]},
    )
    assert report.overall_scores["content_coverage"] >= 0.0


def test_pipeline_auto_stt_when_enabled(monkeypatch, tmp_path: Path):
    cfg = {
        "version": "0.3.0",
        "coherence": {"model_name": "dummy", "threshold": 0.55},
        "attitude": {"wav2vec2": {"use_probe": False}},
        "stt": {"enabled": True, "backend": "faster-whisper", "model_name": "small"},
        "report": {"template": "speechpt/report/templates/feedback_ko.yaml"},
    }
    cfg_path = tmp_path / "pipeline.yaml"
    cfg_path.write_text(json.dumps(cfg))

    _patch_core(monkeypatch)
    monkeypatch.setattr("speechpt.pipeline.transcribe_audio", lambda *args, **kwargs: {"words": [{"word": "테스트", "start": 0.0, "end": 0.4}]})

    pipeline = SpeechPTPipeline(str(cfg_path))
    report = pipeline.analyze(
        document_path="dummy.pdf",
        audio_path="dummy.wav",
        slide_timestamps=[0.0, 5.0],
        whisper_result=None,
    )
    assert report.overall_scores["content_coverage"] >= 0.0


def test_pipeline_uses_wav2vec_when_enabled(monkeypatch, tmp_path: Path):
    cfg = {
        "version": "0.3.0",
        "coherence": {"model_name": "dummy", "threshold": 0.55},
        "attitude": {
            "audio": {"sample_rate": 16000},
            "wav2vec2": {"use_probe": True, "model_name": "facebook/wav2vec2-base", "chunk_duration_sec": 30},
        },
        "stt": {"enabled": False},
        "report": {"template": "speechpt/report/templates/feedback_ko.yaml"},
    }
    cfg_path = tmp_path / "pipeline.yaml"
    cfg_path.write_text(json.dumps(cfg))

    _patch_core(monkeypatch)

    class DummyEmbedder:
        def __init__(self, *args, **kwargs):
            pass

        def encode_with_times(self, *args, **kwargs):
            return np.ones((10, 768), dtype=float), np.linspace(0.0, 5.0, num=10, endpoint=False)

    monkeypatch.setattr("speechpt.pipeline.Wav2Vec2Embedder", DummyEmbedder)

    pipeline = SpeechPTPipeline(str(cfg_path))
    report = pipeline.analyze(
        document_path="dummy.pdf",
        audio_path="dummy.wav",
        slide_timestamps=[0.0, 5.0],
        whisper_result={"words": [{"word": "포인트", "start": 0.0, "end": 0.2}]},
    )
    assert report.overall_scores["content_coverage"] >= 0.0


def test_pipeline_requires_words_if_stt_disabled(tmp_path: Path):
    cfg = {
        "version": "0.3.0",
        "coherence": {"model_name": "dummy", "threshold": 0.55},
        "attitude": {"wav2vec2": {"use_probe": False}},
        "stt": {"enabled": False},
        "report": {"template": "speechpt/report/templates/feedback_ko.yaml"},
    }
    cfg_path = tmp_path / "pipeline.yaml"
    cfg_path.write_text(json.dumps(cfg))

    pipeline = SpeechPTPipeline(str(cfg_path))
    with pytest.raises(ValueError):
        pipeline._resolve_whisper_result("dummy.wav", whisper_result=None)


def test_pipeline_runtime_stt_override_enables_transcription(monkeypatch, tmp_path: Path):
    cfg = {
        "version": "0.3.0",
        "coherence": {"model_name": "dummy", "threshold": 0.55},
        "attitude": {"wav2vec2": {"use_probe": False}},
        "stt": {"enabled": False, "backend": "faster-whisper", "model_name": "small"},
        "report": {"template": "speechpt/report/templates/feedback_ko.yaml"},
    }
    cfg_path = tmp_path / "pipeline.yaml"
    cfg_path.write_text(json.dumps(cfg))

    pipeline = SpeechPTPipeline(str(cfg_path))
    pipeline.apply_runtime_overrides({"stt": {"enabled": True, "model_name": "base"}})

    monkeypatch.setattr(
        "speechpt.pipeline.transcribe_audio",
        lambda *args, **kwargs: {"words": [{"word": "테스트", "start": 0.0, "end": 0.2}], "text": "테스트"},
    )

    result = pipeline._resolve_whisper_result("dummy.wav", whisper_result=None)
    assert pipeline.stt_cfg["enabled"] is True
    assert pipeline.stt_cfg["model_name"] == "base"
    assert result["words"][0]["word"] == "테스트"


def test_pipeline_runtime_vlm_override_enables_high_quality(tmp_path: Path):
    cfg = {
        "version": "0.3.0",
        "coherence": {
            "model_name": "dummy",
            "threshold": 0.55,
            "vlm_caption": {
                "enabled": False,
                "model": "gpt-4.1-mini",
                "detail": "low",
                "cache_enabled": True,
            },
        },
        "attitude": {"wav2vec2": {"use_probe": False}},
        "stt": {"enabled": False},
        "report": {"template": "speechpt/report/templates/feedback_ko.yaml"},
    }
    cfg_path = tmp_path / "pipeline.yaml"
    cfg_path.write_text(json.dumps(cfg))

    pipeline = SpeechPTPipeline(str(cfg_path))
    pipeline.apply_runtime_overrides(
        {
            "vlm_caption": {
                "enabled": True,
                "model": "gpt-4.1",
                "detail": "high",
                "cache_enabled": False,
                "dpi": 180,
                "timeout_sec": 120.0,
            }
        }
    )

    vlm_cfg = pipeline.ce_cfg["vlm_caption"]
    assert vlm_cfg["enabled"] is True
    assert vlm_cfg["model"] == "gpt-4.1"
    assert vlm_cfg["detail"] == "high"
    assert vlm_cfg["cache_enabled"] is False
    assert vlm_cfg["dpi"] == 180
    assert vlm_cfg["timeout_sec"] == 120.0


def test_pipeline_keeps_all_slides_when_middle_segment_is_empty(monkeypatch, tmp_path: Path):
    cfg = {
        "version": "0.3.0",
        "coherence": {"model_name": "dummy", "threshold": 0.55},
        "attitude": {"wav2vec2": {"use_probe": False}},
        "stt": {"enabled": False},
        "report": {"template": "speechpt/report/templates/feedback_ko.yaml"},
    }
    cfg_path = tmp_path / "pipeline.yaml"
    cfg_path.write_text(json.dumps(cfg))

    monkeypatch.setattr(
        "speechpt.pipeline.document_parser.parse_document",
        lambda _: [
            SlideContent(slide_id=1, text="내용1", title="제목1", bullet_points=["포인트1"]),
            SlideContent(slide_id=2, text="내용2", title="제목2", bullet_points=["포인트2"]),
            SlideContent(slide_id=3, text="내용3", title="제목3", bullet_points=["포인트3"]),
        ],
    )
    monkeypatch.setattr(
        "speechpt.pipeline.keypoint_extractor.extract_keypoints",
        lambda slide: [Keypoint(text=slide.title, importance=1.0, source="title")],
    )
    monkeypatch.setattr(
        "speechpt.pipeline.transcript_aligner.align_transcript",
        lambda words, times: [
            TranscriptSegment(slide_id=1, start_sec=0.0, end_sec=5.0, text="첫 슬라이드", words=words[:1]),
            TranscriptSegment(slide_id=2, start_sec=5.0, end_sec=10.0, text="", words=[], warning_flags=["empty_segment"]),
            TranscriptSegment(slide_id=3, start_sec=10.0, end_sec=15.0, text="셋째 슬라이드", words=words[1:]),
        ],
    )
    monkeypatch.setattr(
        "speechpt.pipeline.coherence_scorer.score_slide",
        lambda keypoints, segment, **kwargs: SlideCoherenceResult(
            slide_id=segment.slide_id,
            coverage=0.0 if not segment.text else 0.9,
            missed_keypoints=[] if segment.text else [kp.text for kp in keypoints],
            evidence_spans=[],
        ),
    )
    monkeypatch.setattr(
        "speechpt.pipeline.extract_audio_features",
        lambda *args, **kwargs: AudioFeatures(
            duration_sec=15.0,
            pitch=[100.0, 110.0],
            energy=[-10.0, -11.0],
            speech_rate_per_sec=[3.0, 3.1],
            silence_mask=[False, False],
            frame_times=[0.0, 15.0],
        ),
    )
    monkeypatch.setattr("speechpt.pipeline.detect_change_points", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        "speechpt.pipeline.score_attitude",
        lambda *args, **kwargs: [
            SegmentAttitude(slide_id=1, start_sec=0.0, end_sec=5.0, features={"avg_speech_rate": 3.0, "silence_ratio": 0.1}, change_points=[], trend_label="stable", anomaly_flags=[], fillers=[]),
            SegmentAttitude(slide_id=2, start_sec=5.0, end_sec=10.0, features={"avg_speech_rate": 3.0, "silence_ratio": 0.1}, change_points=[], trend_label="stable", anomaly_flags=[], fillers=[]),
            SegmentAttitude(slide_id=3, start_sec=10.0, end_sec=15.0, features={"avg_speech_rate": 3.0, "silence_ratio": 0.1}, change_points=[], trend_label="stable", anomaly_flags=[], fillers=[]),
        ],
    )

    pipeline = SpeechPTPipeline(str(cfg_path))
    report = pipeline.analyze(
        document_path="dummy.pdf",
        audio_path="dummy.wav",
        slide_timestamps=[0.0, 5.0, 10.0, 15.0],
        whisper_result={"words": [{"word": "첫", "start": 0.0, "end": 0.2}, {"word": "셋째", "start": 10.1, "end": 10.3}]},
    )

    assert len(report.per_slide_detail) == 3
    assert report.per_slide_detail[1]["slide_id"] == 2


def test_pipeline_hybrid_alignment_includes_alignment_metadata(monkeypatch, tmp_path: Path):
    cfg = {
        "version": "0.3.0",
        "coherence": {
            "model_name": "dummy",
            "threshold": 0.55,
            "alignment": {"mode": "hybrid"},
        },
        "attitude": {"wav2vec2": {"use_probe": False}},
        "stt": {"enabled": False},
        "report": {"template": "speechpt/report/templates/feedback_ko.yaml"},
    }
    cfg_path = tmp_path / "pipeline.yaml"
    cfg_path.write_text(json.dumps(cfg))

    _patch_core(monkeypatch)
    monkeypatch.setattr(
        "speechpt.pipeline.auto_aligner.resolve_alignment",
        lambda **kwargs: AlignmentResult(
            mode="hybrid",
            strategy_used="manual_with_auto_proposal",
            final_boundaries=[0.0, 5.0],
            provided_boundaries=[0.0, 5.0],
            proposed_boundaries=[0.0, 4.2],
            confidence=0.82,
            unit_assignments=[{"unit_id": 0, "slide_id": 1, "start_sec": 0.0, "end_sec": 4.2}],
        ),
    )

    pipeline = SpeechPTPipeline(str(cfg_path))
    report = pipeline.analyze(
        document_path="dummy.pdf",
        audio_path="dummy.wav",
        slide_timestamps=[0.0, 5.0],
        whisper_result={"words": [{"word": "포인트", "start": 0.0, "end": 0.2}]},
    )

    assert report.alignment["mode"] == "hybrid"
    assert report.alignment["strategy_used"] == "manual_with_auto_proposal"
    assert report.alignment["proposed_boundaries"] == [0.0, 4.2]


def test_pipeline_uses_vlm_captions_for_alignment_only(monkeypatch, tmp_path: Path):
    cfg = {
        "version": "0.3.0",
        "coherence": {
            "model_name": "dummy",
            "threshold": 0.55,
            "alignment": {"mode": "auto"},
            "vlm_caption": {"enabled": True},
        },
        "attitude": {"wav2vec2": {"use_probe": False}},
        "stt": {"enabled": False},
        "report": {"template": "speechpt/report/templates/feedback_ko.yaml"},
    }
    cfg_path = tmp_path / "pipeline.yaml"
    cfg_path.write_text(json.dumps(cfg))

    monkeypatch.setattr(
        "speechpt.pipeline.document_parser.parse_document",
        lambda _: [SlideContent(slide_id=1, text="내용", title="제목", bullet_points=["포인트"])],
    )
    monkeypatch.setattr(
        "speechpt.pipeline.keypoint_extractor.extract_keypoints",
        lambda _: [Keypoint(text="포인트", importance=1.0, source="title")],
    )
    monkeypatch.setattr(
        "speechpt.pipeline.vlm_caption.caption_document",
        lambda *args, **kwargs: VlmCaptionResult(
            presentation=VlmPresentationCaption(
                topic="테스트",
                core_terminology=["알파"],
                sections=[VlmSection(name="전체", slide_indices=[1])],
                model="mock-model",
            ),
            slides=[
                VlmSlideCaption(
                    slide_id=1,
                    slide_type="chart",
                    role_in_flow="결과를 설명한다.",
                    main_claim="성능이 개선되었다.",
                    visual_kind="line_chart",
                    visual_summary="결과 그래프",
                    entities=["알파"],
                    likely_keywords_in_speech=["알파", "결과"],
                    model="mock-model",
                )
            ],
        ),
    )

    captured_alignment_sources = []

    def fake_resolve_alignment(**kwargs):
        captured_alignment_sources.extend(kp.source for kp in kwargs["slide_keypoints"][0])
        return AlignmentResult(
            mode="auto",
            strategy_used="auto",
            final_boundaries=[0.0, 5.0],
            proposed_boundaries=[0.0, 5.0],
        )

    captured_scoring_sources = []

    def fake_score_slide(keypoints, segment, **kwargs):
        captured_scoring_sources.extend(kp.source for kp in keypoints)
        return SlideCoherenceResult(slide_id=segment.slide_id, coverage=0.9, missed_keypoints=[], evidence_spans=[])

    monkeypatch.setattr("speechpt.pipeline.auto_aligner.resolve_alignment", fake_resolve_alignment)
    monkeypatch.setattr(
        "speechpt.pipeline.transcript_aligner.align_transcript",
        lambda words, times: [TranscriptSegment(slide_id=1, start_sec=0.0, end_sec=5.0, text="포인트", words=words)],
    )
    monkeypatch.setattr("speechpt.pipeline.coherence_scorer.score_slide", fake_score_slide)
    monkeypatch.setattr(
        "speechpt.pipeline.extract_audio_features",
        lambda *args, **kwargs: AudioFeatures(
            duration_sec=5.0,
            pitch=[100.0],
            energy=[-10.0],
            speech_rate_per_sec=[3.0],
            silence_mask=[False],
            frame_times=[0.0],
        ),
    )
    monkeypatch.setattr("speechpt.pipeline.detect_change_points", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        "speechpt.pipeline.score_attitude",
        lambda *args, **kwargs: [
            SegmentAttitude(
                slide_id=1,
                start_sec=0.0,
                end_sec=5.0,
                features={"avg_speech_rate": 3.0, "silence_ratio": 0.0},
                change_points=[],
                trend_label="stable",
                anomaly_flags=[],
                fillers=[],
            )
        ],
    )

    pipeline = SpeechPTPipeline(str(cfg_path))
    pipeline.analyze(
        document_path="dummy.pdf",
        audio_path="dummy.wav",
        whisper_result={"words": [{"word": "포인트", "start": 0.0, "end": 0.2}]},
    )

    assert "vlm_claim" in captured_alignment_sources
    assert "vlm_visual" in captured_alignment_sources
    assert "vlm_role" in captured_alignment_sources
    assert not any(source.startswith("vlm_") for source in captured_scoring_sources)


def test_pipeline_attaches_ae_probe_predictions(monkeypatch, tmp_path: Path):
    cfg = {
        "version": "0.3.0",
        "coherence": {"model_name": "dummy", "threshold": 0.55},
        "attitude": {"ae_probe": {"enabled": True, "model_dir": "models/ae_probe_artifact"}},
        "stt": {"enabled": False},
        "report": {"template": "speechpt/report/templates/feedback_ko.yaml"},
    }
    cfg_path = tmp_path / "pipeline.yaml"
    cfg_path.write_text(json.dumps(cfg))

    _patch_core(monkeypatch)

    class DummyPrediction:
        slide_id = 1

        def to_feature_dict(self):
            return {
                "ae_probe_speech_rate": 1.23,
                "ae_probe_silence_ratio": 0.12,
                "ae_probe_overall_delivery": 0.78,
            }

    captured_segments = []

    def fake_predict(audio_path, segments, config):
        captured_segments.extend(segments)
        return [DummyPrediction()]

    monkeypatch.setattr("speechpt.pipeline.predict_ae_probe_segments", fake_predict)

    pipeline = SpeechPTPipeline(str(cfg_path))
    report = pipeline.analyze(
        document_path="dummy.pdf",
        audio_path="dummy.wav",
        slide_timestamps=[0.0, 5.0],
        whisper_result={"words": [{"word": "포인트", "start": 0.0, "end": 0.2}]},
    )

    assert captured_segments == [{"slide_id": 1, "start_sec": 0.0, "end_sec": 5.0}]
    assert report.per_slide_detail[0]["ae_probe"]["ae_probe_speech_rate"] == 1.23
    assert report.per_slide_detail[0]["ae_probe"]["ae_probe_overall_delivery"] == 0.78
    assert report.overall_scores["delivery_stability"] == 78.0
