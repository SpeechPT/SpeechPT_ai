import json
from pathlib import Path

import numpy as np
import pytest

from speechpt.attitude.attitude_scorer import SegmentAttitude
from speechpt.attitude.audio_feature_extractor import AudioFeatures
from speechpt.attitude.change_point_detector import ChangePoint
from speechpt.coherence.coherence_scorer import SlideCoherenceResult
from speechpt.coherence.document_parser import SlideContent
from speechpt.coherence.keypoint_extractor import Keypoint
from speechpt.coherence.transcript_aligner import TranscriptSegment
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
