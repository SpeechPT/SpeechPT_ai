import numpy as np

from speechpt.attitude.attitude_scorer import score_attitude
from speechpt.attitude.audio_feature_extractor import AudioFeatures
from speechpt.attitude.change_point_detector import ChangePoint


def test_score_attitude_flags_anomalies_and_fillers():
    frame_times = np.array([0.0, 1.0, 2.0, 3.0])
    feats = AudioFeatures(
        duration_sec=3.0,
        pitch=np.array([100, 100, 80, 60], dtype=float),
        energy=np.array([0.0, -2.0, -6.0, -8.0], dtype=float),
        speech_rate_per_sec=np.array([3.0, 2.5, 2.0, 1.0]),
        silence_mask=np.array([False, False, False, True]),
        frame_times=frame_times,
    )
    cps = [ChangePoint(time_sec=2.0, type="energy_drop", magnitude=2.5)]
    segments = [{"slide_id": 1, "start_sec": 0.0, "end_sec": 3.0}]
    words = [
        {"word": "음", "start": 1.5, "end": 1.6},
        {"word": "내용", "start": 0.5, "end": 0.6},
    ]
    config = {"scoring": {"anomaly_z_threshold": 0.5, "filler_patterns": ["음", "어"]}}

    result = score_attitude(feats, segments, change_points=cps, words=words, config=config)
    assert len(result) == 1
    seg = result[0]
    assert isinstance(seg.anomaly_flags, list)
    assert seg.trend_label in {"declining_energy", "decreasing_speed"}
    assert seg.features["filler_count"] == 1
    assert len(seg.change_points) == 1
    assert seg.dwell_sec == 3.0
    assert seg.features["dwell_sec"] == 3.0
    assert seg.features["dwell_ratio"] == 1.0
    assert seg.features["dwell_z"] == 0.0
    assert seg.features["word_count"] == 2
    assert seg.word_count == 2
    assert seg.features["words_per_sec"] == 2 / 3.0


def test_score_attitude_adds_relative_dwell_features():
    frame_times = np.arange(0.0, 15.0, 1.0)
    feats = AudioFeatures(
        duration_sec=14.0,
        pitch=np.ones_like(frame_times),
        energy=np.ones_like(frame_times),
        speech_rate_per_sec=np.ones_like(frame_times),
        silence_mask=np.zeros_like(frame_times, dtype=bool),
        frame_times=frame_times,
    )
    segments = [
        {"slide_id": 1, "start_sec": 0.0, "end_sec": 2.0},
        {"slide_id": 2, "start_sec": 2.0, "end_sec": 6.0},
        {"slide_id": 3, "start_sec": 6.0, "end_sec": 14.0},
    ]
    words = [
        {"word": "하나", "start": 0.5, "end": 0.6},
        {"word": "둘", "start": 2.5, "end": 2.6},
        {"word": "셋", "start": 7.0, "end": 7.1},
        {"word": "넷", "start": 8.0, "end": 8.1},
    ]

    result = score_attitude(feats, segments, change_points=[], words=words, config={})

    assert [seg.features["dwell_sec"] for seg in result] == [2.0, 4.0, 8.0]
    assert np.isclose(sum(seg.features["dwell_ratio"] for seg in result), 1.0)
    assert result[0].features["dwell_z"] < 0
    assert result[-1].features["dwell_z"] > 0
    assert [seg.features["word_count"] for seg in result] == [1, 1, 2]


def test_score_attitude_uses_shared_filler_detector_patterns():
    frame_times = np.array([0.0, 1.0, 2.0, 3.0])
    feats = AudioFeatures(
        duration_sec=3.0,
        pitch=np.ones_like(frame_times),
        energy=np.ones_like(frame_times),
        speech_rate_per_sec=np.ones_like(frame_times),
        silence_mask=np.zeros_like(frame_times, dtype=bool),
        frame_times=frame_times,
    )
    segments = [{"slide_id": 1, "start_sec": 0.0, "end_sec": 3.0}]
    words = [
        {"word": "어어", "start": 0.5, "end": 0.7},
        {"word": "그러니까", "start": 1.0, "end": 1.4},
        {"word": "음악", "start": 1.8, "end": 2.0},
    ]

    result = score_attitude(feats, segments, change_points=[], words=words, config={})

    assert result[0].features["filler_count"] == 2
    assert [item["word"] for item in result[0].fillers] == ["어어", "그러니까"]
