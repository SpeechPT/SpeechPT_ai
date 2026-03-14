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
