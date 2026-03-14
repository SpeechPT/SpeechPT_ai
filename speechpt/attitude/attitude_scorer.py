"""발화 태도(속도/에너지/침묵) 분석 스코어러."""
from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Pattern, Sequence

import numpy as np

from .audio_feature_extractor import AudioFeatures
from .change_point_detector import ChangePoint

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class SegmentAttitude:
    slide_id: int
    start_sec: float
    end_sec: float
    features: Dict[str, float]
    change_points: List[ChangePoint]
    trend_label: str
    anomaly_flags: List[str]
    fillers: List[Dict]


def _compile_patterns(patterns: Sequence[str]) -> List[Pattern[str]]:
    return [re.compile(pattern) for pattern in patterns]


def filler_detector(words: Sequence[Dict], patterns: Sequence[Pattern[str]]) -> List[Dict]:
    hits = []
    for word_item in words:
        text = str(word_item.get("word", ""))
        for pattern in patterns:
            if pattern.fullmatch(text):
                hits.append({"word": text, "time_sec": float(word_item.get("start", 0.0))})
                break
    return hits


def _zscore_series(x: np.ndarray) -> tuple[float, float]:
    mu = float(np.nanmean(x))
    sigma = float(np.nanstd(x) + 1e-8)
    return mu, sigma


def _trend_label(times: np.ndarray, series: np.ndarray, key: str, eps: float) -> str:
    if len(series) < 2:
        return "stable"
    slope, _ = np.polyfit(times, series, 1)
    if key == "energy":
        if slope < -eps:
            return "declining_energy"
        if slope > eps:
            return "rising_energy"
    if key == "speech_rate_per_sec":
        if slope > eps:
            return "increasing_speed"
        if slope < -eps:
            return "decreasing_speed"
    return "stable"


def _attach_wav2vec_features(
    result: SegmentAttitude,
    wav2vec_embeddings: np.ndarray | None,
    wav2vec_times: np.ndarray | None,
) -> None:
    if wav2vec_embeddings is None or wav2vec_times is None or len(wav2vec_embeddings) == 0:
        return
    mask = (wav2vec_times >= result.start_sec) & (wav2vec_times <= result.end_sec)
    if not np.any(mask):
        return
    seg_emb = wav2vec_embeddings[mask]
    norms = np.linalg.norm(seg_emb, axis=1)
    result.features["wav2vec_norm_mean"] = float(np.mean(norms))
    result.features["wav2vec_norm_std"] = float(np.std(norms))


def score_attitude(
    audio_features: AudioFeatures,
    segments: Sequence[Dict],
    change_points: Sequence[ChangePoint],
    words: Sequence[Dict],
    config: Dict | None = None,
    wav2vec_embeddings: np.ndarray | None = None,
    wav2vec_times: np.ndarray | None = None,
) -> List[SegmentAttitude]:
    cfg = config or {}
    scoring_cfg = cfg.get("scoring", {})
    filler_patterns = scoring_cfg.get("filler_patterns", ["음", "어", "그", "아"])
    anomaly_thresh = float(scoring_cfg.get("anomaly_z_threshold", 1.5))
    trend_eps = float(scoring_cfg.get("trend_slope_eps", 1e-3))

    global_stats = {
        "energy": _zscore_series(audio_features.energy),
        "pitch": _zscore_series(audio_features.pitch),
        "speech_rate_per_sec": _zscore_series(audio_features.speech_rate_per_sec),
    }

    compiled_patterns = _compile_patterns(filler_patterns)
    fillers_all = filler_detector(words, compiled_patterns)

    results: List[SegmentAttitude] = []
    for segment in segments:
        start, end = float(segment["start_sec"]), float(segment["end_sec"])
        mask = (audio_features.frame_times >= start) & (audio_features.frame_times <= end)
        if not np.any(mask):
            continue

        def _mean_safe(arr: np.ndarray) -> float:
            sub = arr[mask]
            return float(np.nanmean(sub)) if len(sub) else 0.0

        features = {
            "avg_speech_rate": _mean_safe(audio_features.speech_rate_per_sec),
            "silence_ratio": float(np.mean(audio_features.silence_mask[mask])),
            "pitch_mean": _mean_safe(audio_features.pitch),
            "energy_mean": _mean_safe(audio_features.energy),
        }

        anomaly_flags: List[str] = []
        for key, seg_val in [
            ("energy", features["energy_mean"]),
            ("pitch", features["pitch_mean"]),
            ("speech_rate_per_sec", features["avg_speech_rate"]),
        ]:
            mu, sigma = global_stats[key]
            z = (seg_val - mu) / (sigma + 1e-8)
            if abs(z) > anomaly_thresh:
                anomaly_flags.append(f"{key}_z>{anomaly_thresh}")

        seg_times = audio_features.frame_times[mask]
        trend = _trend_label(seg_times, audio_features.energy[mask], "energy", trend_eps)
        if trend == "stable":
            trend = _trend_label(seg_times, audio_features.speech_rate_per_sec[mask], "speech_rate_per_sec", trend_eps)

        segment_cps = [cp for cp in change_points if start <= cp.time_sec <= end]
        segment_fillers = [f for f in fillers_all if start <= f["time_sec"] <= end]
        features["filler_count"] = len(segment_fillers)

        result = SegmentAttitude(
            slide_id=int(segment.get("slide_id", 0)),
            start_sec=start,
            end_sec=end,
            features=features,
            change_points=segment_cps,
            trend_label=trend,
            anomaly_flags=anomaly_flags,
            fillers=segment_fillers,
        )
        _attach_wav2vec_features(result, wav2vec_embeddings, wav2vec_times)
        results.append(result)
    return results


def main():
    parser = argparse.ArgumentParser(description="Score attitude segments")
    parser.add_argument("--features_json", required=True, help="AudioFeatures serialized as np lists (dev/demo only)")
    args = parser.parse_args()

    data = json.loads(Path(args.features_json).read_text())
    audio_feats = AudioFeatures(
        duration_sec=data["duration_sec"],
        pitch=np.array(data["pitch"]),
        energy=np.array(data["energy"]),
        speech_rate_per_sec=np.array(data["speech_rate_per_sec"]),
        silence_mask=np.array(data["silence_mask"]),
        frame_times=np.array(data["frame_times"]),
    )
    segments = [{"slide_id": 1, "start_sec": 0, "end_sec": audio_feats.frame_times[-1]}]
    result = score_attitude(audio_feats, segments, change_points=[], words=[], config={})
    for item in result:
        logger.info(item)


if __name__ == "__main__":
    main()
