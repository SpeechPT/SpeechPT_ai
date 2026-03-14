"""음성 신호로부터 기본 프로소디 피처를 추출한다."""
from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import librosa
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class AudioFeatures:
    duration_sec: float
    pitch: np.ndarray
    energy: np.ndarray
    speech_rate_per_sec: np.ndarray
    silence_mask: np.ndarray
    frame_times: np.ndarray


def _count_syllables(word: str) -> int:
    """간단한 한글 음절 카운터 (Hangul 코드블록 기준)."""
    return len(re.findall(r"[가-힣]", word))


def _compute_speech_rate(words: Sequence[Dict], frame_times: np.ndarray, window_sec: float = 1.0) -> np.ndarray:
    """Whisper 단어 타임스탬프를 사용해 초당 음절 수를 프레임 타임스탬프에 맞춰 계산."""
    if len(frame_times) == 0:
        return np.array([], dtype=float)
    half = window_sec / 2.0
    rates = np.zeros_like(frame_times, dtype=float)
    syllable_times: List[tuple[float, int]] = []
    for w in words:
        start = float(w.get("start", 0.0))
        syllable_times.append((start, _count_syllables(str(w.get("word", "")))))
    for i, t in enumerate(frame_times):
        total = 0
        for st, syllables in syllable_times:
            if t - half <= st <= t + half:
                total += syllables
        rates[i] = total / window_sec
    return rates


def extract_audio_features(audio_path: str | Path, words: Sequence[Dict] | None = None, config: Dict | None = None) -> AudioFeatures:
    cfg = config or {}
    audio_cfg = cfg.get("audio", {})
    silence_cfg = cfg.get("silence", {})
    pitch_cfg = cfg.get("pitch", {})

    sample_rate = int(audio_cfg.get("sample_rate", 16000))
    hop_length = int(audio_cfg.get("hop_length", 512))
    fmin = float(pitch_cfg.get("fmin", 50))
    fmax = float(pitch_cfg.get("fmax", 500))
    silence_thresh = float(silence_cfg.get("threshold_db", -40))

    y, sr = librosa.load(Path(audio_path), sr=sample_rate)
    duration = len(y) / sr

    # Pitch (F0)
    f0, _, _ = librosa.pyin(
        y,
        sr=sr,
        fmin=fmin,
        fmax=fmax,
        hop_length=hop_length,
    )
    pitch = np.nan_to_num(f0, nan=0.0)

    # Energy -> dB
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    energy_db = librosa.amplitude_to_db(rms + 1e-9, ref=np.max)

    frame_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
    silence_mask = energy_db < silence_thresh

    speech_rate = _compute_speech_rate(words or [], frame_times, window_sec=1.0)

    return AudioFeatures(
        duration_sec=float(duration),
        pitch=pitch,
        energy=energy_db,
        speech_rate_per_sec=speech_rate,
        silence_mask=silence_mask,
        frame_times=frame_times,
    )


def main():
    parser = argparse.ArgumentParser(description="Extract prosody features from audio")
    parser.add_argument("audio", type=str, help="wav/mp3 path")
    parser.add_argument("--words_json", type=str, default=None, help="Whisper words JSON (optional)")
    parser.add_argument("--config", type=str, default=None, help="config yaml/json (optional)")
    args = parser.parse_args()

    words: List[Dict] = []
    if args.words_json:
        words = json.loads(Path(args.words_json).read_text())

    feats = extract_audio_features(args.audio, words=words, config={})
    logger.info(
        "duration=%.2fs pitch_len=%d energy_len=%d speech_rate_len=%d",
        feats.duration_sec,
        len(feats.pitch),
        len(feats.energy),
        len(feats.speech_rate_per_sec),
    )


if __name__ == "__main__":
    main()
