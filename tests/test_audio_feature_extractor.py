import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from speechpt.attitude.audio_feature_extractor import extract_audio_features


def _make_tone_with_silence(path: Path, sr: int = 16000):
    t = np.linspace(0, 1.0, int(sr * 1.0), endpoint=False)
    tone = 0.1 * np.sin(2 * np.pi * 440 * t)
    silence = np.zeros(int(sr * 0.5))
    wav = np.concatenate([tone, silence])
    sf.write(path, wav, sr)


def test_extract_audio_features_basic():
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        _make_tone_with_silence(Path(tmp.name))
        words = [{"word": "안녕", "start": 0.1, "end": 0.3}, {"word": "세계", "start": 0.5, "end": 0.7}]
        feats = extract_audio_features(tmp.name, words=words, config={"audio": {"sample_rate": 16000, "hop_length": 512}})

    assert feats.duration_sec == pytest.approx(1.5, rel=0.05)
    assert np.nanmedian(feats.pitch) == pytest.approx(440, rel=0.1)
    assert feats.speech_rate_per_sec.max() > 0
    # 마지막 구간은 침묵이므로 침묵 마스크가 일부 True
    assert feats.silence_mask[-10:].mean() > 0.5
    Path(tmp.name).unlink()
