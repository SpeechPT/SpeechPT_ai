"""Wav2Vec2 임베딩 추출 (추론 전용)."""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import librosa
import numpy as np
import torch
from transformers import Wav2Vec2Model, Wav2Vec2Processor


class Wav2Vec2Embedder:
    def __init__(self, model_name: str = "facebook/wav2vec2-base", chunk_duration_sec: float = 30.0):
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.chunk_duration_sec = chunk_duration_sec

    def _encode_chunk(self, audio: np.ndarray, sr: int) -> np.ndarray:
        inputs = self.processor(audio, sampling_rate=sr, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**inputs).last_hidden_state  # (1, T, 768)
        return out.cpu().numpy()[0]

    def __call__(self, audio_path: str | Path, sample_rate: int = 16000) -> np.ndarray:
        embeddings, _ = self.encode_with_times(audio_path, sample_rate=sample_rate)
        return embeddings

    def encode_with_times(self, audio_path: str | Path, sample_rate: int = 16000) -> Tuple[np.ndarray, np.ndarray]:
        y, sr = librosa.load(Path(audio_path), sr=sample_rate)
        chunk_size = int(self.chunk_duration_sec * sr)
        embeddings: List[np.ndarray] = []
        for start in range(0, len(y), chunk_size):
            chunk = y[start : start + chunk_size]
            emb = self._encode_chunk(chunk, sr)
            embeddings.append(emb)

        all_embs = np.concatenate(embeddings, axis=0)
        duration_sec = len(y) / sr if sr > 0 else 0.0
        frame_times = np.linspace(0.0, duration_sec, num=len(all_embs), endpoint=False)
        return all_embs, frame_times


if __name__ == "__main__":
    import argparse
    import logging

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("audio")
    args = parser.parse_args()
    embedder = Wav2Vec2Embedder()
    embs, times = embedder.encode_with_times(args.audio)
    print("emb shape", embs.shape, "time shape", times.shape)
