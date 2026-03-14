"""Whisper-based STT wrapper with word-level timestamp output."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


@dataclass
class WhisperConfig:
    backend: str = "faster-whisper"  # faster-whisper | openai-whisper
    model_name: str = "small"
    language: str = "ko"
    compute_type: str = "int8"
    device: str = "cpu"


class WhisperTranscriber:
    def __init__(self, config: WhisperConfig):
        self.config = config

    def transcribe(self, audio_path: str | Path) -> Dict[str, Any]:
        path = str(Path(audio_path))
        if self.config.backend == "faster-whisper":
            return self._transcribe_faster_whisper(path)
        if self.config.backend == "openai-whisper":
            return self._transcribe_openai_whisper(path)
        raise ValueError(f"Unsupported STT backend: {self.config.backend}")

    def _transcribe_faster_whisper(self, audio_path: str) -> Dict[str, Any]:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "faster-whisper is not installed. Install with `pip install faster-whisper`."
            ) from exc

        model = WhisperModel(
            self.config.model_name,
            device=self.config.device,
            compute_type=self.config.compute_type,
        )
        segments, _ = model.transcribe(
            audio_path,
            language=self.config.language,
            word_timestamps=True,
            vad_filter=True,
        )

        words: List[Dict[str, Any]] = []
        texts: List[str] = []
        for segment in segments:
            texts.append(segment.text.strip())
            if segment.words is None:
                continue
            for word in segment.words:
                token = (word.word or "").strip()
                if not token:
                    continue
                words.append(
                    {
                        "word": token,
                        "start": float(word.start),
                        "end": float(word.end),
                        "probability": float(getattr(word, "probability", 0.0)),
                    }
                )

        return {"text": " ".join(texts).strip(), "words": words}

    def _transcribe_openai_whisper(self, audio_path: str) -> Dict[str, Any]:
        try:
            import whisper
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "openai-whisper is not installed. Install with `pip install openai-whisper`."
            ) from exc

        model = whisper.load_model(self.config.model_name)
        result = model.transcribe(audio_path, language=self.config.language, word_timestamps=True)

        words: List[Dict[str, Any]] = []
        for segment in result.get("segments", []):
            for word in segment.get("words", []):
                token = str(word.get("word", "")).strip()
                if not token:
                    continue
                words.append(
                    {
                        "word": token,
                        "start": float(word.get("start", 0.0)),
                        "end": float(word.get("end", 0.0)),
                        "probability": float(word.get("probability", 0.0)),
                    }
                )

        return {"text": str(result.get("text", "")).strip(), "words": words}


def transcribe_audio(audio_path: str | Path, config_dict: Dict[str, Any] | None = None) -> Dict[str, Any]:
    cfg = config_dict or {}
    whisper_cfg = WhisperConfig(
        backend=str(cfg.get("backend", "faster-whisper")),
        model_name=str(cfg.get("model_name", "small")),
        language=str(cfg.get("language", "ko")),
        compute_type=str(cfg.get("compute_type", "int8")),
        device=str(cfg.get("device", "cpu")),
    )
    logger.info(
        "Running STT backend=%s model=%s language=%s",
        whisper_cfg.backend,
        whisper_cfg.model_name,
        whisper_cfg.language,
    )
    transcriber = WhisperTranscriber(whisper_cfg)
    return transcriber.transcribe(audio_path)
