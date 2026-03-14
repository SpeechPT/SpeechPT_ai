"""STT integration modules."""

from .whisper_transcriber import WhisperTranscriber, transcribe_audio

__all__ = ["WhisperTranscriber", "transcribe_audio"]
