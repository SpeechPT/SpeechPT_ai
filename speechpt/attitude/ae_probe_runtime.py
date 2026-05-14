"""Inference-only runtime for the trained AE probe."""
from __future__ import annotations

import logging
import tarfile
from dataclasses import dataclass
from pathlib import Path
from posixpath import normpath
from typing import Dict, Sequence

import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import PeftModel
from transformers import Wav2Vec2Model, Wav2Vec2Processor

logger = logging.getLogger(__name__)
ARTIFACT_URI_MARKER = ".model_artifact_s3"


@dataclass(frozen=True)
class AEProbePrediction:
    slide_id: int
    start_sec: float
    end_sec: float
    speech_rate: float
    silence_ratio: float
    energy_drop: int
    energy_drop_prob: float
    pitch_shift: int
    pitch_shift_prob: float
    overall_delivery: float

    def to_feature_dict(self) -> Dict[str, float]:
        return {
            "ae_probe_speech_rate": self.speech_rate,
            "ae_probe_silence_ratio": self.silence_ratio,
            "ae_probe_energy_drop": float(self.energy_drop),
            "ae_probe_energy_drop_prob": self.energy_drop_prob,
            "ae_probe_pitch_shift": float(self.pitch_shift),
            "ae_probe_pitch_shift_prob": self.pitch_shift_prob,
            "ae_probe_overall_delivery": self.overall_delivery,
        }


class _AEProbe(nn.Module):
    def __init__(self, in_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 256)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(256, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"Invalid S3 URI: {uri}")
    body = uri[5:]
    bucket, _, key = body.partition("/")
    if not bucket or not key:
        raise ValueError(f"Invalid S3 URI: {uri}")
    return bucket, key


def _is_allowed_artifact_member(name: str) -> bool:
    normalized = normpath(name)
    if normalized.startswith("../") or normalized.startswith("/") or normalized == ".":
        return False
    return normalized in {"ae_probe.pt", "meta.pt"} or normalized.startswith("lora_adapter/")


def _clear_model_files(model_dir: Path) -> None:
    for path in [model_dir / "ae_probe.pt", model_dir / "meta.pt", model_dir / "model.tar.gz"]:
        if path.exists():
            path.unlink()
    lora_dir = model_dir / "lora_adapter"
    if lora_dir.exists():
        import shutil

        shutil.rmtree(lora_dir)


def _extract_probe_from_tar(tar_path: Path, model_dir: Path) -> Path:
    probe_path = model_dir / "ae_probe.pt"
    extracted_probe = False
    with tarfile.open(tar_path, "r:gz") as tf:
        for member in tf.getmembers():
            if not member.isfile() or not _is_allowed_artifact_member(member.name):
                continue
            normalized = normpath(member.name)
            source = tf.extractfile(member)
            if source is None:
                continue
            target = model_dir / normalized
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read())
            if normalized == "ae_probe.pt":
                extracted_probe = True
    if not extracted_probe:
        raise FileNotFoundError(f"ae_probe.pt not found in {tar_path}")
    return probe_path


def resolve_probe_path(config: Dict) -> Path:
    model_dir = Path(str(config.get("model_dir", "models/ae_probe_artifact")))
    model_dir.mkdir(parents=True, exist_ok=True)

    probe_path = model_dir / "ae_probe.pt"
    artifact_s3 = config.get("model_artifact_s3")
    marker_path = model_dir / ARTIFACT_URI_MARKER
    if probe_path.exists() and not artifact_s3:
        return probe_path
    if probe_path.exists() and artifact_s3 and marker_path.exists() and marker_path.read_text().strip() == str(artifact_s3):
        return probe_path

    tar_path = model_dir / "model.tar.gz"
    if tar_path.exists() and not artifact_s3:
        return _extract_probe_from_tar(tar_path, model_dir)

    if artifact_s3:
        import boto3

        _clear_model_files(model_dir)
        bucket, key = _parse_s3_uri(str(artifact_s3))
        profile = config.get("aws_profile")
        session = boto3.Session(profile_name=str(profile)) if profile else boto3.Session()
        session.client("s3").download_file(bucket, key, str(tar_path))
        extracted = _extract_probe_from_tar(tar_path, model_dir)
        marker_path.write_text(str(artifact_s3), encoding="utf-8")
        return extracted

    raise FileNotFoundError(f"{probe_path} not found and no model_artifact_s3 provided")


def _load_backbone(hf_model: str, model_dir: Path, device: torch.device) -> Wav2Vec2Model:
    backbone = Wav2Vec2Model.from_pretrained(hf_model).to(device)
    lora_dir = model_dir / "lora_adapter"
    if lora_dir.exists() and (lora_dir / "adapter_config.json").exists():
        backbone = PeftModel.from_pretrained(backbone, str(lora_dir))
        backbone = backbone.merge_and_unload()
        logger.info("Loaded AE LoRA adapter from %s", lora_dir)
    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad = False
    return backbone


def _load_probe(probe_path: Path, device: torch.device) -> _AEProbe:
    state = torch.load(probe_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    if not isinstance(state, dict) or "fc1.weight" not in state:
        raise ValueError(f"Unsupported AE probe checkpoint format: {probe_path}")

    in_dim = int(state["fc1.weight"].shape[1])
    probe = _AEProbe(in_dim=in_dim).to(device)
    probe.load_state_dict(state)
    probe.eval()
    return probe


def _prediction_from_tensor(pred: torch.Tensor, segment: Dict) -> AEProbePrediction:
    energy_prob = torch.sigmoid(pred[2]).item()
    pitch_prob = torch.sigmoid(pred[3]).item()
    return AEProbePrediction(
        slide_id=int(segment.get("slide_id", 0)),
        start_sec=float(segment.get("start_sec", 0.0)),
        end_sec=float(segment.get("end_sec", 0.0)),
        speech_rate=float(pred[0].item()),
        silence_ratio=float(max(0.0, min(1.0, pred[1].item()))),
        energy_drop=int(energy_prob >= 0.5),
        energy_drop_prob=float(energy_prob),
        pitch_shift=int(pitch_prob >= 0.5),
        pitch_shift_prob=float(pitch_prob),
        overall_delivery=float(max(0.0, min(1.0, pred[4].item()))),
    )


def predict_segments(audio_path: str | Path, segments: Sequence[Dict], config: Dict) -> list[AEProbePrediction]:
    """Run the trained AE probe over slide-level segments.

    This function performs inference only. It never trains or updates the probe.
    """
    if not segments:
        return []

    probe_path = resolve_probe_path(config)
    hf_model = str(config.get("hf_model", config.get("model_name", "kresnik/wav2vec2-large-xlsr-korean")))
    sample_rate = int(config.get("sample_rate", 16000))
    device_name = str(config.get("device", "auto"))
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    processor = Wav2Vec2Processor.from_pretrained(hf_model)
    backbone = _load_backbone(hf_model, probe_path.parent, device)

    probe = _load_probe(probe_path, device)
    expected_dim = int(probe.fc1.in_features)
    hidden_dim = int(getattr(backbone.config, "hidden_size", expected_dim))
    if hidden_dim != expected_dim:
        raise ValueError(
            f"AE probe expects {expected_dim}-dim embeddings, but {hf_model} produces {hidden_dim}. "
            "Use the same wav2vec2 backbone used during training."
        )

    wav, _ = librosa.load(str(audio_path), sr=sample_rate)
    duration_sec = len(wav) / sample_rate if sample_rate > 0 else 0.0
    if len(wav) == 0 or duration_sec <= 0.0:
        return []

    with torch.no_grad():
        inputs = processor([wav], sampling_rate=sample_rate, return_tensors="pt", padding=True).to(device)
        hidden = backbone(**inputs).last_hidden_state

        predictions: list[AEProbePrediction] = []
        frame_count = hidden.shape[1]
        for segment in segments:
            start_sec = max(0.0, float(segment.get("start_sec", 0.0)))
            end_sec = min(duration_sec, float(segment.get("end_sec", duration_sec)))
            if end_sec <= start_sec:
                continue
            f_start = int((start_sec / duration_sec) * frame_count)
            f_end = int((end_sec / duration_sec) * frame_count)
            f_start = max(0, min(f_start, frame_count - 1))
            f_end = max(f_start + 1, min(f_end, frame_count))
            pooled = hidden[:, f_start:f_end, :].mean(dim=1)
            pred = probe(pooled)[0]
            predictions.append(_prediction_from_tensor(pred, segment))

    logger.info("AE probe inference completed for %d segments using %s", len(predictions), probe_path)
    return predictions
