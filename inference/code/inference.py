"""SpeechPT 통합 추론 핸들러 (SageMaker SDK 호환).

task 분기:
  task=pipeline → SpeechPTPipeline.analyze() 전체 실행 (권장)
                   문서 파싱 → STT → 정렬 → CE → AE → report → LLM 까지
                   {document_s3, audio_s3} 입력, SpeechReport.to_dict() 반환
  task=stt   → Whisper-small forward            (legacy / 디버깅용)
  task=ce    → ko-sroberta encode + similarity   (legacy / 디버깅용)
  task=ae    → wav2vec2-large + AEProbe forward  (legacy / 디버깅용)

가드레일:
  ① chunk_duration_sec ≤ 30 강제 (task=ae)
  ② Whisper는 HuggingFace 버전 사용 (CTranslate2 회피, task=stt)
  ③ PYTORCH_CUDA_ALLOC_CONF (Dockerfile에서 ENV로)
  ④ MaxConcurrentInvocationsPerInstance=2 (Endpoint config에서)
"""
from __future__ import annotations

import io
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, stream=sys.stdout)


# ──────────────────────────────────────────────────────────────
# AE Probe 모델 정의 (학습 코드와 동일해야 함)
# ──────────────────────────────────────────────────────────────
class AEProbe(nn.Module):
    """ae_probe_train.py의 AEProbe와 동일 구조."""

    def __init__(self, in_dim: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 256)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(256, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)


# ──────────────────────────────────────────────────────────────
# SageMaker 표준 핸들러
# ──────────────────────────────────────────────────────────────
def model_fn(model_dir: str) -> Dict[str, Any]:
    """컨테이너 시작 시 1회 호출. 3 모델 모두 GPU에 로드."""
    logger.info(f"model_fn: loading from {model_dir}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"  device = {device}")

    # 1. STT — HuggingFace Whisper (가드레일 ②)
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    stt_processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
    stt_model = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-large-v3",
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    ).to(device).eval()
    # 한국어 강제 디코딩
    stt_forced_ids = stt_processor.get_decoder_prompt_ids(language="ko", task="transcribe")
    logger.info("  ✓ STT (Whisper-large-v3) loaded")

    # 2. CE — ko-sroberta
    from sentence_transformers import SentenceTransformer

    ce_model = SentenceTransformer(
        "jhgan/ko-sroberta-multitask",
        device=str(device),
    )
    logger.info("  ✓ CE (ko-sroberta) loaded")

    # 3. AE — wav2vec2-large + AEProbe
    from transformers import Wav2Vec2Model, Wav2Vec2Processor

    ae_processor = Wav2Vec2Processor.from_pretrained("kresnik/wav2vec2-large-xlsr-korean")
    ae_backbone = Wav2Vec2Model.from_pretrained(
        "kresnik/wav2vec2-large-xlsr-korean",
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    ).to(device).eval()

    # LoRA adapter 적용 (있으면)
    lora_dir = os.path.join(model_dir, "lora_adapter")
    if os.path.isdir(lora_dir):
        from peft import PeftModel
        ae_backbone = PeftModel.from_pretrained(ae_backbone, lora_dir)
        ae_backbone = ae_backbone.merge_and_unload()
        logger.info("  ✓ LoRA adapter merged into AE backbone")

    # AE Probe
    probe_path = os.path.join(model_dir, "ae_probe.pt")
    if not os.path.isfile(probe_path):
        raise RuntimeError(f"AE probe not found at {probe_path}")

    probe_state = torch.load(probe_path, map_location=device)
    # 학습 시 in_dim 추출 (meta.bin 있으면 거기서)
    meta_path = os.path.join(model_dir, "meta.bin")
    in_dim = 1024
    if os.path.isfile(meta_path):
        meta = torch.load(meta_path, map_location="cpu")
        in_dim = int(meta.get("in_dim", 1024))

    ae_probe = AEProbe(in_dim=in_dim).to(device).eval()
    ae_probe.load_state_dict(probe_state)
    logger.info(f"  ✓ AE Probe loaded (in_dim={in_dim})")

    # JIT warmup — 첫 호출 지연 제거
    with torch.inference_mode():
        dummy = torch.randn(1, 16000, device=device, dtype=ae_backbone.dtype)
        _ = ae_backbone(dummy)
        _ = ae_probe(torch.randn(1, in_dim, device=device))
    logger.info("  ✓ JIT warmup done")

    # 4. SpeechPTPipeline — task=pipeline 분기에서 사용
    # speechpt 패키지는 Dockerfile에서 /opt/ml/code/speechpt 로 복사됨.
    # 모델 가중치는 model_dir(=/opt/ml/model)에서 읽어들임 (ae_probe.pt, meta.bin, lora_adapter/).
    pipeline = None
    try:
        from speechpt.pipeline import SpeechPTPipeline

        runtime_config = os.environ.get(
            "PIPELINE_RUNTIME_CONFIG",
            "/opt/ml/code/configs/pipeline_runtime.yaml",
        )
        pipeline = SpeechPTPipeline(config_path=runtime_config, device=str(device))
        # ae_probe.model_dir / report.template 경로 보정 — 패키징 이미지 기준
        ae_probe_cfg = pipeline.ae_cfg.setdefault("ae_probe", {})
        ae_probe_cfg.setdefault("model_dir", model_dir)
        logger.info(f"  ✓ SpeechPTPipeline loaded (config={runtime_config})")
    except Exception as exc:
        logger.exception("SpeechPTPipeline init failed; task=pipeline will return error")

    return {
        "device": device,
        "model_dir": model_dir,
        "stt_model": stt_model,
        "stt_processor": stt_processor,
        "stt_forced_ids": stt_forced_ids,
        "ce_model": ce_model,
        "ae_processor": ae_processor,
        "ae_backbone": ae_backbone,
        "ae_probe": ae_probe,
        "ae_in_dim": in_dim,
        "pipeline": pipeline,
    }


def input_fn(request_body: bytes, content_type: str) -> Dict[str, Any]:
    """JSON 입력 파싱."""
    if content_type != "application/json":
        raise ValueError(f"Unsupported content type: {content_type}")
    return json.loads(request_body)


def predict_fn(input_data: Dict[str, Any], models: Dict[str, Any]) -> Dict[str, Any]:
    """task 필드로 분기."""
    task = input_data.get("task")
    if not task:
        raise ValueError("missing 'task' field; expected one of: pipeline, stt, ce, ae")

    if task == "pipeline":
        return _run_pipeline(input_data, models)
    if task == "stt":
        return _run_stt(input_data, models)
    if task == "ce":
        return _run_ce(input_data, models)
    if task == "ae":
        # 가드레일 ① 청크 길이 강제
        chunk_sec = float(input_data.get("chunk_sec", 30))
        if chunk_sec > 30:
            raise ValueError(f"chunk_sec must be <= 30 (got {chunk_sec}). OOM 위험.")
        return _run_ae(input_data, models, chunk_sec=chunk_sec)

    raise ValueError(f"unknown task: {task}")


def output_fn(prediction: Dict[str, Any], accept: str) -> bytes:
    """JSON 직렬화."""
    return json.dumps(prediction, ensure_ascii=False).encode("utf-8")


# ──────────────────────────────────────────────────────────────
# 태스크별 처리
# ──────────────────────────────────────────────────────────────
def _download_audio(audio_uri: str) -> np.ndarray:
    """S3 URI 또는 로컬 경로에서 오디오 로드 → 16kHz mono numpy."""
    import librosa
    import boto3

    if audio_uri.startswith("s3://"):
        s3 = boto3.client("s3")
        bucket, key = audio_uri.replace("s3://", "").split("/", 1)
        local = "/tmp/audio_input.wav"
        s3.download_file(bucket, key, local)
        path = local
    else:
        path = audio_uri

    audio, _ = librosa.load(path, sr=16000, mono=True)
    return audio


def _run_stt(input_data: Dict[str, Any], models: Dict[str, Any]) -> Dict[str, Any]:
    """Whisper로 한국어 전사 + word timestamps."""
    audio_uri = input_data.get("audio_s3") or input_data.get("audio_path")
    if not audio_uri:
        raise ValueError("STT: 'audio_s3' or 'audio_path' required")

    audio = _download_audio(audio_uri)
    device = models["device"]
    processor = models["stt_processor"]
    model = models["stt_model"]

    # 30초 청크로 자동 분할 (Whisper 표준)
    chunk_size = 30 * 16000
    all_segments: List[Dict[str, Any]] = []
    full_text = []

    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:i + chunk_size]
        offset_sec = i / 16000.0

        inputs = processor(chunk, sampling_rate=16000, return_tensors="pt")
        input_features = inputs.input_features.to(
            device,
            dtype=torch.float16 if device.type == "cuda" else torch.float32,
        )

        with torch.inference_mode():
            generated_ids = model.generate(
                input_features,
                forced_decoder_ids=models["stt_forced_ids"],
                max_new_tokens=440,
                return_timestamps=True,
            )

        decoded = processor.batch_decode(
            generated_ids, skip_special_tokens=True,
        )[0]
        full_text.append(decoded)

        # 단순화: 청크 단위 segment (word-level은 추가 작업 필요)
        all_segments.append({
            "start": offset_sec,
            "end": offset_sec + len(chunk) / 16000.0,
            "text": decoded,
        })

    # word-level timestamps 근사 (단순 분할 — Step 9에선 충분)
    words = []
    for seg in all_segments:
        tokens = seg["text"].split()
        if not tokens:
            continue
        seg_dur = seg["end"] - seg["start"]
        per_word = seg_dur / max(len(tokens), 1)
        for j, tok in enumerate(tokens):
            words.append({
                "word": tok,
                "start": round(seg["start"] + j * per_word, 3),
                "end": round(seg["start"] + (j + 1) * per_word, 3),
            })

    return {
        "text": " ".join(full_text),
        "words": words,
        "segments": all_segments,
    }


def _run_ce(input_data: Dict[str, Any], models: Dict[str, Any]) -> Dict[str, Any]:
    """ko-sroberta로 키포인트 ↔ 트랜스크립트 청크 유사도 행렬."""
    keypoints: List[str] = input_data.get("keypoints", [])
    chunks: List[str] = input_data.get("transcript_chunks", [])
    threshold = float(input_data.get("threshold", 0.55))

    if not keypoints or not chunks:
        return {
            "similarity_matrix": [],
            "max_sim_per_keypoint": [],
            "covered_mask": [],
            "coverage": 0.0,
        }

    model = models["ce_model"]
    kp_emb = model.encode(keypoints, normalize_embeddings=True, convert_to_numpy=True)
    ch_emb = model.encode(chunks, normalize_embeddings=True, convert_to_numpy=True)

    sim = np.dot(kp_emb, ch_emb.T)               # (K, C)
    max_sim = sim.max(axis=1)                     # (K,)
    covered = (max_sim >= threshold).astype(int)
    coverage = float(covered.sum() / len(keypoints))

    return {
        "similarity_matrix": sim.tolist(),
        "max_sim_per_keypoint": max_sim.tolist(),
        "covered_mask": covered.tolist(),
        "coverage": coverage,
    }


def _run_ae(input_data: Dict[str, Any], models: Dict[str, Any], chunk_sec: float = 30) -> Dict[str, Any]:
    """wav2vec2 임베딩 → AE Probe로 5개 점수 예측."""
    audio_uri = input_data.get("audio_s3") or input_data.get("audio_path")
    segments = input_data.get("segments", [])
    if not audio_uri:
        raise ValueError("AE: 'audio_s3' or 'audio_path' required")
    if not segments:
        raise ValueError("AE: 'segments' (list of {slide_id, start_sec, end_sec}) required")

    audio = _download_audio(audio_uri)
    device = models["device"]
    backbone = models["ae_backbone"]
    processor = models["ae_processor"]
    probe = models["ae_probe"]
    backbone_dtype = next(backbone.parameters()).dtype
    probe_dtype = next(probe.parameters()).dtype

    sr = 16000
    results: List[Dict[str, Any]] = []

    for seg in segments:
        start = float(seg.get("start_sec", 0.0))
        end = float(seg.get("end_sec", start + 30))
        seg_audio = audio[int(start * sr):int(end * sr)]
        if len(seg_audio) < sr:  # 1초 미만은 건너뜀
            results.append({
                "slide_id": seg.get("slide_id"),
                "scores": None,
                "warning": "segment too short",
            })
            continue

        # 30초 청크로 분할
        chunk_size = int(chunk_sec * sr)
        chunk_embeddings: List[torch.Tensor] = []
        for i in range(0, len(seg_audio), chunk_size):
            chunk = seg_audio[i:i + chunk_size]
            inputs = processor(chunk, sampling_rate=sr, return_tensors="pt")
            input_values = inputs.input_values.to(device, dtype=backbone_dtype)

            with torch.inference_mode():
                hidden = backbone(input_values).last_hidden_state   # (1, T, D)
            chunk_embeddings.append(hidden.mean(dim=1))             # (1, D)

        # 청크 평균 → segment-level embedding
        seg_emb = torch.stack(chunk_embeddings).mean(dim=0)         # (1, D)
        seg_emb = seg_emb.to(probe_dtype)

        with torch.inference_mode():
            scores = probe(seg_emb)                                  # (1, 5)
        scores_np = scores.cpu().float().numpy()[0]

        # 5개 출력 의미 (학습 코드와 동일):
        # 0: speech_rate (regression)
        # 1: silence_ratio (regression)
        # 2: energy_drop (binary logit)
        # 3: pitch_shift (binary logit)
        # 4: overall_delivery (regression)
        results.append({
            "slide_id": seg.get("slide_id"),
            "start_sec": start,
            "end_sec": end,
            "scores": {
                "speech_rate": float(scores_np[0]),
                "silence_ratio": float(scores_np[1]),
                "energy_drop_logit": float(scores_np[2]),
                "pitch_shift_logit": float(scores_np[3]),
                "overall_delivery": float(scores_np[4]),
                "energy_drop_prob": float(torch.sigmoid(torch.tensor(scores_np[2])).item()),
                "pitch_shift_prob": float(torch.sigmoid(torch.tensor(scores_np[3])).item()),
            },
        })

    return {"segments": results}


# ──────────────────────────────────────────────────────────────
# task=pipeline — SpeechPTPipeline.analyze() 통째로 실행
# ──────────────────────────────────────────────────────────────
def _download_to_tmp(s3_uri: str, suffix: str = "") -> str:
    """S3 URI(또는 로컬 경로)를 /tmp에 받아 로컬 경로 반환."""
    import boto3
    import tempfile

    if not s3_uri.startswith("s3://"):
        return s3_uri  # 이미 로컬 경로

    bucket, key = s3_uri.replace("s3://", "").split("/", 1)
    if not suffix:
        suffix = os.path.splitext(key)[1] or ""
    fd, local = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    boto3.client("s3").download_file(bucket, key, local)
    return local


def _run_pipeline(input_data: Dict[str, Any], models: Dict[str, Any]) -> Dict[str, Any]:
    """발표 자료 + 음성을 받아 SpeechPTPipeline.analyze() 결과를 그대로 반환.

    입력:
        {
          "task": "pipeline",
          "document_s3": "s3://.../slides.pdf"     # 또는 document_path
          "audio_s3":    "s3://.../audio.wav"      # 또는 audio_path
          "alignment_mode": "auto" | "manual" | "hybrid"   # optional
        }

    출력 (SpeechReport.to_dict()):
        {
          "version", "overall_scores", "highlight_sections",
          "per_slide_detail", "global_summary", "alignment",
          "transcript_segments", "llm_feedback"
        }
    """
    pipeline = models.get("pipeline")
    if pipeline is None:
        raise RuntimeError(
            "SpeechPTPipeline not initialized (see model_fn logs). "
            "task=pipeline cannot run."
        )

    document_uri = input_data.get("document_s3") or input_data.get("document_path")
    audio_uri = input_data.get("audio_s3") or input_data.get("audio_path")
    if not document_uri or not audio_uri:
        raise ValueError("task=pipeline requires 'document_s3' and 'audio_s3' (or *_path).")

    document_local = _download_to_tmp(document_uri)
    audio_local = _download_to_tmp(audio_uri, suffix=".wav")

    try:
        report = pipeline.analyze(
            document_path=document_local,
            audio_path=audio_local,
            alignment_mode=input_data.get("alignment_mode"),
        )
        return report.to_dict()
    finally:
        # 다운로드 파일 정리
        for path in (document_local, audio_local):
            if path and path.startswith("/tmp/"):
                try:
                    os.unlink(path)
                except OSError:
                    pass
