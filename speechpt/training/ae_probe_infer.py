from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path
from typing import Dict, List

import boto3
import librosa
import torch
import torch.nn.functional as F
from transformers import Wav2Vec2Model, Wav2Vec2Processor

from speechpt.training.ae_probe_train import AEProbe


def parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"Invalid S3 URI: {uri}")
    body = uri[5:]
    bucket, _, key = body.partition("/")
    return bucket, key


def load_model_dir(model_dir: Path, model_artifact_s3: str | None) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    probe_path = model_dir / "ae_probe.pt"
    if probe_path.exists():
        return model_dir
    if not model_artifact_s3:
        raise FileNotFoundError(f"{probe_path} not found and --model-artifact-s3 not provided")

    bucket, key = parse_s3_uri(model_artifact_s3)
    tar_path = model_dir / "model.tar.gz"
    boto3.client("s3").download_file(bucket, key, str(tar_path))
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(model_dir)
    if not probe_path.exists():
        raise FileNotFoundError(f"ae_probe.pt not found in artifact: {model_artifact_s3}")
    return model_dir


def resolve_audio_files(audio: Path) -> List[Path]:
    if audio.is_file():
        return [audio]
    if audio.is_dir():
        return sorted(audio.rglob("*.wav"))
    raise FileNotFoundError(f"Audio path not found: {audio}")


def load_wav(audio_path: Path, sample_rate: int, chunk_sec: float) -> torch.Tensor:
    wav, _ = librosa.load(audio_path, sr=sample_rate)
    chunk_len = int(sample_rate * chunk_sec)
    if len(wav) >= chunk_len:
        wav = wav[:chunk_len]
    else:
        pad = chunk_len - len(wav)
        wav = F.pad(torch.tensor(wav), (0, pad)).numpy()
    return torch.tensor(wav, dtype=torch.float32)


def _pred_to_dict(pred: torch.Tensor, audio_path: str, slide_id: int | None = None,
                  start_sec: float | None = None, end_sec: float | None = None) -> Dict:
    energy_prob = torch.sigmoid(pred[2]).item()
    pitch_prob = torch.sigmoid(pred[3]).item()
    out: Dict = {
        "audio_path": audio_path,
        "speech_rate": float(pred[0].item()),
        "silence_ratio": float(max(0.0, min(1.0, pred[1].item()))),
        "energy_drop": int(energy_prob >= 0.5),
        "energy_drop_prob": float(energy_prob),
        "pitch_shift": int(pitch_prob >= 0.5),
        "pitch_shift_prob": float(pitch_prob),
        "overall_delivery": float(max(0.0, min(1.0, pred[4].item()))),
    }
    if slide_id is not None:
        out["slide_id"] = slide_id
    if start_sec is not None:
        out["start_sec"] = start_sec
    if end_sec is not None:
        out["end_sec"] = end_sec
    return out


def main():
    parser = argparse.ArgumentParser(description="AE probe inference (wav -> 5 scores)")
    parser.add_argument("--audio", required=True, help="wav file or directory")
    parser.add_argument("--model-dir", default="artifacts/ae_model")
    parser.add_argument("--model-artifact-s3", help="Optional model.tar.gz S3 URI")
    parser.add_argument("--hf-model", default="facebook/wav2vec2-base")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--chunk-sec", type=float, default=20.0)
    parser.add_argument("--output-jsonl")
    parser.add_argument(
        "--slide-timestamps",
        help="슬라이드 경계 타임스탬프(초), 쉼표 구분. 예: 0,40,50,90,120 → 슬라이드1=0~40, 슬라이드2=40~50, ...",
    )
    args = parser.parse_args()

    model_dir = load_model_dir(Path(args.model_dir), args.model_artifact_s3)
    meta_path = model_dir / "meta.pt"
    if meta_path.exists():
        meta = torch.load(meta_path, map_location="cpu")
        args.hf_model = str(meta.get("model", args.hf_model))
        args.sample_rate = int(meta.get("sample_rate", args.sample_rate))
        args.chunk_sec = float(meta.get("chunk_sec", args.chunk_sec))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = Wav2Vec2Processor.from_pretrained(args.hf_model)
    backbone = Wav2Vec2Model.from_pretrained(args.hf_model).to(device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    in_dim = int(getattr(backbone.config, "hidden_size", 768))
    probe = AEProbe(in_dim=in_dim).to(device)
    state = torch.load(model_dir / "ae_probe.pt", map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        probe.load_state_dict(state["model_state_dict"])
    else:
        probe.load_state_dict(state)
    probe.eval()

    # 슬라이드 타임스탬프 파싱: "0,40,50,90" → [(0,40,1), (40,50,2), (50,90,3)]
    slide_segments: List[tuple] = []
    if args.slide_timestamps:
        ts = sorted(float(t) for t in args.slide_timestamps.split(","))
        for i in range(len(ts) - 1):
            slide_segments.append((ts[i], ts[i + 1], i + 1))

    # wav2vec2는 50 fps (20ms hop) — 초 → 프레임 인덱스 변환
    FRAMES_PER_SEC = 50

    files = resolve_audio_files(Path(args.audio))
    results: List[Dict] = []
    with torch.no_grad():
        for fp in files:
            if slide_segments:
                # 오디오 전체 로드 (chunk 제한 없이)
                wav_full, _ = librosa.load(str(fp), sr=args.sample_rate)
                wav_tensor = torch.tensor(wav_full, dtype=torch.float32)
                inputs = processor(
                    [wav_full], sampling_rate=args.sample_rate, return_tensors="pt", padding=True
                ).to(device)
                hidden = backbone(**inputs).last_hidden_state  # (1, T, hidden)

                for start_sec, end_sec, slide_id in slide_segments:
                    f_start = int(start_sec * FRAMES_PER_SEC)
                    f_end = int(end_sec * FRAMES_PER_SEC)
                    f_end = min(f_end, hidden.shape[1])
                    if f_start >= hidden.shape[1] or f_start >= f_end:
                        continue
                    seg_hidden = hidden[:, f_start:f_end, :]
                    pooled = seg_hidden.mean(dim=1)
                    pred = probe(pooled)[0]
                    out = _pred_to_dict(pred, str(fp), slide_id=slide_id,
                                        start_sec=start_sec, end_sec=end_sec)
                    results.append(out)
                    print(json.dumps(out, ensure_ascii=False))
            else:
                # 타임스탬프 없음 → 전체 오디오 단일 점수
                wav = load_wav(fp, sample_rate=args.sample_rate, chunk_sec=args.chunk_sec)
                inputs = processor(
                    [wav.numpy()], sampling_rate=args.sample_rate, return_tensors="pt", padding=True
                ).to(device)
                hidden = backbone(**inputs).last_hidden_state
                pooled = hidden.mean(dim=1)
                pred = probe(pooled)[0]
                out = _pred_to_dict(pred, str(fp))
                results.append(out)
                print(json.dumps(out, ensure_ascii=False))

    if args.output_jsonl:
        out_path = Path(args.output_jsonl)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for row in results:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
