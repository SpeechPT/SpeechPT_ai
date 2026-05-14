from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import librosa

from speechpt.attitude.ae_probe_runtime import AEProbePrediction, predict_segments


def resolve_audio_files(audio: Path) -> List[Path]:
    if audio.is_file():
        return [audio]
    if audio.is_dir():
        return sorted(audio.rglob("*.wav")) + sorted(audio.rglob("*.mp3"))
    raise FileNotFoundError(f"Audio path not found: {audio}")


def _audio_duration_sec(audio_path: Path) -> float:
    return float(librosa.get_duration(path=str(audio_path)))


def _parse_slide_segments(slide_timestamps: str | None, duration_sec: float, chunk_sec: float) -> List[Dict]:
    if slide_timestamps:
        boundaries = [float(item.strip()) for item in slide_timestamps.split(",") if item.strip()]
        if len(boundaries) < 2:
            raise ValueError("--slide-timestamps must include at least 2 boundaries")
        return [
            {"slide_id": idx + 1, "start_sec": boundaries[idx], "end_sec": boundaries[idx + 1]}
            for idx in range(len(boundaries) - 1)
        ]
    end_sec = duration_sec if chunk_sec <= 0 else min(duration_sec, chunk_sec)
    return [{"slide_id": 1, "start_sec": 0.0, "end_sec": end_sec}]


def _prediction_to_row(prediction: AEProbePrediction, audio_path: Path) -> Dict:
    return {
        "audio_path": str(audio_path),
        "slide_id": prediction.slide_id,
        "start_sec": prediction.start_sec,
        "end_sec": prediction.end_sec,
        "speech_rate": prediction.speech_rate,
        "silence_ratio": prediction.silence_ratio,
        "energy_drop": prediction.energy_drop,
        "energy_drop_prob": prediction.energy_drop_prob,
        "pitch_shift": prediction.pitch_shift,
        "pitch_shift_prob": prediction.pitch_shift_prob,
        "overall_delivery": prediction.overall_delivery,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="AE probe inference via the same runtime used by SpeechPT E2E")
    parser.add_argument("--audio", required=True, help="wav/mp3 file or directory")
    parser.add_argument("--model-dir", default="artifacts/ae_model")
    parser.add_argument("--model-artifact-s3", help="Optional model.tar.gz S3 URI")
    parser.add_argument("--hf-model", default="kresnik/wav2vec2-large-xlsr-korean")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--chunk-sec", type=float, default=20.0, help="No timestamp mode only. <=0 means full audio.")
    parser.add_argument("--probe-chunk-sec", type=float, default=25.0, help="AE probe inference chunk size.")
    parser.add_argument("--probe-min-chunk-sec", type=float, default=1.0, help="Merge tails shorter than this.")
    parser.add_argument("--device", default="auto", help="auto/cpu/cuda")
    parser.add_argument("--aws-profile", default=None, help="Optional AWS profile for S3 artifact download")
    parser.add_argument("--output-jsonl")
    parser.add_argument(
        "--slide-timestamps",
        help="슬라이드 경계 타임스탬프(초), 쉼표 구분. 예: 0,40,90,130",
    )
    args = parser.parse_args()

    config = {
        "model_dir": args.model_dir,
        "model_artifact_s3": args.model_artifact_s3,
        "hf_model": args.hf_model,
        "sample_rate": args.sample_rate,
        "chunk_duration_sec": args.probe_chunk_sec,
        "min_chunk_duration_sec": args.probe_min_chunk_sec,
        "device": args.device,
    }
    if args.aws_profile:
        config["aws_profile"] = args.aws_profile

    results: List[Dict] = []
    for audio_path in resolve_audio_files(Path(args.audio)):
        duration_sec = _audio_duration_sec(audio_path)
        segments = _parse_slide_segments(args.slide_timestamps, duration_sec, args.chunk_sec)
        for prediction in predict_segments(audio_path, segments, config):
            row = _prediction_to_row(prediction, audio_path)
            results.append(row)
            print(json.dumps(row, ensure_ascii=False))

    if args.output_jsonl:
        out_path = Path(args.output_jsonl)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as fp:
            for row in results:
                fp.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
