"""Build AE training JSONL from raw SpeechPT label/audio folders.

This script creates pseudo-targets required by `ae_probe_train.py`:
- speech_rate
- silence_ratio
- energy_drop
- pitch_shift
- overall_delivery

These targets are heuristic and intended for bootstrapping when explicit
human scores are not available.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import wave
from pathlib import Path
from typing import Dict, List

import librosa
import numpy as np


def safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def load_wav_mono(audio_path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(audio_path), "rb") as wf:
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sample_width == 2:
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")

    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)
    return audio, sample_rate


def frame_signal(y: np.ndarray, frame_len: int, hop_len: int) -> np.ndarray:
    if len(y) < frame_len:
        y = np.pad(y, (0, frame_len - len(y)))
    starts = np.arange(0, len(y) - frame_len + 1, hop_len)
    return np.stack([y[s : s + frame_len] for s in starts], axis=0)


def derive_audio_targets(audio_path: Path) -> Dict[str, float]:
    y, sr = load_wav_mono(audio_path)
    if len(y) == 0:
        return {"silence_ratio": 1.0, "energy_drop": 0, "pitch_shift": 0}

    frame_len = int(0.032 * sr)
    hop_len = int(0.010 * sr)
    frames = frame_signal(y, frame_len=frame_len, hop_len=hop_len)

    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
    energy_db = 20.0 * np.log10(rms + 1e-9)
    energy_db = energy_db - float(np.max(energy_db))
    silence_ratio = float(np.mean(energy_db < -40.0))

    times = np.arange(len(energy_db), dtype=np.float32) * (hop_len / float(sr))
    if len(times) > 1:
        slope, _ = np.polyfit(times, energy_db, 1)
    else:
        slope = 0.0
    energy_drop = 1 if slope < -0.02 else 0

    # pyin으로 실제 기본 주파수(F0) 추출 → voiced 구간 F0 표준편차로 억양 변화 판정
    # zero-crossing 방식 대비 훨씬 정확하며 한국어 발화 분포에 맞는 임계값 사용
    try:
        y_float = np.array(y, dtype=np.float32)
        f0, voiced_flag, _ = librosa.pyin(
            y_float,
            fmin=librosa.note_to_hz("C2"),  # 65 Hz
            fmax=librosa.note_to_hz("C7"),  # 2093 Hz
            sr=sr,
        )
        f0_voiced = f0[voiced_flag & ~np.isnan(f0)]
        pitch_shift = 1 if (len(f0_voiced) > 10 and float(np.std(f0_voiced)) > 80.0) else 0
    except Exception:
        pitch_shift = 0

    return {
        "silence_ratio": max(0.0, min(1.0, silence_ratio)),
        "energy_drop": int(energy_drop),
        "pitch_shift": int(pitch_shift),
    }


def compute_overall_delivery(speech_rate: float, silence_ratio: float, energy_drop: int, pitch_shift: int) -> float:
    pace_score = math.exp(-abs(speech_rate - 2.2) / 1.8)
    silence_score = max(0.0, 1.0 - silence_ratio / 0.45)
    stability = 1.0 - 0.5 * (float(energy_drop) + float(pitch_shift))
    overall = 0.45 * pace_score + 0.30 * silence_score + 0.25 * max(0.0, stability)
    return max(0.0, min(1.0, float(overall)))


def fallback_audio_targets(speech_rate: float) -> Dict[str, float]:
    # Label-only heuristic when audio signal cannot be read.
    silence_ratio = max(0.05, min(0.6, 0.45 - 0.08 * speech_rate))
    energy_drop = 0
    pitch_shift = 0
    return {
        "silence_ratio": float(silence_ratio),
        "energy_drop": int(energy_drop),
        "pitch_shift": int(pitch_shift),
    }


def label_to_audio_name(label_file: Path, label_obj: Dict) -> str:
    answer_path = (
        label_obj.get("rawDataInfo", {})
        .get("answer", {})
        .get("audioPath")
    )
    if isinstance(answer_path, str) and answer_path.strip():
        return Path(answer_path).name
    return label_file.name.replace("_d_", "_a_").replace(".json", ".wav")


def row_from_label(
    label_file: Path,
    audio_dir: Path,
    audio_path_mode: str,
    audio_path_prefix: str,
    audio_index: Dict[str, Path] | None = None,
    use_audio_features: bool = True,
) -> Dict | None:
    row, _ = row_from_label_with_reason(
        label_file=label_file,
        audio_dir=audio_dir,
        audio_path_mode=audio_path_mode,
        audio_path_prefix=audio_path_prefix,
        audio_index=audio_index,
        use_audio_features=use_audio_features,
    )
    return row


def row_from_label_with_reason(
    label_file: Path,
    audio_dir: Path,
    audio_path_mode: str,
    audio_path_prefix: str,
    audio_index: Dict[str, Path] | None = None,
    use_audio_features: bool = True,
) -> tuple[Dict | None, str]:
    try:
        text = label_file.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None, "read_file_not_found"
    except OSError as e:
        return None, f"read_os_error:{type(e).__name__}"

    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None, "json_decode_error"
    answer = obj.get("dataSet", {}).get("answer", {})
    word_count = safe_float(answer.get("raw", {}).get("wordCount"), 0.0)
    duration_ms = safe_float(obj.get("rawDataInfo", {}).get("answer", {}).get("duration"), 0.0)
    duration_sec = max(1e-6, duration_ms / 1000.0)
    speech_rate = float(word_count / duration_sec)

    wav_name = label_to_audio_name(label_file, obj)
    wav_path = None
    if use_audio_features:
        if audio_index is not None:
            wav_path = audio_index.get(wav_name)
        if wav_path is None:
            candidate = audio_dir / wav_name
            if candidate.exists():
                wav_path = candidate

    if wav_path is not None and wav_path.exists() and use_audio_features:
        audio_targets = derive_audio_targets(wav_path)
    else:
        audio_targets = fallback_audio_targets(speech_rate=speech_rate)
    overall_delivery = compute_overall_delivery(
        speech_rate=speech_rate,
        silence_ratio=audio_targets["silence_ratio"],
        energy_drop=audio_targets["energy_drop"],
        pitch_shift=audio_targets["pitch_shift"],
    )

    if audio_path_mode == "relative":
        audio_path_value = f"{audio_path_prefix.rstrip('/')}/{wav_name}"
    else:
        audio_path_value = str(wav_path)

    return {
        "audio_path": audio_path_value,
        "speech_rate": float(speech_rate),
        "silence_ratio": float(audio_targets["silence_ratio"]),
        "energy_drop": int(audio_targets["energy_drop"]),
        "pitch_shift": int(audio_targets["pitch_shift"]),
        "overall_delivery": float(overall_delivery),
    }, "ok"


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Prepare AE dataset from raw label/audio folders")
    parser.add_argument("--label-dir", required=True, help="Directory with raw label json files")
    parser.add_argument("--audio-dir", required=True, help="Directory with wav files")
    parser.add_argument("--output-dir", required=True, help="Output dir for train/valid/test jsonl")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-files", type=int, default=0, help="Optional cap for quick experiments")
    parser.add_argument("--audio-path-mode", choices=["relative", "absolute"], default="relative")
    parser.add_argument("--audio-path-prefix", default="audio")
    parser.add_argument("--use-audio-features", choices=["true", "false"], default="true")
    args = parser.parse_args()
    use_audio_features = args.use_audio_features.lower() == "true"

    test_ratio = 1.0 - args.train_ratio - args.valid_ratio
    if args.train_ratio <= 0 or args.valid_ratio <= 0 or test_ratio <= 0:
        raise ValueError("train/valid/test ratios must be positive and sum to 1")

    label_dir = Path(args.label_dir)
    audio_dir = Path(args.audio_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    label_files = sorted(label_dir.rglob("*.json"))
    if args.max_files > 0:
        label_files = label_files[: args.max_files]
    if not label_files:
        raise ValueError(f"No json files found in {label_dir}")

    rows: List[Dict] = []
    missing_wav = 0
    for f in label_files:
        row = row_from_label(
            f,
            audio_dir=audio_dir,
            audio_path_mode=args.audio_path_mode,
            audio_path_prefix=args.audio_path_prefix,
            use_audio_features=use_audio_features,
        )
        if row is None:
            missing_wav += 1
            continue
        rows.append(row)

    if len(rows) < 3:
        raise ValueError(f"Not enough prepared rows: {len(rows)}")

    rng = random.Random(args.seed)
    rng.shuffle(rows)

    n = len(rows)
    n_train = max(1, int(n * args.train_ratio))
    n_valid = max(1, int(n * args.valid_ratio))
    if n_train + n_valid >= n:
        n_valid = max(1, n - n_train - 1)
    n_test = n - n_train - n_valid
    if n_test <= 0:
        n_test = 1
        if n_train > n_valid:
            n_train -= 1
        else:
            n_valid -= 1

    train_rows = rows[:n_train]
    valid_rows = rows[n_train : n_train + n_valid]
    test_rows = rows[n_train + n_valid :]

    write_jsonl(output_dir / "train.jsonl", train_rows)
    write_jsonl(output_dir / "valid.jsonl", valid_rows)
    write_jsonl(output_dir / "test.jsonl", test_rows)

    print(
        json.dumps(
            {
                "label_files": len(label_files),
                "prepared_rows": len(rows),
                "missing_wav": missing_wav,
                "counts": {"train": len(train_rows), "valid": len(valid_rows), "test": len(test_rows)},
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
