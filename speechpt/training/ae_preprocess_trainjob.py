"""Run AE preprocessing inside a SageMaker training job and upload JSONL to S3."""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import boto3

from prepare_ae_from_raws import fallback_audio_targets, label_to_audio_name, row_from_label, write_jsonl


def parse_s3_uri(s3_uri: str) -> tuple[str, str]:
    if not s3_uri.startswith("s3://"):
        raise ValueError(f"Invalid S3 URI: {s3_uri}")
    body = s3_uri[5:]
    bucket, _, key = body.partition("/")
    return bucket, key


def row_from_label_obj(label_obj: dict, wav_name: str) -> dict | None:
    answer = label_obj.get("dataSet", {}).get("answer", {})
    try:
        word_count = float(answer.get("raw", {}).get("wordCount", 0.0))
    except Exception:
        word_count = 0.0
    try:
        duration_ms = float(label_obj.get("rawDataInfo", {}).get("answer", {}).get("duration", 0.0))
    except Exception:
        duration_ms = 0.0
    if duration_ms <= 0:
        return None
    duration_sec = max(1e-6, duration_ms / 1000.0)
    speech_rate = float(word_count / duration_sec)

    audio_targets = fallback_audio_targets(speech_rate=speech_rate)
    overall_delivery = 0.45 * max(0.0, min(1.0, speech_rate / 3.5)) + 0.55 * (1.0 - audio_targets["silence_ratio"])
    overall_delivery = max(0.0, min(1.0, overall_delivery))

    return {
        "audio_path": f"audio/{wav_name}",
        "speech_rate": speech_rate,
        "silence_ratio": float(audio_targets["silence_ratio"]),
        "energy_drop": int(audio_targets["energy_drop"]),
        "pitch_shift": int(audio_targets["pitch_shift"]),
        "overall_delivery": float(overall_delivery),
    }


def build_rows_from_s3(labels_s3_uri: str, max_files: int = 0) -> list[dict]:
    bucket, prefix = parse_s3_uri(labels_s3_uri.rstrip("/") + "/")
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    rows: list[dict] = []
    scanned = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            key = item.get("Key", "")
            if not key.endswith(".json"):
                continue
            body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
            try:
                obj = json.loads(body.decode("utf-8"))
            except Exception:
                continue
            wav_name = label_to_audio_name(Path(key), obj)
            row = row_from_label_obj(obj, wav_name=wav_name)
            if row is not None:
                rows.append(row)
            scanned += 1
            if max_files > 0 and scanned >= max_files:
                return rows
    return rows


def main():
    parser = argparse.ArgumentParser(description="AE preprocessing in SageMaker training job")
    parser.add_argument("--labels-dir", default=os.environ.get("SM_CHANNEL_LABELS", "/opt/ml/input/data/labels"))
    parser.add_argument("--audio-dir", default=os.environ.get("SM_CHANNEL_AUDIO", "/opt/ml/input/data/audio"))
    parser.add_argument("--output-s3-uri", required=True)
    parser.add_argument("--labels-s3-uri", default=os.environ.get("AE_LABELS_S3", ""))
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--use-audio-features", choices=["true", "false"], default="false")
    args = parser.parse_args()
    use_audio_features = args.use_audio_features.lower() == "true"

    test_ratio = 1.0 - args.train_ratio - args.valid_ratio
    if args.train_ratio <= 0 or args.valid_ratio <= 0 or test_ratio <= 0:
        raise ValueError("train/valid/test ratios must be positive and sum to 1")

    label_dir = Path(args.labels_dir)
    audio_dir = Path(args.audio_dir)
    label_files = sorted(label_dir.rglob("*.json"))
    if args.max_files > 0:
        label_files = label_files[: args.max_files]
    if not label_files:
        raise ValueError(f"No label files in {label_dir}")

    audio_files = sorted(audio_dir.rglob("*.wav"))
    audio_index = {p.name: p for p in audio_files}

    rows = []
    missing_wav = 0
    for f in label_files:
        row = row_from_label(
            f,
            audio_dir=audio_dir,
            audio_path_mode="relative",
            audio_path_prefix="audio",
            audio_index=audio_index,
            use_audio_features=use_audio_features,
        )
        if row is None:
            missing_wav += 1
            continue
        rows.append(row)

    if len(rows) < 3 and args.labels_s3_uri:
        rows = build_rows_from_s3(args.labels_s3_uri, max_files=args.max_files)

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
        if n_train > n_valid:
            n_train -= 1
        else:
            n_valid -= 1

    train_rows = rows[:n_train]
    valid_rows = rows[n_train : n_train + n_valid]
    test_rows = rows[n_train + n_valid :]

    out_dir = Path("/tmp/ae_prepared")
    out_dir.mkdir(parents=True, exist_ok=True)
    all_path = out_dir / "all.jsonl"
    train_path = out_dir / "train.jsonl"
    valid_path = out_dir / "valid.jsonl"
    test_path = out_dir / "test.jsonl"
    write_jsonl(all_path, rows)
    write_jsonl(train_path, train_rows)
    write_jsonl(valid_path, valid_rows)
    write_jsonl(test_path, test_rows)

    bucket, prefix = parse_s3_uri(args.output_s3_uri.rstrip("/") + "/")
    s3 = boto3.client("s3")
    s3.upload_file(str(all_path), bucket, f"{prefix}all.jsonl")
    s3.upload_file(str(train_path), bucket, f"{prefix}train.jsonl")
    s3.upload_file(str(valid_path), bucket, f"{prefix}valid.jsonl")
    s3.upload_file(str(test_path), bucket, f"{prefix}test.jsonl")

    print(
        json.dumps(
            {
                "label_files": len(label_files),
                "audio_files": len(audio_files),
                "prepared_rows": len(rows),
                "missing_wav": missing_wav,
                "use_audio_features": use_audio_features,
                "counts": {"train": len(train_rows), "valid": len(valid_rows), "test": len(test_rows)},
                "output_s3": args.output_s3_uri,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
