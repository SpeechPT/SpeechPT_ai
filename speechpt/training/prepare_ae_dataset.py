"""Prepare AE dataset JSONL for training.

Accepted input formats:
- CSV
- JSON (list[object])
- JSONL

Required fields:
- audio_path
- speech_rate
- silence_ratio
- energy_drop
- pitch_shift
- overall_delivery
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List

REQUIRED_FIELDS = [
    "audio_path",
    "speech_rate",
    "silence_ratio",
    "energy_drop",
    "pitch_shift",
    "overall_delivery",
]


def read_rows(path: Path) -> List[Dict]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8") as fp:
            return list(csv.DictReader(fp))
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as fp:
            data = json.load(fp)
        if not isinstance(data, list):
            raise ValueError("JSON input must be a list of objects")
        return data
    if suffix == ".jsonl":
        rows: List[Dict] = []
        with path.open("r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    raise ValueError(f"Unsupported input format: {path.suffix}")


def to_binary(value) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return 1 if float(value) >= 0.5 else 0
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return 1
    if text in {"0", "false", "no", "n", "off"}:
        return 0
    raise ValueError(f"Cannot parse binary value: {value}")


def normalize_row(row: Dict, audio_root: Path | None) -> Dict:
    for key in REQUIRED_FIELDS:
        if key not in row:
            raise ValueError(f"Missing required field: {key}")

    audio_path = Path(str(row["audio_path"]))
    if audio_root is not None and not audio_path.is_absolute():
        audio_path = audio_root / audio_path

    normalized = {
        "audio_path": str(audio_path),
        "speech_rate": float(row["speech_rate"]),
        "silence_ratio": float(row["silence_ratio"]),
        "energy_drop": to_binary(row["energy_drop"]),
        "pitch_shift": to_binary(row["pitch_shift"]),
        "overall_delivery": float(row["overall_delivery"]),
    }
    return normalized


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Prepare AE dataset JSONL")
    parser.add_argument("--input", required=True, help="Path to source csv/json/jsonl")
    parser.add_argument("--output-dir", required=True, help="Output directory for split jsonl files")
    parser.add_argument("--audio-root", help="Optional root dir to prefix relative audio_path")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    test_ratio = 1.0 - args.train_ratio - args.valid_ratio
    if args.train_ratio <= 0 or args.valid_ratio <= 0 or test_ratio <= 0:
        raise ValueError("train/valid/test ratios must be positive and sum to 1")

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_root = Path(args.audio_root) if args.audio_root else None

    rows = [normalize_row(row, audio_root) for row in read_rows(input_path)]
    if len(rows) < 3:
        raise ValueError("Need at least 3 rows for train/valid/test split")

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
                "input": str(input_path),
                "output_dir": str(output_dir),
                "counts": {"train": len(train_rows), "valid": len(valid_rows), "test": len(test_rows)},
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
