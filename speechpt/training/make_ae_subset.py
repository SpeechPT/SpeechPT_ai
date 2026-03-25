"""Create subset train/valid/test manifests from a full AE JSONL."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List


def read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Make AE subset manifests from full all.jsonl")
    parser.add_argument("--input", required=True, help="Path to full all.jsonl")
    parser.add_argument("--output-dir", required=True, help="Output dir for subset manifests")
    parser.add_argument("--max-rows", type=int, required=True, help="Subset size (N)")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    test_ratio = 1.0 - args.train_ratio - args.valid_ratio
    if args.max_rows < 3:
        raise ValueError("--max-rows must be >= 3")
    if args.train_ratio <= 0 or args.valid_ratio <= 0 or test_ratio <= 0:
        raise ValueError("train/valid/test ratios must be positive and sum to 1")

    rows = read_jsonl(Path(args.input))
    if len(rows) < 3:
        raise ValueError(f"Not enough rows in {args.input}: {len(rows)}")

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    subset = rows[: min(args.max_rows, len(rows))]
    n = len(subset)

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

    train_rows = subset[:n_train]
    valid_rows = subset[n_train : n_train + n_valid]
    test_rows = subset[n_train + n_valid :]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "all.jsonl", subset)
    write_jsonl(out_dir / "train.jsonl", train_rows)
    write_jsonl(out_dir / "valid.jsonl", valid_rows)
    write_jsonl(out_dir / "test.jsonl", test_rows)

    print(
        json.dumps(
            {
                "input_rows": len(rows),
                "subset_rows": len(subset),
                "counts": {"train": len(train_rows), "valid": len(valid_rows), "test": len(test_rows)},
                "output_dir": str(out_dir),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
