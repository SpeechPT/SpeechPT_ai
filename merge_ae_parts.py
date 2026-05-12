"""병렬 prep 잡들이 만든 rows_a.jsonl / rows_b.jsonl을 합치고
train/valid/test로 분할한 뒤 S3에 업로드.

사용법:
    python merge_ae_parts.py \\
        --output-s3 s3://aws-s3-speechpt1/pipeline/ae/parallel-XXXX/processed/ \\
        --part-keys rows_a.jsonl rows_b.jsonl \\
        --train-ratio 0.9 --valid-ratio 0.1
"""
from __future__ import annotations

import argparse
import json
import random
import tempfile
from pathlib import Path

import boto3


def parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"Invalid S3 URI: {uri}")
    body = uri[5:]
    bucket, _, prefix = body.partition("/")
    return bucket, prefix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-s3", required=True,
                        help="S3 prefix where part files live and merged files will go.")
    parser.add_argument("--part-keys", nargs="+", default=["rows_a.jsonl", "rows_b.jsonl"],
                        help="Part file names under output-s3.")
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    bucket, prefix = parse_s3_uri(args.output_s3.rstrip("/") + "/")
    s3 = boto3.client("s3")

    # 1. 부분 파일들 다운로드 + 합치기
    all_rows: list[dict] = []
    for key_name in args.part_keys:
        key = f"{prefix}{key_name}"
        print(f"[fetch] s3://{bucket}/{key}")
        try:
            obj = s3.get_object(Bucket=bucket, Key=key)
            body = obj["Body"].read().decode("utf-8")
        except s3.exceptions.NoSuchKey:
            raise SystemExit(f"부분 파일 없음: s3://{bucket}/{key}")
        cnt = 0
        for line in body.splitlines():
            line = line.strip()
            if not line:
                continue
            all_rows.append(json.loads(line))
            cnt += 1
        print(f"  → {cnt} rows")

    print(f"[merge] total rows: {len(all_rows)}")
    if len(all_rows) < 3:
        raise SystemExit(f"Too few rows: {len(all_rows)}")

    # 2. 셔플 + 분할
    test_ratio = max(0.0, 1.0 - args.train_ratio - args.valid_ratio)
    rng = random.Random(args.seed)
    rng.shuffle(all_rows)
    n = len(all_rows)
    n_train = max(1, int(n * args.train_ratio))
    n_valid = max(1, int(n * args.valid_ratio))
    if n_train + n_valid >= n:
        n_valid = max(1, n - n_train - 1)
    n_test = n - n_train - n_valid
    if n_test < 0:
        n_valid = max(1, n - n_train)
        n_test = 0

    train_rows = all_rows[:n_train]
    valid_rows = all_rows[n_train : n_train + n_valid]
    test_rows = all_rows[n_train + n_valid :]
    print(f"[split] train={len(train_rows)} valid={len(valid_rows)} test={len(test_rows)}")

    # 3. 로컬 임시 파일에 쓰고 S3에 업로드
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        for name, rows in [
            ("all.jsonl", all_rows),
            ("train.jsonl", train_rows),
            ("valid.jsonl", valid_rows),
            ("test.jsonl", test_rows),
        ]:
            if not rows:
                continue
            p = tmp / name
            with p.open("w", encoding="utf-8") as fp:
                for r in rows:
                    fp.write(json.dumps(r, ensure_ascii=False) + "\n")
            key = f"{prefix}{name}"
            s3.upload_file(str(p), bucket, key)
            print(f"[upload] s3://{bucket}/{key} ({p.stat().st_size} bytes)")

    print(f"[done] merged & split & uploaded under s3://{bucket}/{prefix}")


if __name__ == "__main__":
    main()
