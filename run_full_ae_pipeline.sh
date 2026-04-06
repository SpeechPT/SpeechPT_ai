#!/usr/bin/env bash
set -euo pipefail

BUCKET="${BUCKET:-aws-s3-speechpt1}"
WORKDIR="${WORKDIR:-/tmp/speechpt_full_ae}"
LABEL_S3="${LABEL_S3:-s3://$BUCKET/datasets/raws/Training/02.라벨링데이터/}"
AUDIO_S3="${AUDIO_S3:-s3://$BUCKET/datasets/raws/Training/01.원천데이터/}"
PROCESSED_S3="${PROCESSED_S3:-s3://$BUCKET/datasets/processed/ae/}"

mkdir -p "$WORKDIR/labels" "$WORKDIR/audio" "$WORKDIR/out"

echo "[1/5] sync labels"
aws s3 sync "$LABEL_S3" "$WORKDIR/labels" --exclude "*" --include "*.json" --no-progress --only-show-errors

echo "[2/5] sync wav"
aws s3 sync "$AUDIO_S3" "$WORKDIR/audio" --exclude "*" --include "*.wav" --no-progress --only-show-errors

echo "[3/5] preprocess"
python3 -m speechpt.training.prepare_ae_from_raws \
  --label-dir "$WORKDIR/labels" \
  --audio-dir "$WORKDIR/audio" \
  --output-dir "$WORKDIR/out" \
  --audio-path-mode relative \
  --audio-path-prefix audio

echo "[4/5] upload processed"
aws s3 cp "$WORKDIR/out/train.jsonl" "$PROCESSED_S3/train.jsonl" --no-progress --only-show-errors
aws s3 cp "$WORKDIR/out/valid.jsonl" "$PROCESSED_S3/valid.jsonl" --no-progress --only-show-errors
aws s3 cp "$WORKDIR/out/test.jsonl" "$PROCESSED_S3/test.jsonl" --no-progress --only-show-errors
aws s3 sync "$WORKDIR/audio" "$PROCESSED_S3/audio/" --exclude "*" --include "*.wav" --no-progress --only-show-errors
aws s3 ls "$PROCESSED_S3"

echo "[5/5] submit training"
AE_INPUT_S3="$PROCESSED_S3" python3 submit_ae_training.py

echo "done"
