"""Run AE preprocessing inside a SageMaker training job and upload JSONL to S3."""
from __future__ import annotations

import argparse
import io
import json
import os
import random
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
import librosa
import numpy as np

from prepare_ae_from_raws import (
    compute_overall_delivery,
    fallback_audio_targets,
    label_to_audio_name,
    row_from_label_with_reason,
    write_jsonl,
)


_S3_CONFIG = boto3.session.Session().client(
    "s3",
    config=boto3.session.Config(
        connect_timeout=10,
        read_timeout=30,
        retries={"max_attempts": 2},
    ),
)


def derive_audio_targets_from_s3(bucket: str, key: str) -> dict | None:
    """S3 오디오를 메모리 스트리밍으로 읽어 silence_ratio, energy_drop, pitch_shift 계산.

    다운로드 없이 boto3 get_object → BytesIO → librosa.load 순서로 처리한다.
    timeout 설정된 _S3_CONFIG 클라이언트를 사용해 hang 방지.
    실패 시 None 반환 (row 자체를 제외하도록 호출부에서 처리).
    """
    try:
        obj = _S3_CONFIG.get_object(Bucket=bucket, Key=key)
        audio_bytes = io.BytesIO(obj["Body"].read())
        # 억양 변화 판정에는 30초로 충분 — 전체 로드 대비 3~5배 빠름
        y, sr = librosa.load(audio_bytes, sr=16000, mono=True, duration=30.0)
    except Exception:
        return None

    if len(y) == 0:
        return None

    # silence_ratio: RMS 에너지 기반
    frame_len = int(0.032 * sr)
    hop_len = int(0.010 * sr)
    if len(y) < frame_len:
        y = np.pad(y, (0, frame_len - len(y)))
    starts = np.arange(0, len(y) - frame_len + 1, hop_len)
    frames = np.stack([y[s:s + frame_len] for s in starts])
    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
    energy_db = 20.0 * np.log10(rms + 1e-9)
    energy_db = energy_db - float(np.max(energy_db))
    silence_ratio = float(np.mean(energy_db < -40.0))

    # energy_drop: 에너지 추세 기울기
    times = np.arange(len(energy_db), dtype=np.float32) * (hop_len / float(sr))
    slope, _ = np.polyfit(times, energy_db, 1) if len(times) > 1 else (0.0, 0.0)
    energy_drop = 1 if slope < -0.02 else 0

    # pitch_shift: pyin F0 표준편차 (zero-crossing 대체)
    try:
        f0, voiced_flag, _ = librosa.pyin(
            y.astype(np.float32),
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr,
        )
        f0_voiced = f0[voiced_flag & ~np.isnan(f0)]
        pitch_shift = 1 if (len(f0_voiced) > 10 and float(np.std(f0_voiced)) > 80.0) else 0
    except Exception:
        pitch_shift = 0

    return {
        "silence_ratio": float(max(0.0, min(1.0, silence_ratio))),
        "energy_drop": int(energy_drop),
        "pitch_shift": int(pitch_shift),
    }


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


def _list_s3_wav_names(audio_s3_uri: str) -> set[str]:
    bucket, prefix = parse_s3_uri(audio_s3_uri.rstrip("/") + "/")
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    names: set[str] = set()
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            key = item.get("Key", "")
            if key.endswith(".wav"):
                names.add(Path(key).name)
    return names


def build_rows_from_s3(
    labels_s3_uri: str,
    max_files: int = 0,
    require_audio_exists: bool = False,
    audio_s3_uri: str = "",
    use_audio_features: bool = False,
    num_workers: int = 16,
) -> tuple[list[dict], dict]:
    label_bucket, label_prefix = parse_s3_uri(labels_s3_uri.rstrip("/") + "/")
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")

    audio_bucket, audio_prefix = ("", "")
    if audio_s3_uri:
        audio_bucket, audio_prefix = parse_s3_uri(audio_s3_uri.rstrip("/") + "/")

    decoded_fail = 0
    row_none = 0
    filtered_missing_audio = 0
    audio_names: set[str] = set()
    if require_audio_exists and audio_s3_uri:
        audio_names = _list_s3_wav_names(audio_s3_uri)

    # Phase 1: 레이블 순차 수집 (I/O 가볍고 순서 보장 필요)
    candidates: list[dict] = []  # {wav_name, speech_rate} 리스트
    scanned = 0
    for page in paginator.paginate(Bucket=label_bucket, Prefix=label_prefix):
        for item in page.get("Contents", []):
            key = item.get("Key", "")
            if not key.endswith(".json"):
                continue
            body = s3.get_object(Bucket=label_bucket, Key=key)["Body"].read()
            try:
                obj = json.loads(body.decode("utf-8"))
            except Exception:
                decoded_fail += 1
                continue

            wav_name = label_to_audio_name(Path(key), obj)
            if require_audio_exists and wav_name not in audio_names:
                filtered_missing_audio += 1
                scanned += 1
                if max_files > 0 and scanned >= max_files:
                    break
                continue

            answer = obj.get("dataSet", {}).get("answer", {})
            try:
                word_count = float(answer.get("raw", {}).get("wordCount", 0.0))
            except Exception:
                word_count = 0.0
            try:
                duration_ms = float(obj.get("rawDataInfo", {}).get("answer", {}).get("duration", 0.0))
            except Exception:
                duration_ms = 0.0
            if duration_ms <= 0:
                row_none += 1
                scanned += 1
                if max_files > 0 and scanned >= max_files:
                    break
                continue

            duration_sec = max(1e-6, duration_ms / 1000.0)
            candidates.append({"wav_name": wav_name, "speech_rate": float(word_count / duration_sec)})
            scanned += 1
            if max_files > 0 and scanned >= max_files:
                break
        else:
            continue
        break

    print(json.dumps({"stage": "label_scan_done", "candidates": len(candidates)}, ensure_ascii=False))

    # Phase 2: 오디오 분석 병렬 처리 (ThreadPoolExecutor)
    audio_signal_fail = 0
    rows: list[dict] = []

    def _process(cand: dict) -> dict | None:
        wav_name = cand["wav_name"]
        speech_rate = cand["speech_rate"]
        if use_audio_features and audio_bucket:
            audio_key = audio_prefix + wav_name
            audio_targets = derive_audio_targets_from_s3(audio_bucket, audio_key)
            if audio_targets is None:
                return None
        else:
            audio_targets = fallback_audio_targets(speech_rate=speech_rate)
        overall_delivery = compute_overall_delivery(
            speech_rate=speech_rate,
            silence_ratio=audio_targets["silence_ratio"],
            energy_drop=audio_targets["energy_drop"],
            pitch_shift=audio_targets["pitch_shift"],
        )
        return {
            "audio_path": f"audio/{wav_name}",
            "speech_rate": speech_rate,
            "silence_ratio": float(audio_targets["silence_ratio"]),
            "energy_drop": int(audio_targets["energy_drop"]),
            "pitch_shift": int(audio_targets["pitch_shift"]),
            "overall_delivery": float(overall_delivery),
        }

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_process, c): c for c in candidates}
        done = 0
        for future in as_completed(futures):
            result = future.result()
            if result is None:
                audio_signal_fail += 1
            else:
                rows.append(result)
            done += 1
            if done % 500 == 0:
                print(json.dumps({"stage": "audio_analysis_progress",
                                  "done": done, "total": len(candidates),
                                  "rows_ok": len(rows)}, ensure_ascii=False))

    return rows, {
        "scanned": scanned,
        "decoded_fail": decoded_fail,
        "row_none": row_none,
        "filtered_missing_audio": filtered_missing_audio,
        "audio_signal_fail": audio_signal_fail,
        "audio_name_count": len(audio_names),
    }


def main():
    parser = argparse.ArgumentParser(description="AE preprocessing in SageMaker training job")
    parser.add_argument("--labels-dir", default=os.environ.get("SM_CHANNEL_LABELS", "/opt/ml/input/data/labels"))
    parser.add_argument("--audio-dir", default=os.environ.get("SM_CHANNEL_AUDIO", "/opt/ml/input/data/audio"))
    parser.add_argument("--output-s3-uri", default="")
    parser.add_argument("--output-dir", default="", help="Local output dir (pipeline mode). Set to skip S3 upload.")
    parser.add_argument("--labels-s3-uri", default=os.environ.get("AE_LABELS_S3", ""))
    parser.add_argument("--audio-s3-uri", default=os.environ.get("AE_AUDIO_S3", ""))
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--target-rows", type=int, default=0)
    parser.add_argument("--require-readable-audio", choices=["true", "false"], default="false")
    parser.add_argument("--use-audio-features", choices=["true", "false"], default="false")
    parser.add_argument("--allow-label-only-fallback", choices=["true", "false"], default="false")
    parser.add_argument("--s3-rescue-on-empty", choices=["true", "false"], default="true")
    parser.add_argument("--val-labels-s3-uri", default="", help="Validation labels S3 URI (별도 평가셋)")
    parser.add_argument("--val-audio-s3-uri", default="", help="Validation audio S3 URI (별도 평가셋)")
    args = parser.parse_args()
    use_audio_features = args.use_audio_features.lower() == "true"
    require_readable_audio = args.require_readable_audio.lower() == "true"
    allow_label_only_fallback = args.allow_label_only_fallback.lower() == "true"
    s3_rescue_on_empty = args.s3_rescue_on_empty.lower() == "true"

    test_ratio = 1.0 - args.train_ratio - args.valid_ratio
    if args.train_ratio <= 0 or args.valid_ratio <= 0 or test_ratio < 0:
        raise ValueError("train/valid ratios must be positive and sum to <= 1")

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
    row_none = 0
    unreadable_wav = 0
    readable_missing_path = 0
    readable_open_fail = 0
    sample_row_none: list[str] = []
    row_none_reason_counts: dict[str, int] = {}
    sample_missing_path: list[str] = []
    sample_open_fail: list[str] = []
    # 오디오 채널이 마운트되지 않았으면 로컬 루프를 건너뛰고
    # S3 직접 스트리밍 경로(build_rows_from_s3)를 사용한다.
    # use_audio_features=true + 오디오 채널 없음 조합도 S3 스트리밍으로 처리 가능.
    skip_local_loop = (
        len(audio_files) == 0
        and bool(args.labels_s3_uri)
        and (s3_rescue_on_empty or use_audio_features)
    )

    for f in ([] if skip_local_loop else label_files):
        row, reason = row_from_label_with_reason(
            f,
            audio_dir=audio_dir,
            audio_path_mode="relative",
            audio_path_prefix="audio",
            audio_index=audio_index,
            use_audio_features=use_audio_features,
        )
        if row is None:
            row_none += 1
            row_none_reason_counts[reason] = row_none_reason_counts.get(reason, 0) + 1
            if len(sample_row_none) < 5:
                sample_row_none.append(f"{str(f)} [{reason}]")
            continue
        if require_readable_audio:
            wav_name = Path(row.get("audio_path", "")).name
            wav_path = audio_index.get(wav_name)
            if wav_path is None:
                wav_path = audio_dir / wav_name
            if not wav_path.exists():
                readable_missing_path += 1
                unreadable_wav += 1
                if len(sample_missing_path) < 5:
                    sample_missing_path.append(str(wav_path))
                continue
            try:
                with wave.open(str(wav_path), "rb"):
                    pass
            except Exception:
                readable_open_fail += 1
                unreadable_wav += 1
                if len(sample_open_fail) < 5:
                    sample_open_fail.append(str(wav_path))
                continue
        rows.append(row)
        if args.target_rows > 0 and len(rows) >= args.target_rows:
            break

    print(
        json.dumps(
            {
                "stage": "pre_fallback_summary",
                "label_files": len(label_files),
                "audio_files": len(audio_files),
                "prepared_rows": len(rows),
                "row_none": row_none,
                "row_none_reason_counts": row_none_reason_counts,
                "unreadable_wav": unreadable_wav,
                "readable_missing_path": readable_missing_path,
                "readable_open_fail": readable_open_fail,
                "require_readable_audio": require_readable_audio,
                "use_audio_features": use_audio_features,
                "samples": {
                    "row_none": sample_row_none,
                    "missing_path": sample_missing_path,
                    "open_fail": sample_open_fail,
                },
                "skip_local_loop": skip_local_loop,
            },
            ensure_ascii=False,
        )
    )

    fallback_used = False
    s3_fallback_stats = None
    use_s3_fallback = len(rows) < 3 and args.labels_s3_uri and (allow_label_only_fallback or s3_rescue_on_empty or skip_local_loop)
    if use_s3_fallback:
        rows, s3_fallback_stats = build_rows_from_s3(
            args.labels_s3_uri,
            max_files=args.max_files,
            require_audio_exists=require_readable_audio,
            audio_s3_uri=args.audio_s3_uri,
            use_audio_features=use_audio_features,
        )
        fallback_used = True
        print(
            json.dumps(
                {
                    "stage": "s3_fallback_summary",
                    "prepared_rows": len(rows),
                    "require_audio_exists": require_readable_audio,
                    "stats": s3_fallback_stats,
                },
                ensure_ascii=False,
            )
        )

    if len(rows) < 3:
        print(
            json.dumps(
                {
                    "stage": "final_failure_summary",
                    "prepared_rows": len(rows),
                    "allow_label_only_fallback": allow_label_only_fallback,
                    "s3_rescue_on_empty": s3_rescue_on_empty,
                    "fallback_used": fallback_used,
                    "s3_fallback_stats": s3_fallback_stats,
                    "target_rows": args.target_rows,
                },
                ensure_ascii=False,
            )
        )
        raise ValueError(f"Not enough prepared rows: {len(rows)}")

    rng = random.Random(args.seed)
    rng.shuffle(rows)

    n = len(rows)
    n_train = max(1, int(n * args.train_ratio))
    n_valid = max(1, int(n * args.valid_ratio))
    if n_train + n_valid >= n:
        n_valid = max(1, n - n_train - 1)
    n_test = n - n_train - n_valid
    if n_test < 0:
        # train + valid이 n보다 큰 경우 valid에서 조정
        n_valid = max(1, n - n_train)
        n_test = 0

    train_rows = rows[:n_train]
    valid_rows = rows[n_train : n_train + n_valid]
    test_rows = rows[n_train + n_valid :]

    # Pipeline mode: write to --output-dir, skip S3 upload
    # Standalone mode: write to /tmp, upload to S3 via --output-s3-uri
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = Path("/tmp/ae_prepared")
    out_dir.mkdir(parents=True, exist_ok=True)
    all_path = out_dir / "all.jsonl"
    train_path = out_dir / "train.jsonl"
    valid_path = out_dir / "valid.jsonl"
    test_path = out_dir / "test.jsonl"
    write_jsonl(all_path, rows)
    write_jsonl(train_path, train_rows)
    write_jsonl(valid_path, valid_rows)
    if test_rows:
        write_jsonl(test_path, test_rows)

    # Validation 데이터 처리 (별도 AIHub Validation 세트 → eval_validation.jsonl)
    eval_val_rows: list[dict] = []
    val_stats: dict | None = None
    if args.val_labels_s3_uri:
        print(json.dumps({"stage": "validation_processing_start", "val_labels_s3_uri": args.val_labels_s3_uri}, ensure_ascii=False))
        eval_val_rows, val_stats = build_rows_from_s3(
            args.val_labels_s3_uri,
            max_files=0,
            audio_s3_uri=args.val_audio_s3_uri,
            use_audio_features=use_audio_features,
        )
        eval_val_path = out_dir / "eval_validation.jsonl"
        write_jsonl(eval_val_path, eval_val_rows)
        print(json.dumps({"stage": "validation_processing_done", "eval_validation_rows": len(eval_val_rows), "stats": val_stats}, ensure_ascii=False))

    if not args.output_dir and args.output_s3_uri:
        bucket, prefix = parse_s3_uri(args.output_s3_uri.rstrip("/") + "/")
        s3 = boto3.client("s3")
        s3.upload_file(str(all_path), bucket, f"{prefix}all.jsonl")
        s3.upload_file(str(train_path), bucket, f"{prefix}train.jsonl")
        s3.upload_file(str(valid_path), bucket, f"{prefix}valid.jsonl")
        if test_rows:
            s3.upload_file(str(test_path), bucket, f"{prefix}test.jsonl")
        if eval_val_rows:
            s3.upload_file(str(out_dir / "eval_validation.jsonl"), bucket, f"{prefix}eval_validation.jsonl")

    print(
        json.dumps(
            {
                "label_files": len(label_files),
                "audio_files": len(audio_files),
                "prepared_rows": len(rows),
                "row_none": row_none,
                "row_none_reason_counts": row_none_reason_counts,
                "unreadable_wav": unreadable_wav,
                "readable_missing_path": readable_missing_path,
                "readable_open_fail": readable_open_fail,
                "target_rows": args.target_rows,
                "require_readable_audio": require_readable_audio,
                "use_audio_features": use_audio_features,
                "allow_label_only_fallback": allow_label_only_fallback,
                "s3_rescue_on_empty": s3_rescue_on_empty,
                "fallback_used": fallback_used,
                "s3_fallback_stats": s3_fallback_stats,
                "counts": {"train": len(train_rows), "valid": len(valid_rows), "test": len(test_rows), "eval_validation": len(eval_val_rows)},
                "output_dir": args.output_dir or "",
                "output_s3": args.output_s3_uri,
                "val_stats": val_stats,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
