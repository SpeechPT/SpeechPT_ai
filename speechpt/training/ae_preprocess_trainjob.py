"""Run AE preprocessing inside a SageMaker training job and upload JSONL to S3."""
from __future__ import annotations

import argparse
import io
import json
import os
import random
import threading
import wave
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
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


def _make_s3_client():
    """timeout/retry가 설정된 boto3 S3 클라이언트 생성.

    ProcessPoolExecutor 워커마다 fork-safe 하게 새로 만들기 위해 함수로 분리.
    """
    return boto3.session.Session().client(
        "s3",
        config=boto3.session.Config(
            connect_timeout=10,
            read_timeout=30,
            retries={"max_attempts": 2},
        ),
    )


# 클라이언트 캐싱 전략:
# - 메인 프로세스: 모듈 로드 시 생성 안 함 (lazy). spawn 워커가 모듈 임포트할 때
#   IMDS 자격 증명 호출 폭주를 막기 위함.
# - ProcessPool 워커: _init_worker에서 _WORKER_S3_CLIENT 생성 (process-local).
# - ThreadPool 워커: threading.local()로 thread-local 클라이언트 (fork 후 손상된
#   글로벌 클라이언트 공유로 인한 hang 방지).
_S3_CONFIG = None  # 메인 스레드 호출용 (lazy)
_WORKER_S3_CLIENT = None  # ProcessPool 자식 프로세스용
_thread_local = threading.local()  # ThreadPool 스레드용


def _init_worker():
    """ProcessPoolExecutor initializer — 자식 프로세스 진입 시 1회 호출."""
    global _WORKER_S3_CLIENT
    _WORKER_S3_CLIENT = _make_s3_client()


def _get_s3_client():
    """컨텍스트별 fork/thread-safe S3 클라이언트 반환."""
    # ProcessPool 워커 — process-local
    if _WORKER_S3_CLIENT is not None:
        return _WORKER_S3_CLIENT
    # ThreadPool 워커 또는 메인 — thread-local 우선
    cli = getattr(_thread_local, "client", None)
    if cli is None:
        cli = _make_s3_client()
        _thread_local.client = cli
    return cli


# 오디오 분석 구간 (초). 30s → 15s 단축 시 librosa.pyin이 약 50% 빨라짐.
# 발화 평가 라벨링 컨텍스트에서 15s도 silence_ratio/F0std/energy slope 통계량으로 충분.
_AUDIO_ANALYSIS_DURATION_SEC = float(os.environ.get("AE_PREP_AUDIO_DURATION", "15.0"))


def derive_audio_targets_from_s3(bucket: str, key: str) -> dict | None:
    """S3 오디오를 메모리 스트리밍으로 읽어 silence_ratio, energy_drop, pitch_shift 계산.

    다운로드 없이 boto3 get_object → BytesIO → librosa.load 순서로 처리한다.
    timeout 설정된 _S3_CONFIG 클라이언트를 사용해 hang 방지.
    실패 시 None 반환 (row 자체를 제외하도록 호출부에서 처리).
    """
    try:
        obj = _get_s3_client().get_object(Bucket=bucket, Key=key)
        audio_bytes = io.BytesIO(obj["Body"].read())
        # 분석 구간 (기본 15s) — 발화 평가 통계량(silence_ratio/F0 std/energy slope)에 충분.
        y, sr = librosa.load(audio_bytes, sr=16000, mono=True, duration=_AUDIO_ANALYSIS_DURATION_SEC)
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


def _process_audio_candidate(arg_tuple) -> dict | None:
    """모듈 레벨 — ProcessPoolExecutor가 pickle해서 워커로 보낼 수 있도록 클로저 외부화.

    arg_tuple: (cand, audio_bucket, audio_prefix, use_audio_features)
    """
    cand, audio_bucket, audio_prefix, use_audio_features = arg_tuple
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
    start_index: int = 0,
    end_index: int = 0,
    checkpoint_interval: int = 0,
    checkpoint_s3_uri: str = "",
    checkpoint_name: str = "rows_partial",
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

    total_scanned = len(candidates)
    print(json.dumps({"stage": "label_scan_done", "candidates": total_scanned}, ensure_ascii=False))

    # Phase 1.5: 인덱스 범위 슬라이싱 (병렬 처리 / 부분 재시도)
    sliced_start = max(0, start_index)
    sliced_end = end_index if end_index > 0 else len(candidates)
    sliced_end = min(sliced_end, len(candidates))
    if sliced_start > 0 or sliced_end < len(candidates):
        candidates = candidates[sliced_start:sliced_end]
        print(json.dumps({
            "stage": "candidates_sliced",
            "start_index": sliced_start,
            "end_index": sliced_end,
            "sliced_count": len(candidates),
            "total_scanned": total_scanned,
        }, ensure_ascii=False))

    # Phase 1.6: S3 체크포인트 resume — 이미 처리된 행 로드 후 남은 인덱스만 처리
    rows: list[dict] = []
    resume_skip = 0
    audio_signal_fail = 0
    checkpoint_key = None
    checkpoint_bucket = None
    if checkpoint_interval > 0 and checkpoint_s3_uri:
        checkpoint_bucket, ckpt_prefix = parse_s3_uri(checkpoint_s3_uri.rstrip("/") + "/")
        checkpoint_key = f"{ckpt_prefix}{checkpoint_name}.jsonl"
        try:
            obj = boto3.client("s3").get_object(Bucket=checkpoint_bucket, Key=checkpoint_key)
            body = obj["Body"].read().decode("utf-8")
            for line in body.splitlines():
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
            resume_skip = len(rows)
            print(json.dumps({
                "stage": "checkpoint_resumed",
                "checkpoint_s3": f"s3://{checkpoint_bucket}/{checkpoint_key}",
                "rows_loaded": resume_skip,
            }, ensure_ascii=False))
        except boto3.client("s3").exceptions.NoSuchKey:
            print(json.dumps({
                "stage": "checkpoint_none",
                "checkpoint_s3": f"s3://{checkpoint_bucket}/{checkpoint_key}",
            }, ensure_ascii=False))
        except Exception as ckpt_exc:
            print(json.dumps({
                "warning": "checkpoint_load_failed",
                "msg": str(ckpt_exc)[:200],
            }, ensure_ascii=False))
        # resume_skip 만큼은 이미 처리됨 — 남은 candidates만 처리
        if resume_skip > 0:
            candidates = candidates[resume_skip:]

    # Phase 2: 오디오 분석 병렬 처리
    use_processes = os.environ.get("AE_PREP_USE_PROCESSES", "true").lower() == "true"
    if use_processes:
        # ProcessPool: vCPU 수에 맞춰 워커 수 캡 (16개를 4코어에 띄우면 컨텍스트 스위칭 손실)
        cpu_count = os.cpu_count() or 8
        worker_count = min(num_workers, cpu_count)
    else:
        worker_count = num_workers

    print(json.dumps({
        "stage": "audio_analysis_start",
        "executor": "ProcessPool" if use_processes else "ThreadPool",
        "workers": worker_count,
        "audio_duration_sec": _AUDIO_ANALYSIS_DURATION_SEC,
        "candidates": len(candidates),
        "resume_skip": resume_skip,
        "checkpoint_interval": checkpoint_interval,
        "checkpoint_s3": f"s3://{checkpoint_bucket}/{checkpoint_key}" if checkpoint_key else None,
        "mp_start_method": "forkserver" if use_processes else "n/a",
    }, ensure_ascii=False))

    task_args_all = [(c, audio_bucket, audio_prefix, use_audio_features) for c in candidates]
    n_total = len(task_args_all)

    def _save_checkpoint(rows_so_far: list[dict]) -> None:
        """현재까지 처리된 rows 전체를 체크포인트로 S3에 덮어쓰기.

        클라이언트 충돌 시 다음 시작 때 이 파일에서 이어서 처리할 수 있다.
        """
        if not checkpoint_key or not checkpoint_bucket:
            return
        try:
            body = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows_so_far) + "\n"
            boto3.client("s3").put_object(
                Bucket=checkpoint_bucket,
                Key=checkpoint_key,
                Body=body.encode("utf-8"),
            )
            print(json.dumps({
                "stage": "checkpoint_saved",
                "rows": len(rows_so_far),
                "key": checkpoint_key,
            }, ensure_ascii=False))
        except Exception as exc:
            print(json.dumps({
                "warning": "checkpoint_save_failed",
                "msg": str(exc)[:200],
            }, ensure_ascii=False))

    def _run_threadpool(args_list: list, start_offset: int = 0) -> tuple[int, list]:
        """단일 ThreadPool로 args_list 처리. start_offset은 진행 로그 누적용."""
        local_rows: list[dict] = []
        local_fail = 0
        with ThreadPoolExecutor(max_workers=worker_count) as ex:
            # ex.map은 결과를 입력 순서대로 yield. chunksize=16으로 배치 IPC.
            for i, result in enumerate(ex.map(_process_audio_candidate, args_list, chunksize=16)):
                if result is None:
                    local_fail += 1
                else:
                    local_rows.append(result)
                done_total = start_offset + i + 1
                if done_total % 500 == 0:
                    print(json.dumps({
                        "stage": "audio_analysis_progress",
                        "executor": "ThreadPool",
                        "done": done_total,
                        "total": n_total + resume_skip,
                    }, ensure_ascii=False))
                # 체크포인트: 누적 rows (resume + 이전 풀의 결과 + 현재 local_rows)
                if checkpoint_interval > 0 and (i + 1) % checkpoint_interval == 0:
                    _save_checkpoint(rows + local_rows)
        return local_fail, local_rows

    def _run_processpool_resumable(args_list: list) -> tuple[int, list, int]:
        """단일 ProcessPool(forkserver) 처리. 풀이 죽으면 (fail, rows, last_idx) 반환,
        호출부가 last_idx부터 ThreadPool로 잔여 처리.

        forkserver: 자식이 클린 템플릿에서 fork → 메인의 fd/state 상속 안 함 +
        spawn처럼 매번 인터프리터 부팅하지 않아 빠름.
        """
        local_rows: list[dict] = []
        local_fail = 0
        last_completed = 0
        forkserver_ctx = mp.get_context("forkserver")
        try:
            with ProcessPoolExecutor(
                max_workers=worker_count,
                mp_context=forkserver_ctx,
                initializer=_init_worker,
            ) as ex:
                results_iter = ex.map(_process_audio_candidate, args_list, chunksize=64)
                for i, result in enumerate(results_iter):
                    if result is None:
                        local_fail += 1
                    else:
                        local_rows.append(result)
                    last_completed = i + 1
                    if (i + 1) % 500 == 0:
                        print(json.dumps({
                            "stage": "audio_analysis_progress",
                            "executor": "ProcessPool",
                            "done": i + 1,
                            "total": n_total + resume_skip,
                            "rows_ok": len(local_rows),
                            "fail": local_fail,
                        }, ensure_ascii=False))
                    # 체크포인트: 누적 rows
                    if checkpoint_interval > 0 and (i + 1) % checkpoint_interval == 0:
                        _save_checkpoint(rows + local_rows)
        except BrokenProcessPool as exc:
            print(json.dumps({
                "warning": "broken_process_pool",
                "completed_via_processpool": last_completed,
                "remaining": n_total - last_completed,
                "msg": str(exc)[:300],
            }, ensure_ascii=False))
        except Exception as exc:
            # 기타 예외 (e.g. 워커 메모리 폭주) — 진단용 로그 + 폴백 진입
            print(json.dumps({
                "warning": "processpool_unexpected_exception",
                "type": type(exc).__name__,
                "completed": last_completed,
                "msg": str(exc)[:300],
            }, ensure_ascii=False))
        return local_fail, local_rows, last_completed

    if use_processes:
        pp_fail, pp_rows, completed_idx = _run_processpool_resumable(task_args_all)
        audio_signal_fail += pp_fail
        rows.extend(pp_rows)
        if completed_idx < n_total:
            print(json.dumps({
                "stage": "threadpool_fallback_start",
                "remaining": n_total - completed_idx,
            }, ensure_ascii=False))
            tp_fail, tp_rows = _run_threadpool(task_args_all[completed_idx:], start_offset=completed_idx)
            audio_signal_fail += tp_fail
            rows.extend(tp_rows)
    else:
        tp_fail, tp_rows = _run_threadpool(task_args_all, start_offset=0)
        audio_signal_fail += tp_fail
        rows.extend(tp_rows)

    print(json.dumps({
        "stage": "audio_analysis_done",
        "rows_ok_total": len(rows),
        "fail_total": audio_signal_fail,
        "total": n_total + resume_skip,
    }, ensure_ascii=False))

    # 최종 체크포인트 갱신 (완료 표식) — 다음 resume 시 모두 로드됨
    if checkpoint_interval > 0:
        _save_checkpoint(rows)

    return rows, {
        "scanned": scanned,
        "decoded_fail": decoded_fail,
        "row_none": row_none,
        "filtered_missing_audio": filtered_missing_audio,
        "audio_signal_fail": audio_signal_fail,
        "audio_name_count": len(audio_names),
        "resume_skip": resume_skip,
        "start_index": sliced_start,
        "end_index": sliced_end,
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
    # 병렬 처리 / 체크포인팅 / Validation 분리
    parser.add_argument("--start-index", type=int, default=0,
                        help="candidates 슬라이스 시작 (병렬 처리용). 0=처음부터.")
    parser.add_argument("--end-index", type=int, default=0,
                        help="candidates 슬라이스 끝 (exclusive). 0=끝까지.")
    parser.add_argument("--output-suffix", default="",
                        help="병렬 잡 구분용 파일명 접미사 (예: '_a', '_b').")
    parser.add_argument("--checkpoint-interval", type=int, default=5000,
                        help="매 N개 처리 후 부분 JSONL을 S3에 업로드해 작업 손실 방어. 0=비활성.")
    parser.add_argument("--skip-validation-prep", choices=["true", "false"], default="false",
                        help="true면 Validation 데이터 처리 스킵 (Training 전용 잡).")
    parser.add_argument("--validation-only", choices=["true", "false"], default="false",
                        help="true면 Validation 데이터만 처리 (Training 스킵). val-labels-s3-uri 필수.")
    args = parser.parse_args()
    use_audio_features = args.use_audio_features.lower() == "true"
    require_readable_audio = args.require_readable_audio.lower() == "true"
    allow_label_only_fallback = args.allow_label_only_fallback.lower() == "true"
    s3_rescue_on_empty = args.s3_rescue_on_empty.lower() == "true"

    test_ratio = 1.0 - args.train_ratio - args.valid_ratio
    # 부동소수점 오차 허용 (예: 1.0 - 0.9 - 0.1 = -2.78e-17)
    if args.train_ratio <= 0 or args.valid_ratio <= 0 or test_ratio < -1e-9:
        raise ValueError("train/valid ratios must be positive and sum to <= 1")
    test_ratio = max(0.0, test_ratio)

    validation_only = args.validation_only.lower() == "true"
    skip_validation_prep = args.skip_validation_prep.lower() == "true"
    suffix = args.output_suffix  # 예: "_a", "_b", 또는 "" (분할 모드)
    parallel_mode = bool(suffix) or args.start_index > 0 or args.end_index > 0

    # 출력 경로 계산 (Training 부분)
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = Path("/tmp/ae_prepared")
    out_dir.mkdir(parents=True, exist_ok=True)

    def _upload_to_s3(local_path: Path, key_name: str) -> None:
        """output_s3_uri가 설정됐을 때 로컬 파일을 S3에 업로드."""
        if not args.output_s3_uri:
            return
        bucket, prefix = parse_s3_uri(args.output_s3_uri.rstrip("/") + "/")
        boto3.client("s3").upload_file(str(local_path), bucket, f"{prefix}{key_name}")
        print(json.dumps({
            "stage": "s3_upload",
            "key": f"{prefix}{key_name}",
            "local": str(local_path),
            "size": local_path.stat().st_size,
        }, ensure_ascii=False))

    label_dir = Path(args.labels_dir)
    audio_dir = Path(args.audio_dir)
    label_files = sorted(label_dir.rglob("*.json"))
    if args.max_files > 0:
        label_files = label_files[: args.max_files]
    # validation_only이면 Training 라벨 디렉토리가 비어있어도 OK
    if not label_files and not validation_only:
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
    # validation_only=true면 Training 처리 자체를 스킵
    use_s3_fallback = (
        not validation_only
        and len(rows) < 3
        and args.labels_s3_uri
        and (allow_label_only_fallback or s3_rescue_on_empty or skip_local_loop)
    )
    if use_s3_fallback:
        rows, s3_fallback_stats = build_rows_from_s3(
            args.labels_s3_uri,
            max_files=args.max_files,
            require_audio_exists=require_readable_audio,
            audio_s3_uri=args.audio_s3_uri,
            use_audio_features=use_audio_features,
            start_index=args.start_index,
            end_index=args.end_index,
            checkpoint_interval=args.checkpoint_interval,
            checkpoint_s3_uri=args.output_s3_uri,
            checkpoint_name=f"rows_partial{suffix}",
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

    # === Training rows를 즉시 S3에 업로드 === (Validation 처리 전에 보존)
    # parallel_mode(suffix 또는 인덱스 범위): rows{suffix}.jsonl로 저장, 분할 안 함
    # 풀 모드(suffix=""): 기존대로 train/valid/test 분할 후 업로드
    if not validation_only and len(rows) >= 1:
        if parallel_mode:
            # 부분 jsonl만 저장 — 분할은 merge 단계에서.
            rows_path = out_dir / f"rows{suffix}.jsonl"
            write_jsonl(rows_path, rows)
            _upload_to_s3(rows_path, f"rows{suffix}.jsonl")
            print(json.dumps({
                "stage": "training_partial_uploaded",
                "rows": len(rows),
                "suffix": suffix,
                "start_index": args.start_index,
                "end_index": args.end_index,
            }, ensure_ascii=False))

    # === 풀 모드 (parallel_mode=False) only: train/valid/test 분할 + 업로드 ===
    train_rows: list[dict] = []
    valid_rows: list[dict] = []
    test_rows: list[dict] = []
    all_path = out_dir / "all.jsonl"
    train_path = out_dir / "train.jsonl"
    valid_path = out_dir / "valid.jsonl"
    test_path = out_dir / "test.jsonl"

    if not validation_only and not parallel_mode:
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
            n_valid = max(1, n - n_train)
            n_test = 0

        train_rows = rows[:n_train]
        valid_rows = rows[n_train : n_train + n_valid]
        test_rows = rows[n_train + n_valid :]

        write_jsonl(all_path, rows)
        write_jsonl(train_path, train_rows)
        write_jsonl(valid_path, valid_rows)
        if test_rows:
            write_jsonl(test_path, test_rows)
        # 즉시 S3 업로드 (Validation 처리 전에 보존)
        _upload_to_s3(all_path, "all.jsonl")
        _upload_to_s3(train_path, "train.jsonl")
        _upload_to_s3(valid_path, "valid.jsonl")
        if test_rows:
            _upload_to_s3(test_path, "test.jsonl")
        print(json.dumps({
            "stage": "training_split_uploaded",
            "counts": {"train": len(train_rows), "valid": len(valid_rows), "test": len(test_rows)},
        }, ensure_ascii=False))

    # === Validation 데이터 처리 (별도 AIHub Validation 세트 → eval_validation.jsonl) ===
    # skip_validation_prep=true (병렬 잡)이면 스킵. 별도 Validation-only 잡에서 처리.
    # parallel_mode이면 자동 스킵 (rows{suffix}.jsonl만 출력하고 종료).
    eval_val_rows: list[dict] = []
    val_stats: dict | None = None
    should_run_validation = (
        not skip_validation_prep
        and not parallel_mode
        and bool(args.val_labels_s3_uri)
    )
    if validation_only:
        # validation_only 모드: Training 스킵, Validation만 처리
        should_run_validation = bool(args.val_labels_s3_uri)
    if should_run_validation:
        print(json.dumps({"stage": "validation_processing_start", "val_labels_s3_uri": args.val_labels_s3_uri}, ensure_ascii=False))
        eval_val_rows, val_stats = build_rows_from_s3(
            args.val_labels_s3_uri,
            max_files=0,
            audio_s3_uri=args.val_audio_s3_uri,
            use_audio_features=use_audio_features,
            checkpoint_interval=args.checkpoint_interval,
            checkpoint_s3_uri=args.output_s3_uri,
            checkpoint_name="rows_partial_val",
        )
        eval_val_path = out_dir / "eval_validation.jsonl"
        write_jsonl(eval_val_path, eval_val_rows)
        _upload_to_s3(eval_val_path, "eval_validation.jsonl")
        print(json.dumps({"stage": "validation_processing_done", "eval_validation_rows": len(eval_val_rows), "stats": val_stats}, ensure_ascii=False))

    # (즉시 업로드 패턴으로 변경됨 — Training/Validation 각 처리 직후 _upload_to_s3 호출)

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
