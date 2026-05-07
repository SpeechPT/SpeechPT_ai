"""SpeechPT 분석 Worker.

Step 8 단계: SageMaker EP 미배포 상태이므로 mock 결과로 흐름만 검증한다.
Step 9에서 process()의 mock 부분을 실제 SageMaker invoke로 교체한다.

흐름:
1. SQS 폴링 (long poll 20s)
2. analyses UPDATE → status=running, progress 단계별 갱신
3. (Step 9: S3 다운로드 + SageMaker 호출 + 결과 합산)
4. analysis_results INSERT (점수 + report_json + result_blob_s3_key)
5. analyses UPDATE → status=done, progress=100
6. SQS 메시지 delete
"""
from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any

import boto3
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker


# ──────────────────────────────────────────────────────────────
# 환경 변수
# ──────────────────────────────────────────────────────────────
REGION = os.environ.get("AWS_REGION", "ap-northeast-2")
QUEUE_URL = os.environ["ANALYSIS_QUEUE_URL"]
RESULTS_BUCKET = os.environ["S3_BUCKET_RESULTS"]
DATABASE_URL = os.environ["DATABASE_URL"]

WORKER_ID = f"worker-{os.environ.get('HOSTNAME', uuid.uuid4().hex[:8])}"

# ──────────────────────────────────────────────────────────────
# AWS / DB 클라이언트 (모듈 로드 시 1회)
# ──────────────────────────────────────────────────────────────
sqs = boto3.client("sqs", region_name=REGION)
s3 = boto3.client("s3", region_name=REGION)

engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_recycle=3600)
SessionLocal = sessionmaker(bind=engine)


def log(msg: str) -> None:
    print(f"[{datetime.utcnow().isoformat()}][{WORKER_ID}] {msg}", flush=True)


# ──────────────────────────────────────────────────────────────
# DB 헬퍼 — raw SQL로 BE 모델 import 회피
# ──────────────────────────────────────────────────────────────
def update_analysis(analysis_id: str, **fields: Any) -> None:
    """analyses 테이블의 한 행을 부분 갱신."""
    if not fields:
        return
    cols = ", ".join(f"{k} = :{k}" for k in fields)
    with SessionLocal() as db:
        db.execute(
            text(f"UPDATE analyses SET {cols} WHERE analysis_id = :id"),
            {**fields, "id": analysis_id},
        )
        db.commit()


def upsert_analysis_result(
    analysis_id: str,
    *,
    content_coverage: int,
    delivery_stability: int,
    pacing_score: int,
    overall_score: int,
    severity_json: dict,
    report_json: dict,
    result_blob_s3_key: str,
) -> None:
    """analysis_results 테이블에 결과 INSERT (충돌 시 UPDATE)."""
    sql = text("""
        INSERT INTO analysis_results (
            analysis_id, content_coverage, delivery_stability, pacing_score,
            overall_score, severity_json, report_json, result_blob_s3_key
        ) VALUES (
            :analysis_id, :content_coverage, :delivery_stability, :pacing_score,
            :overall_score, CAST(:severity_json AS JSONB), CAST(:report_json AS JSONB),
            :result_blob_s3_key
        )
        ON CONFLICT (analysis_id) DO UPDATE SET
            content_coverage = EXCLUDED.content_coverage,
            delivery_stability = EXCLUDED.delivery_stability,
            pacing_score = EXCLUDED.pacing_score,
            overall_score = EXCLUDED.overall_score,
            severity_json = EXCLUDED.severity_json,
            report_json = EXCLUDED.report_json,
            result_blob_s3_key = EXCLUDED.result_blob_s3_key
    """)
    with SessionLocal() as db:
        db.execute(sql, {
            "analysis_id": analysis_id,
            "content_coverage": content_coverage,
            "delivery_stability": delivery_stability,
            "pacing_score": pacing_score,
            "overall_score": overall_score,
            "severity_json": json.dumps(severity_json, ensure_ascii=False),
            "report_json": json.dumps(report_json, ensure_ascii=False),
            "result_blob_s3_key": result_blob_s3_key,
        })
        db.commit()


# ──────────────────────────────────────────────────────────────
# Mock 분석 — Step 9에서 실제 SageMaker 호출로 교체
# ──────────────────────────────────────────────────────────────
def run_mock_analysis(payload: dict) -> dict:
    """Step 9 교체 지점.

    실제로는 다음을 수행해야 함:
    - S3에서 audio/document 다운로드
    - SageMaker Async Endpoint 호출 (STT → CE → AE)
    - 결과 합산 → report
    """
    return {
        "scores": {
            "content_coverage": 75,
            "delivery_stability": 80,
            "pacing_score": 72,
            "overall_score": 76,
        },
        "severity": {
            "high_severity_count": 0,
            "medium_severity_count": 1,
            "low_severity_count": 2,
        },
        "report": {
            "summary": "MOCK 결과입니다. Step 9에서 실제 SageMaker 추론으로 교체됩니다.",
            "strengths": [
                {"text": "MOCK: 도입부 발화가 안정적이었습니다."},
                {"text": "MOCK: 슬라이드와 발화의 연결이 자연스러웠습니다."},
            ],
            "improvements": [
                {"text": "MOCK: 후반부 말속도가 빨라졌습니다."},
                {"text": "MOCK: 일부 키포인트가 충분히 강조되지 않았습니다."},
            ],
            "sections": [
                {
                    "section_index": 1,
                    "title": "MOCK 도입부",
                    "start_time_sec": 0,
                    "end_time_sec": 30,
                    "score": 82,
                    "feedback": "MOCK: 안정적인 도입부였습니다.",
                },
                {
                    "section_index": 2,
                    "title": "MOCK 본론",
                    "start_time_sec": 30,
                    "end_time_sec": 90,
                    "score": 75,
                    "feedback": "MOCK: 핵심 내용은 전달되었으나 속도가 일정하지 않았습니다.",
                },
                {
                    "section_index": 3,
                    "title": "MOCK 결론",
                    "start_time_sec": 90,
                    "end_time_sec": 120,
                    "score": 70,
                    "feedback": "MOCK: 마무리에서 발화가 빨라져 전달력이 떨어졌습니다.",
                },
            ],
        },
    }


# ──────────────────────────────────────────────────────────────
# 메인 처리
# ──────────────────────────────────────────────────────────────
def process(payload: dict) -> None:
    aid = payload["analysis_id"]
    log(f"START analysis {aid}")

    # 1. 작업 시작 마킹
    update_analysis(
        aid,
        status="running",
        progress=10,
        stage="ingest",
        worker_id=WORKER_ID,
        started_at=datetime.now(timezone.utc),
    )

    # 2. Step 9에서 여기에 S3 다운로드 + SageMaker 호출 들어감
    #    지금은 단순 sleep으로 progress만 갱신
    time.sleep(3)
    update_analysis(aid, progress=40, stage="analyzing")
    time.sleep(3)
    update_analysis(aid, progress=70)
    time.sleep(3)

    # 3. Mock 분석 결과 생성
    mock = run_mock_analysis(payload)

    # 4. 원본 report를 S3에 업로드 (백업/감사용)
    result_blob_key = f"results/{aid}.json"
    s3.put_object(
        Bucket=RESULTS_BUCKET,
        Key=result_blob_key,
        Body=json.dumps(mock, ensure_ascii=False).encode("utf-8"),
        ContentType="application/json",
    )
    log(f"uploaded result blob → s3://{RESULTS_BUCKET}/{result_blob_key}")

    # 5. analysis_results 테이블에 점수 + report_json INSERT
    upsert_analysis_result(
        aid,
        content_coverage=mock["scores"]["content_coverage"],
        delivery_stability=mock["scores"]["delivery_stability"],
        pacing_score=mock["scores"]["pacing_score"],
        overall_score=mock["scores"]["overall_score"],
        severity_json=mock["severity"],
        report_json=mock["report"],
        result_blob_s3_key=result_blob_key,
    )

    # 6. analyses 마무리
    update_analysis(
        aid,
        status="done",
        progress=100,
        stage="finished",
        finished_at=datetime.now(timezone.utc),
    )
    log(f"DONE analysis {aid}")


def main() -> None:
    log(f"Worker started")
    log(f"  QUEUE_URL = {QUEUE_URL}")
    log(f"  RESULTS_BUCKET = {RESULTS_BUCKET}")
    log(f"  DB connected = {engine.url.host}")

    while True:
        try:
            resp = sqs.receive_message(
                QueueUrl=QUEUE_URL,
                MaxNumberOfMessages=1,
                WaitTimeSeconds=20,        # long polling
                VisibilityTimeout=900,     # 15분 (Step 9의 SageMaker 호출 시간 대비)
            )
            for msg in resp.get("Messages", []):
                try:
                    payload = json.loads(msg["Body"])
                except json.JSONDecodeError as exc:
                    log(f"INVALID JSON in message: {exc}, body={msg['Body'][:200]}")
                    sqs.delete_message(
                        QueueUrl=QUEUE_URL, ReceiptHandle=msg["ReceiptHandle"],
                    )
                    continue

                try:
                    process(payload)
                except Exception as exc:
                    log(f"FAILED analysis {payload.get('analysis_id')}: {type(exc).__name__}: {exc}")
                    aid = payload.get("analysis_id")
                    if aid:
                        try:
                            update_analysis(
                                aid,
                                status="failed",
                                error_code=type(exc).__name__,
                                error_message=str(exc)[:500],
                                finished_at=datetime.now(timezone.utc),
                            )
                        except Exception as db_exc:
                            log(f"failed to mark analysis as failed: {db_exc}")
                    # 메시지는 삭제 (재시도 안 함). DLQ는 maxReceiveCount=3으로 자동.
                finally:
                    sqs.delete_message(
                        QueueUrl=QUEUE_URL, ReceiptHandle=msg["ReceiptHandle"],
                    )
        except KeyboardInterrupt:
            log("Worker shutting down (KeyboardInterrupt)")
            break
        except Exception as exc:
            log(f"polling loop error: {type(exc).__name__}: {exc}, sleep 5s")
            time.sleep(5)


if __name__ == "__main__":
    main()
