"""SpeechPT 분석 Worker — SQS 컨슈머.

책임은 큐 컨슈머 수준으로 얇게:
1. SQS 메시지 수신
2. analyses 상태 업데이트 (queued → running → done/failed)
3. SageMaker EP에 task=pipeline 호출 (한 번)
4. EP가 반환한 SpeechReport JSON을 그대로 S3/DB에 저장

AI 도메인 로직(문서 파싱, 슬라이드 정렬, 점수 환산, LLM 피드백 작성)은
모두 SageMaker EP 안 SpeechPTPipeline.analyze()가 담당한다.
"""
from __future__ import annotations

import json
import os
import time
import traceback
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List

import boto3
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from worker.sagemaker_client import invoke as sm_invoke

# ──────────────────────────────────────────────────────────────
# 환경 변수
# ──────────────────────────────────────────────────────────────
REGION = os.environ.get("AWS_REGION", "ap-northeast-2")
QUEUE_URL = os.environ["ANALYSIS_QUEUE_URL"]
RESULTS_BUCKET = os.environ["S3_BUCKET_RESULTS"]
DATABASE_URL = os.environ["DATABASE_URL"]
PIPELINE_TIMEOUT_SEC = int(os.environ.get("PIPELINE_TIMEOUT_SEC", "1500"))

WORKER_ID = f"worker-{os.environ.get('HOSTNAME', uuid.uuid4().hex[:8])}"

sqs = boto3.client("sqs", region_name=REGION)
s3 = boto3.client("s3", region_name=REGION)

engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_recycle=3600)
SessionLocal = sessionmaker(bind=engine)


def log(msg: str) -> None:
    print(f"[{datetime.utcnow().isoformat()}][{WORKER_ID}] {msg}", flush=True)


# ──────────────────────────────────────────────────────────────
# DB 헬퍼
# ──────────────────────────────────────────────────────────────
def update_analysis(analysis_id: str, **fields: Any) -> None:
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
# SpeechReport → BE 스키마 매핑 (얇은 어댑터, 비즈니스 로직 없음)
# ──────────────────────────────────────────────────────────────
def _to_be_schema(report: Dict[str, Any]) -> Dict[str, Any]:
    """EP가 반환한 SpeechReport.to_dict() 를 BE analysis_results 행 형식으로 매핑.

    점수 계산식·임계값 판정·텍스트 생성은 EP 안에서 이미 끝나 있다.
    여기서는 키 이름만 BE 스키마에 맞춘다.
    """
    overall = report.get("overall_scores", {}) or {}
    coverage_pct = int(round(float(overall.get("content_coverage", 0))))
    delivery_pct = int(round(float(overall.get("delivery_stability", 0))))
    pacing_pct = int(round(float(overall.get("pacing_score", 0))))
    overall_pct = int(round((coverage_pct + delivery_pct + pacing_pct) / 3))

    highlights: List[Dict[str, Any]] = list(report.get("highlight_sections", []) or [])
    high = sum(1 for h in highlights if int(h.get("severity", 0)) >= 5)
    medium = sum(1 for h in highlights if 2 <= int(h.get("severity", 0)) < 5)
    low = sum(1 for h in highlights if int(h.get("severity", 0)) < 2)
    severity = {
        "high_severity_count": high,
        "medium_severity_count": medium,
        "low_severity_count": low,
    }

    llm = report.get("llm_feedback") or {}
    global_summary = report.get("global_summary") or {}

    # FE가 기대하는 report_json 구조: summary, strengths, improvements, sections
    summary = llm.get("overall_comment") or global_summary.get("summary_text") or ""
    strengths = [{"text": s} for s in (llm.get("strengths") or []) if s]
    improvements = [{"text": a} for a in (llm.get("priority_actions") or []) if a]

    slide_comment_map = {
        int(item.get("slide_id")): item.get("comment", "")
        for item in (llm.get("slide_comments") or [])
        if item.get("slide_id") is not None
    }

    sections: List[Dict[str, Any]] = []
    transcript_segments = report.get("transcript_segments") or []
    seg_by_slide = {int(seg.get("slide_id", 0)): seg for seg in transcript_segments}

    for detail in report.get("per_slide_detail", []) or []:
        slide_id = int(detail.get("slide_id", 0))
        seg = seg_by_slide.get(slide_id, {})
        # 슬라이드별 점수: coverage 그대로 (없으면 None)
        coverage = detail.get("coverage")
        score = int(round(float(coverage) * 100)) if coverage is not None else None
        sections.append({
            "section_index": slide_id,
            "title": f"슬라이드 {slide_id}",
            "start_time_sec": int(seg.get("start_sec", 0) or 0),
            "end_time_sec": int(seg.get("end_sec", 0) or 0),
            "score": score,
            "feedback": slide_comment_map.get(slide_id, ""),
        })

    report_json = {
        "summary": summary,
        "strengths": strengths,
        "improvements": improvements,
        "sections": sections,
        "raw": report,  # 원본 SpeechReport 통째로 보관 (FE/디버깅용)
    }

    return {
        "content_coverage": coverage_pct,
        "delivery_stability": delivery_pct,
        "pacing_score": pacing_pct,
        "overall_score": overall_pct,
        "severity": severity,
        "report_json": report_json,
    }


# ──────────────────────────────────────────────────────────────
# 메인 처리
# ──────────────────────────────────────────────────────────────
def process(payload: dict) -> None:
    aid = payload["analysis_id"]
    log(f"START analysis {aid}")

    update_analysis(
        aid,
        status="running", progress=10, stage="ingest",
        worker_id=WORKER_ID,
        started_at=datetime.now(timezone.utc),
    )

    audio_uri = f"s3://{payload['audio']['bucket']}/{payload['audio']['object_key']}"
    doc_uri = f"s3://{payload['document']['bucket']}/{payload['document']['object_key']}"

    update_analysis(aid, progress=30, stage="analyzing")

    # SageMaker EP에 통합 파이프라인 한 번 호출
    report = sm_invoke(
        "pipeline",
        {"audio_s3": audio_uri, "document_s3": doc_uri},
        timeout_sec=PIPELINE_TIMEOUT_SEC,
    )

    update_analysis(aid, progress=85)

    # 원본 결과 S3 백업
    result_blob_key = f"results/{aid}.json"
    s3.put_object(
        Bucket=RESULTS_BUCKET,
        Key=result_blob_key,
        Body=json.dumps(report, ensure_ascii=False).encode("utf-8"),
        ContentType="application/json",
    )

    # BE 스키마로 얇게 매핑 후 저장
    mapped = _to_be_schema(report)
    upsert_analysis_result(
        aid,
        content_coverage=mapped["content_coverage"],
        delivery_stability=mapped["delivery_stability"],
        pacing_score=mapped["pacing_score"],
        overall_score=mapped["overall_score"],
        severity_json=mapped["severity"],
        report_json=mapped["report_json"],
        result_blob_s3_key=result_blob_key,
    )

    update_analysis(
        aid, status="done", progress=100, stage="finished",
        finished_at=datetime.now(timezone.utc),
    )
    log(f"DONE analysis {aid}")


def main() -> None:
    log("Worker started")
    log(f"  QUEUE_URL = {QUEUE_URL}")
    log(f"  RESULTS_BUCKET = {RESULTS_BUCKET}")
    log(f"  AE_ENDPOINT_NAME = {os.environ.get('AE_ENDPOINT_NAME', 'speechpt-unified-async')}")

    while True:
        try:
            resp = sqs.receive_message(
                QueueUrl=QUEUE_URL,
                MaxNumberOfMessages=1,
                WaitTimeSeconds=20,
                VisibilityTimeout=1800,
            )
            for msg in resp.get("Messages", []):
                try:
                    payload = json.loads(msg["Body"])
                except json.JSONDecodeError as exc:
                    log(f"INVALID JSON: {exc}")
                    sqs.delete_message(QueueUrl=QUEUE_URL, ReceiptHandle=msg["ReceiptHandle"])
                    continue

                try:
                    process(payload)
                except Exception as exc:
                    log(f"FAILED {payload.get('analysis_id')}: {type(exc).__name__}: {exc}\n{traceback.format_exc()}")
                    aid = payload.get("analysis_id")
                    if aid:
                        try:
                            update_analysis(
                                aid, status="failed",
                                error_code=type(exc).__name__,
                                error_message=str(exc)[:500],
                                finished_at=datetime.now(timezone.utc),
                            )
                        except Exception as db_exc:
                            log(f"db update failed: {db_exc}")
                finally:
                    sqs.delete_message(QueueUrl=QUEUE_URL, ReceiptHandle=msg["ReceiptHandle"])
        except KeyboardInterrupt:
            log("Worker shutting down")
            break
        except Exception as exc:
            log(f"polling error: {exc}, sleep 5s")
            time.sleep(5)


if __name__ == "__main__":
    main()
