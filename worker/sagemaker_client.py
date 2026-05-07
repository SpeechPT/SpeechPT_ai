"""SageMaker Async Endpoint 호출 래퍼.

Async invoke 패턴:
1. 입력 JSON을 S3에 업로드
2. invoke_endpoint_async (즉시 200 OK + OutputLocation)
3. OutputLocation의 S3 객체 polling
4. 도착하면 JSON 로드 후 반환
"""
from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any, Dict

import boto3

REGION = os.environ.get("AWS_REGION", "ap-northeast-2")
RESULTS_BUCKET = os.environ["S3_BUCKET_RESULTS"]
ENDPOINT_NAME = os.environ.get("AE_ENDPOINT_NAME", "speechpt-unified-async")

sm_runtime = boto3.client("sagemaker-runtime", region_name=REGION)
s3 = boto3.client("s3", region_name=REGION)


class AsyncInvocationError(Exception):
    pass


def invoke(task: str, payload: Dict[str, Any], timeout_sec: int = 600) -> Dict[str, Any]:
    """통합 EP를 동기처럼 호출. S3-in/invoke/S3-out polling.

    Args:
        task: "stt" | "ce" | "ae"
        payload: task별 추가 입력 (audio_s3, segments, etc.)
        timeout_sec: 결과 대기 한도

    Returns:
        EP가 반환한 JSON dict
    """
    request_id = f"{int(time.time()*1000)}-{uuid.uuid4().hex[:8]}"
    input_key = f"async-input/{task}/{request_id}.json"

    body = {"task": task, **payload}
    s3.put_object(
        Bucket=RESULTS_BUCKET,
        Key=input_key,
        Body=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        ContentType="application/json",
    )

    try:
        response = sm_runtime.invoke_endpoint_async(
            EndpointName=ENDPOINT_NAME,
            InputLocation=f"s3://{RESULTS_BUCKET}/{input_key}",
            ContentType="application/json",
        )
    except Exception as exc:
        raise AsyncInvocationError(f"invoke_endpoint_async failed for task={task}: {exc}") from exc

    output_uri = response["OutputLocation"]
    failure_uri = response.get("FailureLocation")

    bucket, output_key = output_uri.replace("s3://", "").split("/", 1)
    failure_key = None
    if failure_uri:
        _, failure_key = failure_uri.replace("s3://", "").split("/", 1)

    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        # 성공 출력 체크
        try:
            obj = s3.get_object(Bucket=bucket, Key=output_key)
            return json.loads(obj["Body"].read())
        except s3.exceptions.NoSuchKey:
            pass

        # 실패 출력 체크
        if failure_key:
            try:
                fobj = s3.get_object(Bucket=bucket, Key=failure_key)
                error_body = fobj["Body"].read().decode("utf-8", errors="replace")
                raise AsyncInvocationError(f"task={task} EP error: {error_body[:1000]}")
            except s3.exceptions.NoSuchKey:
                pass

        time.sleep(2)

    raise AsyncInvocationError(f"task={task} timeout after {timeout_sec}s")
