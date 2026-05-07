# SpeechPT 통합 추론 컨테이너 (STT + CE + AE)
#
# 모델 가중치는 model_fn() 첫 호출 시 다운로드 (한 번만 발생, ~5분).
# minCap=1로 EP를 항상 warm 유지하므로 사용자 체감 영향 없음.
FROM 763104351884.dkr.ecr.ap-northeast-2.amazonaws.com/pytorch-inference:2.3.0-gpu-py311-cu121-ubuntu20.04-sagemaker

ENV PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,expandable_segments:True"
ENV HF_HOME=/opt/hf_cache
ENV TRANSFORMERS_CACHE=/opt/hf_cache
ENV PYTHONUNBUFFERED=1

# 시스템 의존성
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성 (모델 다운로드는 안 함)
COPY code/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# 추론 코드
COPY code/ /opt/ml/code/

# 캐시 디렉토리 미리 생성 (model_fn이 여기 다운로드)
RUN mkdir -p /opt/hf_cache && chmod 777 /opt/hf_cache

ENV SAGEMAKER_PROGRAM=inference.py
ENV SAGEMAKER_SUBMIT_DIRECTORY=/opt/ml/code

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD curl -f http://localhost:8080/ping || exit 1
