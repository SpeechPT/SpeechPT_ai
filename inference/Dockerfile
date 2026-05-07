# SpeechPT 통합 추론 컨테이너 (STT + CE + AE)
#
# 베이스: AWS Deep Learning Container (PyTorch 2.1 + CUDA 12.1 + SageMaker)
# 한국 리전(ap-northeast-2)의 ECR 미러에서 풀.
#
# 빌드:
#   $(aws ecr get-login-password --region ap-northeast-2 \
#     | docker login --username AWS --password-stdin \
#     763104351884.dkr.ecr.ap-northeast-2.amazonaws.com)
#   docker build -t speechpt-inference:v1 .
FROM 763104351884.dkr.ecr.ap-northeast-2.amazonaws.com/pytorch-inference:2.3.0-gpu-py311-cu121-ubuntu20.04-sagemaker

# 환경변수 — 가드레일 ③: 메모리 단편화 억제
ENV PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,expandable_segments:True"
ENV TRANSFORMERS_CACHE=/opt/hf_cache
ENV HF_HOME=/opt/hf_cache
ENV PYTHONUNBUFFERED=1

# 시스템 의존성 (librosa, soundfile용 ffmpeg + libsndfile)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성
COPY code/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# 가드레일 ②: HuggingFace Whisper + ko-sroberta를 *빌드 타임*에 다운로드
# → 컨테이너 시작 시 HF Hub 호출 0초
RUN python -c "\
from transformers import WhisperForConditionalGeneration, WhisperProcessor; \
WhisperForConditionalGeneration.from_pretrained('openai/whisper-small'); \
WhisperProcessor.from_pretrained('openai/whisper-small'); \
print('✓ Whisper-small cached')"

RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('jhgan/ko-sroberta-multitask'); \
print('✓ ko-sroberta cached')"

RUN python -c "\
from transformers import Wav2Vec2Model, Wav2Vec2Processor; \
Wav2Vec2Model.from_pretrained('kresnik/wav2vec2-large-xlsr-korean'); \
Wav2Vec2Processor.from_pretrained('kresnik/wav2vec2-large-xlsr-korean'); \
print('✓ wav2vec2-large-xlsr-korean cached')"

# 우리 추론 코드 (model_fn, predict_fn 등)
COPY code/ /opt/ml/code/

# SageMaker가 진입점으로 사용
ENV SAGEMAKER_PROGRAM=inference.py
ENV SAGEMAKER_SUBMIT_DIRECTORY=/opt/ml/code

# 헬스체크 (SageMaker 표준)
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD curl -f http://localhost:8080/ping || exit 1
