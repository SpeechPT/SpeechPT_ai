# SpeechPT AI Engine

SpeechPT AI Engine은 발표 문서(PPT/PDF)와 음성을 함께 분석해 코칭 리포트를 생성합니다.

---

## 목차
1. [프로젝트 구조](#프로젝트-구조)
2. [사전 준비 (처음 한 번만)](#사전-준비-처음-한-번만)
3. [S3 / 모델 경로 레퍼런스](#s3--모델-경로-레퍼런스)
4. [AE 학습 파이프라인](#ae-학습-파이프라인)
   - [Step 1. 전처리](#step-1-전처리)
   - [Step 2. 학습](#step-2-학습)
   - [Step 3. 평가](#step-3-평가)
   - [Step 4. 잡 상태 확인](#step-4-잡-상태-확인)
5. [파인튜닝 (기존 모델에서 이어서)](#파인튜닝-기존-모델에서-이어서)
6. [추론 (모델로 점수 뽑기)](#추론-모델로-점수-뽑기)
7. [엔진 파이프라인 실행](#엔진-파이프라인-실행)

---

## 프로젝트 구조

```
speechpt/
├── coherence/       CE: 슬라이드 내용 ↔ 발화 정합성 분석
├── attitude/        AE: 음성 전달력 분석 (속도/침묵/에너지/억양/간투사)
├── stt/             Whisper 기반 STT
├── report/          통합 리포트 생성
└── training/        AE 학습/평가/추론 스크립트

submit_ae_preprocessing.py   전처리 잡 SageMaker 제출
submit_ae_training.py        학습 잡 SageMaker 제출
submit_ae_eval.py            평가 잡 SageMaker 제출
```

**AE 출력 점수 5개:**
| 점수 | 설명 |
|------|------|
| `speech_rate` | 발화 속도 |
| `silence_ratio` | 침묵 비율 |
| `energy_drop` | 에너지 하락 여부 (0/1) |
| `pitch_shift` | 억양 변화 여부 (0/1) |
| `overall_delivery` | 종합 전달력 |

---

## 사전 준비 (처음 한 번만)

### 1. AWS CLI 설정
```bash
aws configure
# AWS Access Key ID, Secret Access Key, Region: ap-northeast-2 입력
```

### 2. Python 환경
```bash
# Python 3.10+ 필요
pip install sagemaker boto3 torch transformers librosa
```

> SageMaker 잡은 로컬에서 코드를 실행하지 않고 AWS 클라우드에 제출만 함.
> GPU나 대용량 데이터를 로컬에 받을 필요 없음.

### 3. 레포 클론
```bash
git clone https://github.com/SpeechPT/SpeechPT_ai.git
cd SpeechPT_ai
```

---

## S3 / 모델 경로 레퍼런스

| 용도 | S3 경로 |
|------|---------|
| Raw 라벨 JSON | `s3://aws-s3-speechpt1/datasets/raws/Training/02.라벨링데이터/` |
| Raw 오디오 WAV | `s3://aws-s3-speechpt1/datasets/raws/Training/01.원천데이터/` |
| 전처리 결과 (최신) | `s3://aws-s3-speechpt1/datasets/processed/ae/audio-v3/` |
| **현재 최신 모델** | `s3://aws-s3-speechpt1/models/ae/v1/speechpt-ae-train-v1-20260407-090927/output/model.tar.gz` |
| 전체 모델 목록 | `s3://aws-s3-speechpt1/models/ae/v1/` |
| 체크포인트 | `s3://aws-s3-speechpt1/checkpoints/ae/v1/` |

> 학습된 모델은 모두 S3에 저장됨. 로컬에 없어도 S3 URI로 바로 사용 가능.

S3 모델 목록 확인:
```bash
aws s3 ls s3://aws-s3-speechpt1/models/ae/v1/
```

---

## AE 학습 파이프라인

### Step 1. 전처리

> 데이터가 바뀌었을 때만 실행. 이미 완료된 `audio-v3`가 있으면 스킵 가능.

```bash
# 기본 실행
/usr/local/bin/python3 submit_ae_preprocessing.py
```

커스터마이즈 (환경변수로 설정):
```bash
AE_PREP_OUTPUT_S3=s3://aws-s3-speechpt1/datasets/processed/ae/audio-v4/ \
AE_PREP_MAX_FILES=20000 \
AE_PREP_INSTANCE_TYPE=ml.c5.4xlarge \
AE_PREP_VOLUME_SIZE_GB=100 \
/usr/local/bin/python3 submit_ae_preprocessing.py
```

완료 확인:
```bash
aws s3 ls s3://aws-s3-speechpt1/datasets/processed/ae/audio-v3/
# train.jsonl / valid.jsonl / test.jsonl 세 파일이 있으면 완료
```

---

### Step 2. 학습

```bash
# 기본 실행 (audio-v3 데이터, kresnik 한국어 모델, 10 epochs)
AE_INPUT_S3=s3://aws-s3-speechpt1/datasets/processed/ae/audio-v3/ \
/usr/local/bin/python3 submit_ae_training.py
```

커스터마이즈:
```bash
AE_INPUT_S3=s3://aws-s3-speechpt1/datasets/processed/ae/audio-v3/ \
AE_EPOCHS=15 \
AE_LR=1e-3 \
AE_BATCH_SIZE=8 \
AE_INSTANCE_TYPE=ml.g5.xlarge \
/usr/local/bin/python3 submit_ae_training.py
```

**주요 환경변수:**
| 변수 | 기본값 | 설명 |
|------|--------|------|
| `AE_INPUT_S3` | `datasets/processed/ae/` | 전처리된 JSONL 위치 |
| `AE_MODEL` | `kresnik/wav2vec2-large-xlsr-korean` | 백본 모델 (변경 금지) |
| `AE_EPOCHS` | `10` | 학습 epoch 수 |
| `AE_LR` | `1e-3` | learning rate |
| `AE_BATCH_SIZE` | `8` | 배치 크기 |
| `AE_INSTANCE_TYPE` | `ml.g5.xlarge` | GPU 인스턴스 |

학습이 완료되면 모델이 자동으로 S3에 저장됨:
```
s3://aws-s3-speechpt1/models/ae/v1/<job-name>/output/model.tar.gz
```

job-name 확인:
```bash
/usr/local/bin/python3 -c "
import boto3
sm = boto3.client('sagemaker', region_name='ap-northeast-2')
for j in sm.list_training_jobs(SortBy='CreationTime', SortOrder='Descending', MaxResults=5)['TrainingJobSummaries']:
    print(j['TrainingJobName'], '|', j['TrainingJobStatus'])
"
```

---

### Step 3. 평가

학습 완료 후 바로 실행:
```bash
AE_MODEL_ARTIFACT_S3=s3://aws-s3-speechpt1/models/ae/v1/<job-name>/output/model.tar.gz \
AE_INPUT_S3=s3://aws-s3-speechpt1/datasets/processed/ae/audio-v3/ \
AE_EVAL_WAIT=false \
/usr/local/bin/python3 submit_ae_eval.py
```

결과 확인 (완료 후):
```bash
/usr/local/bin/python3 -c "
import boto3, tarfile, json
from pathlib import Path

# eval job name으로 교체
EVAL_JOB = 'speechpt-ae-eval-v1-XXXXXXXX-XXXXXX'

s3 = boto3.client('s3', region_name='ap-northeast-2')
out = Path('/tmp/ae_eval_result')
out.mkdir(exist_ok=True)
tar = out / 'model.tar.gz'
s3.download_file('aws-s3-speechpt1', f'models/ae-eval/v1/{EVAL_JOB}/output/model.tar.gz', str(tar))
with tarfile.open(tar, 'r:gz') as tf:
    tf.extractall(out)
print(json.loads((out / 'eval_result.json').read_text()))
"
```

결과 예시:
```json
{
  "num_test_rows": 653,
  "base_loss": 1.6794,
  "finetuned_loss": 0.3668,
  "improvement_abs": 1.3126,
  "improvement_pct": 78.16
}
```

---

### Step 4. 잡 상태 확인

```bash
# 최근 잡 목록
/usr/local/bin/python3 -c "
import boto3
sm = boto3.client('sagemaker', region_name='ap-northeast-2')
for j in sm.list_training_jobs(SortBy='CreationTime', SortOrder='Descending', MaxResults=10)['TrainingJobSummaries']:
    print(j['TrainingJobName'], '|', j['TrainingJobStatus'])
"

# CloudWatch 실시간 로그
aws logs tail /aws/sagemaker/TrainingJobs \
  --log-stream-name-prefix <job-name> \
  --follow
```

**잡 상태 의미:**
| 상태 | 의미 |
|------|------|
| `Starting` | 인스턴스 부팅 중 (5~10분) |
| `Downloading` | 코드/데이터 준비 중 |
| `Training` | 실제 학습/평가 실행 중 |
| `Completed` | 완료, 결과 S3에 저장됨 |
| `Failed` | 실패, FailureReason 확인 필요 |

---

## 파인튜닝 (기존 모델에서 이어서)

기존에 학습된 모델 가중치를 불러와서 새 데이터로 추가 학습:

```bash
AE_INPUT_S3=s3://aws-s3-speechpt1/datasets/processed/ae/audio-v3/ \
AE_RESUME_FROM=s3://aws-s3-speechpt1/models/ae/v1/speechpt-ae-train-v1-20260407-090927/output/model.tar.gz \
AE_EPOCHS=5 \
AE_LR=3e-4 \
/usr/local/bin/python3 submit_ae_training.py
```

> 파인튜닝 시 `AE_LR`은 신규학습(1e-3)보다 낮게 설정 (3e-4 권장).

---

## 추론 (모델로 점수 뽑기)

S3에서 모델을 자동 다운로드해서 추론:
```bash
python -m speechpt.training.ae_probe_infer \
  --audio /path/to/sample.wav \
  --model-dir /tmp/ae_model \
  --model-artifact-s3 s3://aws-s3-speechpt1/models/ae/v1/speechpt-ae-train-v1-20260407-090927/output/model.tar.gz
```

슬라이드 섹션별 점수 (슬라이드1=0~40s, 슬라이드2=40~90s):
```bash
python -m speechpt.training.ae_probe_infer \
  --audio /path/to/sample.wav \
  --model-dir /tmp/ae_model \
  --model-artifact-s3 s3://aws-s3-speechpt1/models/ae/v1/speechpt-ae-train-v1-20260407-090927/output/model.tar.gz \
  --slide-timestamps 0,40,90,130
```

---

## 엔진 파이프라인 실행

발표 음성 + 슬라이드 PDF → 코칭 리포트:
```bash
python -m speechpt.pipeline \
  --config configs/pipeline_v0.1.yaml \
  --document /path/to/slides.pdf \
  --audio /path/to/audio.wav \
  --slide-timestamps 0,40,90,130 \
  --whisper-json /path/to/whisper_words.json
```

`whisper_words.json` 형식:
```json
{
  "words": [
    {"word": "안녕하세요", "start": 0.0, "end": 0.4},
    {"word": "발표", "start": 0.5, "end": 0.8}
  ]
}
```

---

## Filler Word Detection (간투사 탐지)

STT 출력에서 "어", "음", "그", "저" 등을 탐지하고 슬라이드별로 집계:

```python
from speechpt.attitude.filler_detector import detect_fillers

words = [
    {"word": "안녕하세요", "start": 0.0, "end": 0.4},
    {"word": "어", "start": 1.2, "end": 1.4},
]

result = detect_fillers(words, slide_timestamps=[0, 40, 90])
# result: {total_fillers, filler_rate, filler_words, per_slide}
```
