# SpeechPT AI Engine

SpeechPT AI Engine은 발표 문서(PPT/PDF)와 음성을 함께 분석해 코칭 리포트를 생성합니다.

---

## 목차
1. [프로젝트 구조](#프로젝트-구조)
2. [사전 준비 (처음 한 번만)](#사전-준비-처음-한-번만)
3. [S3 / 모델 경로 레퍼런스](#s3--모델-경로-레퍼런스)
4. [AE 학습 파이프라인 (Pipeline)](#ae-학습-파이프라인-pipeline)
   - [파이프라인 실행](#파이프라인-실행)
   - [LoRA 파인튜닝](#lora-파인튜닝)
   - [진행 상황 확인](#진행-상황-확인)
   - [평가 결과 확인](#평가-결과-확인)
   - [등록된 모델 확인](#등록된-모델-확인)
5. [AE 개별 실행 (단독 모드)](#ae-개별-실행-단독-모드)
6. [파인튜닝 (기존 모델에서 이어서)](#파인튜닝-기존-모델에서-이어서)
7. [추론 (모델로 점수 뽑기)](#추론-모델로-점수-뽑기)
8. [엔진 파이프라인 실행](#엔진-파이프라인-실행)

---

## 프로젝트 구조

```
speechpt/
├── coherence/       CE: 슬라이드 내용 ↔ 발화 정합성 분석
├── attitude/        AE: 음성 전달력 분석 (속도/침묵/에너지/억양/간투사)
├── stt/             Whisper 기반 STT
├── report/          통합 리포트 생성
└── training/        AE 학습/평가/추론 스크립트

pipeline_ae.py                   SageMaker Pipeline (전처리→학습→평가→품질게이트→모델등록)
submit_ae_preprocessing.py       전처리 잡 SageMaker 제출 (단독 실행용)
submit_ae_training.py            학습 잡 SageMaker 제출 (단독 실행용)
submit_ae_eval.py                평가 잡 SageMaker 제출 (단독 실행용)
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
pip install sagemaker boto3 torch transformers librosa peft
```

> SageMaker 잡은 로컬에서 코드를 실행하지 않고 AWS 클라우드에 제출만 함.
> GPU나 대용량 데이터를 로컬에 받을 필요 없음.

### 3. 레포 클론
```bash
git clone https://github.com/SpeechPT/SpeechPT_ai.git
cd SpeechPT_ai
```

### 4. 모델 레지스트리 생성 (최초 1회)
```bash
aws sagemaker create-model-package-group \
  --model-package-group-name SpeechPT-AE-Models \
  --region ap-northeast-2
```

---

## S3 / 모델 경로 레퍼런스

| 용도 | S3 경로 |
|------|---------|
| Training 라벨 JSON | `s3://aws-s3-speechpt1/datasets/raws/Training/02.라벨링데이터/` |
| Training 오디오 WAV | `s3://aws-s3-speechpt1/datasets/raws/Training/01.원천데이터/` |
| Validation 라벨 JSON | `s3://aws-s3-speechpt1/datasets/raws/Validation/02.라벨링데이터/` |
| Validation 오디오 WAV | `s3://aws-s3-speechpt1/datasets/raws/Validation/01.원천데이터/` |
| 파이프라인 출력 | `s3://aws-s3-speechpt1/pipeline/ae/{timestamp}/` |
| 전처리 결과 (단독) | `s3://aws-s3-speechpt1/datasets/processed/ae/audio-v3/` |
| 전체 모델 목록 (단독) | `s3://aws-s3-speechpt1/models/ae/v1/` |
| 모델 레지스트리 | `SpeechPT-AE-Models` (SageMaker Model Registry) |

> 학습된 모델은 모두 S3에 저장됨. 로컬에 없어도 S3 URI로 바로 사용 가능.

---

## AE 학습 파이프라인 (Pipeline)

`pipeline_ae.py` 한 번 실행으로 **전처리 → 학습 → 평가 → 품질 게이트 → 모델 등록**을 자동화.

```
Step 1          →   Step 2          →   Step 3          →   Step 4          →   Step 5
전처리               학습                 평가                 품질 게이트          모델 등록
ml.c5.4xlarge        ml.g5.2xlarge        ml.g5.xlarge         자동 판정            자동
(Training+           (Linear Probing      (Validation          (improvement_pct     (ModelPackage
 Validation 처리)     또는 LoRA)           독립 평가셋)          >= 50%)              Group에 등록)
```

- **Training 데이터** (84,134건): 학습용 (8:1:1 → train/valid/test 분할)
- **Validation 데이터** (8,028건): 독립 평가셋 (eval_validation.jsonl)

### 파이프라인 실행

```bash
# 소규모 테스트 (라벨 100개만)
python pipeline_ae.py --max-files 100

# 전체 데이터, Linear Probing (기본)
python pipeline_ae.py

# 파라미터 커스텀
python pipeline_ae.py --epochs 15 --lr 1e-4 --batch-size 16

# 파이프라인 등록만 (실행 안 함)
python pipeline_ae.py --upsert-only
```

**주요 파라미터:**
| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `--max-files` | 0 (무제한) | 전처리할 라벨 파일 수 제한 |
| `--epochs` | 10 | 학습 epoch 수 |
| `--lr` | 1e-3 | learning rate |
| `--batch-size` | 8 | 배치 크기 |
| `--use-lora` | off | LoRA 파인튜닝 활성화 |
| `--lora-r` | 16 | LoRA rank (8, 16, 32) |
| `--quality-threshold` | 50.0 | 모델 등록 기준 improvement_pct (%) |
| `--approval-status` | PendingManualApproval | 모델 등록 승인 상태 |

---

### LoRA 파인튜닝

LoRA는 wav2vec2 backbone의 q_proj/v_proj에 경량 어댑터를 추가하여 학습하는 방식.
Linear Probing(20만 파라미터)과 달리 LoRA는 ~300만 파라미터를 추가로 학습한다.

```bash
# LoRA 기본 실행
python pipeline_ae.py --use-lora --lr 1e-4

# LoRA r=8 (파라미터 적게)
python pipeline_ae.py --use-lora --lora-r 8 --lr 1e-4

# LoRA r=32 (파라미터 많이)
python pipeline_ae.py --use-lora --lora-r 32 --lr 1e-4
```

> LoRA 사용 시 `--lr`은 1e-4로 낮춰야 함 (Linear Probing의 1e-3보다 낮게).

---

### 진행 상황 확인

```bash
# 파이프라인 실행 목록
aws sagemaker list-pipeline-executions \
  --pipeline-name SpeechPT-AE-Pipeline \
  --region ap-northeast-2

# 각 Step 상태
aws sagemaker list-pipeline-execution-steps \
  --pipeline-execution-arn <execution-arn> \
  --region ap-northeast-2

# CloudWatch 실시간 로그
aws logs tail /aws/sagemaker/TrainingJobs \
  --log-stream-name-prefix <job-name> \
  --follow --region ap-northeast-2
```

**Step 상태 의미:**
| 상태 | 의미 |
|------|------|
| `Starting` | 인스턴스 부팅 중 (5~10분) |
| `Executing` | 실행 중 |
| `Succeeded` | 완료 |
| `Failed` | 실패, FailureReason 확인 필요 |

---

### 평가 결과 확인

파이프라인 완료 후 평가 결과는 S3에 자동 저장됨:

```bash
# 최신 평가 결과 다운로드
aws s3 ls s3://aws-s3-speechpt1/pipeline/ae/ --region ap-northeast-2
# 가장 최근 timestamp 폴더 확인 후:

aws s3 cp s3://aws-s3-speechpt1/pipeline/ae/<timestamp>/evaluation/eval_result.json - \
  --region ap-northeast-2
```

결과 예시:
```json
{
  "num_test_rows": 8028,
  "backbone_model": "kresnik/wav2vec2-large-xlsr-korean",
  "use_lora": true,
  "base_loss": 1.724,
  "finetuned_loss": 0.408,
  "improvement_pct": 76.36
}
```

- `base_loss`: 랜덤 초기화 probe의 loss (기준선)
- `finetuned_loss`: 학습된 모델의 loss (낮을수록 좋음)
- `improvement_pct`: 개선율 (이 값이 `quality-threshold` 이상이면 모델 자동 등록)

---

### 등록된 모델 확인

품질 게이트를 통과한 모델은 SageMaker Model Registry에 자동 등록됨:

```bash
aws sagemaker list-model-packages \
  --model-package-group-name SpeechPT-AE-Models \
  --region ap-northeast-2
```

---

## AE 개별 실행 (단독 모드)

> Pipeline을 사용하지 않고 각 Step을 개별 실행할 때 사용. 디버깅이나 특정 Step만 재실행할 때 유용.

### 전처리

```bash
/usr/local/bin/python3 submit_ae_preprocessing.py
```

커스터마이즈:
```bash
AE_PREP_OUTPUT_S3=s3://aws-s3-speechpt1/datasets/processed/ae/audio-v4/ \
AE_PREP_MAX_FILES=20000 \
AE_PREP_INSTANCE_TYPE=ml.c5.4xlarge \
AE_PREP_VOLUME_SIZE_GB=100 \
/usr/local/bin/python3 submit_ae_preprocessing.py
```

### 학습

```bash
AE_INPUT_S3=s3://aws-s3-speechpt1/datasets/processed/ae/audio-v3/ \
/usr/local/bin/python3 submit_ae_training.py
```

**주요 환경변수:**
| 변수 | 기본값 | 설명 |
|------|--------|------|
| `AE_INPUT_S3` | `datasets/processed/ae/` | 전처리된 JSONL 위치 |
| `AE_MODEL` | `kresnik/wav2vec2-large-xlsr-korean` | 백본 모델 |
| `AE_EPOCHS` | `10` | 학습 epoch 수 |
| `AE_LR` | `1e-3` | learning rate |
| `AE_BATCH_SIZE` | `8` | 배치 크기 |
| `AE_INSTANCE_TYPE` | `ml.g5.xlarge` | GPU 인스턴스 |

### 평가

```bash
AE_MODEL_ARTIFACT_S3=s3://aws-s3-speechpt1/models/ae/v1/<job-name>/output/model.tar.gz \
AE_INPUT_S3=s3://aws-s3-speechpt1/datasets/processed/ae/audio-v3/ \
AE_EVAL_WAIT=false \
/usr/local/bin/python3 submit_ae_eval.py
```

### 잡 상태 확인

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
