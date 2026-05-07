# Step 3~5. 평가 → 품질 게이트 → 모델 등록 상세

## 개요

학습이 완료된 모델을 **독립 평가셋(Validation)**으로 검증하고,
품질 기준을 통과하면 자동으로 모델 레지스트리에 등록한다.

```
Step 3                       Step 4                         Step 5
평가 (AE-Evaluate)       →   품질 게이트 (AE-QualityGate) →   모델 등록 (AE-RegisterModel)
ProcessingStep               ConditionStep                   ModelStep
ml.g5.xlarge (GPU)           (자동 판정, 인스턴스 없음)         (자동 등록, 인스턴스 없음)

                             improvement_pct >= 50%?
                               ├── Yes → 모델 등록 실행
                               └── No  → 파이프라인 종료 (등록 스킵)
```

---

# Step 3. 평가 (AE-Evaluate)

## 목적

학습된 모델이 **"아무것도 학습하지 않은 모델"보다 얼마나 나은지** 측정한다.

```
비교 대상 A: base_probe        → 랜덤 초기화된 AEProbe (학습 안 함)
비교 대상 B: finetuned_probe   → Step 2에서 학습된 AEProbe (+ LoRA 시 backbone도)

improvement_pct = (base_loss - finetuned_loss) / base_loss × 100
```

- improvement_pct가 높을수록 학습이 잘 된 것
- 0%이면 학습 효과 없음, 음수이면 학습이 오히려 성능을 악화시킴

---

## 실행 환경

| 항목 | 값 |
|------|-----|
| SageMaker Job 유형 | **Processing Job** |
| 인스턴스 | `ml.g5.xlarge` (A10G 24GB GPU, 4vCPU, 16GB RAM) |
| 프레임워크 | PyTorch 2.1, Python 3.10 |
| 스크립트 | `ae_probe_eval.py` |
| 볼륨 크기 | 30GB |

> 이전에는 `ml.t3.xlarge` (CPU)를 사용했으나, wav2vec2 forward를 GPU로 가속하기 위해 업그레이드.

---

## 입력 데이터

### SageMaker ProcessingInput으로 마운트되는 데이터

```
/opt/ml/processing/input/
├── model/                          ← Step 2의 model.tar.gz (자동 마운트)
│   ├── ae_probe.pt
│   ├── meta.pt
│   └── lora_adapter/  (LoRA 시)
│       ├── adapter_config.json
│       └── adapter_model.safetensors
│
└── data/                           ← Step 1의 출력 JSONL (자동 마운트)
    ├── train.jsonl
    ├── valid.jsonl
    └── eval_validation.jsonl       ← 이 파일을 평가에 사용
```

### Validation 오디오 (S3 스트리밍)

```
s3://aws-s3-speechpt1/datasets/raws/Validation/01.원천데이터/
└── ckmk_a_ard_f_e_14751.wav, ckmk_a_ard_f_e_14752.wav, ...  (8,028개)
```

- ProcessingInput으로 마운트하지 않고, `--audio-s3` 인자로 S3 URI를 전달
- AudioDataset이 각 샘플마다 boto3로 S3에서 직접 스트리밍
- Validation 오디오 S3 경로를 사용 (Training 오디오가 아님)

---

## 처리 흐름 상세

### 전체 흐름도

```
main()
  │
  ├── 1. 평가 데이터 로딩
  │     └── eval_validation.jsonl → rows (8,028건)
  │
  ├── 2. DataLoader 생성
  │     └── AudioDataset (chunk_sec=20, batch_size=2)
  │
  ├── 3. base_loss 측정
  │     ├── wav2vec2 로드 (frozen)
  │     ├── AEProbe 랜덤 초기화 (seed=42)
  │     └── eval_loss() → base_loss
  │
  ├── 4. 모델 artifact 로드
  │     ├── model.tar.gz 압축 해제 (tar_path가 있으면)
  │     ├── ae_probe.pt → finetuned_probe
  │     └── lora_adapter/ 자동 감지
  │
  ├── 5. LoRA 적용 (감지된 경우)
  │     ├── wav2vec2 새로 로드
  │     ├── PeftModel.from_pretrained(backbone, lora_adapter/)
  │     └── merge_and_unload() → 원본 backbone에 LoRA를 합침
  │
  ├── 6. finetuned_loss 측정
  │     └── eval_loss() → finetuned_loss
  │
  ├── 7. 결과 계산 + 저장
  │     ├── improvement_pct 계산
  │     └── eval_result.json 출력
  │
  └── 8. PropertyFile로 파이프라인에 결과 전달
```

---

### 평가 파일 선택 로직

```
--eval-file 인자가 있으면 → 해당 파일 사용
없으면 → eval_validation.jsonl 사용
```

파이프라인에서는 `--eval-file eval_validation.jsonl`을 명시적으로 전달한다.

---

### base_loss 측정 (기준선)

"학습을 전혀 하지 않은 모델"의 성능을 측정한다.

```python
torch.manual_seed(42)                     # 동일 seed로 재현 가능
base_probe = AEProbe(in_dim=1024)          # 랜덤 초기화
base_loss = eval_loss(loader, processor, backbone, base_probe, device)
```

- backbone은 사전학습된 wav2vec2 (frozen)
- probe만 랜덤 초기화 → 당연히 loss가 높음
- 이 값이 "아무것도 안 했을 때의 loss" 기준선

---

### 모델 artifact 로드

Pipeline 모드에서는 model.tar.gz가 ProcessingInput으로 마운트된다.

```
/opt/ml/processing/input/model/
├── model.tar.gz     ← SageMaker가 마운트한 원본
```

로드 과정:
```
1. model/ 디렉토리에서 .tar.gz 파일 검색
2. 있으면 → /tmp/ae_eval_model/에 압축 해제
3. 없으면 → model/ 디렉토리에서 직접 ae_probe.pt 검색
4. ae_probe.pt 로드 → finetuned_probe에 가중치 적용
```

ae_probe.pt의 형식은 두 가지:
- `state_dict` 직접 저장: `probe.load_state_dict(state)`
- epoch/loss 포함 dict: `probe.load_state_dict(state["model_state_dict"])`

---

### LoRA 자동 감지 및 적용

model artifact 내 `lora_adapter/` 디렉토리와 `adapter_config.json`이 존재하면 LoRA 모드로 판단.

```python
lora_dir = model_base_dir / "lora_adapter"
use_lora = lora_dir.exists() and (lora_dir / "adapter_config.json").exists()
```

**LoRA 적용 과정:**
```python
# 1. 새로운 wav2vec2 로드 (base_loss 측정에 사용한 것과 별도)
finetuned_backbone = Wav2Vec2Model.from_pretrained("kresnik/wav2vec2-large-xlsr-korean")

# 2. LoRA 어댑터 로드
finetuned_backbone = PeftModel.from_pretrained(finetuned_backbone, "lora_adapter/")

# 3. LoRA를 원본 가중치에 합치기 (merge)
finetuned_backbone = finetuned_backbone.merge_and_unload()
#   → 추론 시 LoRA 오버헤드 제거, 순수 wav2vec2와 동일한 속도
```

**merge_and_unload()의 의미:**
```
학습 시:  output = W × input + (B × A) × input    ← 두 경로 분리
merge:   W_new = W + B × A                        ← 하나로 합침
추론 시:  output = W_new × input                   ← 원본과 동일 구조, 속도 동일
```

**Linear Probing의 경우:**
- lora_adapter/ 없음 → LoRA 감지 안 됨
- finetuned_backbone = backbone (base와 동일한 frozen wav2vec2 사용)
- probe만 학습된 가중치로 교체

---

### eval_loss 함수

base_loss와 finetuned_loss 모두 동일한 함수로 측정한다.

```python
def eval_loss(loader, processor, backbone, probe, device, sample_rate):
    probe.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch in loader:
            wavs, targets = batch
            # WAV → wav2vec2 → pooled embedding
            inputs = processor(wavs, sampling_rate=16000, return_tensors="pt", padding=True)
            hidden = backbone(**inputs).last_hidden_state
            pooled = hidden.mean(dim=1)

            # probe → 점수 5개
            preds = probe(pooled)

            # 손실 계산 (학습과 동일한 공식)
            reg_pred = preds[:, [0, 1, 4]]   # speech_rate, silence_ratio, overall_delivery
            reg_true = targets[:, [0, 1, 4]]
            cls_pred = preds[:, [2, 3]]       # energy_drop, pitch_shift
            cls_true = targets[:, [2, 3]]

            loss = MSELoss(reg_pred, reg_true) + BCEWithLogitsLoss(cls_pred, cls_true)
            total_loss += loss
            n_batches += 1

    return total_loss / n_batches
```

- 학습 Step과 **동일한 손실 함수** (MSE + BCE) 사용
- `torch.no_grad()`로 gradient 계산 없이 forward만 수행

---

### 결과 계산

```python
improvement_abs = base_loss - finetuned_loss
improvement_pct = improvement_abs / base_loss × 100
```

| 지표 | 의미 | 예시 |
|------|------|------|
| `base_loss` | 랜덤 probe의 loss | 1.724 |
| `finetuned_loss` | 학습된 모델의 loss | 0.408 |
| `improvement_abs` | 절대 개선량 | 1.316 |
| `improvement_pct` | 개선율 (%) | 76.36% |

---

## 출력

### eval_result.json

```json
{
  "num_test_rows": 8028,
  "audio_indexed_files": 0,
  "backbone_model": "kresnik/wav2vec2-large-xlsr-korean",
  "probe_input_dim": 1024,
  "use_lora": true,
  "base_loss": 1.724,
  "finetuned_loss": 0.408,
  "improvement_abs": 1.316,
  "improvement_pct": 76.36
}
```

| 필드 | 설명 |
|------|------|
| `num_test_rows` | 평가에 사용된 Validation 데이터 수 |
| `audio_indexed_files` | 로컬에 마운트된 오디오 수 (S3 스트리밍 시 0) |
| `backbone_model` | 사용된 backbone 모델명 |
| `probe_input_dim` | backbone 출력 차원 (1024) |
| `use_lora` | LoRA 사용 여부 (자동 감지) |
| `base_loss` | 랜덤 probe loss (기준선) |
| `finetuned_loss` | 학습된 모델 loss (**낮을수록 좋음**) |
| `improvement_abs` | 절대 개선량 |
| `improvement_pct` | 개선율 (%) → **품질 게이트에서 사용** |

### 출력 위치

```
s3://aws-s3-speechpt1/pipeline/ae/{timestamp}/evaluation/
└── eval_result.json
```

### PropertyFile 등록

eval_result.json은 SageMaker **PropertyFile**로 등록되어,
파이프라인 내에서 JSON 필드를 직접 참조할 수 있다.

```python
eval_report = PropertyFile(
    name="EvalReport",
    output_name="evaluation",      # ProcessingOutput의 output_name과 일치
    path="eval_result.json",       # 출력 디렉토리 내 파일명
)
```

---

## 하이퍼파라미터 (pipeline_ae.py에서 전달)

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| `--input-dir` | `/opt/ml/processing/input/data/` | JSONL이 마운트된 경로 |
| `--model-local-dir` | `/opt/ml/processing/input/model/` | model.tar.gz가 마운트된 경로 |
| `--output-dir` | `/opt/ml/processing/output/` | eval_result.json 출력 경로 |
| `--eval-file` | `eval_validation.jsonl` | 평가할 JSONL 파일명 |
| `--audio-s3` | Validation 오디오 S3 URI | S3 스트리밍 경로 (Training이 아닌 **Validation**) |
| `--model` | kresnik/wav2vec2-large-xlsr-korean | backbone 모델명 |
| `--chunk-sec` | 20 | 오디오 최대 길이 (초) |
| `--batch-size` | 2 | 배치 크기 (GPU 메모리 고려) |

---

## 관련 소스 파일

| 파일 | 역할 |
|------|------|
| `speechpt/training/ae_probe_eval.py` | 평가 메인 로직 |
| `speechpt/training/ae_probe_train.py` | AEProbe, AudioDataset 등 공유 클래스 |

---

## 로그 예시

```json
{"eval_file": "eval_validation.jsonl"}
{"lora_adapter_loaded": true, "lora_dir": "/tmp/ae_eval_model/lora_adapter"}
{"num_test_rows": 8028, "use_lora": true, "base_loss": 1.724, "finetuned_loss": 0.408, "improvement_pct": 76.36}
```

---

---

# Step 4. 품질 게이트 (AE-QualityGate)

## 목적

평가 결과가 **품질 기준을 충족하는지 자동 판정**하여,
기준 미달 모델이 레지스트리에 등록되는 것을 방지한다.

---

## 동작

SageMaker **ConditionStep**으로 구현. 별도 인스턴스를 사용하지 않는다.

```python
condition = ConditionGreaterThanOrEqualTo(
    left=JsonGet(
        step_name="AE-Evaluate",
        property_file=eval_report,
        json_path="improvement_pct",       # eval_result.json에서 추출
    ),
    right=param_quality_threshold,          # 기본값 50.0
)
```

### 판정 로직

```
eval_result.json의 improvement_pct >= QualityThreshold ?

  Yes (예: 76.36 >= 50.0)
    └── Step 5 (모델 등록) 실행

  No (예: 23.5 < 50.0)
    └── 파이프라인 종료 (등록 스킵, 실패가 아닌 정상 종료)
```

### 파라미터

| 파라미터 | 기본값 | CLI 인자 |
|----------|--------|---------|
| `QualityThreshold` | 50.0 (%) | `--quality-threshold 70` |

- 50%: "랜덤 probe 대비 loss가 절반 이상 개선" → 학습이 의미 있게 됐다는 최소 기준
- 엄격하게 하려면 70~80%, 느슨하게 하려면 30~40%로 조정

---

---

# Step 5. 모델 등록 (AE-RegisterModel)

## 목적

품질 게이트를 통과한 모델을 **SageMaker Model Registry**에 등록하여
버전 관리 및 배포 준비를 자동화한다.

---

## 동작

SageMaker **ModelStep**으로 구현. 별도 인스턴스를 사용하지 않는다.

```python
model = Model(
    image_uri=IMAGE_URI,
    model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
    role=ROLE,
)

step_register = ModelStep(
    name="AE-RegisterModel",
    step_args=model.register(
        content_types=["application/json"],
        response_types=["application/json"],
        inference_instances=["ml.t2.medium", "ml.m5.large"],
        transform_instances=["ml.m5.large"],
        model_package_group_name="SpeechPT-AE-Models",
        approval_status=param_approval_status,
    ),
)
```

### 등록 정보

| 항목 | 값 |
|------|-----|
| ModelPackageGroup | `SpeechPT-AE-Models` |
| model_data | Step 2의 model.tar.gz S3 경로 (자동 참조) |
| image_uri | 커스텀 Docker 이미지 |
| 승인 상태 | `PendingManualApproval` (기본) 또는 `Approved` |

### 승인 상태

| 상태 | 의미 | CLI 인자 |
|------|------|---------|
| `PendingManualApproval` | 등록만 하고 배포 대기. 사람이 콘솔에서 Approve 해야 배포 가능 | 기본값 |
| `Approved` | 등록 즉시 배포 가능 상태 | `--approval-status Approved` |

### 등록된 모델 확인

```bash
aws sagemaker list-model-packages \
  --model-package-group-name SpeechPT-AE-Models \
  --region ap-northeast-2
```

### 사전 준비 (최초 1회)

ModelPackageGroup이 없으면 등록 Step이 실패한다. 최초 실행 전 생성 필요:

```bash
aws sagemaker create-model-package-group \
  --model-package-group-name SpeechPT-AE-Models \
  --region ap-northeast-2
```

---

---

# Step 3~5 전체 흐름 요약

```
Step 2 완료 (model.tar.gz 생성)
  │
  ▼
Step 3: 평가 (AE-Evaluate)  ─── ml.g5.xlarge, ~30분
  │  eval_validation.jsonl (Validation 8,028건) + Validation WAV
  │  base_loss (랜덤) vs finetuned_loss (학습된 모델)
  │  → eval_result.json 출력
  │
  ▼
Step 4: 품질 게이트 (AE-QualityGate)  ─── 인스턴스 없음, 즉시 판정
  │  improvement_pct >= 50% ?
  │
  ├── Yes ──▶ Step 5: 모델 등록 (AE-RegisterModel)  ─── 인스턴스 없음
  │             model.tar.gz를 SpeechPT-AE-Models에 등록
  │             승인 상태: PendingManualApproval
  │
  └── No ───▶ 파이프라인 정상 종료 (등록 스킵)
```

### 이전 학습 결과 참고

| 날짜 | 방식 | 데이터 | 평가 데이터 | finetuned_loss | improvement_pct | 게이트 통과? |
|------|------|--------|-----------|----------------|-----------------|-------------|
| 4/8 | Linear Probing | ~6,500건 | Training 10% split | 0.412 | 76.1% | 통과 |
| 4/24 | Linear Probing | ~6,500건 | Training 10% split | 0.408 | 76.4% | 통과 |

기존 결과 기준으로 improvement_pct가 76%대이므로 기본 임계값 50%는 무난히 통과.
Validation 데이터(8,028건)를 사용하면 기존보다 더 신뢰할 수 있는 평가 수치가 나올 것으로 예상.
