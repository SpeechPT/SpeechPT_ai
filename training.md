# Step 2. 학습 (AE-Train) 상세

## 개요

학습 Step은 전처리에서 생성된 JSONL + S3의 WAV 오디오를 읽어서
**wav2vec2 backbone → AEProbe** 구조를 학습하고, 모델 artifact를 출력한다.

두 가지 학습 모드를 지원한다:

```
모드 A: Linear Probing (기본)
┌──────────────────────────────────────────────────────────────────────┐
│  WAV → Wav2Vec2Processor → wav2vec2 (frozen) → pooled → AEProbe → 5점수 │
│                             3.15억 params        mean     학습 대상     │
│                             (전부 고정)           pool     (20만)       │
└──────────────────────────────────────────────────────────────────────┘

모드 B: LoRA 파인튜닝 (--use-lora)
┌──────────────────────────────────────────────────────────────────────┐
│  WAV → Wav2Vec2Processor → wav2vec2 + LoRA → pooled → AEProbe → 5점수  │
│                             원본 고정           mean     학습 대상     │
│                             LoRA 학습 (~300만)   pool     (20만)       │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 실행 환경

| 항목 | 값 |
|------|-----|
| SageMaker Job 유형 | **Training Job** |
| 인스턴스 | `ml.g5.2xlarge` (A10G 24GB GPU, 8vCPU, 32GB RAM) |
| 프레임워크 | PyTorch 2.1, Python 3.10 (SageMaker 프레임워크 이미지) |
| 엔트리포인트 | `train.py` → `ae_probe_train.py`로 라우팅 |
| 입력 모드 | FastFile (S3에서 필요할 때 스트리밍) |
| 최대 실행 시간 | 24시간 |
| 체크포인트 | S3에 주기적 동기화 |

### 라우팅 구조

```
train.py (엔트리포인트)
  └── "--output-s3-uri" 인자 없음 → ae_probe_train.py의 main() 호출
```

---

## 입력 데이터

### Step 1에서 받는 JSONL

```
/opt/ml/input/data/training/     ← S3에서 FastFile로 마운트
├── train.jsonl                   (학습용, ~44,000건)
└── valid.jsonl                   (검증용, ~5,500건)
```

각 행의 형식:
```json
{"audio_path": "audio/ckmk_a_ard_f_e_101874.wav", "speech_rate": 1.087, "silence_ratio": 0.183, "energy_drop": 0, "pitch_shift": 1, "overall_delivery": 0.742}
```

### S3 WAV 오디오

```
/opt/ml/input/data/audio/        ← S3에서 FastFile로 마운트
└── (Training WAV 파일들)
```

FastFile 모드에서는 한글 경로의 FUSE 버그로 직접 읽기가 실패할 수 있다.
이 경우 boto3로 S3에서 직접 스트리밍하는 fallback이 자동 동작한다.

---

## 처리 흐름 상세

### 전체 흐름도

```
main()
  │
  ├── 1. 데이터 로딩
  │     ├── train.jsonl → train_rows (Sample 리스트)
  │     ├── valid.jsonl → valid_rows
  │     └── S3 오디오 인덱스 구축 (FastFile 버그 우회)
  │
  ├── 2. DataLoader 생성
  │     ├── AudioDataset: WAV 로드 + chunk + 타겟 텐서
  │     └── collate_skip_none: 로드 실패 샘플 제거
  │
  ├── 3. 모델 초기화
  │     ├── Wav2Vec2Processor.from_pretrained()
  │     ├── Wav2Vec2Model.from_pretrained()
  │     ├── (LoRA) get_peft_model(backbone, lora_config)
  │     └── AEProbe(768 → 256 → 5)
  │
  ├── 4. 체크포인트 복원 (있으면)
  │
  ├── 5. 학습 루프 (epochs)
  │     ├── train_epoch() → train_loss
  │     ├── eval_epoch()  → valid_loss
  │     ├── 체크포인트 저장 (매 epoch)
  │     └── best model 저장 (valid_loss 갱신 시)
  │
  └── 6. 메타 정보 저장 (meta.pt)
```

---

### 1. 데이터 로딩

#### JSONL → Sample 변환

```python
@dataclass
class Sample:
    audio_path: str          # "audio/ckmk_a_ard_f_e_101874.wav"
    speech_rate: float       # 1.087
    silence_ratio: float     # 0.183
    energy_drop: float       # 0
    pitch_shift: float       # 1
    overall_delivery: float  # 0.742
```

#### S3 오디오 인덱스 구축

FastFile 모드에서 `rglob("*.wav")`가 0개를 반환하면, S3 오브젝트 목록을 직접 조회하여
`{파일명: FastFile 마운트 경로}` 매핑을 구축한다.

```
S3: s3://bucket/01.원천데이터/ckmk_a_ard_f_e_101874.wav
  → 인덱스: {"ckmk_a_ard_f_e_101874.wav": "/opt/ml/input/data/audio/ckmk_a_ard_f_e_101874.wav"}
```

---

### 2. AudioDataset

각 샘플에 대해 WAV 로딩 → chunk → 타겟 텐서 변환을 수행한다.

```
AudioDataset.__getitem__(idx)
  │
  ├── resolve_audio_path()
  │     ├── audio_dir에서 직접 찾기
  │     ├── audio_index에서 basename으로 찾기
  │     └── rglob fallback
  │
  ├── WAV 로딩 시도
  │     ├── 1차: librosa.load(local_path, sr=16000)
  │     └── 2차 (실패 시): S3 직접 스트리밍 → BytesIO → librosa.load
  │           └── 이것도 실패 → None 반환 (배치에서 제외)
  │
  ├── 오디오 길이 처리
  │     ├── chunk_len 이상: 앞부분 chunk_len만큼 자르기
  │     └── chunk_len 미만: 뒤에 zero padding
  │
  └── 반환: (wav_array, target_tensor)
            wav: [chunk_len] 크기의 numpy array
            target: [speech_rate, silence_ratio, energy_drop, pitch_shift, overall_delivery]
```

**chunk_sec 설정:**
- 파이프라인 기본값: `30초` × 16000Hz = **480,000 샘플**
- 30초보다 긴 오디오는 앞 30초만 사용
- 30초보다 짧은 오디오는 뒤에 0을 채움

**collate_skip_none:**
- WAV 로딩에 실패한 샘플(None)을 배치에서 제거
- 배치 전체가 None이면 배치 자체를 None으로 반환 → 학습 루프에서 스킵

---

### 3. 모델 구조

#### wav2vec2 backbone

```
kresnik/wav2vec2-large-xlsr-korean
  ├── Feature Extractor (CNN): raw wav → feature frames
  └── Transformer Encoder (24 layers): feature frames → hidden states
      └── hidden_size = 1024 (xlsr-large)
      └── 총 파라미터: ~3.15억 개
```

입력: raw waveform (16kHz)
출력: `last_hidden_state` shape = `[batch, seq_len, 1024]`

#### Mean Pooling

```python
pooled = hidden.mean(dim=1)   # [batch, seq_len, 1024] → [batch, 1024]
```

시퀀스 차원을 평균하여 고정 크기 벡터로 변환.

#### AEProbe

```
입력 (1024) → Linear(1024, 256) → ReLU → Dropout(0.1) → Linear(256, 5) → 출력 (5)
```

| 레이어 | 입력 → 출력 | 파라미터 수 |
|--------|------------|-----------|
| fc1 | 1024 → 256 | 262,400 |
| fc2 | 256 → 5 | 1,285 |
| **합계** | | **~263,685** |

출력 5개:
```
preds[:, 0] = speech_rate        (회귀)
preds[:, 1] = silence_ratio      (회귀)
preds[:, 2] = energy_drop        (분류, logit)
preds[:, 3] = pitch_shift        (분류, logit)
preds[:, 4] = overall_delivery   (회귀)
```

---

### 4. LoRA 설정

LoRA 모드에서는 wav2vec2의 Transformer 레이어에 경량 어댑터를 삽입한다.

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,                              # rank (저차원 분해 크기)
    lora_alpha=32,                     # 스케일링 팩터
    lora_dropout=0.05,                 # dropout
    target_modules=["q_proj", "v_proj"],  # Attention의 Q, V 행렬에 적용
)
backbone = get_peft_model(backbone, lora_config)
```

**LoRA 수학적 원리:**
```
원래:  output = W × input              (W: 1024×1024)
LoRA:  output = W × input + (B × A) × input
       A: 1024 × 16  (down projection)
       B: 16 × 1024  (up projection)
       → 학습 파라미터: 16×1024 + 16×1024 = 32,768 per module
```

| 항목 | 값 |
|------|-----|
| 적용 대상 | 24 Transformer 레이어 × 2 모듈 (q_proj, v_proj) = 48개 |
| 모듈당 파라미터 | 2 × r × hidden_size = 2 × 16 × 1024 = 32,768 |
| **LoRA 총 파라미터** | 48 × 32,768 = **~157만 개** (r=16 기준) |
| 원본 파라미터 | 3.15억 개 (전부 frozen) |
| 학습 비율 | ~0.5% |

**Linear Probing vs LoRA 비교:**

| | Linear Probing | LoRA (r=16) |
|---|---|---|
| backbone | 완전 frozen | 원본 frozen + LoRA 어댑터 학습 |
| 학습 파라미터 | ~26만 (probe만) | ~183만 (probe 26만 + LoRA 157만) |
| backbone.train() | 호출 안 함 | 호출함 (dropout 등 활성화) |
| torch.no_grad() | backbone forward 시 사용 | 사용 안 함 (gradient 필요) |
| 학습 속도 | 빠름 | ~2배 느림 |
| 권장 LR | 1e-3 | 1e-4 |

---

### 5. 손실 함수

5개 출력을 **회귀(3개) + 분류(2개)**로 나누어 손실을 계산한다.

```python
# 회귀 타겟: speech_rate(0), silence_ratio(1), overall_delivery(4)
reg_pred = preds[:, [0, 1, 4]]
reg_true = targets[:, [0, 1, 4]]

# 분류 타겟: energy_drop(2), pitch_shift(3)
cls_pred = preds[:, [2, 3]]
cls_true = targets[:, [2, 3]]

loss = MSELoss(reg_pred, reg_true) + BCEWithLogitsLoss(cls_pred, cls_true)
```

| 손실 | 대상 | 설명 |
|------|------|------|
| `MSE Loss` | speech_rate, silence_ratio, overall_delivery | 연속값 회귀. 예측값과 실제값의 차이 제곱 평균 |
| `BCE with Logits Loss` | energy_drop, pitch_shift | 이진 분류. 시그모이드 + 크로스엔트로피 |

두 손실을 **동일 가중치(1:1)**로 합산한다.

---

### 6. 학습 루프

```
for epoch in 1..epochs:
  │
  ├── train_epoch()
  │     ├── probe.train()
  │     ├── (LoRA) backbone.train()
  │     └── for batch in train_loader:
  │           ├── WAV → Processor → backbone → pooled
  │           ├── pooled → probe → preds
  │           ├── loss = MSE + BCE
  │           ├── optimizer.zero_grad()
  │           ├── loss.backward()
  │           └── optimizer.step()
  │
  ├── eval_epoch()
  │     ├── probe.eval()
  │     ├── (LoRA) backbone.eval()
  │     └── with torch.no_grad():
  │           └── 동일 forward, loss 계산만 (backward 없음)
  │
  ├── 로그 출력
  │     └── {"epoch": 3, "train_loss": 0.542, "valid_loss": 0.489}
  │
  ├── 체크포인트 저장 (매 epoch)
  │     └── checkpoint_dir/ae_probe_latest.pt
  │
  └── best model 저장 (valid_loss < best_loss 일 때)
        ├── output/ae_probe.pt        ← probe 가중치
        ├── (LoRA) output/lora_adapter/  ← LoRA 어댑터
        └── checkpoint_dir/ae_probe_best.pt
```

**Optimizer:** Adam

| 모드 | 학습 대상 | 기본 LR |
|------|----------|---------|
| Linear Probing | `probe.parameters()` | 1e-3 |
| LoRA | `probe.parameters() + backbone.parameters()` (requires_grad만) | 1e-4 |

---

### 7. 체크포인트와 복원

**체크포인트 구조:**
```python
{
    "epoch": 5,
    "best_loss": 0.412,
    "model_state_dict": probe.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
}
```

**복원 우선순위:**
1. `--resume-from` 인자로 지정된 경로
2. `checkpoint_dir/ae_probe_latest.pt` (SageMaker Spot 인스턴스 복구용)

복원 시 `start_epoch`와 `best_loss`를 이어받아 학습을 계속한다.

**SageMaker 체크포인트:**
- `checkpoint_s3_uri`가 설정되면 SageMaker가 `/opt/ml/checkpoints/`를 S3에 주기적으로 동기화
- Spot 인스턴스 중단 후 재시작 시 자동 복구

---

## 출력 (model artifact)

학습 완료 후 `/opt/ml/model/` 디렉토리의 파일이 SageMaker에 의해 **model.tar.gz**로 자동 패키징된다.

### Linear Probing 모드

```
model.tar.gz
├── ae_probe.pt     ← probe 가중치 (~800KB)
└── meta.pt         ← 메타 정보
```

### LoRA 모드

```
model.tar.gz
├── ae_probe.pt     ← probe 가중치 (~800KB)
├── meta.pt         ← 메타 정보 (LoRA 설정 포함)
└── lora_adapter/   ← LoRA 어댑터 (~12MB)
    ├── adapter_config.json
    └── adapter_model.safetensors
```

### meta.pt 내용

```python
# Linear Probing
{"model": "kresnik/wav2vec2-large-xlsr-korean", "sample_rate": 16000, "chunk_sec": 30, "dropout": 0.1, "use_lora": False}

# LoRA
{"model": "kresnik/wav2vec2-large-xlsr-korean", "sample_rate": 16000, "chunk_sec": 30, "dropout": 0.1, "use_lora": True, "lora_r": 16, "lora_alpha": 32, "lora_dropout": 0.05}
```

### 출력 위치

```
s3://aws-s3-speechpt1/pipeline/ae/{timestamp}/models/
└── {job-name}/output/model.tar.gz
```

---

## 하이퍼파라미터 (pipeline_ae.py에서 전달)

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `epochs` | 10 | 학습 epoch 수 |
| `lr` | 1e-3 (LP) / 1e-4 (LoRA) | learning rate |
| `batch-size` | 8 | 배치 크기 |
| `model` | kresnik/wav2vec2-large-xlsr-korean | 백본 모델 |
| `chunk-sec` | 30 | 오디오 최대 길이 (초) |
| `audio-s3` | Training 오디오 S3 URI | S3 스트리밍 fallback 경로 |
| `use-lora` | false / true | LoRA 활성화 |
| `lora-r` | 16 | LoRA rank |

---

## 관련 소스 파일

| 파일 | 역할 |
|------|------|
| `speechpt/training/train.py` | 엔트리포인트 라우터 |
| `speechpt/training/ae_probe_train.py` | 학습 메인 로직 (모델, 데이터, 학습 루프) |

---

## 로그 예시

CloudWatch에서 확인할 수 있는 학습 로그:

```json
{"train_path": "/opt/ml/input/data/training/train.jsonl", "valid_path": "/opt/ml/input/data/training/valid.jsonl"}
{"audio_dir": "/opt/ml/input/data/audio", "audio_indexed_files": 0}
{"message": "audio_index_empty_building_from_s3", "audio_s3": "s3://aws-s3-speechpt1/datasets/raws/Training/01.원천데이터/"}
{"s3_audio_index_built": 46958, "audio_s3_uri": "s3://..."}
{"lora_enabled": true, "lora_r": 16, "lora_trainable_params": 1572864, "total_params": 315473920}
{"probe_input_dim": 1024, "use_lora": true}
{"epoch": 1, "train_loss": 1.234, "valid_loss": 1.089}
{"epoch": 2, "train_loss": 0.876, "valid_loss": 0.723}
{"epoch": 3, "train_loss": 0.654, "valid_loss": 0.589}
...
{"epoch": 10, "train_loss": 0.412, "valid_loss": 0.408}
```

**학습 진행 판단:**
- `valid_loss`가 epoch마다 감소하면 정상
- `train_loss`만 감소하고 `valid_loss`가 증가하면 과적합 → epoch 줄이거나 LR 낮추기
- best model은 `valid_loss`가 가장 낮은 epoch의 가중치
