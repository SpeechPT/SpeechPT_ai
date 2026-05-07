# Step 1. 전처리 (AE-Preprocess) 상세

## 개요

전처리 Step은 S3에 저장된 **라벨 JSON + WAV 오디오**를 읽어서
학습에 필요한 **5개 점수(타겟)**를 계산하고, JSONL 파일로 출력한다.

```
입력                                     처리                                      출력
┌─────────────────────┐                                                      ┌──────────────────────┐
│ Training 라벨 JSON   │─┐                                                    │ train.jsonl (90%)     │
│ (68,078건)          │ │    ┌──────────────────────────────┐                │ valid.jsonl (10%)     │
│                     │ ├──▶│ 라벨 파싱 → 음향 분석 → 점수 계산 │──────────────▶│ all.jsonl   (전체)    │
│ Training WAV 오디오  │ │    └──────────────────────────────┘                │                      │
│ (68,078개)          │─┘                                                    │                      │
├─────────────────────┤                                                      ├──────────────────────┤
│ Validation 라벨 JSON │─┐    ┌──────────────────────────────┐                │ eval_validation.jsonl │
│ (8,028건)           │ ├──▶│ 동일 처리 (분할 없이 전체 출력)  │──────────────▶│ (8,028건, 분할 없음)   │
│ Validation WAV 오디오│─┘    └──────────────────────────────┘                │                      │
│ (8,028개)           │                                                      └──────────────────────┘
└─────────────────────┘
```

---

## 실행 환경

| 항목 | 값 |
|------|-----|
| SageMaker Job 유형 | **Training Job** (Processing Job에 ml.c5 할당량이 없으므로) |
| 인스턴스 | `ml.c5.4xlarge` (16 vCPU, 32GB RAM) |
| Docker 이미지 | `242071452299.dkr.ecr.ap-northeast-2.amazonaws.com/speechpt-training:latest` |
| 엔트리포인트 | `train.py` → `ae_preprocess_trainjob.py`로 라우팅 |
| 볼륨 크기 | 50GB |
| 최대 실행 시간 | 24시간 |

### 라우팅 구조

```
train.py (엔트리포인트)
  └── "--output-s3-uri" 인자 감지 → ae_preprocess_trainjob.py의 main() 호출
```

`train.py`는 인자에 `--output-s3-uri` 또는 `--output-dir`이 포함되어 있으면 전처리 모드로 판단하고
`ae_preprocess_trainjob.py`를 실행한다. 없으면 학습 모드(`ae_probe_train.py`)로 분기한다.

---

## 입력 데이터

### 라벨 JSON 구조 (AIHub 포맷)

```json
{
  "version": "1.0",
  "dataSet": {
    "info": {
      "date": "20230116",
      "occupation": "ARD",
      "channel": "MOCK",
      "place": "ONLINE",
      "gender": "FEMALE",
      "ageRange": "35-44",
      "experience": "EXPERIENCED"
    },
    "question": {
      "raw": { "text": "디자이너로서 앞으로의 목표에 관해서 설명해 주세요", "wordCount": 6 }
    },
    "answer": {
      "raw": {
        "text": "저는 디자이너로서 어 많은 경험들을 쌓아서 ...",
        "wordCount": 63         // ← 전처리에서 사용
      }
    }
  },
  "rawDataInfo": {
    "question": {
      "duration": 7540,         // ms
      "audioPath": "/Mock/06.Design/Female/Experienced/ckmk_q_ard_f_e_101874.wav"
    },
    "answer": {
      "duration": 57970,        // ← 전처리에서 사용 (ms)
      "audioPath": "/Mock/06.Design/Female/Experienced/ckmk_a_ard_f_e_101874.wav"
                                // ← 전처리에서 사용 (WAV 파일명 추출)
    }
  }
}
```

**전처리에서 사용하는 필드:**
- `dataSet.answer.raw.wordCount` → 단어 수
- `rawDataInfo.answer.duration` → 발화 시간 (밀리초)
- `rawDataInfo.answer.audioPath` → WAV 파일명 매핑 (예: `ckmk_a_ard_f_e_101874.wav`)

### 파일명 매핑 규칙

라벨 JSON에서 WAV 파일명을 추출하는 로직 (`label_to_audio_name`):

1. `rawDataInfo.answer.audioPath` 필드가 있으면 → 해당 경로의 파일명 사용
2. 없으면 → 라벨 파일명에서 `_d_` → `_a_`로 치환, `.json` → `.wav`로 변환
   - 예: `ckmk_d_ard_f_e_101874.json` → `ckmk_a_ard_f_e_101874.wav`

### S3 경로

| 데이터 | S3 경로 |
|--------|---------|
| Training 라벨 | `s3://aws-s3-speechpt1/datasets/raws/Training/02.라벨링데이터/` |
| Training 오디오 | `s3://aws-s3-speechpt1/datasets/raws/Training/01.원천데이터/` |
| Validation 라벨 | `s3://aws-s3-speechpt1/datasets/raws/Validation/02.라벨링데이터/` |
| Validation 오디오 | `s3://aws-s3-speechpt1/datasets/raws/Validation/01.원천데이터/` |

---

## 처리 흐름 상세

### 전체 흐름도

```
main()
  │
  ├── 1. 인자 파싱
  │
  ├── 2. 로컬 파일 체크 (skip_local_loop 판정)
  │     └── 오디오 파일이 마운트 안 되었으면 → S3 직접 스트리밍 경로로 전환
  │
  ├── 3. build_rows_from_s3() ← 메인 처리 함수
  │     ├── Phase 1: 라벨 순차 수집 (S3에서 JSON 읽기)
  │     └── Phase 2: 오디오 분석 병렬 처리 (S3에서 WAV 스트리밍)
  │
  ├── 4. 데이터 분할 (9:1)
  │     └── shuffle → train / valid
  │
  ├── 5. Validation 데이터 처리 (val-labels-s3-uri가 있을 때)
  │     └── build_rows_from_s3() 다시 호출 (Validation S3 URI로)
  │
  ├── 6. JSONL 파일 저장
  │     └── train.jsonl, valid.jsonl, eval_validation.jsonl
  │
  └── 7. S3 업로드
```

---

### Phase 1: 라벨 순차 수집

S3에서 라벨 JSON을 하나씩 읽어서 `speech_rate`와 WAV 파일명을 추출한다.

```
S3 라벨 JSON 순회 (paginator)
  │
  ├── JSON 파싱 실패 → decoded_fail 카운트, 스킵
  │
  ├── WAV 파일명 추출 (label_to_audio_name)
  │
  ├── duration_ms <= 0 → row_none 카운트, 스킵
  │
  └── speech_rate 계산
        speech_rate = wordCount / (duration_ms / 1000)
        → candidates 리스트에 {wav_name, speech_rate} 추가
```

**speech_rate 계산 공식:**
```
speech_rate = wordCount / duration_sec
```
- `wordCount`: 라벨 JSON의 `dataSet.answer.raw.wordCount` (단어 수)
- `duration_sec`: `rawDataInfo.answer.duration` / 1000 (밀리초 → 초)
- 예: wordCount=63, duration=57970ms → speech_rate = 63 / 57.97 = **1.087 단어/초**

**max-files 제한:**
- `--max-files 10000` 이면 라벨 10,000개까지만 스캔 후 중단
- 0이면 전체 스캔

---

### Phase 2: 오디오 분석 병렬 처리

Phase 1에서 수집한 candidates에 대해 **ThreadPoolExecutor(16 workers)**로 병렬 음향 분석.

```
각 candidate (wav_name, speech_rate)에 대해:
  │
  ├── use_audio_features=true 이고 audio_bucket이 있으면
  │     └── derive_audio_targets_from_s3() 호출
  │           ├── S3에서 WAV를 BytesIO로 스트리밍 (다운로드 없음)
  │           ├── librosa.load(sr=16000, duration=30초)
  │           ├── silence_ratio 계산
  │           ├── energy_drop 계산
  │           ├── pitch_shift 계산
  │           └── 실패 시 None 반환 → audio_signal_fail 카운트
  │
  ├── use_audio_features=false 이면
  │     └── fallback_audio_targets() 호출
  │           └── speech_rate 기반 추정값 사용
  │
  └── overall_delivery 계산
```

---

### 음향 분석 상세 (derive_audio_targets_from_s3)

WAV 오디오에서 3가지 음향 특성을 추출한다. S3에서 직접 스트리밍으로 읽어 디스크 I/O 없이 처리.

#### 1. silence_ratio (침묵 비율) — 연속값 [0.0, 1.0]

**의미:** 전체 프레임 중 에너지가 낮은(침묵) 프레임의 비율. 높을수록 말을 안 하는 구간이 많다.

**계산 과정:**
```
1. WAV를 16kHz 모노로 로드 (최대 30초)

2. 프레이밍:
   - frame_len = 0.032초 × 16000 = 512 샘플
   - hop_len   = 0.010초 × 16000 = 160 샘플
   - 30초 오디오 → 약 2,994 프레임

3. 각 프레임의 RMS 에너지 계산:
   rms = sqrt(mean(frame²) + 1e-12)

4. dB 변환 + 정규화:
   energy_db = 20 × log10(rms + 1e-9)
   energy_db = energy_db - max(energy_db)   ← 최대값이 0dB이 되도록

5. 침묵 판정:
   silence_ratio = mean(energy_db < -40.0)
   → energy_db가 -40dB 미만인 프레임의 비율
```

**예시:**
- 2,994 프레임 중 898 프레임이 -40dB 미만 → silence_ratio = 0.30

#### 2. energy_drop (에너지 하락) — 이진값 {0, 1}

**의미:** 발화가 진행됨에 따라 에너지가 떨어지는지 여부. 1이면 "갈수록 목소리가 작아짐".

**계산 과정:**
```
1. 에너지 dB 시계열에 대해 1차 선형 회귀 (np.polyfit)
   times = [0, 0.01, 0.02, ..., 29.94]   ← 각 프레임의 시간(초)
   slope, _ = np.polyfit(times, energy_db, 1)

2. 판정:
   energy_drop = 1  if slope < -0.02
   energy_drop = 0  otherwise
```

**해석:**
- slope = -0.03: 초당 0.03dB씩 에너지 감소 → energy_drop = 1
- slope = +0.01: 에너지가 오히려 증가 → energy_drop = 0

#### 3. pitch_shift (억양 변화) — 이진값 {0, 1}

**의미:** 발화 중 목소리 높낮이(F0)의 변화가 큰지 여부. 1이면 "억양 변화가 풍부함".

**계산 과정:**
```
1. librosa.pyin으로 기본 주파수(F0) 추출
   - fmin = C2 (65Hz), fmax = C7 (2093Hz)
   - voiced_flag: 유성음 구간 표시

2. 유성음 구간의 F0만 필터링
   f0_voiced = f0[voiced_flag & ~isnan(f0)]

3. 판정:
   pitch_shift = 1  if len(f0_voiced) > 10 AND std(f0_voiced) > 80.0
   pitch_shift = 0  otherwise
```

**해석:**
- F0 표준편차 120Hz: 목소리 높낮이 변화가 큼 → pitch_shift = 1
- F0 표준편차 40Hz: 단조로운 억양 → pitch_shift = 0
- 유성음 프레임이 10개 미만이면 판정 불가 → pitch_shift = 0

---

### fallback 모드 (오디오 없이 추정)

WAV 파일을 읽을 수 없는 경우 (S3에 없거나 스트리밍 실패) `fallback_audio_targets()`로 추정값 생성.

```python
silence_ratio = max(0.05, min(0.6, 0.45 - 0.08 × speech_rate))
energy_drop = 0     # 항상 0 (추정 불가)
pitch_shift = 0     # 항상 0 (추정 불가)
```

- 빠르게 말하면 (speech_rate 높으면) → silence_ratio 낮게 추정
- 느리게 말하면 (speech_rate 낮으면) → silence_ratio 높게 추정
- energy_drop, pitch_shift는 오디오 없이 추정할 수 없으므로 항상 0

**주의:** fallback 값은 실제 음향 분석보다 부정확하므로, 학습 데이터 품질에 영향을 준다.
현재 파이프라인은 `use-audio-features: true`로 실행되어 실제 음향 분석을 사용하고,
WAV가 S3에 없는 경우에만 해당 샘플이 제외된다 (`audio_signal_fail`).

---

### overall_delivery 계산 (종합 전달력)

4개 점수를 종합하여 0~1 사이의 종합 점수를 계산한다.

```python
pace_score    = exp(-|speech_rate - 2.2| / 1.8)          # 가중치 45%
silence_score = max(0, 1 - silence_ratio / 0.45)         # 가중치 30%
stability     = 1 - 0.5 × (energy_drop + pitch_shift)    # 가중치 25%

overall_delivery = 0.45 × pace_score + 0.30 × silence_score + 0.25 × max(0, stability)
overall_delivery = clamp(0.0, 1.0)
```

| 구성 요소 | 가중치 | 의미 |
|-----------|--------|------|
| `pace_score` | 45% | speech_rate=2.2일 때 최고점(1.0). 너무 빠르거나 느리면 감소 |
| `silence_score` | 30% | silence_ratio=0이면 1.0, 0.45이면 0. 침묵이 적을수록 좋음 |
| `stability` | 25% | energy_drop=0, pitch_shift=0이면 1.0. 둘 다 1이면 0 |

**예시:**
- speech_rate=2.2, silence_ratio=0.1, energy_drop=0, pitch_shift=1
  - pace_score = 1.0
  - silence_score = 0.778
  - stability = 0.5
  - overall = 0.45×1.0 + 0.30×0.778 + 0.25×0.5 = **0.808**

---

### 데이터 분할

처리 완료된 rows를 셔플 후 9:1 비율로 분할한다.
Validation 데이터(8,028건)가 독립 평가셋으로 사용되므로 test split은 불필요하다.

```
rng = Random(seed=42)    ← 동일 seed로 재현 가능
rng.shuffle(rows)

n = len(rows)
n_train = int(n × 0.9)
n_valid = n - n_train

train_rows = rows[:n_train]
valid_rows = rows[n_train:]
```

| 분할 | 비율 | 용도 |
|------|------|------|
| `train.jsonl` | 90% | Step 2 학습 |
| `valid.jsonl` | 10% | Step 2 학습 중 best model 선택 (early stopping) |

---

### Validation 데이터 처리

Training 데이터 처리 후, `--val-labels-s3-uri`가 지정되어 있으면 Validation 데이터도 처리한다.

```
build_rows_from_s3(val_labels_s3_uri, val_audio_s3_uri)
  └── Training과 동일한 Phase 1 + Phase 2 처리
  └── 분할 없이 전체를 eval_validation.jsonl로 출력
```

- Training과 **완전히 독립된** 데이터셋
- 분할하지 않고 **전체 8,028건**을 그대로 평가용으로 사용
- Step 3 (평가)에서 이 파일을 읽어 모델의 일반화 성능을 측정

---

## 출력 형식

### JSONL 레코드 구조

각 행은 하나의 오디오 샘플에 대한 5개 타겟 점수를 포함한다.

```json
{
  "audio_path": "audio/ckmk_a_ard_f_e_101874.wav",
  "speech_rate": 1.087,
  "silence_ratio": 0.183,
  "energy_drop": 0,
  "pitch_shift": 1,
  "overall_delivery": 0.742
}
```

| 필드 | 타입 | 범위 | 설명 |
|------|------|------|------|
| `audio_path` | string | - | WAV 파일 상대 경로 (`audio/{파일명}`) |
| `speech_rate` | float | 0~ | 발화 속도 (단어/초). 일반적으로 1.0~4.0 |
| `silence_ratio` | float | [0, 1] | 침묵 비율 |
| `energy_drop` | int | {0, 1} | 에너지 하락 여부 |
| `pitch_shift` | int | {0, 1} | 억양 변화 여부 |
| `overall_delivery` | float | [0, 1] | 종합 전달력 점수 |

### 출력 파일 목록

| 파일 | 설명 | 다음 Step에서 사용 |
|------|------|-------------------|
| `train.jsonl` | Training 데이터 90% | Step 2 학습 |
| `valid.jsonl` | Training 데이터 10% | Step 2 학습 (validation) |
| `all.jsonl` | Training 데이터 전체 | 참고용 |
| `eval_validation.jsonl` | Validation 데이터 전체 | **Step 3 평가** |

### 출력 위치

```
s3://aws-s3-speechpt1/pipeline/ae/{timestamp}/processed/
├── train.jsonl
├── valid.jsonl
├── all.jsonl
└── eval_validation.jsonl
```

---

## 실패 케이스 및 카운터

전처리 중 발생하는 실패를 추적하는 카운터들:

| 카운터 | 의미 | 원인 |
|--------|------|------|
| `decoded_fail` | JSON 파싱 실패 | 파일 인코딩 오류, 손상된 JSON |
| `row_none` | duration_ms <= 0 | 라벨에 duration 정보 없음 |
| `filtered_missing_audio` | 매칭 WAV 없음 | S3에 해당 WAV 파일 미존재 |
| `audio_signal_fail` | 오디오 분석 실패 | WAV 읽기/스트리밍 실패, 빈 오디오 |

**데이터 손실 흐름:**
```
라벨 JSON 전체  →  [JSON 파싱 실패 제외]  →  [duration 없음 제외]  →  [WAV 스트리밍 실패 제외]  →  최종 rows
  68,078건           -decoded_fail            -row_none              -audio_signal_fail          ~55,000건
```

**알려진 이슈:** bm/sm 카테고리의 WAV 파일이 S3에 없으면 ~34% 데이터 손실 발생.
누락 WAV를 S3에 업로드하면 해결됨.

---

## 하이퍼파라미터 (pipeline_ae.py에서 전달)

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| `output-s3-uri` | `s3://.../processed/` | JSONL 업로드 위치 |
| `labels-s3-uri` | Training 라벨 S3 URI | 라벨 JSON 읽기 경로 |
| `audio-s3-uri` | Training 오디오 S3 URI | WAV 스트리밍 경로 |
| `use-audio-features` | `true` | 실제 음향 분석 수행 여부 |
| `s3-rescue-on-empty` | `true` | 로컬 오디오 없으면 S3 스트리밍 폴백 |
| `max-files` | 파이프라인 파라미터 | 라벨 스캔 제한 (0=무제한) |
| `val-labels-s3-uri` | Validation 라벨 S3 URI | Validation 라벨 경로 |
| `val-audio-s3-uri` | Validation 오디오 S3 URI | Validation WAV 경로 |

---

## 관련 소스 파일

| 파일 | 역할 |
|------|------|
| `speechpt/training/train.py` | 엔트리포인트 라우터 (전처리/학습 분기) |
| `speechpt/training/ae_preprocess_trainjob.py` | 전처리 메인 로직 |
| `speechpt/training/prepare_ae_from_raws.py` | 공통 유틸 (음향 분석, 점수 계산, JSONL 쓰기) |

---

## 로그 예시

CloudWatch에서 확인할 수 있는 전처리 로그:

```json
{"stage": "pre_fallback_summary", "label_files": 84134, "audio_files": 0, "prepared_rows": 0, "skip_local_loop": true}
{"stage": "label_scan_done", "candidates": 68078}
{"stage": "audio_analysis_progress", "done": 500, "total": 68078, "rows_ok": 437}
{"stage": "audio_analysis_progress", "done": 1000, "total": 68078, "rows_ok": 892}
...
{"stage": "s3_fallback_summary", "prepared_rows": 55234, "stats": {"audio_signal_fail": 12844}}
{"stage": "validation_processing_start", "val_labels_s3_uri": "s3://..."}
{"stage": "validation_processing_done", "eval_validation_rows": 7821}
{"counts": {"train": 49710, "valid": 5524, "test": 0, "eval_validation": 7821}}
```
